/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "call_config.h"
#include "orchestrator.h"
#include "recovery_coordinator.h"
#include "ring.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"
#include "task_args.h"

namespace {

TEST(RecoveryConfigTest, SchedulerConfigDefaultsOperatorRecoveryOff) {
    Scheduler::Config config{};
    EXPECT_FALSE(config.operator_recovery_enabled);
}

Tensor wire_tensor(uint64_t buffer_id) {
    Tensor t{};
    t.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    t.buffer.address_space = static_cast<uint8_t>(AddressSpace::HOST);
    t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    t.buffer.nbytes = 1;
    t.buffer.identity.buffer_id = buffer_id;
    t.buffer.identity.generation = 1;
    const std::string shm_name = "psm_recovery_" + std::to_string(buffer_id);
    t.buffer.body_len = static_cast<uint16_t>(shm_name.size());
    std::memcpy(t.buffer.body, shm_name.data(), shm_name.size());
    t.ndims = 1;
    t.shapes[0] = 1;
    t.strides[0] = 1;
    t.dtype = DataType::UINT8;
    return t;
}

TaskArgs single_tensor_args(uint64_t buffer_id, TensorArgType tag = TensorArgType::OUTPUT) {
    TaskArgs args;
    args.add_tensor(wire_tensor(buffer_id), tag);
    return args;
}

CallableIdentity chip_callable(uint8_t seed) {
    CallableIdentity callable;
    callable.digest.fill(seed);
    callable.kind = CallableKind::CHIP_CALLABLE;
    callable.target_namespace = TargetNamespace::LOCAL_CHIP;
    return callable;
}

CallableIdentity non_local_chip_callable(uint8_t seed) {
    CallableIdentity callable = chip_callable(seed);
    callable.target_namespace = TargetNamespace::LOCAL_PYTHON;
    return callable;
}

class RecoveryEndpoint final : public WorkerEndpoint {
public:
    explicit RecoveryEndpoint(int32_t worker_id = 0) {
        caps_.worker_id = worker_id;
        caps_.max_inflight_tasks = 1;
        caps_.supports_frame_staging = false;
    }

    const WorkerEndpointCaps &caps() const override { return caps_; }

    void submit_progress(Ring *, const WorkerDispatch &dispatch) override {
        std::lock_guard<std::mutex> lk(mu_);
        dispatch_ = dispatch;
        submitted_ = true;
        outstanding_ = true;
        cv_.notify_all();
    }

    bool poll_progress(WorkerEndpointProgress &progress) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (events_.empty()) return false;
        progress = std::move(events_.front());
        events_.pop_front();
        if (progress.kind == WorkerProgressKind::COMPLETED) {
            outstanding_ = false;
            completed_ = true;
            cv_.notify_all();
        }
        return true;
    }

    bool activate_progress(RunId) override { return true; }

    uint64_t endpoint_generation() const override { return generation_.load(std::memory_order_acquire); }

    void release_faulted_endpoint(const EndpointRecoveryRequest &request) override {
        if (request.recovery_id == 0 || request.expected_endpoint_generation != endpoint_generation()) {
            throw std::runtime_error("stale release transaction");
        }
        releases_.fetch_add(1, std::memory_order_relaxed);
    }
    uint32_t releases() const { return releases_.load(std::memory_order_relaxed); }

    void request_progress_stop() noexcept override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!outstanding_) return;
        WorkerEndpointProgress progress;
        progress.kind = WorkerProgressKind::COMPLETED;
        progress.dispatch = dispatch_;
        progress.completion.task_slot = dispatch_.task_slot;
        progress.completion.group_index = dispatch_.group_index;
        progress.completion.outcome = EndpointOutcome::ENDPOINT_FAILURE;
        progress.completion.error_message = "recovery test endpoint stopped";
        events_.push_back(std::move(progress));
    }

    bool wait_submitted(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout, [this] {
            return submitted_;
        });
    }

    bool wait_completed() {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, std::chrono::seconds(3), [this] {
            return completed_;
        });
    }

    void emit_completion(
        EndpointOutcome outcome, const NativeExecutionFault &fault, bool launch_accepted, const std::string &message
    ) {
        std::lock_guard<std::mutex> lk(mu_);
        WorkerEndpointProgress progress;
        progress.kind = WorkerProgressKind::COMPLETED;
        progress.dispatch = dispatch_;
        progress.completion.task_slot = dispatch_.task_slot;
        progress.completion.group_index = dispatch_.group_index;
        progress.completion.outcome = outcome;
        progress.completion.error_message = message;
        progress.completion.launch_accepted = launch_accepted;
        progress.completion.native_execution_fault = fault;
        events_.push_back(std::move(progress));
    }

    void fail_native(const NativeExecutionFault &fault, bool launch_accepted) {
        emit_completion(EndpointOutcome::TASK_FAILURE, fault, launch_accepted, "injected native execution fault");
    }

    void succeed() { emit_completion(EndpointOutcome::SUCCESS, NativeExecutionFault{}, true, ""); }

private:
    WorkerEndpointCaps caps_{};
    std::mutex mu_;
    std::condition_variable cv_;
    WorkerDispatch dispatch_{};
    bool submitted_{false};
    bool outstanding_{false};
    bool completed_{false};
    std::deque<WorkerEndpointProgress> events_;
    std::atomic<uint64_t> generation_{1};
    std::atomic<uint32_t> releases_{0};
};

RecoveryRequest episode(bool group = false) {
    RecoveryRequest r;
    r.ticket = {7, 3, 11, 0};
    r.failure.outcome = EndpointOutcome::TASK_FAILURE;
    r.failure.error_message = "original native fault";
    r.members.push_back({0, 1, true, r.failure});
    if (group) r.members.push_back({1, 1, false, {}});
    return r;
}
TEST(RecoveryCoordinatorTest, HoldsUntilExplicitGiveUpAndRejectsStaleTickets) {
    RecoveryCoordinator c;
    auto r = episode();
    r.ticket = c.submit(r);
    ASSERT_TRUE(c.wait(0).has_value());
    EXPECT_TRUE(c.has_work());
    RecoveryRequest out;
    std::string reason;
    EXPECT_FALSE(c.try_pop_resolution(out, reason));
    auto stale = r.ticket;
    ++stale.run_id;
    EXPECT_FALSE(c.give_up(stale, "owner gives up"));
    EXPECT_TRUE(c.give_up(r.ticket, "owner gives up"));
    EXPECT_FALSE(c.give_up(r.ticket, "duplicate"));
    EXPECT_TRUE(c.try_pop_resolution(out, reason));
    EXPECT_EQ(reason, "owner gives up");
    EXPECT_EQ(out.failure.error_message, "original native fault");
    EXPECT_FALSE(c.has_work());
}
TEST(RecoveryCoordinatorTest, BusyPeerBlocksResolutionAndCancellationWaitsForTerminal) {
    RecoveryCoordinator c;
    auto r = episode(true);
    r.ticket = c.submit(r);
    EXPECT_FALSE(c.wait(0).has_value());
    EXPECT_THROW(c.give_up(r.ticket, "unsafe"), std::logic_error);
    c.cancel_all();
    RecoveryRequest out;
    std::string reason;
    EXPECT_FALSE(c.try_pop_resolution(out, reason));
    WorkerCompletion peer;
    peer.outcome = EndpointOutcome::SUCCESS;
    c.complete_member(11, 1, peer);
    EXPECT_TRUE(c.try_pop_resolution(out, reason));
    EXPECT_TRUE(out.all_members_terminal());
    EXPECT_EQ(out.members[1].completion.outcome, EndpointOutcome::SUCCESS);
}
TEST(RecoveryCoordinatorTest, RejectsInvalidWaitAndDuplicateEpisode) {
    RecoveryCoordinator c;
    auto r = episode();
    r.ticket = c.submit(r);
    EXPECT_THROW(c.submit(r), std::logic_error);
    EXPECT_THROW(c.wait(-1), std::invalid_argument);
    EXPECT_THROW(c.wait(std::numeric_limits<double>::infinity()), std::invalid_argument);
}
TEST(EndpointRecoveryCommandTest, StaleReleaseDoesNotPublishControl) {
    std::vector<char> mailbox(MAILBOX_SIZE, 0);
    LocalMailboxEndpoint endpoint(4, mailbox.data());
    EXPECT_THROW(endpoint.release_faulted_endpoint({4, 91, 0}), std::runtime_error);
    uint64_t command = 0;
    std::memcpy(&command, mailbox.data() + MAILBOX_OFF_CALLABLE, sizeof(command));
    EXPECT_NE(command, CTRL_RELEASE_FAULTED_ENDPOINT);
}
TEST(RecoveryCoordinatorTest, AnotherWorkerTicketCannotResolveMatchingNumericIds) {
    RecoveryCoordinator one;
    RecoveryCoordinator two;
    auto first = one.submit(episode());
    auto second = two.submit(episode());
    EXPECT_EQ(first.recovery_id, second.recovery_id);
    EXPECT_NE(first.owner_id, second.owner_id);
    EXPECT_FALSE(two.give_up(first, "foreign owner"));
    EXPECT_TRUE(two.has_work());
    EXPECT_TRUE(two.give_up(second, "own decision"));
    EXPECT_TRUE(one.give_up(first, "own decision"));
}
TEST(OrchestratorRecoveryTest, GlobalFreezeBlocksNewRunAdmissionUntilThaw) {
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orch;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    ready_next.reset({0});
    orch.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next);
    orch.set_global_recovery_freeze(true);

    auto begin = std::async(std::launch::async, [&] {
        return orch.begin_run();
    });
    EXPECT_EQ(begin.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    orch.set_global_recovery_freeze(false);
    ASSERT_EQ(begin.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    RunId run = begin.get();
    orch.close_run_submission(run);
    orch.wait_run(run);
    orch.release_run(run);
    allocator.shutdown();
}

TEST(OrchestratorRecoveryTest, RecoveryHoldBlocksSubmitUntilGiveUpReleasesEpisode) {
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orch;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    ready_next.reset({0});
    orch.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next);

    RunId run = orch.begin_run();
    CallConfig config;
    auto task = orch.submit_next_level(chip_callable(1), single_tensor_args(0x1001), config, 0);
    TaskSlot ready = INVALID_SLOT;
    ASSERT_TRUE(ready_next.try_pop_single(0, run, ready));
    ASSERT_EQ(ready, task.task_slot);
    TaskSlotState *state = allocator.slot_state(task.task_slot);
    ASSERT_NE(state, nullptr);
    state->state.store(TaskState::RUNNING, std::memory_order_release);

    std::optional<uint32_t> attempt = orch.begin_task_recovery(task.task_slot, 41);
    ASSERT_TRUE(attempt.has_value());
    EXPECT_EQ(*attempt, 0u);
    EXPECT_EQ(state->state.load(std::memory_order_acquire), TaskState::RETRY_PENDING);
    EXPECT_FALSE(claim_task_failure(*state, "cancelled").has_value());

    auto blocked_submit = std::async(std::launch::async, [&] {
        try {
            (void)orch.submit_next_level(chip_callable(2), single_tensor_args(0x1002), config, 0);
            return std::string("submitted");
        } catch (const std::exception &e) {
            return std::string(e.what());
        }
    });
    EXPECT_EQ(blocked_submit.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    orch.report_task_error(task.task_slot, "recovery give up");
    ASSERT_EQ(blocked_submit.wait_for(std::chrono::seconds(1)), std::future_status::ready)
        << "committed first_error must wake a recovery-gated submitter";
    EXPECT_EQ(blocked_submit.get(), "recovery give up");
    EXPECT_TRUE(orch.finish_task_recovery(task.task_slot, 41));
    EXPECT_EQ(state->state.load(std::memory_order_acquire), TaskState::RETRY_PENDING)
        << "finishing the recovery episode must not commit FAILED itself";

    state->state.store(TaskState::FAILED, std::memory_order_release);
    ASSERT_TRUE(orch.on_consumed(task.task_slot));
    orch.close_run_submission(run);
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    orch.release_run(run);
    allocator.shutdown();
}

class SchedulerRecoveryTest : public ::testing::Test {
protected:
    TensorMap map;
    Ring ring;
    Scope scope;
    ReadyQueue sub;
    NextLevelReadyQueues next;
    Orchestrator orch;
    WorkerManager manager;
    Scheduler scheduler;
    RecoveryEndpoint *first{nullptr};
    RecoveryEndpoint *second{nullptr};
    RunId run{INVALID_RUN_ID};
    TaskHandle a{}, b{}, c{};
    uint64_t backing{0};
    Tensor resource{};
    virtual bool enabled() const { return true; }
    void SetUp() override {
        ring.init(1ULL << 20);
        auto one = std::make_unique<RecoveryEndpoint>(0);
        auto two = std::make_unique<RecoveryEndpoint>(1);
        first = one.get();
        second = two.get();
        manager.add_next_level_endpoint(std::move(one));
        manager.add_next_level_endpoint(std::move(two));
        manager.start(
            &ring,
            [this](WorkerCompletion event) {
                scheduler.worker_done(std::move(event));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            }
        );
        next.reset({0, 1});
        orch.init(&map, &ring, &scope, &sub, &next, &manager, [this] {
            scheduler.notify_ready();
        });
        Scheduler::Config cfg{};
        cfg.ring = &ring;
        cfg.ready_sub_queue = &sub;
        cfg.ready_next_level_queues = &next;
        cfg.manager = &manager;
        cfg.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        cfg.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        cfg.preparable_run_cb = [this] {
            return orch.preparable_run_id();
        };
        cfg.on_consumed_cb = [this](TaskSlot slot) {
            orch.on_consumed(slot);
        };
        cfg.on_task_failed_cb = [this](TaskSlot slot, const std::string &message) {
            orch.report_task_error(slot, message);
        };
        cfg.operator_recovery_enabled = enabled();
        cfg.begin_task_recovery_cb = [this](TaskSlot slot, uint64_t id) {
            return orch.begin_task_recovery(slot, id);
        };
        cfg.finish_task_recovery_cb = [this](TaskSlot slot, uint64_t id) {
            orch.finish_task_recovery(slot, id);
        };
        cfg.on_global_recovery_freeze_cb = [this](bool frozen) {
            orch.set_global_recovery_freeze(frozen);
        };
        cfg.recovery_graph_cb = [this](const RecoveryTicket &ticket) {
            return orch.recovery_graph(ticket);
        };
        orch.set_recovery_cancel_callback([this](RunId run_id, const std::string &reason) {
            scheduler.cancel_run_recovery(run_id, reason);
        });
        scheduler.start(cfg);
        orch.set_scheduler_loop_mutex(&scheduler.loop_mutex());
    }
    void TearDown() override {
        scheduler.stop();
        manager.stop();
        ring.shutdown();
    }
    void graph(bool group = false) {
        run = orch.begin_run();
        orch.scope_begin();
        resource = wire_tensor(42);
        resource.buffer.backend_kind = static_cast<uint8_t>(BackendKind::FORK_SHM);
        resource.buffer.nbytes = 4;
        resource.shapes[0] = 4;
        resource.buffer.body_len = 8;
        backing = orch.alloc({4}, DataType::UINT8, resource.buffer.identity);
        std::memcpy(resource.buffer.body, &backing, 8);
        *reinterpret_cast<uint32_t *>(backing) = 0x1234abcd;
        TaskArgs args;
        args.add_tensor(resource, TensorArgType::INPUT);
        args.add_scalar(42);
        CallConfig cfg;
        a = group ? orch.submit_next_level_group(chip_callable(1), {args, args}, cfg, {0, 1}) :
                    orch.submit_next_level(chip_callable(1), args, cfg, 0);
        TaskArgs successor = args;
        successor.add_dep(a);
        b = orch.submit_next_level(chip_callable(2), successor, cfg, 0);
        successor.explicit_deps_.clear();
        successor.add_dep(b);
        c = orch.submit_next_level(chip_callable(3), successor, cfg, 0);
        orch.scope_end();
        orch.close_run_submission(run);
        ASSERT_TRUE(first->wait_submitted());
        if (group) ASSERT_TRUE(second->wait_submitted());
    }
    void native_failure(RecoveryEndpoint *endpoint = nullptr) {
        NativeExecutionFault fault{};
        fault.raw_rc = -12345;
        fault.runtime_status = -9;
        (endpoint ? endpoint : first)->fail_native(fault, false);
    }
    RecoveryRequest held() {
        auto r = scheduler.wait_for_recovery(3);
        if (!r) {
            ADD_FAILURE() << "no retained episode";
            return {};
        }
        return *r;
    }
    void expect_graph(const RecoveryRequest &request) {
        const auto snapshot = scheduler.recovery_graph(request.ticket);
        EXPECT_FALSE(snapshot.run_failed);
        EXPECT_EQ(snapshot.active_recoveries, 1u);
        EXPECT_TRUE(snapshot.submission_closed);
        auto find = [&](TaskHandle h) -> const RecoveryTaskSnapshot * {
            for (const auto &item : snapshot.tasks)
                if (item.handle.task_slot == h.task_slot) return &item;
            return nullptr;
        };
        auto *aa = find(a);
        auto *bb = find(b);
        auto *cc = find(c);
        ASSERT_NE(aa, nullptr);
        ASSERT_NE(bb, nullptr);
        ASSERT_NE(cc, nullptr);
        EXPECT_EQ(aa->state, TaskState::RETRY_PENDING);
        EXPECT_EQ(bb->state, TaskState::PENDING);
        EXPECT_EQ(cc->state, TaskState::PENDING);
        EXPECT_NE(std::find(bb->producers.begin(), bb->producers.end(), a.task_slot), bb->producers.end());
        EXPECT_NE(std::find(cc->producers.begin(), cc->producers.end(), b.task_slot), cc->producers.end());
        EXPECT_EQ(aa->args[0].scalar(0), 42u);
        EXPECT_EQ(bb->args[0].tensor(0).buffer.identity, resource.buffer.identity);
        EXPECT_EQ(bb->fanin_released, 0);
        EXPECT_EQ(cc->fanin_released, 0);
        EXPECT_EQ(scheduler.dispatched_total(), request.members.size());
        EXPECT_EQ(*reinterpret_cast<uint32_t *>(backing), 0x1234abcdu);
        EXPECT_FALSE(orch.wait_run_for(run, 0.03));
        EXPECT_FALSE(orch.run_failed(run));
    }
    void give_up(const RecoveryRequest &request) {
        EXPECT_TRUE(scheduler.give_up_recovery(request.ticket, "owner rejects replay"));
        EXPECT_FALSE(scheduler.give_up_recovery(request.ticket, "duplicate"));
        EXPECT_THROW(orch.wait_run(run), std::runtime_error);
        EXPECT_TRUE(orch.run_failed(run));
        EXPECT_EQ(ring.slot_state(a.task_slot)->state.load(), TaskState::CONSUMED);
        EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::CONSUMED);
        EXPECT_EQ(ring.slot_state(c.task_slot)->state.load(), TaskState::CONSUMED);
        EXPECT_THROW(scheduler.recovery_graph(request.ticket), std::invalid_argument);
        orch.release_run(run);
    }
};
TEST_F(SchedulerRecoveryTest, HoldsOriginalGraphArgsDependenciesAndManagedBufferUntilGiveUp) {
    graph();
    native_failure();
    auto r = held();
    expect_graph(r);
    give_up(r);
    EXPECT_EQ(first->releases(), 1u);
}
TEST_F(SchedulerRecoveryTest, GroupHealthyPeerEvidenceBeforeFaultIsPreserved) {
    graph(true);
    second->succeed();
    ASSERT_TRUE(second->wait_completed());
    native_failure();
    auto r = held();
    ASSERT_EQ(r.members.size(), 2u);
    EXPECT_TRUE(r.members[1].terminal);
    EXPECT_EQ(r.members[1].completion.outcome, EndpointOutcome::SUCCESS);
    EXPECT_TRUE(r.members[1].completion.launch_accepted);
    expect_graph(r);
    give_up(r);
    EXPECT_EQ(first->releases(), 1u);
    EXPECT_EQ(second->releases(), 0u);
}
TEST_F(SchedulerRecoveryTest, GroupFaultFreezesGraphWhilePeerIsStillRunning) {
    graph(true);
    native_failure();
    EXPECT_FALSE(scheduler.wait_for_recovery(0.03).has_value());
    auto pending = scheduler.pending_recoveries();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_FALSE(pending[0].all_members_terminal());
    auto snapshot = scheduler.recovery_graph(pending[0].ticket);
    EXPECT_FALSE(snapshot.run_failed);
    EXPECT_THROW(scheduler.give_up_recovery(pending[0].ticket, "unsafe"), std::logic_error);
    second->succeed();
    auto r = held();
    expect_graph(r);
    give_up(r);
}
TEST_F(SchedulerRecoveryTest, GroupBothFaultsReleaseBothHeldChildrenExactlyOnce) {
    graph(true);
    native_failure();
    native_failure(second);
    auto r = held();
    expect_graph(r);
    give_up(r);
    EXPECT_EQ(first->releases(), 1u);
    EXPECT_EQ(second->releases(), 1u);
}
TEST_F(SchedulerRecoveryTest, StaleTicketCannotEndCurrentEpisode) {
    graph();
    native_failure();
    auto r = held();
    auto stale = r.ticket;
    ++stale.attempt;
    EXPECT_FALSE(scheduler.give_up_recovery(stale, "stale"));
    expect_graph(r);
    give_up(r);
}
TEST_F(SchedulerRecoveryTest, StopCancelsHeldEpisodeAndReleasesTheRun) {
    graph();
    native_failure();
    auto r = held();
    expect_graph(r);
    scheduler.request_stop();
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    scheduler.stop();
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, StopWithLiveGroupPeerWaitsForItsTerminalCompletion) {
    graph(true);
    native_failure();
    EXPECT_FALSE(scheduler.wait_for_recovery(0.03).has_value());
    scheduler.request_stop();
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    scheduler.stop();
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, GraphSubmissionFailureCancelsTheHeldEpisode) {
    graph();
    native_failure();
    auto request = held();
    expect_graph(request);
    orch.fail_run_submission(run, std::make_exception_ptr(std::runtime_error("graph construction cancelled")));
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    EXPECT_TRUE(scheduler.pending_recoveries().empty());
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, AnotherTerminalRunErrorCancelsTheHeldEpisode) {
    graph();
    native_failure();
    auto request = held();
    expect_graph(request);
    orch.report_task_error(b.task_slot, "terminal run error");
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    EXPECT_TRUE(scheduler.pending_recoveries().empty());
    orch.release_run(run);
}
class RecoveryDisabledTest : public SchedulerRecoveryTest {
    bool enabled() const override { return false; }
};
TEST_F(RecoveryDisabledTest, DefaultFailurePropagationRemainsTerminal) {
    graph();
    native_failure();
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    EXPECT_TRUE(scheduler.pending_recoveries().empty());
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, UnstructuredEndpointFailureIsNotHeld) {
    graph();
    first->emit_completion(EndpointOutcome::ENDPOINT_FAILURE, {}, false, "endpoint died");
    EXPECT_THROW(orch.wait_run(run), std::runtime_error);
    EXPECT_TRUE(scheduler.pending_recoveries().empty());
    orch.release_run(run);
}
}  // namespace
