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
#include "execution_control.h"
#include "orchestrator.h"
#include "recovery_coordinator.h"
#include "ring.h"
#include "scheduler.h"
#include "recovery_dispatch.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
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
        history_.push_back(dispatch);
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

    std::optional<RecoveryControlPhase> fail_phase;
    bool stale_reply{false};
    bool ignore_stop{false};
    bool stale_stop_reply{false};
    bool stop_proof{true};
    std::vector<RecoveryControlPhase> phases;
    bool wait_count(size_t count) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, std::chrono::seconds(3), [&] {
            return history_.size() >= count;
        });
    }
    std::vector<WorkerDispatch> history() {
        std::lock_guard<std::mutex> lk(mu_);
        return history_;
    }
    void control_generic(uint64_t command, const char *name, size_t size, double, const uint8_t *) override {
        if (command != CTRL_RECOVER_ENDPOINT || size != sizeof(RecoveryControlPacket))
            throw std::runtime_error("unexpected recovery command");
        const int fd = shm_open(name, O_RDWR, 0);
        if (fd < 0) throw std::runtime_error("test control shm open failed");
        void *address = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (address == MAP_FAILED) throw std::runtime_error("test control shm map failed");
        RecoveryControlPacket packet;
        std::memcpy(&packet, address, size);
        bool fail = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            phases.push_back(packet.phase);
            fail = fail_phase && *fail_phase == packet.phase;
            if (packet.phase == RecoveryControlPhase::STOP) {
                if (packet.dispatch_id != dispatch_.dispatch_id || packet.task_slot != dispatch_.task_slot ||
                    packet.group_index != dispatch_.group_index)
                    fail = true;
                packet.stop_accepted = 1;
                if (outstanding_ && events_.empty() && !ignore_stop) {
                    WorkerEndpointProgress stopped;
                    stopped.kind = WorkerProgressKind::COMPLETED;
                    stopped.dispatch = dispatch_;
                    stopped.completion.task_slot = dispatch_.task_slot;
                    stopped.completion.group_index = dispatch_.group_index;
                    stopped.completion.outcome = EndpointOutcome::TASK_FAILURE;
                    stopped.completion.error_message = "peer cooperatively stopped";
                    stopped.completion.native_execution_fault.raw_rc = 507018;
                    stopped.completion.native_execution_fault.runtime_status = -SIMPLER_ERROR_COORDINATED_STOP;
                    faulted_ = true;
                    events_.push_back(std::move(stopped));
                }
                if (stale_stop_reply) ++packet.dispatch_id;
            }
            packet.faulted = faulted_ ? 1 : 0;
            packet.execution.communication_context = 0x100;
            packet.execution.execution_context = 0x200;
            packet.execution.aicpu_stream = 0x300;
            packet.execution.aicore_stream = 0x400;
            packet.execution.quiesced = stop_proof ? 1 : 0;
            packet.execution.failed_run_epoch = 1;
            packet.execution.status = packet.phase == RecoveryControlPhase::INSPECT && faulted_ ?
                                          ExecutionRecoveryStatus::RECOVERY_REQUIRED :
                                          ExecutionRecoveryStatus::RECOVERED;
            if (packet.has_domain) {
                auto &domain = packet.domain;
                domain.recovery_id = packet.recovery_id;
                domain.allocation_id = packet.before.allocation_id;
                domain.previous_generation = packet.before.generation;
                domain.generation = packet.before.generation;
                domain.device_ctx = packet.before.device_ctx;
                domain.local_window_base = packet.before.local_window_base;
                domain.window_size = packet.before.window_size;
                domain.workspace = packet.before.workspace;
                domain.domain_preserved = 1;
                if (packet.impact == DomainRecoveryImpact::DOMAIN_TRANSPORT) {
                    domain.status = packet.phase == RecoveryControlPhase::PREPARE ? DomainRecoveryStatus::PREPARED :
                                                                                    DomainRecoveryStatus::REARMED;
                    if (packet.phase != RecoveryControlPhase::PREPARE) ++domain.generation;
                    domain.rebind_required = domain.transport_stopped = 1;
                    domain.transport_channel_count = domain.stopped_channel_count = 48;
                } else domain.status = DomainRecoveryStatus::PRESERVED;
            }
        }
        if (stale_reply && packet.phase == RecoveryControlPhase::PREPARE) ++packet.endpoint_generation;
        std::memcpy(address, &packet, size);
        munmap(address, size);
        if (fail) throw std::runtime_error("injected resource control failure");
    }

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
        faulted_ = fault.raw_rc != 0 || fault.runtime_status != 0;
        events_.push_back(std::move(progress));
    }

    void fail_native(const NativeExecutionFault &fault, bool launch_accepted) {
        emit_completion(EndpointOutcome::TASK_FAILURE, fault, launch_accepted, "injected native execution fault");
    }

    void succeed() { emit_completion(EndpointOutcome::SUCCESS, NativeExecutionFault{}, true, ""); }

private:
    bool faulted_{false};
    std::vector<WorkerDispatch> history_;
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
    virtual bool automatic() const { return false; }
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
        cfg.automatic_operator_recovery = automatic();
        cfg.recovery_timeout_s = 0.2;
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
        cfg.start_task_replay_cb = [this](const RecoveryTicket &ticket) {
            orch.start_task_replay(ticket);
        };
        cfg.rebind_recovery_graph_cb = [this](const RecoveryTicket &ticket, const RecoveryBindingPlan &plan) {
            orch.rebind_recovery_graph(ticket, plan);
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
    void graph(bool group = false, bool annotate = false) {
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
        TaskArgs peer = args;
        if (automatic()) {
            TaskDomainBinding binding;
            binding.domain = {0, 1, 42, 0x2000, 0x1000, 0x4000};
            binding.arguments = {{RecoveryArgKind::COMM_CONTEXT, 0, 0}};
            args.domain_bindings = peer.domain_bindings = {binding};
        }
        if (annotate) {
            RecoveryMetadata metadata{
                91, 0, 1, 0, group ? std::vector<int32_t>{0, 1} : std::vector<int32_t>{0}, ReplaySafety::REPLAY_SAFE
            };
            args.set_recovery_metadata(metadata);
            metadata.rank = 1;
            if (group) peer.set_recovery_metadata(metadata);
        }
        a = group ? orch.submit_next_level_group(chip_callable(1), {args, peer}, cfg, {0, 1}) :
                    orch.submit_next_level(chip_callable(1), args, cfg, 0);
        TaskArgs successor = args;
        successor.recovery_metadata.reset();
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
TEST_F(SchedulerRecoveryTest, ReplayAssessmentUsesRetainedMetadataWithoutDispatching) {
    graph(true, true);
    native_failure();
    second->succeed();
    auto request = held();
    auto result = scheduler.assess_replay(request.ticket);
    ASSERT_EQ(result.status, ReplayStatus::REPLAY_ALLOWED) << result.reason;
    ASSERT_TRUE(result.op);
    ASSERT_EQ(result.op->members.size(), 2u);
    EXPECT_EQ(result.op->logical_run_id, 91u);
    expect_graph(request);
    auto stale = request.ticket;
    ++stale.attempt;
    EXPECT_THROW(scheduler.assess_replay(stale), std::invalid_argument);
    give_up(request);
    EXPECT_THROW(scheduler.assess_replay(request.ticket), std::invalid_argument);
}
TEST_F(SchedulerRecoveryTest, ReplayAssessmentWaitsForBusyGroupPeer) {
    graph(true, true);
    native_failure();
    EXPECT_FALSE(scheduler.wait_for_recovery(0.03).has_value());
    auto pending = scheduler.pending_recoveries();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(scheduler.assess_replay(pending[0].ticket).status, ReplayStatus::WAITING_MEMBERS);
    second->succeed();
    auto request = held();
    EXPECT_EQ(scheduler.assess_replay(request.ticket).status, ReplayStatus::REPLAY_ALLOWED);
    give_up(request);
}
TEST_F(SchedulerRecoveryTest, ReplayAssessmentMissingMetadataFailsClosed) {
    graph(true);
    native_failure();
    second->succeed();
    auto request = held();
    EXPECT_EQ(scheduler.assess_replay(request.ticket).status, ReplayStatus::NACK_METADATA);
    expect_graph(request);
    give_up(request);
}
TEST_F(SchedulerRecoveryTest, ReplayCommitsOriginalAThenOriginalBandCExactlyOnce) {
    graph(true, true);
    native_failure();
    second->succeed();
    const auto request = held();
    ASSERT_TRUE(scheduler.retry_recovery(request.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3));
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    EXPECT_TRUE(first->history()[1].recovery);
    EXPECT_GT(first->history()[1].dispatch_id, first->history()[0].dispatch_id);
    EXPECT_EQ(first->history()[1].task_slot, a.task_slot);
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    EXPECT_EQ(ring.slot_state(c.task_slot)->state.load(), TaskState::PENDING);
    EXPECT_THROW(
        scheduler.retry_recovery(request.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3), std::invalid_argument
    );
    first->succeed();
    EXPECT_FALSE(orch.wait_run_for(run, 0.03));
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    second->succeed();
    ASSERT_TRUE(first->wait_count(3));
    EXPECT_EQ(first->history()[2].task_slot, b.task_slot);
    first->succeed();
    ASSERT_TRUE(first->wait_count(4));
    EXPECT_EQ(first->history()[3].task_slot, c.task_slot);
    first->succeed();
    ASSERT_TRUE(orch.wait_run_for(run, 3));
    EXPECT_FALSE(orch.run_failed(run));
    EXPECT_EQ(first->history().size(), 4u);
    EXPECT_EQ(second->history().size(), 2u);
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, PersistentFailureIsReportedWithNewTicketAndExternalBudget) {
    graph(true, true);
    native_failure();
    second->succeed();
    const auto one = held();
    ASSERT_TRUE(scheduler.retry_recovery(one.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3));
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    native_failure();
    second->succeed();
    const auto two = held();
    EXPECT_EQ(two.ticket.attempt, 1u);
    EXPECT_NE(two.ticket.recovery_id, one.ticket.recovery_id);
    EXPECT_THROW(scheduler.assess_replay(one.ticket), std::invalid_argument);
    EXPECT_FALSE(scheduler.retry_recovery(two.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3));
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 2u);
    EXPECT_EQ(second->history().size(), 2u);
    EXPECT_TRUE(scheduler.pending_recoveries().empty());
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, PartialResourceFailureNeverArmsOrDispatches) {
    graph(true, true);
    native_failure();
    second->succeed();
    auto request = held();
    second->fail_phase = RecoveryControlPhase::PREPARE;
    EXPECT_THROW(
        scheduler.retry_recovery(request.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3), std::runtime_error
    );
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(second->history().size(), 1u);
    EXPECT_EQ(std::count(first->phases.begin(), first->phases.end(), RecoveryControlPhase::ARM), 0);
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::ARM), 0);
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, TransportCommitRebindsUnlaunchedConsumersBeforeReplay) {
    graph(true, true);
    native_failure();
    second->succeed();
    auto request = held();
    for (auto handle : {a, b, c}) {
        auto *task = ring.slot_state(handle.task_slot);
        TaskDomainBinding binding;
        binding.domain = {0, 1, 42, 0x2000, 0x1000, 0x4000};
        binding.arguments = {{RecoveryArgKind::COMM_CONTEXT, 0, 0}};
        if (task->is_group())
            for (auto &args : task->task_args_list)
                args.domain_bindings = {binding};
        else task->task_args.domain_bindings = {binding};
    }
    ASSERT_TRUE(scheduler.retry_recovery(request.ticket, 1, 0, DomainRecoveryImpact::DOMAIN_TRANSPORT, 3));
    EXPECT_EQ(ring.slot_state(b.task_slot)->task_args.domain_bindings[0].domain.generation, 2u);
    EXPECT_EQ(ring.slot_state(c.task_slot)->task_args.domain_bindings[0].domain.generation, 2u);
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    scheduler.request_stop();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    scheduler.stop();
    orch.release_run(run);
}

TEST_F(SchedulerRecoveryTest, StaleRecoveryControlReplyCannotArmOrDispatch) {
    graph(true, true);
    native_failure();
    second->succeed();
    const auto request = held();
    second->stale_reply = true;
    EXPECT_THROW(
        scheduler.retry_recovery(request.ticket, 1, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3), std::runtime_error
    );
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(second->history().size(), 1u);
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, CloseCancelsOwnerBackoffWithoutWaitingForTheDelay) {
    graph(true, true);
    native_failure();
    second->succeed();
    const auto request = held();
    auto owner = std::async(std::launch::async, [&] {
        try {
            scheduler.retry_recovery(request.ticket, 1, 30, DomainRecoveryImpact::EXECUTION_ONLY, 3);
        } catch (const std::exception &) {
            return true;
        }
        return false;
    });
    EXPECT_EQ(owner.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    scheduler.request_stop();
    ASSERT_EQ(owner.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(owner.get());
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    scheduler.stop();
    orch.release_run(run);
}
TEST_F(SchedulerRecoveryTest, CallerCanBudgetMoreThanOneRetryWithoutAnL2Credit) {
    graph(true, true);
    for (uint32_t attempt = 0; attempt <= 2; ++attempt) {
        native_failure();
        second->succeed();
        auto request = held();
        EXPECT_EQ(request.ticket.attempt, attempt);
        const bool dispatched = scheduler.retry_recovery(request.ticket, 2, 0, DomainRecoveryImpact::EXECUTION_ONLY, 3);
        EXPECT_EQ(dispatched, attempt < 2);
        if (dispatched) {
            ASSERT_TRUE(first->wait_count(attempt + 2));
            ASSERT_TRUE(second->wait_count(attempt + 2));
        }
    }
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 3u);
    EXPECT_EQ(second->history().size(), 3u);
    orch.release_run(run);
}

class AutomaticRecoveryTest : public SchedulerRecoveryTest {
    bool automatic() const override { return true; }
};
TEST_F(AutomaticRecoveryTest, StopsLivePeerAndCommitsOriginalGraphWithoutAnExternalOwner) {
    graph(true, true);
    native_failure();
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    first->succeed();
    EXPECT_FALSE(orch.wait_run_for(run, 0.03));
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    second->succeed();
    ASSERT_TRUE(first->wait_count(3));
    EXPECT_EQ(first->history()[2].task_slot, b.task_slot);
    first->succeed();
    ASSERT_TRUE(first->wait_count(4));
    first->succeed();
    EXPECT_TRUE(orch.wait_run_for(run, 3));
    EXPECT_EQ(first->history().size(), 4u);
    EXPECT_EQ(second->history().size(), 2u);
    EXPECT_FALSE(orch.run_failed(run));
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, HostWaitTimeoutAloneStopsPeerAndReplaysOriginalGraph) {
    graph(true, true);
    NativeExecutionFault timeout{};
    timeout.raw_rc = PTO_RUNTIME_ERR_EXECUTION_WAIT_TIMEOUT;
    // No device error code is needed to trigger the normal recovery transaction.
    first->fail_native(timeout, true);
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    EXPECT_NE(
        std::find(second->phases.begin(), second->phases.end(), RecoveryControlPhase::STOP), second->phases.end()
    );
    first->succeed();
    EXPECT_FALSE(orch.wait_run_for(run, 0.03));
    EXPECT_EQ(ring.slot_state(b.task_slot)->state.load(), TaskState::PENDING);
    second->succeed();
    ASSERT_TRUE(first->wait_count(3));
    EXPECT_EQ(first->history()[2].task_slot, b.task_slot);
    first->succeed();
    ASSERT_TRUE(first->wait_count(4));
    first->succeed();
    EXPECT_TRUE(orch.wait_run_for(run, 3));
    EXPECT_FALSE(orch.run_failed(run));
    EXPECT_EQ(first->history().size(), 4u);
    EXPECT_EQ(second->history().size(), 2u);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, HostWaitTimeoutWithoutQuiescenceProofCannotReplay) {
    first->stop_proof = false;
    graph(true, true);
    NativeExecutionFault timeout{};
    timeout.raw_rc = PTO_RUNTIME_ERR_EXECUTION_WAIT_TIMEOUT;
    first->fail_native(timeout, true);
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(second->history().size(), 1u);
    EXPECT_EQ(std::count(first->phases.begin(), first->phases.end(), RecoveryControlPhase::REARM), 0);
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::ARM), 0);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, PersistentFailureUsesTheConfiguredL3Budget) {
    graph(true, true);
    native_failure();
    ASSERT_TRUE(first->wait_count(2));
    ASSERT_TRUE(second->wait_count(2));
    native_failure();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 2u);
    EXPECT_EQ(second->history().size(), 2u);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, MissingMetadataStillStopsThePeerThenGivesUp) {
    graph(true, false);
    native_failure();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(second->history().size(), 1u);
    EXPECT_NE(
        std::find(second->phases.begin(), second->phases.end(), RecoveryControlPhase::STOP), second->phases.end()
    );
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::ARM), 0);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, DeliveryAckWithoutTerminalOutcomeCannotStartResourceRecovery) {
    second->ignore_stop = true;
    graph(true, true);
    native_failure();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::PREPARE), 0);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, AStaleStopReplyNeverRearmsOrRedispatches) {
    second->stale_stop_reply = true;
    graph(true, true);
    native_failure();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::ARM), 0);
    orch.release_run(run);
}
TEST_F(AutomaticRecoveryTest, AStoppedPeerWithoutProofCannotRecover) {
    second->stop_proof = false;
    graph(true, true);
    native_failure();
    EXPECT_THROW(orch.wait_run_for(run, 3), std::runtime_error);
    EXPECT_EQ(first->history().size(), 1u);
    EXPECT_EQ(std::count(second->phases.begin(), second->phases.end(), RecoveryControlPhase::ARM), 0);
    orch.release_run(run);
}
TEST(RecoveryCoordinatorTest, FaultEventArrivesBeforePeerTermination) {
    RecoveryCoordinator c;
    auto r = episode(true);
    r.ticket = c.submit(r);
    const auto early = c.wait_next(0);
    ASSERT_TRUE(early);
    EXPECT_FALSE(early->all_members_terminal());
    EXPECT_FALSE(c.wait_terminal(r.ticket, 0));
    c.complete_member(r.ticket.recovery_id, 1, WorkerCompletion{});
    EXPECT_TRUE(c.wait_terminal(r.ticket, 0)->all_members_terminal());
    auto waiter = std::async(std::launch::async, [&] {
        return c.wait_next(r.ticket.recovery_id);
    });
    c.cancel_all();
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(waiter.get());
}
TEST(ExecutionStopControlTest, ZeroAndStaleEpochCannotCancelAnotherGeneration) {
    ExecutionStopControl control;
    EXPECT_FALSE(execution_stop_requested(control));
    control.run_epoch = 7;
    control.requested_epoch = 6;
    EXPECT_FALSE(execution_stop_requested(control));
    control.requested_epoch = 7;
    EXPECT_TRUE(execution_stop_requested(control));
    control.run_epoch = 8;
    EXPECT_FALSE(execution_stop_requested(control));
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
