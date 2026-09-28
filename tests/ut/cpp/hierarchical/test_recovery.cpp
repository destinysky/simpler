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
#include "chip_worker.h"
#include "orchestrator.h"
#include "recovery_coordinator.h"
#include "ring.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"
#include "task_args.h"

class ChipWorkerRecoveryTestPeer {
public:
    static void seed(ChipWorker &worker, std::vector<std::string> &events, int retire_rc = 0) {
        event_sink() = &events;
        active_worker() = &worker;
        retire_result() = retire_rc;
        worker.initialized_ = true;
        worker.finalized_ = false;
        worker.device_ctx_ = reinterpret_cast<void *>(0x10);
        worker.finalize_device_fn_ = &healthy_finalize;
        worker.recovery_finalize_device_fn_ = &recovery_finalize;
        worker.destroy_comm_stream_fn_ = &destroy_stream;
        worker.comm_destroy_fn_ = &normal_destroy;
        worker.comm_abandon_after_device_reset_fn_ = &abandon_after_reset;
        worker.comm_retire_after_peer_reset_fn_ = &retire_after_peer_reset;
        auto *session = worker.create_comm_session(
            reinterpret_cast<void *>(0x20), reinterpret_cast<void *>(0x30), true
        );
        ASSERT_NE(session, nullptr);
        worker.base_comm_handle_ = reinterpret_cast<uint64_t>(session->handle);
    }

    static void discard(ChipWorker &worker) {
        worker.comm_sessions_.clear();
        worker.comm_session_index_.clear();
        worker.base_comm_handle_ = 0;
        worker.device_ctx_ = nullptr;
        worker.initialized_ = false;
        worker.finalized_ = true;
        event_sink() = nullptr;
        active_worker() = nullptr;
    }

    static size_t session_count(const ChipWorker &worker) { return worker.comm_sessions_.size(); }
    static void *session_handle(const ChipWorker &worker) { return worker.comm_sessions_.at(0).handle; }
    static uint64_t base_handle(const ChipWorker &worker) { return worker.base_comm_handle_; }

private:
    static std::vector<std::string> *&event_sink() {
        static std::vector<std::string> *value = nullptr;
        return value;
    }
    static ChipWorker *&active_worker() {
        static ChipWorker *value = nullptr;
        return value;
    }
    static int &retire_result() {
        static int value = 0;
        return value;
    }
    static void record(const char *event) {
        if (event_sink() != nullptr) event_sink()->emplace_back(event);
    }
    static int healthy_finalize(void *) {
        record("healthy-finalize");
        return 0;
    }
    static int recovery_finalize(void *) {
        ChipWorker *worker = active_worker();
        record(worker != nullptr && !worker->comm_sessions_.empty() &&
                       worker->comm_sessions_.front().handle != nullptr
                   ? "reset-with-owned-handle"
                   : "reset-without-owned-handle");
        return 0;
    }
    static int abandon_after_reset(void *) {
        record("abandon-after-reset");
        return 0;
    }
    static int retire_after_peer_reset(void *) {
        record("retire-after-peer-reset");
        return retire_result();
    }
    static int normal_destroy(void *) {
        record("normal-destroy");
        return 0;
    }
    static int destroy_stream(void *, void *) {
        record("destroy-stream");
        return 0;
    }
};

namespace {

TEST(ChipWorkerCommRecoveryTest, FaultedEndpointResetsBeforeHostOnlyAbandon) {
    ChipWorker worker;
    std::vector<std::string> events;
    ChipWorkerRecoveryTestPeer::seed(worker, events);

    worker.recovery_finalize();

    EXPECT_EQ(events, (std::vector<std::string>{"reset-with-owned-handle", "abandon-after-reset"}));
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::session_count(worker), 0u);
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::base_handle(worker), 0u);
}

TEST(ChipWorkerCommRecoveryTest, HealthyPeerRetirementFailurePreservesOwnershipAndStops) {
    ChipWorker worker;
    std::vector<std::string> events;
    ChipWorkerRecoveryTestPeer::seed(worker, events, -17);

    EXPECT_THROW(worker.comm_retire_after_peer_reset(), std::runtime_error);

    EXPECT_EQ(events, (std::vector<std::string>{"retire-after-peer-reset"}));
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::session_count(worker), 1u);
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::session_handle(worker), reinterpret_cast<void *>(0x20));
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::base_handle(worker), 0x20u);
    ChipWorkerRecoveryTestPeer::discard(worker);
}

TEST(ChipWorkerCommRecoveryTest, HealthyPeerRetirementConsumesOwnershipOnlyAfterSuccess) {
    ChipWorker worker;
    std::vector<std::string> events;
    ChipWorkerRecoveryTestPeer::seed(worker, events);

    worker.comm_retire_after_peer_reset();

    EXPECT_EQ(events, (std::vector<std::string>{"retire-after-peer-reset", "destroy-stream"}));
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::session_count(worker), 0u);
    EXPECT_EQ(ChipWorkerRecoveryTestPeer::base_handle(worker), 0u);
    ChipWorkerRecoveryTestPeer::discard(worker);
}

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

struct ConcurrentCommProbe {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t entered{0};
    uint32_t expected{0};
    bool observed{false};
};

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
        if (progress.kind == WorkerProgressKind::COMPLETED) outstanding_ = false;
        return true;
    }

    bool activate_progress(RunId) override { return true; }

    uint64_t endpoint_generation() const override { return generation_.load(std::memory_order_acquire); }

    EndpointRecoveryResult rebuild_endpoint(const EndpointRecoveryRequest &request) override {
        recovery_calls_.fetch_add(1, std::memory_order_relaxed);
        uint64_t current = generation_.load(std::memory_order_acquire);
        if (request.recovery_id == 0 || request.expected_endpoint_generation != current) {
            return {request.worker_id, request.recovery_id, false, current, "stale generation"};
        }
        if (fail_recovery_) {
            return {request.worker_id, request.recovery_id, false, current, "injected reset/probe failure"};
        }
        generation_.store(current + 1, std::memory_order_release);
        return {request.worker_id, request.recovery_id, true, current + 1, {}};
    }

    LocalCommEndpointResult rebuild_local_comm(const LocalCommEndpointRequest &request) override {
        comm_recovery_calls_.fetch_add(1, std::memory_order_relaxed);
        if (comm_probe_ != nullptr) {
            std::unique_lock<std::mutex> lk(comm_probe_->mu);
            ++comm_probe_->entered;
            comm_probe_->cv.notify_all();
            comm_probe_->observed = comm_probe_->cv.wait_for(lk, std::chrono::seconds(1), [this] {
                return comm_probe_->entered == comm_probe_->expected;
            });
        }
        if (fail_comm_recovery_) {
            return {request.worker_id, request.recovery_id, false, request.expected_local_comm_generation,
                    "injected communication rebuild failure"};
        }
        return {request.worker_id, request.recovery_id, true, request.expected_local_comm_generation + 1, {}};
    }

    void release_faulted_endpoint(const EndpointRecoveryRequest &request) override {
        if (request.recovery_id == 0 || request.expected_endpoint_generation != endpoint_generation()) {
            throw std::runtime_error("stale release transaction");
        }
        releases_.fetch_add(1, std::memory_order_relaxed);
    }
    void set_recovery_failure(bool fail) { fail_recovery_ = fail; }
    void set_comm_recovery_failure(bool fail) { fail_comm_recovery_ = fail; }
    void set_comm_probe(ConcurrentCommProbe *probe) { comm_probe_ = probe; }
    uint32_t recovery_calls() const { return recovery_calls_.load(std::memory_order_relaxed); }
    uint32_t comm_recovery_calls() const { return comm_recovery_calls_.load(std::memory_order_relaxed); }
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
        return cv_.wait_for(lk, timeout, [this] { return submitted_; });
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
        emit_completion(
            EndpointOutcome::TASK_FAILURE, fault, launch_accepted, "injected native execution fault"
        );
    }

    void succeed() { emit_completion(EndpointOutcome::SUCCESS, NativeExecutionFault{}, true, ""); }

private:
    WorkerEndpointCaps caps_{};
    std::mutex mu_;
    std::condition_variable cv_;
    WorkerDispatch dispatch_{};
    bool submitted_{false};
    bool outstanding_{false};
    std::deque<WorkerEndpointProgress> events_;
    std::atomic<uint64_t> generation_{1};
    std::atomic<uint32_t> recovery_calls_{0};
    std::atomic<uint32_t> comm_recovery_calls_{0};
    std::atomic<uint32_t> releases_{0};
    bool fail_recovery_{false};
    bool fail_comm_recovery_{false};
    ConcurrentCommProbe *comm_probe_{nullptr};
};

TEST(LocalCommRecoveryManagerTest, InitialCommitAndInvalidationPreserveGeneration) {
    LocalCommRecoveryManager manager;
    EXPECT_EQ(manager.state().state, LocalCommState::UNINITIALIZED);
    EXPECT_EQ(manager.state().generation, 0u);
    EXPECT_EQ(manager.commit_initial_ready().state, LocalCommState::READY);
    EXPECT_EQ(manager.state().generation, 1u);
    ASSERT_EQ(manager.mark_stale(17), std::optional<uint64_t>(1));
    EXPECT_EQ(manager.state().state, LocalCommState::STALE);
    EXPECT_EQ(manager.state().generation, 1u);
    manager.fail_stale_episode(17, 1);
    EXPECT_EQ(manager.state().state, LocalCommState::BROKEN);
    EXPECT_EQ(manager.state().generation, 1u);
}

TEST(LocalCommRecoveryManagerTest, StaleRequestDoesNotMutateTheHeldEpisode) {
    LocalCommRecoveryManager manager;
    manager.commit_initial_ready();
    ASSERT_TRUE(manager.mark_stale(17).has_value());
    WorkerManager workers;
    EXPECT_FALSE(manager.rebuild({18, 1}, workers).ok);
    EXPECT_EQ(manager.state().state, LocalCommState::STALE);
    EXPECT_FALSE(manager.rebuild({17, 2}, workers).ok);
    EXPECT_EQ(manager.state().state, LocalCommState::STALE);
    EXPECT_EQ(manager.state().generation, 1u);
}

TEST(LocalCommRecoveryManagerTest, FanoutIsConcurrentAndCommitsOnlyAfterEveryEndpoint) {
    Ring ring;
    ring.init(1ULL << 20);
    WorkerManager workers;
    ConcurrentCommProbe probe;
    probe.expected = 2;
    auto first = std::make_unique<RecoveryEndpoint>(0);
    auto second = std::make_unique<RecoveryEndpoint>(1);
    RecoveryEndpoint *first_ptr = first.get();
    RecoveryEndpoint *second_ptr = second.get();
    first_ptr->set_comm_probe(&probe);
    second_ptr->set_comm_probe(&probe);
    workers.add_next_level_endpoint(std::move(first));
    workers.add_next_level_endpoint(std::move(second));
    workers.start(&ring, [](WorkerCompletion) {}, [](WorkerDispatch) {});

    LocalCommRecoveryManager manager;
    manager.commit_initial_ready();
    ASSERT_TRUE(manager.mark_stale(19).has_value());
    LocalCommRecoveryResult result = manager.rebuild({19, 1}, workers);
    EXPECT_TRUE(result.ok);
    EXPECT_TRUE(probe.observed);
    EXPECT_EQ(first_ptr->comm_recovery_calls(), 1u);
    EXPECT_EQ(second_ptr->comm_recovery_calls(), 1u);
    EXPECT_EQ(manager.state().generation, 2u);

    workers.stop_workers();
    workers.stop();
    ring.shutdown();
}

TEST(LocalCommRecoveryManagerTest, PartialEndpointSuccessDoesNotCommitGroupGeneration) {
    Ring ring;
    ring.init(1ULL << 20);
    WorkerManager workers;
    auto first = std::make_unique<RecoveryEndpoint>(0);
    auto second = std::make_unique<RecoveryEndpoint>(1);
    second->set_comm_recovery_failure(true);
    workers.add_next_level_endpoint(std::move(first));
    workers.add_next_level_endpoint(std::move(second));
    workers.start(&ring, [](WorkerCompletion) {}, [](WorkerDispatch) {});

    LocalCommRecoveryManager manager;
    manager.commit_initial_ready();
    ASSERT_TRUE(manager.mark_stale(21).has_value());
    LocalCommRecoveryResult result = manager.rebuild({21, 1}, workers);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(manager.state().state, LocalCommState::BROKEN);
    EXPECT_EQ(manager.state().generation, 1u);

    workers.stop_workers();
    workers.stop();
    ring.shutdown();
}

TEST(LocalCommRecoveryManagerTest, BusyHealthyPeerDeclinesDestructiveFanout) {
    Ring ring;
    ring.init(1ULL << 20);
    WorkerManager workers;
    workers.add_next_level_endpoint(std::make_unique<RecoveryEndpoint>(0));
    workers.start(&ring, [](WorkerCompletion) {}, [](WorkerDispatch) {});
    WorkerThread *worker = workers.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
    ASSERT_NE(worker, nullptr);
    worker->dispatch(WorkerDispatch{});

    LocalCommRecoveryManager manager;
    manager.commit_initial_ready();
    ASSERT_TRUE(manager.mark_stale(23).has_value());
    LocalCommRecoveryResult result = manager.rebuild({23, 1}, workers);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(manager.state().state, LocalCommState::BROKEN);
    EXPECT_EQ(manager.state().generation, 1u);

    workers.stop_workers();
    workers.stop();
    ring.shutdown();
}

TEST(RecoveryCoordinatorTest, RequestResolutionIsAsynchronousAndEndsInGiveUp) {
    RecoveryCoordinator coordinator;
    RecoveryRequest request;
    request.ticket.run_id = 7;
    request.ticket.task_slot = 3;
    request.ticket.recovery_id = 11;
    request.ticket.attempt = 0;
    request.worker_id = 2;
    request.failure.outcome = EndpointOutcome::TASK_FAILURE;
    request.failure.launch_accepted = false;
    request.failure.native_execution_fault.runtime_status = -9;

    coordinator.submit(std::move(request));
    RecoveryResolution resolution;
    EXPECT_FALSE(coordinator.try_pop_resolution(resolution));

    coordinator.progress();
    ASSERT_TRUE(coordinator.try_pop_resolution(resolution));
    EXPECT_EQ(resolution.kind, RecoveryResolutionKind::GIVE_UP);
    EXPECT_EQ(resolution.request.stage, RecoveryStage::GIVE_UP);
    EXPECT_EQ(resolution.request.ticket.recovery_id, 11u);
    EXPECT_EQ(resolution.request.ticket.attempt, 0u);
    EXPECT_FALSE(resolution.request.failure.launch_accepted);
    EXPECT_EQ(resolution.request.failure.native_execution_fault.runtime_status, -9);
    EXPECT_FALSE(coordinator.has_work());
}

TEST(RecoveryCoordinatorTest, EligibleRequestProducesEndpointActionWithoutReplay) {
    RecoveryCoordinator coordinator;
    coordinator.set_eligibility_decision([](const RecoveryRequest &) { return true; });
    RecoveryRequest request;
    request.ticket.recovery_id = 51;
    request.ticket.expected_endpoint_generation = 1;
    coordinator.submit(request);
    coordinator.progress();
    RecoveryResolution resolution;
    ASSERT_TRUE(coordinator.try_pop_resolution(resolution));
    EXPECT_EQ(resolution.kind, RecoveryResolutionKind::REBUILD_ENDPOINT);
    EXPECT_EQ(resolution.request.stage, RecoveryStage::ENDPOINT_REBUILD);
    EXPECT_EQ(resolution.request.ticket.recovery_id, 51u);

    EndpointRecoveryResult result{resolution.request.worker_id, 51, true, 2, {}};
    coordinator.submit_endpoint_result(std::move(resolution.request), std::move(result));
    coordinator.progress();
    ASSERT_TRUE(coordinator.try_pop_resolution(resolution));
    EXPECT_EQ(resolution.kind, RecoveryResolutionKind::GIVE_UP);
    ASSERT_TRUE(resolution.endpoint_result.has_value());
    EXPECT_TRUE(resolution.endpoint_result->ok);
}

TEST(EndpointRecoveryCommandTest, StaleGenerationDoesNotPublishResetCommand) {
    std::vector<char> mailbox(MAILBOX_SIZE, 0);
    LocalMailboxEndpoint endpoint(4, mailbox.data());
    const EndpointRecoveryResult result = endpoint.rebuild_endpoint({4, 91, 0});
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.endpoint_generation, 1u);
    EXPECT_EQ(endpoint.endpoint_generation(), 1u);
    uint64_t command = 0;
    std::memcpy(&command, mailbox.data() + MAILBOX_OFF_CALLABLE, sizeof(command));
    EXPECT_NE(command, CTRL_REBUILD_ENDPOINT);

    EXPECT_THROW(endpoint.release_faulted_endpoint({4, 91, 0}), std::runtime_error);
    std::memcpy(&command, mailbox.data() + MAILBOX_OFF_CALLABLE, sizeof(command));
    EXPECT_NE(command, CTRL_RELEASE_FAULTED_ENDPOINT);

    const LocalCommEndpointResult comm_result = endpoint.rebuild_local_comm({4, 91, 0, 1});
    EXPECT_FALSE(comm_result.ok);
    std::memcpy(&command, mailbox.data() + MAILBOX_OFF_CALLABLE, sizeof(command));
    EXPECT_NE(command, CTRL_REBUILD_COMM_STATE);
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

    auto begin = std::async(std::launch::async, [&] { return orch.begin_run(); });
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
    virtual bool allow_endpoint_recovery() const { return false; }
    virtual bool fail_endpoint_recovery() const { return false; }
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orch;
    WorkerManager manager;
    LocalCommRecoveryManager local_comm_manager;
    Scheduler scheduler;
    RecoveryEndpoint *endpoint{nullptr};
    std::atomic<uint32_t> recovery_begin_calls{0};

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        auto owned_endpoint = std::make_unique<RecoveryEndpoint>();
        endpoint = owned_endpoint.get();
        endpoint->set_recovery_failure(fail_endpoint_recovery());
        manager.add_next_level_endpoint(std::move(owned_endpoint));
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) { scheduler.worker_done(std::move(completion)); },
            [this](WorkerDispatch dispatch) { orch.mark_task_accepted(dispatch.task_slot); }
        );
        ready_next.reset(manager.next_level_worker_ids());
        orch.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next, &manager, [this] {
            scheduler.notify_ready();
        });

        Scheduler::Config config;
        config.ring = &allocator;
        config.ready_sub_queue = &ready_sub;
        config.ready_next_level_queues = &ready_next;
        config.manager = &manager;
        config.enqueue_ready_cb = [this](TaskSlot slot) { orch.enqueue_ready(slot); };
        config.active_run_cb = [this] { return orch.dispatchable_run_id(); };
        config.preparable_run_cb = [this] { return orch.preparable_run_id(); };
        config.on_consumed_cb = [this](TaskSlot slot) { (void)orch.on_consumed(slot); };
        config.on_task_failed_cb = [this](TaskSlot slot, const std::string &message) {
            orch.report_task_error(slot, message);
        };
        config.operator_recovery_enabled = true;
        config.begin_task_recovery_cb = [this](TaskSlot slot, uint64_t recovery_id) {
            recovery_begin_calls.fetch_add(1, std::memory_order_relaxed);
            return orch.begin_task_recovery(slot, recovery_id);
        };
        config.finish_task_recovery_cb = [this](TaskSlot slot, uint64_t recovery_id) {
            (void)orch.finish_task_recovery(slot, recovery_id);
        };
        config.recovery_eligibility_cb = [this](const RecoveryRequest &) { return allow_endpoint_recovery(); };
        config.on_global_recovery_freeze_cb = [this](bool frozen) {
            orch.set_global_recovery_freeze(frozen);
        };
        config.local_comm_recovery_manager = &local_comm_manager;
        scheduler.start(config);
        orch.set_scheduler_loop_mutex(&scheduler.loop_mutex());
    }

    void TearDown() override {
        scheduler.request_stop();
        manager.stop_workers();
        scheduler.stop();
        manager.stop();
        allocator.shutdown();
    }

    std::string run_failure(
        const CallableIdentity &callable, EndpointOutcome outcome, const NativeExecutionFault &fault,
        bool launch_accepted, const std::string &error_message
    ) {
        RunId run = orch.begin_run();
        CallConfig config;
        auto task = orch.submit_next_level(callable, single_tensor_args(0x2000 + run), config, 0);
        orch.close_run_submission(run);
        if (!endpoint->wait_submitted()) {
            ADD_FAILURE() << "timed out waiting for recovery test dispatch";
            return {};
        }
        endpoint->emit_completion(outcome, fault, launch_accepted, error_message);

        std::string message;
        try {
            orch.wait_run(run);
            ADD_FAILURE() << "injected failure unexpectedly completed successfully";
        } catch (const std::runtime_error &e) {
            message = e.what();
        }
        EXPECT_TRUE(orch.run_failed(run));
        TaskSlotState *state = allocator.slot_state(task.task_slot);
        EXPECT_NE(state, nullptr);
        if (state != nullptr) {
            EXPECT_EQ(state->state.load(std::memory_order_acquire), TaskState::CONSUMED);
        }
        orch.release_run(run);
        return message;
    }

    std::string run_fault(const NativeExecutionFault &fault, bool launch_accepted) {
        return run_failure(
            chip_callable(9), EndpointOutcome::TASK_FAILURE, fault, launch_accepted,
            "injected native execution fault"
        );
    }
};

class SchedulerEndpointRebuildTest : public SchedulerRecoveryTest {
protected:
    bool allow_endpoint_recovery() const override { return true; }
};

class SchedulerEndpointRebuildFailureTest : public SchedulerEndpointRebuildTest {
protected:
    bool fail_endpoint_recovery() const override { return true; }
};

class SchedulerCommRebuildTest : public SchedulerEndpointRebuildTest {
protected:
    void SetUp() override {
        SchedulerEndpointRebuildTest::SetUp();
        local_comm_manager.commit_initial_ready();
    }
};

class SchedulerCommRebuildFailureTest : public SchedulerCommRebuildTest {
protected:
    void SetUp() override {
        SchedulerCommRebuildTest::SetUp();
        endpoint->set_comm_recovery_failure(true);
    }
};

TEST_F(SchedulerEndpointRebuildTest, RebuildCommitsGenerationAndStillFailsOriginalTask) {
    NativeExecutionFault fault{};
    fault.raw_rc = 507018;
    fault.device_unusable = 1;
    const std::string message = run_fault(fault, /*launch_accepted=*/true);
    EXPECT_EQ(endpoint->recovery_calls(), 1u);
    EXPECT_EQ(endpoint->releases(), 0u);
    EXPECT_EQ(endpoint->endpoint_generation(), 2u);
    EXPECT_NE(message.find("ENDPOINT_READY generation=2"), std::string::npos);
    EXPECT_NE(message.find("ENDPOINT_REBUILD->ENDPOINT_READY->GIVE_UP"), std::string::npos);
}

TEST_F(SchedulerEndpointRebuildFailureTest, FailedResetKeepsGenerationAndReleasesHeldChild) {
    NativeExecutionFault fault{};
    fault.runtime_status = -9;
    const std::string message = run_fault(fault, /*launch_accepted=*/true);
    EXPECT_EQ(endpoint->recovery_calls(), 1u);
    EXPECT_EQ(endpoint->releases(), 1u);
    EXPECT_EQ(endpoint->endpoint_generation(), 1u);
    EXPECT_NE(message.find("injected reset/probe failure"), std::string::npos);
    EXPECT_NE(message.find("ENDPOINT_REBUILD->GIVE_UP"), std::string::npos);
}

TEST_F(SchedulerCommRebuildTest, CommGenerationCommitsOnlyAfterEndpointFanoutSucceeds) {
    NativeExecutionFault fault{};
    fault.raw_rc = 507018;
    const std::string message = run_fault(fault, /*launch_accepted=*/true);
    EXPECT_EQ(endpoint->comm_recovery_calls(), 1u);
    EXPECT_EQ(local_comm_manager.state().state, LocalCommState::READY);
    EXPECT_EQ(local_comm_manager.state().generation, 2u);
    EXPECT_NE(message.find("COMM_READY generation=2"), std::string::npos);
    EXPECT_NE(message.find("ENDPOINT_READY->COMM_REBUILD->COMM_READY->GIVE_UP"), std::string::npos);
}

TEST_F(SchedulerCommRebuildFailureTest, PartialCommFailureLeavesOldGenerationBroken) {
    NativeExecutionFault fault{};
    fault.runtime_status = -9;
    const std::string message = run_fault(fault, /*launch_accepted=*/true);
    EXPECT_EQ(local_comm_manager.state().state, LocalCommState::BROKEN);
    EXPECT_EQ(local_comm_manager.state().generation, 1u);
    EXPECT_NE(message.find("injected communication rebuild failure"), std::string::npos);
    EXPECT_EQ(message.find("COMM_READY"), std::string::npos);
}

TEST_F(SchedulerRecoveryTest, SemanticOnlyFaultWithUnacceptedLaunchStillUsesRecoveryPath) {
    NativeExecutionFault fault{};
    fault.runtime_status = -9;
    fault.device_unusable = 1;
    std::string message = run_fault(fault, /*launch_accepted=*/false);
    EXPECT_NE(message.find("injected native execution fault"), std::string::npos);
    EXPECT_NE(message.find("[recovery id="), std::string::npos);
    EXPECT_NE(message.find("attempt=0"), std::string::npos);
    EXPECT_NE(message.find("trace=DETECTED->ELIGIBILITY_CHECK->GIVE_UP"), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 1u);
}

TEST_F(SchedulerRecoveryTest, RawRcOnlyFaultAlsoUsesRecoveryPath) {
    NativeExecutionFault fault{};
    fault.raw_rc = 507018;
    fault.device_unusable = 1;
    std::string message = run_fault(fault, /*launch_accepted=*/true);
    EXPECT_NE(message.find("[recovery id="), std::string::npos);
    EXPECT_NE(message.find("trace=DETECTED->ELIGIBILITY_CHECK->GIVE_UP"), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 1u);
}

TEST_F(SchedulerRecoveryTest, EndpointFailureWithNativeFaultDoesNotEnterRecovery) {
    NativeExecutionFault fault{};
    fault.raw_rc = 507018;
    fault.runtime_status = -9;
    fault.device_unusable = 1;
    std::string message = run_failure(
        chip_callable(10), EndpointOutcome::ENDPOINT_FAILURE, fault, /*launch_accepted=*/true,
        "injected endpoint failure"
    );
    EXPECT_NE(message.find("injected endpoint failure"), std::string::npos);
    EXPECT_EQ(message.find("[recovery id="), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 0u);
}

TEST_F(SchedulerRecoveryTest, TaskFailureWithoutRawOrRuntimeStatusDoesNotEnterRecovery) {
    NativeExecutionFault fault{};
    // detail_code and device_unusable are recovery facts, but they do not make
    // a NativeExecutionFault exist without raw_rc or runtime_status.
    fault.detail_code = 1;
    fault.device_unusable = 1;
    std::string message = run_failure(
        chip_callable(11), EndpointOutcome::TASK_FAILURE, fault, /*launch_accepted=*/true,
        "injected task failure without raw or runtime status"
    );
    EXPECT_NE(message.find("without raw or runtime status"), std::string::npos);
    EXPECT_EQ(message.find("[recovery id="), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 0u);
}

TEST_F(SchedulerRecoveryTest, NonLocalChipTaskFailureDoesNotEnterRecovery) {
    NativeExecutionFault fault{};
    fault.runtime_status = -9;
    fault.device_unusable = 1;
    std::string message = run_failure(
        non_local_chip_callable(12), EndpointOutcome::TASK_FAILURE, fault, /*launch_accepted=*/true,
        "injected non-local-chip native fault"
    );
    EXPECT_NE(message.find("non-local-chip"), std::string::npos);
    EXPECT_EQ(message.find("[recovery id="), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 0u);
}

class SchedulerRecoveryGroupNegativeTest : public ::testing::Test {
protected:
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orch;
    WorkerManager manager;
    Scheduler scheduler;
    RecoveryEndpoint *endpoint0{nullptr};
    RecoveryEndpoint *endpoint1{nullptr};
    std::atomic<uint32_t> recovery_begin_calls{0};

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        auto first = std::make_unique<RecoveryEndpoint>(0);
        auto second = std::make_unique<RecoveryEndpoint>(1);
        endpoint0 = first.get();
        endpoint1 = second.get();
        manager.add_next_level_endpoint(std::move(first));
        manager.add_next_level_endpoint(std::move(second));
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) { scheduler.worker_done(std::move(completion)); },
            [this](WorkerDispatch dispatch) { orch.mark_task_accepted(dispatch.task_slot); }
        );
        ready_next.reset(manager.next_level_worker_ids());
        orch.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next, &manager, [this] {
            scheduler.notify_ready();
        });

        Scheduler::Config config;
        config.ring = &allocator;
        config.ready_sub_queue = &ready_sub;
        config.ready_next_level_queues = &ready_next;
        config.manager = &manager;
        config.enqueue_ready_cb = [this](TaskSlot slot) { orch.enqueue_ready(slot); };
        config.active_run_cb = [this] { return orch.dispatchable_run_id(); };
        config.preparable_run_cb = [this] { return orch.preparable_run_id(); };
        config.on_consumed_cb = [this](TaskSlot slot) { (void)orch.on_consumed(slot); };
        config.on_task_failed_cb = [this](TaskSlot slot, const std::string &message) {
            orch.report_task_error(slot, message);
        };
        config.operator_recovery_enabled = true;
        config.begin_task_recovery_cb = [this](TaskSlot slot, uint64_t recovery_id) {
            recovery_begin_calls.fetch_add(1, std::memory_order_relaxed);
            return orch.begin_task_recovery(slot, recovery_id);
        };
        config.finish_task_recovery_cb = [this](TaskSlot slot, uint64_t recovery_id) {
            (void)orch.finish_task_recovery(slot, recovery_id);
        };
        scheduler.start(config);
        orch.set_scheduler_loop_mutex(&scheduler.loop_mutex());
    }

    void TearDown() override {
        scheduler.request_stop();
        manager.stop_workers();
        scheduler.stop();
        manager.stop();
        allocator.shutdown();
    }
};

TEST_F(SchedulerRecoveryGroupNegativeTest, GroupTaskFailureDoesNotEnterRecovery) {
    RunId run = orch.begin_run();
    CallConfig config;
    auto group = orch.submit_next_level_group(
        chip_callable(13),
        {single_tensor_args(0x3101), single_tensor_args(0x3102)},
        config,
        {0, 1}
    );
    orch.close_run_submission(run);
    ASSERT_TRUE(endpoint0->wait_submitted());
    ASSERT_TRUE(endpoint1->wait_submitted());

    NativeExecutionFault fault{};
    fault.runtime_status = -9;
    fault.device_unusable = 1;
    endpoint0->fail_native(fault, /*launch_accepted=*/true);
    endpoint1->succeed();

    std::string message;
    try {
        orch.wait_run(run);
        ADD_FAILURE() << "group native failure unexpectedly completed successfully";
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    EXPECT_NE(message.find("injected native execution fault"), std::string::npos);
    EXPECT_EQ(message.find("[recovery id="), std::string::npos);
    EXPECT_EQ(recovery_begin_calls.load(std::memory_order_relaxed), 0u);
    EXPECT_TRUE(orch.run_failed(run));
    TaskSlotState *state = allocator.slot_state(group.task_slot);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->state.load(std::memory_order_acquire), TaskState::CONSUMED);
    orch.release_run(run);
}

}  // namespace
