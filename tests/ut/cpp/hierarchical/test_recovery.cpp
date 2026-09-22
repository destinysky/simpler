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
        if (progress.kind == WorkerProgressKind::COMPLETED) outstanding_ = false;
        return true;
    }

    bool activate_progress(RunId) override { return true; }

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
};

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
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orch;
    WorkerManager manager;
    Scheduler scheduler;
    RecoveryEndpoint *endpoint{nullptr};
    std::atomic<uint32_t> recovery_begin_calls{0};

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        auto owned_endpoint = std::make_unique<RecoveryEndpoint>();
        endpoint = owned_endpoint.get();
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
