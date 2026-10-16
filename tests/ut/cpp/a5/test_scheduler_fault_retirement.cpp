/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>
#include "runtime_core.h"
#include "../aicpu/run_completion.h"

TEST(A5RunCompletion, EarlyFailureIsReportedOnlyByLastParticipant) {
    AicpuRunCompletion completion;
    const auto early = completion.arrive(2, -101);
    EXPECT_FALSE(early.last);
    EXPECT_EQ(early.report_rc, 0);
    const auto last = completion.arrive(2, 0);
    EXPECT_TRUE(last.last);
    EXPECT_EQ(last.report_rc, -101);
    EXPECT_EQ(last.finished, 2);
}

TEST(A5RunCompletion, FirstLocalFailureSurvivesLaterFailure) {
    AicpuRunCompletion completion;
    EXPECT_EQ(completion.arrive(3, -7).report_rc, 0);
    EXPECT_EQ(completion.arrive(3, -8).report_rc, 0);
    EXPECT_EQ(completion.arrive(3, 0).report_rc, -7);
}

TEST(A5RunCompletion, SingleThreadFailureIsNotLost) {
    AicpuRunCompletion completion;
    const auto only = completion.arrive(1, -9);
    EXPECT_TRUE(only.last);
    EXPECT_EQ(only.report_rc, -9);
}

TEST(A5RunCompletion, ResetDoesNotReusePriorFailure) {
    AicpuRunCompletion completion;
    EXPECT_EQ(completion.arrive(1, -101).report_rc, -101);
    completion.reset();
    EXPECT_EQ(completion.finished(), 0);
    EXPECT_EQ(completion.arrive(2, 0).report_rc, 0);
    const auto last = completion.arrive(2, 0);
    EXPECT_TRUE(last.last);
    EXPECT_EQ(last.report_rc, 0);
}

TEST(A5RunCompletion, ConcurrentArrivalsHaveOneErrorOwner) {
    AicpuRunCompletion completion;
    std::atomic<int> owners{0};
    std::atomic<int> reports{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            const auto arrival = completion.arrive(8, i == 3 ? -101 : 0);
            if (arrival.last) owners.fetch_add(1);
            if (arrival.report_rc != 0) {
                EXPECT_TRUE(arrival.last);
                EXPECT_EQ(arrival.report_rc, -101);
                reports.fetch_add(1);
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(owners.load(), 1);
    EXPECT_EQ(reports.load(), 1);
}

class A5SchedulerFaultWait : public ::testing::Test {
protected:
    RuntimeContext rt{};
    DeviceArena sm_arena;
    DeviceArena arena;
    SharedMemoryHandle *sm = nullptr;
    std::vector<char> heap;
    TaskDescriptor producer{};
    uint32_t value{42};
    simpler::tmr::Tensor tensor{};

    void SetUp() override {
        sm = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm, nullptr);
        int32_t windows[CHIP_MAX_RING_DEPTH];
        for (auto &window : windows)
            window = CHIP_TASK_WINDOW_SIZE;
        auto orch_layout = OrchestratorState::reserve_layout(arena, windows);
        auto sched_layout = SchedulerState::reserve_layout(arena);
        ASSERT_NE(arena.commit(), nullptr);
        heap.resize(4096 * CHIP_MAX_RING_DEPTH);
        ASSERT_TRUE(rt.orchestrator.init_data_from_layout(
            orch_layout, arena, sm->sm_base, heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(rt.scheduler.init_data_from_layout(sched_layout, arena, sm->sm_base));
        rt.scheduler.wire_arena_pointers(sched_layout, arena);
        rt.orchestrator.wire_arena_pointers(orch_layout, arena, &rt.scheduler);
        rt.sm_handle = sm;
        runtime_finalize_after_wire(&rt, 1, 2);
        producer.task_id = TaskId::make(0, 0);
        auto &state = slot();
        state.task = &producer;
        state.ring_id = 0;
        state.task_state.store(CHIP_TASK_PENDING);
        state.fanout_count = 0;
        state.fanout_refcount.store(0);
        const uint32_t shape[1] = {1};
        tensor = simpler::tmr::make_tensor_external(&value, shape, 1);
        tensor.dtype = DataType::UINT32;
        tensor.owner_task_id = producer.task_id;
    }

    ChipTaskSlotState &slot() { return sm->header->rings[0].get_slot_state_by_task_id(0); }

    void TearDown() override {
        rt.orchestrator.destroy();
        rt.scheduler.destroy();
        arena.release();
        sm_arena.release();
    }

    template <typename Operation>
    void expect_scheduler_fault_interrupts_wait(Operation operation) {
        sm->header->sched_error_code.store(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID, std::memory_order_release);
        auto result = std::async(std::launch::async, operation);
        const auto status = result.wait_for(std::chrono::milliseconds(200));
        // Release an unfixed implementation so the regression fails cleanly,
        // rather than leaving the test blocked on its 15-second watchdog.
        slot().task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
        slot().fanout_refcount.store(1, std::memory_order_release);
        const auto value_returned = result.get();
        EXPECT_EQ(status, std::future_status::ready);
        EXPECT_EQ(value_returned, 0u);
        EXPECT_TRUE(rt.orchestrator.fatal);
        EXPECT_TRUE(rt.ops->is_fatal(&rt));
        EXPECT_EQ(sm->header->orch_error_code.load(), SIMPLER_ERROR_NONE);
        EXPECT_EQ(sm->header->sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
};

TEST_F(A5SchedulerFaultWait, ProducerReadStopsWithoutMaskingSchedulerFault) {
    expect_scheduler_fault_interrupts_wait([&] {
        const uint32_t index[1] = {0};
        return get_tensor_data(&rt, tensor, 1, index);
    });
}

TEST_F(A5SchedulerFaultWait, ProducerWriteStopsWithoutTouchingData) {
    expect_scheduler_fault_interrupts_wait([&] {
        const uint32_t index[1] = {0};
        set_tensor_data(&rt, tensor, 1, index, 99);
        return value == 42 ? 0u : 1u;
    });
    EXPECT_EQ(value, 42u);
}

TEST_F(A5SchedulerFaultWait, ConsumerWriteStopsWithoutTouchingData) {
    slot().task_state.store(CHIP_TASK_COMPLETED);
    slot().fanout_count = 1;
    slot().fanout_refcount.store(0);
    expect_scheduler_fault_interrupts_wait([&] {
        const uint32_t index[1] = {0};
        set_tensor_data(&rt, tensor, 1, index, 99);
        return value == 42 ? 0u : 1u;
    });
    EXPECT_EQ(value, 42u);
}

TEST_F(A5SchedulerFaultWait, HealthyReadStillWaitsForCompletion) {
    auto result = std::async(std::launch::async, [&] {
        const uint32_t index[1] = {0};
        return get_tensor_data(&rt, tensor, 1, index);
    });
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
    slot().task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    EXPECT_EQ(result.get(), 42u);
    EXPECT_FALSE(rt.orchestrator.fatal);
}
