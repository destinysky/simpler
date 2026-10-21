/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "orchestration_api.h"
#include "async_wait.h"
#include "backend/urma/urma_completion_kernel.h"

extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &);
[[noreturn]] void assert_impl(const char *, const char *, int) { throw "assert_impl"; }

namespace {
// Wire images follow PTO-ISA 138dbf86; the runtime decodes these separately.
struct Wq {
    uint32_t id{};
    uint64_t buffer{};
    uint32_t shift{6}, depth{8};
    uint64_t head{}, tail{};
    int32_t mode{1};
    uint64_t doorbell{};
    uint32_t sl{}, submitted{};
};
struct Cq {
    uint32_t id{};
    uint64_t buffer{};
    uint32_t shift{6}, depth{8};
    uint64_t head{}, tail{};
    int32_t mode{1};
    uint64_t doorbell{};
};
struct Info {
    uint32_t qp{1}, token{4}, notify_token{0}, ranks{2};
    uint64_t sq{}, rq{}, scq{}, rcq{}, mem{}, notify_pool{};
};
static_assert(sizeof(Info) == 64 && sizeof(Wq) == 64 && sizeof(Cq) == 56);

class UrmaCompletion : public ::testing::Test {
protected:
    alignas(64) std::array<uint32_t, 128> entries{};
    std::array<Wq, 2> sq{};
    std::array<Cq, 2> cq{};
    uint32_t sq_head{3}, sq_tail{}, cq_tail{}, cq_db{};
    Info info{};

    void SetUp() override {
        info.sq = reinterpret_cast<uint64_t>(sq.data());
        info.scq = reinterpret_cast<uint64_t>(cq.data());
        sq[1].head = reinterpret_cast<uint64_t>(&sq_head);
        sq[1].tail = reinterpret_cast<uint64_t>(&sq_tail);
        sq[1].submitted = 2;
        cq[1].buffer = reinterpret_cast<uint64_t>(entries.data());
        cq[1].tail = reinterpret_cast<uint64_t>(&cq_tail);
        cq[1].doorbell = reinterpret_cast<uint64_t>(&cq_db);
        entries[0] = 1U << 2;
        entries[16] = 1U << 2;
    }
    CompletionPollResult poll(uint32_t bb = 3, uint32_t count = 2) {
        CompletionCondition condition{};
        condition.addr = (uint64_t{1} << 32) | bb;
        condition.expected_value = count;
        condition.backend_cookie = reinterpret_cast<uint64_t>(&info);
        return urma_event_handle_poll_op(condition);
    }
};

TEST_F(UrmaCompletion, NotifyUsesDistinctBasicBlockAndCompletionSequences) {
    EXPECT_EQ(poll().state, CompletionPollState::READY);
    EXPECT_EQ(sq_tail, 3U);
    EXPECT_EQ(cq_tail, 2U);
    EXPECT_EQ(cq_db, 2U);
}
TEST_F(UrmaCompletion, TransferUsesTheSameWireHeader) {
    sq_head = sq[1].submitted = 1;
    EXPECT_EQ(poll(1, 1).state, CompletionPollState::READY);
    EXPECT_EQ(sq_tail, 1U);
    EXPECT_EQ(cq_tail, 1U);
}
TEST_F(UrmaCompletion, PendingPrefixRetainsBothTails) {
    entries[16] = 0;
    EXPECT_EQ(poll().state, CompletionPollState::PENDING);
    EXPECT_EQ(sq_tail, 0U);
    EXPECT_EQ(cq_tail, 0U);
}
TEST_F(UrmaCompletion, ErrorCqeRemainsVisibleToRecovery) {
    info.notify_token = 2;
    entries[16] |= 7U << 24;
    EXPECT_EQ(poll().state, CompletionPollState::FAILED);
    EXPECT_EQ(sq_tail, 0U);
    EXPECT_EQ(cq_tail, 0U);
    EXPECT_EQ(cq_db, 0U);
    EXPECT_EQ(poll().state, CompletionPollState::FAILED);
}
TEST_F(UrmaCompletion, AlreadyRetiredEventRequiresBothSequences) {
    info.notify_token = 2;
    cq_tail = 2;
    sq_tail = 3;
    EXPECT_EQ(poll().state, CompletionPollState::READY);
    sq_tail = 2;
    EXPECT_EQ(poll().state, CompletionPollState::FAILED);
}
TEST_F(UrmaCompletion, RejectsInvalidQueueGeometry) {
    cq[1].shift = 1;
    EXPECT_EQ(poll().state, CompletionPollState::FAILED);
    EXPECT_EQ(cq_tail, 0U);
}
TEST_F(UrmaCompletion, RejectsTargetsBeyondSubmittedWork) {
    EXPECT_EQ(poll(4, 2).state, CompletionPollState::FAILED);
    EXPECT_EQ(poll(3, 3).state, CompletionPollState::FAILED);
    EXPECT_EQ(sq_tail, 0U);
    EXPECT_EQ(cq_tail, 0U);
}
TEST_F(UrmaCompletion, SequenceWrapDoesNotTurnAnOutstandingEventIntoANoop) {
    sq_head = sq[1].submitted = 0;
    sq_tail = cq_tail = UINT32_MAX;
    entries[7 * 16] = 0;
    EXPECT_EQ(poll(0, 0).state, CompletionPollState::READY);
    EXPECT_EQ(sq_tail, 0U);
    EXPECT_EQ(cq_tail, 0U);
}

struct Event {
    uint64_t handle{(uint64_t{1} << 32) | 3};
    pto::comm::DmaEngine engine{pto::comm::DmaEngine::URMA};
    uint32_t urmaTargetCqe{2};
    bool wait_result{true};
    mutable int wait_calls{0};
    bool Wait(int) const {
        ++wait_calls;
        return wait_result;
    }
};

TEST(UrmaRegistration, CarriesCqeTargetWithoutWaiting) {
    uint32_t count = 0;
    int32_t error = 0;
    DeferredCompletionEntry entry{};
    uint8_t workspace{};
    AsyncCtx ctx{};
    ctx.task_token = TaskId{17};
    ctx.completion_count = &count;
    ctx.completion_error_code = &error;
    ctx.completion_entries = &entry;
    ctx.completion_capacity = 1;
    Event event;
    ASSERT_TRUE(urma_backend::register_urma_async_event(ctx, event, 0, &workspace));
    ASSERT_EQ(count, 1U);
    EXPECT_EQ(entry.addr, event.handle);
    EXPECT_EQ(entry.expected_value, 2U);
    EXPECT_EQ(entry.backend_cookie, reinterpret_cast<uint64_t>(&workspace));
    EXPECT_EQ(entry.completion_type, COMPLETION_TYPE_URMA_EVENT_HANDLE);
    EXPECT_EQ(error, 0);
    EXPECT_EQ(event.wait_calls, 0);
}

TEST(UrmaRegistration, SynchronousFallbackPropagatesWaitFailure) {
    AsyncCtx ctx{};
    Event event;
    event.wait_result = false;
    EXPECT_FALSE(urma_backend::register_urma_async_event(ctx, event, 0, nullptr));
    EXPECT_EQ(event.wait_calls, 1);
}

struct ReplayArgsState {
    RuntimeContext *runtime{};
    bool fatal{};
    uint32_t prior_attempt{};
    uint64_t counter_addr{};
    uint64_t flags_addr{};
    int submissions{};
    int completion_submissions{};
};
ReplayArgsState replay_args;

extern "C" RuntimeContext *framework_current_runtime(void) { return replay_args.runtime; }
extern "C" void framework_bind_runtime(RuntimeContext *rt) { replay_args.runtime = rt; }

TaskOutputTensors inspect_replay_args(RuntimeContext *, const MixedKernels &kernels, const CoreTaskArgs &args) {
    ++replay_args.submissions;
    EXPECT_FALSE(args.has_error) << (args.error_msg ? args.error_msg : "");
    if (args.has_error) {
        replay_args.fatal = true;
        return {};
    }
    EXPECT_EQ(args.tensor_count(), 2U);
    if (kernels.aiv0_kernel_id == 2) {
        ++replay_args.completion_submissions;
        EXPECT_EQ(args.scalar_count(), 6U);
        EXPECT_EQ(args.scalar<uint64_t>(0), 0xDEADU);
        EXPECT_EQ(args.scalar<uint64_t>(3), 0xFEEDU);
        EXPECT_EQ(args.scalar<uint64_t>(4), replay_args.flags_addr);
    }
    return {};
}

TEST(UrmaReplayArgs, ActualOrchestrationPreservesTheOrderKernelPrefix) {
    RuntimeOps ops{};
    ops.submit_task = inspect_replay_args;
    ops.is_fatal = [](RuntimeContext *) {
        return replay_args.fatal;
    };
    ops.get_tensor_data = [](RuntimeContext *, const simpler::tmr::Tensor &tensor, uint32_t,
                             const uint32_t *) -> uint64_t {
        return tensor.buffer.addr == replay_args.counter_addr ? replay_args.prior_attempt : 0;
    };
    ops.set_tensor_data = [](RuntimeContext *, const simpler::tmr::Tensor &, uint32_t, const uint32_t *, uint64_t) {};
    RuntimeContext runtime{&ops, ScopeMode::AUTO};
    struct Unbind {
        ~Unbind() { framework_bind_runtime(nullptr); }
    } unbind;
    for (uint64_t mode = 0; mode <= 4; ++mode) {
        for (uint64_t rank = 0; rank < 2; ++rank) {
            for (uint32_t attempt = 1; attempt <= 2; ++attempt) {
                SCOPED_TRACE(::testing::Message() << "mode=" << mode << " rank=" << rank << " attempt=" << attempt);
                replay_args = {};
                replay_args.runtime = &runtime;
                replay_args.prior_attempt = attempt - 1;
                std::array<simpler::tmr::Tensor, 7> tensors;
                const uint32_t shape[] = {128};
                for (uint32_t i = 0; i < tensors.size(); ++i) {
                    tensors[i] = simpler::tmr::make_tensor_external(
                        reinterpret_cast<void *>(uint64_t{0x124000000000} + 0x1000U * i), shape, 1, DataType::UINT32
                    );
                }
                tensors[6].start_offset = 7;
                replay_args.counter_addr = tensors[3].buffer.addr;
                replay_args.flags_addr = tensors[6].buffer.addr + 7 * sizeof(uint32_t);
                ChipTaskArgs args;
                for (auto &tensor : tensors)
                    args.add_input(tensor);
                args.add_scalar(uint64_t{0xFEED}, uint64_t{0xDEAD}, uint64_t{0}, uint64_t{1}, mode, rank);
                ASSERT_FALSE(args.has_error);
                aicpu_orchestration_entry(args);
                EXPECT_FALSE(replay_args.fatal);
                EXPECT_EQ(replay_args.completion_submissions, 1);
                EXPECT_EQ(replay_args.submissions, rank == 0 ? 4 : 3);
            }
        }
    }
}
}  // namespace
