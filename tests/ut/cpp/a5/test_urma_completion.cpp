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

#include "async_wait.h"
#include "backend/urma/urma_completion_kernel.h"

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
}  // namespace
