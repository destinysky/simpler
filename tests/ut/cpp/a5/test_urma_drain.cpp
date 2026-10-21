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

#include "host/urma_drain.h"
#include "common/urma_cqe_probe.h"

extern "C" int simpler_aicpu_read_urma_cqe(void *);
#include "platform_comm/domain_recovery.h"

namespace {
class UrmaDrain : public ::testing::Test {
protected:
    urma_drain::Queue queue{3, 0, 2, 0, 8, 8};
    std::array<uint32_t, 8> cqes{4, 4};
    int reads{};
    urma_drain::Observation inspect() {
        return urma_drain::inspect(queue, [&](uint32_t sequence, uint32_t &dw0) {
            ++reads;
            dw0 = cqes[sequence & 7];
            return 0;
        });
    }
};

TEST_F(UrmaDrain, CoversActualCqesWithoutEquatingBbAndCqeCounts) {
    const auto result = inspect();
    EXPECT_EQ(result.state, urma_drain::State::DRAINED);
    EXPECT_EQ(result.checked, 2U);
    EXPECT_EQ(urma_drain::proof_detail(result, false), DomainRecoveryDetail::NONE);
    EXPECT_EQ(queue.retired_bb, 0U);
    EXPECT_EQ(queue.retired_cqe, 0U);
}
TEST_F(UrmaDrain, SoftwareEqualityWithoutHardwareCompletionIsNotProof) {
    queue.retired_bb = 3;
    queue.retired_cqe = 2;
    cqes[1] = 0;
    EXPECT_EQ(inspect().state, urma_drain::State::PENDING);
}
TEST_F(UrmaDrain, ErrorCqeCannotBecomeASuccessfulDrain) {
    queue.retired_bb = 3;
    queue.retired_cqe = 2;
    cqes[1] |= 7U << 24;
    const auto result = inspect();
    EXPECT_EQ(result.state, urma_drain::State::CQE_ERROR);
    EXPECT_EQ(result.dw0, cqes[1]);
    EXPECT_EQ(cqes[1], (7U << 24) | 4U);
}
TEST_F(UrmaDrain, PendingPrefixDoesNotRetireAnything) {
    cqes[1] = 0;
    const auto pending = inspect();
    EXPECT_EQ(pending.state, urma_drain::State::PENDING);
    EXPECT_EQ(urma_drain::proof_detail(pending, false), DomainRecoveryDetail::IN_FLIGHT);
    EXPECT_EQ(queue.retired_bb, 0U);
    EXPECT_EQ(queue.retired_cqe, 0U);
}
TEST_F(UrmaDrain, InvalidGeometryAndImpossibleProgressRejectBeforeReading) {
    queue.cq_depth = 3;
    EXPECT_EQ(inspect().state, urma_drain::State::INVALID);
    queue.cq_depth = 8;
    queue.submitted_cqe = 9;
    EXPECT_EQ(inspect().state, urma_drain::State::INVALID);
    EXPECT_EQ(reads, 0);
}
TEST_F(UrmaDrain, ReadFailureKeepsItsOriginalCode) {
    const auto result = urma_drain::inspect(queue, [](uint32_t, uint32_t &) {
        return -77;
    });
    EXPECT_EQ(result.state, urma_drain::State::READ_FAILED);
    EXPECT_EQ(result.read_rc, -77);
}
TEST_F(UrmaDrain, WrapPreservesTheOutstandingCompletion) {
    queue = {0, UINT32_MAX, 0, UINT32_MAX, 8, 8};
    cqes[7] = 0;
    EXPECT_EQ(inspect().state, urma_drain::State::DRAINED);
    EXPECT_EQ(reads, 1);
}
TEST_F(UrmaDrain, NeverSubmittedQueueNeedsNoCqe) {
    queue = {0, 0, 0, 0, 8, 8};
    EXPECT_EQ(inspect().state, urma_drain::State::DRAINED);
    EXPECT_EQ(reads, 0);
}
TEST_F(UrmaDrain, UnprovedDrainCannotReachRearmOrCommit) {
    DomainRecoveryManager manager;
    DomainRecoveryRequest request;
    request.recovery_id = 1;
    request.expected_generation = 1;
    request.collective_quiescence = 1;
    request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
    request.original_rc = 507018;
    DomainRecoveryResult binding;
    binding.engine = DomainRecoveryEngine::URMA;
    DomainRecoveryManager::Operations operations;
    operations.rearm_supported = true;
    cqes[1] = 0;
    operations.quiesce = [&](DomainRecoveryResult &) {
        return inspect().state == urma_drain::State::DRAINED ? 0 : -1;
    };
    int rearms = 0;
    operations.rearm = [&] {
        ++rearms;
        return 0;
    };
    EXPECT_EQ(manager.recover(request, binding, operations).status, DomainRecoveryStatus::RECOVERY_FAILED);
    request.phase = DomainRecoveryPhase::COMMIT;
    EXPECT_EQ(manager.recover(request, binding, operations).status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(rearms, 0);
    EXPECT_EQ(manager.generation(), 1U);
}
TEST_F(UrmaDrain, FailedProofPreservesDiagnosticsAndCannotCommit) {
    struct Case {
        urma_drain::State state;
        bool expired;
        DomainRecoveryDetail detail;
        int rc;
    };
    const Case cases[] = {
        {urma_drain::State::PENDING, true, DomainRecoveryDetail::DEADLINE_EXCEEDED, -1},
        {urma_drain::State::CQE_ERROR, false, DomainRecoveryDetail::CQE_ERROR, -1},
        {urma_drain::State::READ_FAILED, false, DomainRecoveryDetail::READ_FAILED, -77},
        {urma_drain::State::INVALID, false, DomainRecoveryDetail::INVALID_STATE, -1},
        {urma_drain::State::DRAINED, true, DomainRecoveryDetail::DEADLINE_EXCEEDED, -1},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(static_cast<unsigned>(test.detail));
        DomainRecoveryManager manager;
        DomainRecoveryRequest request;
        request.recovery_id = request.expected_generation = 1;
        request.collective_quiescence = 1;
        request.original_rc = 507018;
        request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
        DomainRecoveryResult binding;
        binding.engine = DomainRecoveryEngine::URMA;
        DomainRecoveryManager::Operations operations;
        operations.rearm_supported = true;
        operations.quiesce = [&](DomainRecoveryResult &proof) {
            urma_drain::Observation observation;
            observation.state = test.state;
            observation.read_rc = test.state == urma_drain::State::READ_FAILED ? test.rc : 0;
            proof.transport_detail = urma_drain::proof_detail(observation, test.expired);
            return test.rc;
        };
        int rearms = 0;
        operations.rearm = [&] {
            ++rearms;
            return 0;
        };
        const auto failed = manager.recover(request, binding, operations);
        EXPECT_EQ(failed.status, DomainRecoveryStatus::RECOVERY_FAILED);
        EXPECT_EQ(failed.transport_detail, test.detail);
        EXPECT_EQ(failed.original_rc, 507018);
        EXPECT_EQ(failed.recovery_rc, test.rc);
        request.phase = DomainRecoveryPhase::COMMIT;
        const auto repeated = manager.recover(request, binding, operations);
        EXPECT_EQ(repeated.status, DomainRecoveryStatus::RECOVERY_FAILED);
        EXPECT_EQ(repeated.transport_detail, test.detail);
        EXPECT_EQ(rearms, 0);
        EXPECT_EQ(manager.generation(), 1U);
    }
}

TEST(UrmaCqeProbe, DeviceEntryPublishesRawCqeWithRequestIdentity) {
    alignas(64) uint32_t cqe = (7U << 24) | 4U;
    UrmaCqeProbeResult result{};
    UrmaCqeProbeArgs args{reinterpret_cast<uint64_t>(&cqe), reinterpret_cast<uint64_t>(&result), 19};
    EXPECT_EQ(simpler_aicpu_read_urma_cqe(&args), 0);
    EXPECT_TRUE(urma_cqe_probe_matches(args, result));
    EXPECT_EQ(result.dw0, cqe);
    EXPECT_EQ(cqe, (7U << 24) | 4U);
    ++args.request_id;
    EXPECT_FALSE(urma_cqe_probe_matches(args, result));
    args.request_id = 19;
    args.cqe_addr += 4;
    EXPECT_FALSE(urma_cqe_probe_matches(args, result));
}
TEST(UrmaCqeProbe, InvalidRequestDoesNotReadTheQueue) {
    EXPECT_EQ(simpler_aicpu_read_urma_cqe(nullptr), -1);
    UrmaCqeProbeArgs args{};
    EXPECT_EQ(simpler_aicpu_read_urma_cqe(&args), -1);
    args.cqe_addr = 1;
    args.result_addr = 64;
    args.request_id = 1;
    EXPECT_EQ(simpler_aicpu_read_urma_cqe(&args), -1);
}
}  // namespace
