/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "platform_comm/domain_recovery.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
class DomainRecoveryTest : public ::testing::Test {
protected:
    DomainRecoveryManager manager;
    DomainRecoveryRequest request;
    DomainRecoveryResult binding;
    DomainRecoveryManager::Operations ops;
    std::vector<int> events;
    int stop_rc{0};
    int rearm_rc{0};
    uint32_t stopped{48};

    void SetUp() override {
        request.recovery_id = 10;
        request.expected_generation = 1;
        request.failed_run_epoch = 99;
        request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
        request.collective_quiescence = 1;
        request.original_rc = 507018;
        binding.engine = DomainRecoveryEngine::SDMA;
        binding.allocation_id = 23;
        binding.communication_context = 31;
        binding.hccl_comm = 32;
        binding.comm_stream = 33;
        binding.device_ctx = 34;
        binding.local_window_base = 35;
        binding.workspace = 36;
        ops.rearm_supported = true;
        ops.quiesce = [&](DomainRecoveryResult &proof) {
            events.push_back(1);
            proof.transport_channel_count = 48;
            proof.stopped_channel_count = stopped;
            proof.transport_stopped = stopped == 48 && stop_rc == 0;
            return stop_rc;
        };
        ops.rearm = [&] {
            events.push_back(2);
            return rearm_rc;
        };
    }
    DomainRecoveryResult recover() { return manager.recover(request, binding, ops); }
    void expect_no_bindings(const DomainRecoveryResult &result) {
        EXPECT_EQ(result.device_ctx, 0u);
        EXPECT_EQ(result.local_window_base, 0u);
        EXPECT_EQ(result.workspace, 0u);
        EXPECT_EQ(result.domain_preserved, 0u);
    }
};

TEST_F(DomainRecoveryTest, PrepareStopsBeforeRearmAndCommitOnlyAdvancesGeneration) {
    const auto prepared = recover();
    ASSERT_EQ(prepared.status, DomainRecoveryStatus::PREPARED);
    EXPECT_TRUE(manager.prepared());
    EXPECT_EQ(events, (std::vector<int>{1, 2}));
    EXPECT_EQ(prepared.generation, 1u);
    EXPECT_EQ(manager.generation(), 1u);
    EXPECT_EQ(prepared.transport_stopped, 1u);
    EXPECT_EQ(prepared.domain_preserved, 0u);
    EXPECT_EQ(recover().status, DomainRecoveryStatus::PREPARED);
    EXPECT_EQ(events.size(), 2u);
    request.phase = DomainRecoveryPhase::COMMIT;
    const auto committed = recover();
    ASSERT_EQ(committed.status, DomainRecoveryStatus::REARMED);
    EXPECT_FALSE(manager.prepared());
    EXPECT_EQ(committed.previous_generation, 1u);
    EXPECT_EQ(committed.generation, 2u);
    EXPECT_EQ(committed.domain_preserved, 1u);
    EXPECT_EQ(committed.rebind_required, 1u);
    EXPECT_EQ(committed.communication_context, binding.communication_context);
    EXPECT_EQ(committed.hccl_comm, binding.hccl_comm);
    EXPECT_EQ(committed.comm_stream, binding.comm_stream);
    EXPECT_EQ(committed.device_ctx, binding.device_ctx);
    EXPECT_EQ(committed.local_window_base, binding.local_window_base);
    EXPECT_EQ(committed.workspace, binding.workspace);
    EXPECT_EQ(recover().status, DomainRecoveryStatus::ALREADY_RECOVERED);
    request.phase = DomainRecoveryPhase::PREPARE;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::ALREADY_RECOVERED);
    EXPECT_EQ(events.size(), 2u);
}

TEST_F(DomainRecoveryTest, CommitAndAbortCannotCreateARecoveryTransaction) {
    request.phase = DomainRecoveryPhase::COMMIT;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    request.phase = DomainRecoveryPhase::ABORT;
    request.collective_rc = 77;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, BusyPreparationCannotBeSupersededOrChanged) {
    ASSERT_EQ(recover().status, DomainRecoveryStatus::PREPARED);
    ++request.recovery_id;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::BUSY);
    --request.recovery_id;
    ++request.failed_run_epoch;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::STALE_GENERATION);
    EXPECT_EQ(events.size(), 2u);
}

TEST_F(DomainRecoveryTest, CollectiveProducerFenceIsRequiredBeforeTouchingTransport) {
    request.collective_quiescence = 0;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::QUIESCENCE_UNPROVEN);
    EXPECT_TRUE(events.empty());
    request.collective_quiescence = 1;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::PREPARED);
}

TEST_F(DomainRecoveryTest, PartialStopCannotRequeryAndPermanentlyQuarantines) {
    stopped = 47;
    const auto failed = recover();
    EXPECT_EQ(failed.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(failed.original_rc, 507018);
    EXPECT_NE(failed.recovery_rc, 0);
    EXPECT_EQ(events, (std::vector<int>{1}));
    expect_no_bindings(failed);
    request.phase = DomainRecoveryPhase::COMMIT;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::RECOVERY_FAILED);
    ++request.recovery_id;
    request.phase = DomainRecoveryPhase::PREPARE;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::GIVE_UP);
    EXPECT_EQ(events.size(), 1u);
}

TEST_F(DomainRecoveryTest, StopAndRearmErrorsPreserveBothFaults) {
    for (bool stop_failed : {true, false}) {
        DomainRecoveryManager local;
        stop_rc = stop_failed ? 73 : 0;
        rearm_rc = stop_failed ? 0 : 81;
        const auto result = local.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
        EXPECT_EQ(result.original_rc, 507018);
        EXPECT_EQ(result.recovery_rc, stop_failed ? 73 : 81);
        expect_no_bindings(result);
    }
}

TEST_F(DomainRecoveryTest, ThrowingProviderFailsWithoutPublishingBindings) {
    ops.rearm = []() -> int {
        throw std::runtime_error("query failure");
    };
    const auto result = recover();
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    expect_no_bindings(result);
}

TEST_F(DomainRecoveryTest, PeerAbortQuarantinesPreparedAndLocallyCommittedMembers) {
    for (bool committed : {false, true}) {
        DomainRecoveryManager local;
        request.phase = DomainRecoveryPhase::PREPARE;
        request.collective_rc = 0;
        ASSERT_EQ(local.recover(request, binding, ops).status, DomainRecoveryStatus::PREPARED);
        if (committed) {
            request.phase = DomainRecoveryPhase::COMMIT;
            ASSERT_EQ(local.recover(request, binding, ops).status, DomainRecoveryStatus::REARMED);
        }
        request.phase = DomainRecoveryPhase::ABORT;
        request.collective_rc = 93;
        const auto result = local.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
        EXPECT_EQ(result.original_rc, 507018);
        EXPECT_EQ(result.recovery_rc, 93);
        expect_no_bindings(result);
        EXPECT_EQ(local.recover(request, binding, ops).recovery_rc, 93);
        request.phase = DomainRecoveryPhase::COMMIT;
        request.collective_rc = 0;
        EXPECT_EQ(local.recover(request, binding, ops).status, DomainRecoveryStatus::RECOVERY_FAILED);
    }
}

TEST_F(DomainRecoveryTest, InvalidAbiPhaseAndReservedFieldsDoNotMutateResources) {
    auto original = request;
    request.abi_version = 2;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::UNSUPPORTED);
    request = original;
    request.reserved = 1;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    request = original;
    request.phase = static_cast<DomainRecoveryPhase>(99);
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    request = original;
    request.collective_rc = 93;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, RepeatedTransactionsUseGenerationsWithoutRetryBudgetsOrCodeWhitelists) {
    for (int original_rc : {507018, 507014, -42}) {
        request.original_rc = original_rc;
        request.phase = DomainRecoveryPhase::PREPARE;
        const auto prepared = recover();
        ASSERT_EQ(prepared.status, DomainRecoveryStatus::PREPARED);
        request.phase = DomainRecoveryPhase::COMMIT;
        const auto committed = recover();
        ASSERT_EQ(committed.status, DomainRecoveryStatus::REARMED);
        EXPECT_EQ(committed.original_rc, original_rc);
        EXPECT_EQ(committed.generation, request.expected_generation + 1);
        ++request.recovery_id;
        EXPECT_EQ(recover().status, DomainRecoveryStatus::STALE_GENERATION);
        request.expected_generation = committed.generation;
    }
    EXPECT_EQ(events.size(), 6u);
}

TEST_F(DomainRecoveryTest, ExecutionOnlyPreservesWithoutCallingProviderOrCommitting) {
    request.impact = DomainRecoveryImpact::EXECUTION_ONLY;
    request.collective_quiescence = 0;
    const auto result = recover();
    EXPECT_EQ(result.status, DomainRecoveryStatus::PRESERVED);
    EXPECT_EQ(result.generation, 1u);
    EXPECT_EQ(result.workspace, binding.workspace);
    EXPECT_TRUE(events.empty());
    EXPECT_EQ(recover().status, DomainRecoveryStatus::ALREADY_RECOVERED);
    request.phase = DomainRecoveryPhase::COMMIT;
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
}

TEST_F(DomainRecoveryTest, UnsupportedTransportAndSevereDomainsGiveUpWithoutResourceOperations) {
    for (auto engine : {DomainRecoveryEngine::URMA, DomainRecoveryEngine::CCU}) {
        DomainRecoveryManager local;
        binding.engine = engine;
        ops.rearm_supported = false;
        const auto result = local.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::GIVE_UP);
        expect_no_bindings(result);
    }
    for (auto impact : {DomainRecoveryImpact::SHARED_COMMUNICATION, DomainRecoveryImpact::DEVICE}) {
        DomainRecoveryManager local;
        request.impact = impact;
        EXPECT_EQ(local.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    }
    request.impact = static_cast<DomainRecoveryImpact>(1);
    EXPECT_EQ(recover().status, DomainRecoveryStatus::INVALID_STATE);
    EXPECT_TRUE(events.empty());
}
}  // namespace
