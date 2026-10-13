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
    int build_rc{0};
    int retire_rc{0};
    int disagree_phase{0};
    int discard_count{0};

    void SetUp() override {
        request.recovery_id = 10;
        request.expected_generation = 1;
        request.failed_run_epoch = 99;
        request.impact = DomainRecoveryImpact::DOMAIN_BINDINGS;
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
        ops.rebuild_supported = true;
        ops.agree = [&](DomainRecoveryManager::Phase phase, int local_rc) {
            events.push_back(static_cast<int>(phase));
            return disagree_phase == static_cast<int>(phase) ? 77 : local_rc;
        };
        ops.build = [&](DomainRecoveryResult &candidate) {
            events.push_back(10);
            candidate.device_ctx = 134;
            candidate.local_window_base = 135;
            return build_rc;
        };
        ops.retire = [&] {
            events.push_back(20);
            return retire_rc;
        };
        ops.discard = [&] {
            ++discard_count;
        };
    }
};

TEST_F(DomainRecoveryTest, RebuildPublishesOnlyAfterAllMembersRetire) {
    const auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::REBUILT);
    EXPECT_EQ(events, (std::vector<int>{1, 10, 2, 20, 3}));
    EXPECT_EQ(result.previous_generation, 1u);
    EXPECT_EQ(result.generation, 2u);
    EXPECT_EQ(result.rebind_required, 1u);
    EXPECT_EQ(result.domain_preserved, 0u);
    EXPECT_EQ(result.device_ctx, 134u);
    EXPECT_EQ(result.local_window_base, 135u);
    EXPECT_EQ(result.workspace, binding.workspace);
    EXPECT_EQ(result.hccl_comm, binding.hccl_comm);
    EXPECT_EQ(result.comm_stream, binding.comm_stream);
    EXPECT_EQ(result.communication_context, binding.communication_context);
    EXPECT_EQ(discard_count, 0);
}

TEST_F(DomainRecoveryTest, ExecutionOnlyPreservesWithoutCallingBackend) {
    request.impact = DomainRecoveryImpact::EXECUTION_ONLY;
    request.collective_quiescence = 0;
    for (auto engine : {DomainRecoveryEngine::SDMA, DomainRecoveryEngine::URMA, DomainRecoveryEngine::CCU}) {
        DomainRecoveryManager separate;
        binding.engine = engine;
        auto result = separate.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::PRESERVED);
        EXPECT_EQ(result.generation, 1u);
        EXPECT_EQ(result.domain_preserved, 1u);
        EXPECT_EQ(result.rebind_required, 0u);
        EXPECT_EQ(result.device_ctx, binding.device_ctx);
        EXPECT_EQ(result.workspace, binding.workspace);
    }
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, DuplicateReturnsCommittedResultWithoutNewResources) {
    auto first = manager.recover(request, binding, ops);
    events.clear();
    auto duplicate = manager.recover(request, binding, ops);
    EXPECT_EQ(duplicate.status, DomainRecoveryStatus::ALREADY_RECOVERED);
    EXPECT_EQ(duplicate.generation, first.generation);
    EXPECT_EQ(duplicate.device_ctx, first.device_ctx);
    EXPECT_TRUE(events.empty());
    request.original_rc = 123;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::STALE_GENERATION);
}

TEST_F(DomainRecoveryTest, OldGenerationAndEpisodeCannotRebuildAgain) {
    ASSERT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::REBUILT);
    events.clear();
    ++request.recovery_id;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::STALE_GENERATION);
    request.expected_generation = 2;
    request.recovery_id = 9;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::STALE_GENERATION);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, QuiescenceMustCoverAllMembersAndAsyncUsers) {
    request.collective_quiescence = 0;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::QUIESCENCE_UNPROVEN);
    EXPECT_TRUE(events.empty());
    EXPECT_EQ(manager.generation(), 1u);
}

TEST_F(DomainRecoveryTest, PeerBuildFailureLeavesOldResourcesUnretired) {
    disagree_phase = 2;
    const auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(events, (std::vector<int>{1, 10, 2}));
    EXPECT_EQ(result.original_rc, 507018);
    EXPECT_EQ(result.recovery_rc, 77);
    EXPECT_EQ(result.generation, 1u);
    EXPECT_EQ(result.device_ctx, 0u);
    EXPECT_EQ(result.local_window_base, 0u);
    EXPECT_EQ(discard_count, 1);
    events.clear();
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::RECOVERY_FAILED);
    ++request.recovery_id;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, LocalBuildFailureIsVotedAndKeepsBothFaults) {
    build_rc = 507014;
    auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(result.original_rc, 507018);
    EXPECT_EQ(result.recovery_rc, 507014);
    EXPECT_EQ(events, (std::vector<int>{1, 10, 2}));
    EXPECT_EQ(discard_count, 1);
}

TEST_F(DomainRecoveryTest, RetireFailureCannotPublishCandidate) {
    retire_rc = 88;
    auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(result.recovery_rc, 88);
    EXPECT_EQ(result.generation, 1u);
    EXPECT_EQ(result.device_ctx, 0u);
    EXPECT_EQ(result.rebind_required, 0u);
    EXPECT_EQ(discard_count, 1);
    EXPECT_EQ(events, (std::vector<int>{1, 10, 2, 20, 3}));
}

TEST_F(DomainRecoveryTest, PeerRetireFailureCannotPublishCandidate) {
    disagree_phase = 3;
    auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(result.generation, 1u);
    EXPECT_EQ(result.device_ctx, 0u);
    EXPECT_EQ(discard_count, 1);
}

TEST_F(DomainRecoveryTest, ExceptionsFollowFailureVoteAndDiscard) {
    ops.build = [](auto &) -> int {
        throw std::runtime_error("allocation");
    };
    auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(events, (std::vector<int>{1, 2}));
    EXPECT_EQ(discard_count, 1);
}

TEST_F(DomainRecoveryTest, SharedAndDeviceFaultsDoNotCallSoftBackend) {
    for (auto impact :
         {DomainRecoveryImpact::SHARED_COMMUNICATION, DomainRecoveryImpact::DEVICE,
          DomainRecoveryImpact::DOMAIN_TRANSPORT}) {
        DomainRecoveryManager separate;
        request.impact = impact;
        EXPECT_EQ(separate.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    }
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, SevereFaultCannotBeBypassedByOldDuplicate) {
    request.impact = DomainRecoveryImpact::EXECUTION_ONLY;
    ASSERT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::PRESERVED);
    auto old = request;
    ++request.recovery_id;
    request.impact = DomainRecoveryImpact::SHARED_COMMUNICATION;
    ASSERT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    EXPECT_EQ(manager.recover(old, binding, ops).status, DomainRecoveryStatus::STALE_GENERATION);
    ++request.recovery_id;
    request.impact = DomainRecoveryImpact::EXECUTION_ONLY;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, ProviderCapabilitiesDoNotAssumeSdmaForUrmaOrCcu) {
    ops.rebuild_supported = false;
    for (auto engine : {DomainRecoveryEngine::URMA, DomainRecoveryEngine::CCU}) {
        binding.engine = engine;
        EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::UNSUPPORTED);
    }
    EXPECT_TRUE(events.empty());
    ops.rebuild_supported = true;
    ops.transport_rebuild_supported = true;
    ops.quiesce = [] {
        return 0;
    };
    request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::REBUILT);
}

TEST_F(DomainRecoveryTest, HostErrorCodeNeverSelectsRecoveryEligibility) {
    for (int code : {-17, 507014, 507018, 507899}) {
        DomainRecoveryManager separate;
        request.original_rc = code;
        auto result = separate.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::REBUILT);
        EXPECT_EQ(result.original_rc, code);
    }
}

TEST_F(DomainRecoveryTest, FailedReleaseCannotReturnPreviouslyPreservedBinding) {
    request.impact = DomainRecoveryImpact::EXECUTION_ONLY;
    ASSERT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::PRESERVED);
    manager.quarantine(88);
    auto duplicate = manager.recover(request, binding, ops);
    EXPECT_EQ(duplicate.status, DomainRecoveryStatus::RECOVERY_FAILED);
    EXPECT_EQ(duplicate.original_rc, request.original_rc);
    EXPECT_EQ(duplicate.recovery_rc, 88);
    EXPECT_EQ(duplicate.device_ctx, 0u);
    ++request.recovery_id;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::GIVE_UP);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, IncompatibleAbiHasNoResourceEffects) {
    request.abi_version = 1;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::UNSUPPORTED);
    request.abi_version = 2;
    request.struct_bytes = 0;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::UNSUPPORTED);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, MalformedRequestHasNoResourceEffects) {
    request.recovery_id = 0;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::INVALID_STATE);
    request.recovery_id = 10;
    request.impact = static_cast<DomainRecoveryImpact>(99);
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::INVALID_STATE);
    request.impact = DomainRecoveryImpact::DOMAIN_BINDINGS;
    request.collective_quiescence = 2;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::INVALID_STATE);
    EXPECT_TRUE(events.empty());
}

TEST_F(DomainRecoveryTest, TransportAbortAgreementPrecedesAnyWorkspaceMutation) {
    request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
    request.transport_action = DomainTransportAction::REARM;
    ops.transport_rearm_supported = true;
    ops.quiesce = [&] {
        events.push_back(30);
        return 0;
    };
    ops.build = [&](auto &) {
        events.push_back(10);
        return 0;
    };
    const auto result = manager.recover(request, binding, ops);
    EXPECT_EQ(result.status, DomainRecoveryStatus::REARMED);
    EXPECT_EQ(events, (std::vector<int>{1, 30, 4, 10, 2, 20, 3}));
    EXPECT_EQ(result.workspace, binding.workspace);
    EXPECT_EQ(result.device_ctx, binding.device_ctx);
    EXPECT_EQ(result.local_window_base, binding.local_window_base);
    EXPECT_EQ(result.generation, 2u);
    events.clear();
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::ALREADY_RECOVERED);
    EXPECT_TRUE(events.empty());
    request.transport_action = DomainTransportAction::REPLACE;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::STALE_GENERATION);
}

TEST_F(DomainRecoveryTest, PartialAbortOrPeerAbortFailureCannotRequeryOrReplace) {
    request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
    request.transport_action = DomainTransportAction::REARM;
    ops.transport_rearm_supported = true;
    for (bool local_failure : {true, false}) {
        DomainRecoveryManager separate;
        events.clear();
        ops.quiesce = [&] {
            events.push_back(30);
            return local_failure ? 88 : 0;
        };
        disagree_phase = local_failure ? 0 : 4;
        const auto result = separate.recover(request, binding, ops);
        EXPECT_EQ(result.status, DomainRecoveryStatus::RECOVERY_FAILED);
        EXPECT_EQ(result.recovery_rc, local_failure ? 88 : 77);
        EXPECT_EQ(result.original_rc, request.original_rc);
        EXPECT_EQ(events, (std::vector<int>{1, 30, 4}));
        EXPECT_EQ(result.device_ctx, 0u);
        EXPECT_EQ(result.workspace, 0u);
    }
}

TEST_F(DomainRecoveryTest, TransportProducerFenceAndProviderAreMandatory) {
    request.impact = DomainRecoveryImpact::DOMAIN_TRANSPORT;
    ops.transport_rebuild_supported = true;
    request.collective_quiescence = 0;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::QUIESCENCE_UNPROVEN);
    request.collective_quiescence = 1;
    EXPECT_EQ(manager.recover(request, binding, ops).status, DomainRecoveryStatus::UNSUPPORTED);
    EXPECT_TRUE(events.empty());
}

}  // namespace
