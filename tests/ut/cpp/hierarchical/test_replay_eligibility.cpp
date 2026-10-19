/*
 * Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <gtest/gtest.h>

#include "recovery_coordinator.h"
#include "task_args_wire.h"

namespace {
RecoveryMetadata metadata(int32_t rank, uint64_t seq = 11) {
    return RecoveryMetadata{99, 0, seq, rank, {0, 1}, ReplaySafety::REPLAY_SAFE};
}
TaskArgs args(int32_t rank, uint64_t seq = 11) {
    TaskArgs value;
    value.add_scalar(42);
    value.set_recovery_metadata(metadata(rank, seq));
    return value;
}

class ReplayEligibilityTest : public ::testing::Test {
protected:
    RecoveryRequest request;
    RecoveryGraphSnapshot graph;
    void SetUp() override {
        request.ticket = {1, 10, 7, 0, 19};
        request.pipeline_lease.generation = 3;
        request.members = {{0, 1, true, {}}, {1, 1, true, {}}};
        request.members[0].completion.native_execution_fault.raw_rc = 507018;
        request.members[0].completion.native_execution_fault.runtime_status = -9;
        graph.ticket = request.ticket;
        graph.submission_closed = true;
        graph.active_recoveries = 1;
        for (int32_t slot = 10; slot <= 12; ++slot) {
            RecoveryTaskSnapshot task;
            task.handle = {1, slot};
            task.state = slot == 10 ? TaskState::RETRY_PENDING : TaskState::PENDING;
            task.local_chip = true;
            task.worker_ids = {0, 1};
            task.args = {args(0, slot + 1), args(1, slot + 1)};
            task.pipeline_generation = 3;
            if (slot < 12) task.consumers = {slot + 1};
            if (slot == 10) task.active_recovery_id = 7;
            graph.tasks.push_back(task);
        }
    }
    ReplayAssessment assess() const { return assess_replay_eligibility(request, graph); }
};

TEST_F(ReplayEligibilityTest, FullGroupIsEligibleWithoutReleasingTheHold) {
    auto result = assess();
    ASSERT_EQ(result.status, ReplayStatus::REPLAY_ALLOWED) << result.reason;
    ASSERT_TRUE(result.op);
    EXPECT_EQ(result.op->logical_run_id, 99u);
    EXPECT_EQ(result.op->domain_id, 0u);
    EXPECT_EQ(result.op->op_seq, 11u);
    ASSERT_EQ(result.op->members.size(), 2u);
    EXPECT_EQ(result.op->members[1].rank, 1);
    EXPECT_EQ(result.op->members[1].worker_id, 1);
    EXPECT_EQ(graph.tasks[0].state, TaskState::RETRY_PENDING);
    EXPECT_EQ(graph.tasks[1].state, TaskState::PENDING);
    EXPECT_EQ(graph.active_recoveries, 1u);
}

TEST_F(ReplayEligibilityTest, HostFaultCodeDoesNotSelectEligibility) {
    request.members[0].completion.native_execution_fault.raw_rc = -123456;
    request.members[0].completion.native_execution_fault.runtime_status = -103;
    EXPECT_EQ(assess().status, ReplayStatus::REPLAY_ALLOWED);
}

TEST_F(ReplayEligibilityTest, MissingMetadataIsNotImplicitPermission) {
    graph.tasks[0].args[0].recovery_metadata.reset();
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}

TEST_F(ReplayEligibilityTest, PartialGroupMetadataIsRejected) {
    graph.tasks[0].args[1].recovery_metadata.reset();
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}

TEST_F(ReplayEligibilityTest, MemberIdentityAndSafetyMustAgree) {
    ++graph.tasks[0].args[1].recovery_metadata->op_seq;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
    graph.tasks[0].args[1] = args(1);
    graph.tasks[0].args[1].recovery_metadata->replay_safety = ReplaySafety::UNKNOWN;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}

TEST_F(ReplayEligibilityTest, DuplicateRanksAndEndpointsAreRejected) {
    graph.tasks[0].args[1] = args(0);
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
    graph.tasks[0].args[1] = args(1);
    graph.tasks[0].worker_ids[1] = 0;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}

TEST_F(ReplayEligibilityTest, MissingRemoteRanksNeedCoordination) {
    for (auto &arg : graph.tasks[0].args)
        arg.recovery_metadata->participants = {0, 1, 2};
    EXPECT_EQ(assess().status, ReplayStatus::NEEDS_COORDINATION);
}

TEST_F(ReplayEligibilityTest, LivePeerNeedsStopCoordination) {
    request.members[1].terminal = false;
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_MEMBERS);
}

TEST_F(ReplayEligibilityTest, UnknownForbiddenAndRestoreContractsDoNotAllowReplay) {
    for (auto &arg : graph.tasks[0].args)
        arg.recovery_metadata->replay_safety = ReplaySafety::UNKNOWN;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_UNSAFE);
    for (auto &arg : graph.tasks[0].args)
        arg.recovery_metadata->replay_safety = ReplaySafety::FORBIDDEN;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_UNSAFE);
    for (auto &arg : graph.tasks[0].args)
        arg.recovery_metadata->replay_safety = ReplaySafety::REQUIRES_RESTORE;
    EXPECT_EQ(assess().status, ReplayStatus::NEEDS_RESTORE);
}

TEST_F(ReplayEligibilityTest, DeviceUnusableEvidenceIsNotAnErrorCodeWhitelist) {
    request.members[1].completion.native_execution_fault.device_unusable = 1;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_DEVICE_UNUSABLE);
}

TEST_F(ReplayEligibilityTest, TransitiveUnannotatedConsumerCannotHaveAdvanced) {
    graph.tasks[2].args.clear();
    graph.tasks[2].state = TaskState::RUNNING;
    auto result = assess();
    ASSERT_EQ(result.status, ReplayStatus::NACK_ADVANCED);
    ASSERT_EQ(result.blocking_tasks.size(), 1u);
    EXPECT_EQ(result.blocking_tasks[0].task_slot, 12);
}

TEST_F(ReplayEligibilityTest, CompletedOrConsumedConsumerIsAlreadyAdvanced) {
    for (auto state : {TaskState::COMPLETED, TaskState::CONSUMED, TaskState::FAILED, TaskState::RETRY_PENDING}) {
        graph.tasks[1].state = state;
        EXPECT_EQ(assess().status, ReplayStatus::NACK_ADVANCED);
    }
}

TEST_F(ReplayEligibilityTest, LaterDomainOpIsCheckedEvenWithoutAnEdge) {
    graph.tasks[0].consumers.clear();
    graph.tasks[1].state = TaskState::RUNNING;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_ADVANCED);
    for (auto &arg : graph.tasks[1].args)
        arg.recovery_metadata->domain_id = 2;
    EXPECT_EQ(assess().status, ReplayStatus::REPLAY_ALLOWED);
}

TEST_F(ReplayEligibilityTest, ReusedGenerationAndResolvedRunAreRejected) {
    ++graph.tasks[0].pipeline_generation;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
    --graph.tasks[0].pipeline_generation;
    graph.run_failed = true;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
    graph.run_failed = false;
    graph.tasks[0].state = TaskState::COMPLETED;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
}

TEST_F(ReplayEligibilityTest, GraphMustBeClosedAndFullyObservable) {
    graph.submission_closed = false;
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_GRAPH);
    graph.submission_closed = true;
    graph.tasks[1].state = TaskState::BUILDING;
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_GRAPH);
    graph.tasks.erase(graph.tasks.begin() + 1);
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
}

TEST_F(ReplayEligibilityTest, PerRankSlotsAggregateByLogicalIdentity) {
    graph.tasks[0].args.resize(1);
    graph.tasks[0].worker_ids.resize(1);
    request.members.resize(1);
    RecoveryTaskSnapshot peer = graph.tasks[0];
    peer.handle.task_slot = 13;
    peer.args = {args(1)};
    peer.worker_ids = {1};
    peer.state = TaskState::COMPLETED;
    peer.active_recovery_id = 0;
    graph.tasks.push_back(peer);
    auto result = assess();
    ASSERT_EQ(result.status, ReplayStatus::REPLAY_ALLOWED) << result.reason;
    ASSERT_TRUE(result.op);
    ASSERT_EQ(result.op->members.size(), 2u);
    EXPECT_EQ(result.op->members[1].task.task_slot, 13);
    graph.tasks.back().state = TaskState::CONSUMED;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
    graph.tasks[1].state = TaskState::RUNNING;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_ADVANCED);
}

TEST_F(ReplayEligibilityTest, UnpublishedIndependentTaskCannotHideLaterDomainWork) {
    graph.tasks[0].consumers.clear();
    graph.tasks[1].state = TaskState::BUILDING;
    graph.tasks[1].args.clear();
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_GRAPH);
}

TEST_F(ReplayEligibilityTest, AdvancedUnannotatedIndependentWorkIsNotAssumedSafe) {
    graph.tasks[0].consumers.clear();
    graph.tasks[1].state = TaskState::COMPLETED;
    for (auto &value : graph.tasks[1].args)
        value.recovery_metadata.reset();
    EXPECT_EQ(assess().status, ReplayStatus::NEEDS_COORDINATION);
    graph.tasks[0].producers = {11};
    EXPECT_EQ(assess().status, ReplayStatus::REPLAY_ALLOWED);
}

TEST_F(ReplayEligibilityTest, MissingEndpointGenerationIsRejected) {
    request.members[0].endpoint_generation = 0;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_NOT_RETAINED);
}

TEST_F(ReplayEligibilityTest, ExtraFailureEvidenceDoesNotCreateAPhantomMember) {
    request.members.push_back({2, 1, true, {}});
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}

TEST_F(ReplayEligibilityTest, CoordinatorRejectsForeignStaleAndResolvingTickets) {
    RecoveryCoordinator coordinator;
    request.ticket.owner_id = 0;
    request.ticket = coordinator.submit(request);
    graph.ticket = request.ticket;
    auto other = request.ticket;
    ++other.owner_id;
    EXPECT_THROW(coordinator.assess(other, graph), std::invalid_argument);
    ++other.attempt;
    EXPECT_THROW(coordinator.assess(other, graph), std::invalid_argument);
    EXPECT_EQ(coordinator.assess(request.ticket, graph).status, ReplayStatus::REPLAY_ALLOWED);
    EXPECT_TRUE(coordinator.give_up(request.ticket, "test terminal decision"));
    EXPECT_THROW(coordinator.assess(request.ticket, graph), std::invalid_argument);
}

TEST_F(ReplayEligibilityTest, UnusableEndpointInAnotherHeldEpisodeIsStillRelevant) {
    RecoveryCoordinator coordinator;
    request.ticket.owner_id = 0;
    request.ticket = coordinator.submit(request);
    graph.ticket = request.ticket;
    auto peer = request;
    peer.ticket = {request.ticket.run_id, 12, request.ticket.recovery_id + 1, 0, 0};
    peer.members[1].completion.native_execution_fault.device_unusable = 1;
    coordinator.submit(peer);
    EXPECT_EQ(coordinator.assess(request.ticket, graph).status, ReplayStatus::NACK_DEVICE_UNUSABLE);
}

TEST(RecoveryMetadataTest, ValidationClearAndWireBoundaryAreExplicit) {
    auto value = metadata(0);
    value.participants = {1, 0};
    EXPECT_THROW(value.validate(), std::invalid_argument);
    value = metadata(0);
    value.rank = 2;
    EXPECT_THROW(value.validate(), std::invalid_argument);
    auto original = args(0);
    auto copy = original;
    copy.recovery_metadata->op_seq = 99;
    EXPECT_EQ(original.recovery_metadata->op_seq, 11u);
    std::vector<uint8_t> blob(task_args_blob_size(original));
    write_blob(blob.data(), original);
    auto decoded = read_blob(blob.data(), blob.size());
    EXPECT_EQ(decoded.scalar_count, 1);
    original.clear();
    EXPECT_FALSE(original.recovery_metadata);
}
TEST_F(ReplayEligibilityTest, OpenGraphRequiresFrozenPublicationAndStillRejectsBuildingSlots) {
    graph.submission_closed = false;
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_GRAPH);
    graph.publication_frozen = true;
    EXPECT_EQ(assess().status, ReplayStatus::REPLAY_ALLOWED);
    graph.tasks.back().state = TaskState::BUILDING;
    EXPECT_EQ(assess().status, ReplayStatus::WAITING_GRAPH);
}
TEST_F(ReplayEligibilityTest, IndependentComputeDoesNotAdvanceTheCommunicationProtocol) {
    graph.tasks[0].consumers.clear();
    graph.tasks[1].state = TaskState::COMPLETED;
    for (auto &arg : graph.tasks[1].args) arg.recovery_metadata->uses_communication = false;
    EXPECT_EQ(assess().status, ReplayStatus::REPLAY_ALLOWED);
    graph.tasks[0].consumers = {graph.tasks[1].handle.task_slot};
    EXPECT_EQ(assess().status, ReplayStatus::NACK_ADVANCED);
}
TEST_F(ReplayEligibilityTest, FailureDomainScopeMustAgreeAcrossMembers) {
    graph.tasks[0].args[1].recovery_metadata->uses_communication = false;
    EXPECT_EQ(assess().status, ReplayStatus::NACK_METADATA);
}
}  // namespace
