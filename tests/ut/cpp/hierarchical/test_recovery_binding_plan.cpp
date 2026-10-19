/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <gtest/gtest.h>
#include <future>
#include <chrono>
#include "recovery_binding_plan.h"
#include "orchestrator.h"

namespace {
RecoveryDomainBinding original() { return {0, 1, 0x1000, 0x2000, 0x1000, 0x4000}; }
DomainRecoveryResult committed(uint64_t id = 11, bool moved = false) {
    const auto before = original();
    DomainRecoveryResult result;
    result.status = DomainRecoveryStatus::REARMED;
    result.engine = DomainRecoveryEngine::SDMA;
    result.recovery_id = id;
    result.allocation_id = before.allocation_id;
    result.previous_generation = before.generation;
    result.generation = 2;
    result.device_ctx = moved ? 0x5000 : before.device_ctx;
    result.local_window_base = moved ? 0x6000 : before.local_window_base;
    result.window_size = before.window_size;
    result.workspace = before.workspace;
    result.domain_preserved = result.rebind_required = result.transport_stopped = 1;
    result.transport_channel_count = result.stopped_channel_count = 48;
    return result;
}
Tensor window_tensor(uint64_t base = 0x2020, uint32_t generation = 1) {
    Tensor tensor{};
    tensor.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    tensor.buffer.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    tensor.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    tensor.buffer.backend_kind = static_cast<uint8_t>(BackendKind::VMM_WINDOW);
    tensor.buffer.identity.owner_instance_id[0] = 1;
    tensor.buffer.identity.buffer_id = 71;
    tensor.buffer.identity.generation = generation;
    tensor.buffer.nbytes = 16;
    tensor.buffer.body_len = 8;
    std::memcpy(tensor.buffer.body, &base, 8);
    tensor.ndims = 1;
    tensor.shapes[0] = 4;
    tensor.strides[0] = 1;
    tensor.dtype = DataType::FLOAT32;
    return tensor;
}
TaskDomainBinding manifest() {
    return {
        original(),
        {{RecoveryArgKind::WINDOW_TENSOR, 0, 0x20},
         {RecoveryArgKind::COMM_CONTEXT, 0, 0},
         {RecoveryArgKind::WINDOW_ADDRESS, 1, 0x30}}
    };
}
TaskArgs retained() {
    TaskArgs args;
    args.add_tensor(window_tensor(), TensorArgType::OUTPUT_EXISTING);
    args.add_scalar(0x1000);
    args.add_scalar(0x2030);
    args.add_scalar(0x1000);  // Equal-valued ordinary data is deliberately unbound.
    args.add_dep(TaskHandle{7, 8});
    args.domain_bindings.push_back(manifest());
    return args;
}
TEST(RecoveryBindingPlanTest, PreservedAddressesAdvanceGenerationAndRemainIdempotent) {
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    auto source = retained();
    source.set_recovery_metadata(RecoveryMetadata{91, 0, 1, 0, {0, 1}, ReplaySafety::REPLAY_SAFE});
    auto bound = plan.rebind(source, 0);
    ASSERT_TRUE(bound.recovery_metadata);
    EXPECT_TRUE(source.recovery_metadata->same_contract(*bound.recovery_metadata));
    EXPECT_EQ(bound.recovery_metadata->rank, 0);
    EXPECT_EQ(source.domain_bindings[0].domain.generation, 1u);
    EXPECT_EQ(bound.domain_bindings[0].domain.generation, 2u);
    EXPECT_EQ(bound.tensor(0).buffer.identity, source.tensor(0).buffer.identity);
    EXPECT_EQ(bound.tag(0), source.tag(0));
    EXPECT_EQ(bound.explicit_dep(0).task_slot, 8);
    EXPECT_EQ(bound.scalar(2), 0x1000u);
    EXPECT_EQ(plan.rebind(bound, 0).domain_bindings[0].domain.generation, 2u);
    EXPECT_EQ(plan.rebind(bound, 0).recovery_metadata->op_seq, 1u);
}
TEST(RecoveryBindingPlanTest, WireRelocationRequiresFreshRegisteredDescriptor) {
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed(11, true));
    auto source = retained();
    EXPECT_THROW(plan.rebind(source, 0), std::invalid_argument);
    EXPECT_EQ(source.scalar(0), 0x1000u);
    plan.add_tensor_replacement(0, 0, 0, window_tensor(0x6020, 2));
    auto next = plan.rebind(source, 0);
    EXPECT_EQ(next.scalar(0), 0x5000u);
    EXPECT_EQ(next.scalar(1), 0x6030u);
    EXPECT_EQ(next.scalar(2), 0x1000u);
    EXPECT_EQ(next.tensor(0).buffer.identity.generation, 2u);
    EXPECT_EQ(next.tensor(0).shapes[0], 4u);
}
TEST(RecoveryBindingPlanTest, MaterializedTensorAndScalarAddressesChangeWithoutMutatingSource) {
    ChipStorageTaskArgs source{};
    auto wire = retained();
    ChipTensor tensor{};
    uint32_t shape[] = {4};
    tensor.init_external(reinterpret_cast<void *>(0x2020), 16, shape, 1, DataType::FLOAT32, AddressSpace::DEVICE);
    source.add_tensor(tensor);
    for (int i = 0; i < 3; ++i)
        source.add_scalar(wire.scalar(i));
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed(11, true));
    auto next = plan.rebind_materialized(source, {manifest()}, 0);
    EXPECT_EQ(source.tensor(0).buffer.addr, 0x2020u);
    EXPECT_EQ(next.args.tensor(0).buffer.addr, 0x6020u);
    EXPECT_EQ(next.args.scalar(0), 0x5000u);
    EXPECT_EQ(next.args.scalar(2), 0x1000u);
    EXPECT_EQ(next.domain_bindings[0].domain.generation, 2u);
}
TEST(RecoveryBindingPlanTest, SameAddressReplacementStillRequiresFreshCanonicalIdentity) {
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    plan.add_tensor_replacement(0, 0, 0, window_tensor(0x2020, 2));
    EXPECT_EQ(plan.rebind(retained(), 0).tensor(0).buffer.identity.generation, 2u);
    RecoveryBindingPlan bad(11);
    bad.add_domain(0, original(), committed());
    bad.add_tensor_replacement(0, 0, 0, window_tensor());
    EXPECT_THROW(bad.rebind(retained(), 0), std::invalid_argument);
}
TEST(RecoveryBindingPlanTest, PreservationAndBackendNeutralResultsDoNotUseFaultCodePolicy) {
    for (auto engine : {DomainRecoveryEngine::SDMA, DomainRecoveryEngine::URMA, DomainRecoveryEngine::CCU}) {
        auto result = committed();
        result.engine = engine;
        result.original_rc = -56789;
        RecoveryBindingPlan plan(11);
        plan.add_domain(0, original(), result);
        EXPECT_EQ(plan.rebind(retained(), 0).domain_bindings[0].domain.generation, 2u);
    }
    auto result = committed();
    result.status = DomainRecoveryStatus::PRESERVED;
    result.generation = 1;
    result.rebind_required = 0;
    result.transport_stopped = result.transport_channel_count = result.stopped_channel_count = 0;
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), result);
    EXPECT_EQ(plan.rebind(retained(), 0).domain_bindings[0].domain.generation, 1u);
}
TEST(RecoveryBindingPlanTest, RejectsPreparedFailedStalePartialAndWrongIdentityResults) {
    for (auto status :
         {DomainRecoveryStatus::PREPARED, DomainRecoveryStatus::RECOVERY_FAILED, DomainRecoveryStatus::GIVE_UP}) {
        auto result = committed();
        result.status = status;
        RecoveryBindingPlan plan(11);
        EXPECT_THROW(plan.add_domain(0, original(), result), std::invalid_argument);
    }
    auto result = committed();
    result.stopped_channel_count = 47;
    EXPECT_THROW(RecoveryBindingPlan(11).add_domain(0, original(), result), std::invalid_argument);
    result = committed(12);
    EXPECT_THROW(RecoveryBindingPlan(11).add_domain(0, original(), result), std::invalid_argument);
    result = committed();
    result.generation = 3;
    EXPECT_THROW(RecoveryBindingPlan(11).add_domain(0, original(), result), std::invalid_argument);
    auto stale = retained();
    stale.domain_bindings[0].domain.generation = 4;
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    EXPECT_THROW(plan.rebind(stale, 0), std::invalid_argument);
    EXPECT_THROW(plan.rebind(retained(), 1), std::invalid_argument);
}
TEST(RecoveryBindingPlanTest, RejectsBadIndicesOffsetsAliasesAndMissingManifests) {
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    auto source = retained();
    source.domain_bindings[0].arguments[0].index = 9;
    EXPECT_THROW(plan.rebind(source, 0), std::out_of_range);
    source = retained();
    source.domain_bindings[0].arguments[0].offset = 0xfff;
    EXPECT_THROW(plan.rebind(source, 0), std::invalid_argument);
    source = retained();
    source.domain_bindings[0].arguments.push_back({RecoveryArgKind::COMM_CONTEXT, 0, 0});
    EXPECT_THROW(plan.rebind(source, 0), std::invalid_argument);
    source = retained();
    source.domain_bindings.clear();
    EXPECT_THROW(plan.rebind(source, 0), std::invalid_argument);
}
TEST(RecoveryBindingPlanTest, OrdinaryWireBlobAndClearContractRemainUnchanged) {
    auto source = retained();
    const size_t bytes = task_args_blob_size(source);
    EXPECT_EQ(bytes, 8 + sizeof(Tensor) + 24);
    std::vector<uint8_t> blob(bytes);
    write_blob(blob.data(), source);
    auto view = read_blob(blob.data(), bytes);
    EXPECT_EQ(view.scalar_count, 3);
    EXPECT_EQ(view.scalars[2], 0x1000u);
    source.clear();
    EXPECT_TRUE(source.domain_bindings.empty());
    EXPECT_EQ(source.tensor_count(), 0);
}

class RetainedSlot : public ::testing::Test {
protected:
    TensorMap map;
    Ring ring;
    Scope scope;
    ReadyQueue sub;
    NextLevelReadyQueues next;
    Orchestrator orch;
    RunId run;
    void SetUp() override {
        ring.init(1ULL << 20);
        next.reset({0, 1});
        orch.init(&map, &ring, &scope, &sub, &next);
        run = orch.begin_run();
    }
    void TearDown() override { ring.shutdown(); }
    TaskHandle task(bool group = false) {
        CallableIdentity callable{};
        callable.kind = CallableKind::CHIP_CALLABLE;
        callable.target_namespace = TargetNamespace::LOCAL_CHIP;
        callable.digest[0] = 1;
        auto args = retained();
        args.explicit_deps_.clear();
        return group ? orch.submit_next_level_group(callable, {args, args}, CallConfig{}, {0, 1}) :
                       orch.submit_next_level(callable, args, CallConfig{}, 0);
    }
    void hold(TaskHandle handle) {
        auto *state = ring.slot_state(handle.task_slot);
        state->state.store(TaskState::RUNNING);
        ASSERT_TRUE(orch.begin_task_recovery(handle.task_slot, 11).has_value());
    }
};
TEST_F(RetainedSlot, AppliesWithoutChangingIdentityLeaseGraphOrRecoveryHold) {
    auto handle = task();
    hold(handle);
    auto *state = ring.slot_state(handle.task_slot);
    auto lease = state->pipeline_lease;
    auto keys = state->output_keys;
    auto callable = state->callable.digest;
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    orch.rebind_task_args(handle, plan);
    EXPECT_EQ(state->run_id, run);
    EXPECT_EQ(state->state.load(), TaskState::RETRY_PENDING);
    EXPECT_TRUE(state->recovery_active);
    EXPECT_EQ(state->active_recovery_id.load(), 11u);
    EXPECT_EQ(state->pipeline_lease.generation, lease.generation);
    EXPECT_EQ(state->output_keys, keys);
    EXPECT_EQ(state->callable.digest, callable);
    EXPECT_EQ(state->task_args.domain_bindings[0].domain.generation, 2u);
    EXPECT_NO_THROW(orch.rebind_task_args(handle, plan));
    EXPECT_TRUE(orch.finish_task_recovery(handle.task_slot, 11));
    EXPECT_THROW(orch.rebind_task_args(handle, plan), std::logic_error);
}
TEST_F(RetainedSlot, GroupValidationIsAtomicAndRejectsExecutingPeer) {
    auto handle = task(true);
    hold(handle);
    auto *state = ring.slot_state(handle.task_slot);
    state->group_member_states.assign(2, GroupMemberState::FAILED);
    RecoveryBindingPlan incomplete(11);
    incomplete.add_domain(0, original(), committed());
    EXPECT_THROW(orch.rebind_task_args(handle, incomplete), std::invalid_argument);
    EXPECT_EQ(state->task_args_list[0].domain_bindings[0].domain.generation, 1u);
    RecoveryBindingPlan complete(11);
    complete.add_domain(0, original(), committed());
    complete.add_domain(1, original(), committed());
    state->group_member_states[1] = GroupMemberState::RUNNING;
    EXPECT_THROW(orch.rebind_task_args(handle, complete), std::logic_error);
    state->group_member_states[1] = GroupMemberState::FAILED;
    orch.rebind_task_args(handle, complete);
    EXPECT_EQ(state->task_args_list[0].domain_bindings[0].domain.generation, 2u);
    EXPECT_EQ(state->task_args_list[1].domain_bindings[0].domain.generation, 2u);
}
TEST_F(RetainedSlot, RejectsStaleHandleAndWrongRecoveryEpisode) {
    auto handle = task();
    hold(handle);
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    EXPECT_THROW(orch.rebind_task_args(TaskHandle{run + 1, handle.task_slot}, plan), std::invalid_argument);
    RecoveryBindingPlan other(12);
    other.add_domain(0, original(), committed(12));
    EXPECT_THROW(orch.rebind_task_args(handle, other), std::logic_error);
}
TEST_F(RetainedSlot, RejectsBackingIdentityChangeUntilGraphAliasPublicationExists) {
    auto handle = task();
    hold(handle);
    auto *state = ring.slot_state(handle.task_slot);
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed(11, true));
    plan.add_tensor_replacement(0, 0, 0, window_tensor(0x6020, 2));
    EXPECT_THROW(orch.rebind_task_args(handle, plan), std::logic_error);
    EXPECT_EQ(state->task_args.scalar(0), 0x1000u);
    EXPECT_EQ(state->task_args.domain_bindings[0].domain.generation, 1u);
    EXPECT_EQ(state->task_args.tensor(0).buffer.identity.generation, 1u);
}
TEST_F(RetainedSlot, IncrementalPrefixRecoversBeforeClosureAndAdvancesFutureBindings) {
    ASSERT_TRUE(orch.enable_incremental_recovery());
    auto root = task();
    hold(root);
    RecoveryTicket ticket{run, root.task_slot, 11, 0, 1};
    auto snapshot = orch.recovery_graph(ticket);
    EXPECT_FALSE(snapshot.submission_closed);
    EXPECT_TRUE(snapshot.publication_frozen);
    RecoveryBindingPlan plan(11);
    plan.add_domain(0, original(), committed());
    orch.rebind_recovery_graph(ticket, plan);
    auto future = std::async(std::launch::async, [&] { return task(); });
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_TRUE(orch.finish_task_recovery(root.task_slot, 11));
    ASSERT_EQ(future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    auto next = future.get();
    EXPECT_EQ(ring.slot_state(next.task_slot)->task_args.domain_bindings[0].domain.generation, 2u);
    EXPECT_FALSE(orch.recovery_graph(ticket).submission_closed);
}

TEST(RecoveryBindingPlanTest, HistoricalChainAcceptsIntermediateButRejectsFabricatedFuture) {
    RecoveryBindingPlan first(11);
    first.add_domain(0, original(), committed());
    auto before = original();
    before.generation = 2;
    auto result = committed(12);
    result.previous_generation = 2;
    result.generation = 3;
    RecoveryBindingPlan second(12);
    second.add_domain(0, before, result);
    auto source = retained();
    auto current = second.advance(first.advance(source, 0), 0);
    EXPECT_EQ(current.domain_bindings[0].domain.generation, 3u);
    EXPECT_NO_THROW(second.validate_current(current, 0));
    auto intermediate = first.rebind(source, 0);
    EXPECT_EQ(second.advance(first.advance(intermediate, 0), 0).domain_bindings[0].domain.generation, 3u);
    source.domain_bindings[0].domain.generation = 4;
    EXPECT_THROW(second.validate_current(second.advance(first.advance(source, 0), 0), 0), std::invalid_argument);
}
}  // namespace
