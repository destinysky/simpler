/*
 * Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "recovery_coordinator.h"

#include <map>
#include <unordered_set>

namespace {
bool advanced(TaskState state) {
    return state == TaskState::RUNNING || state == TaskState::COMPLETED || state == TaskState::CONSUMED ||
           state == TaskState::FAILED || state == TaskState::RETRY_PENDING;
}
bool terminal_member(const RecoveryTaskSnapshot &task, size_t index) {
    if (!task.member_states.empty()) {
        if (index >= task.member_states.size()) return false;
        const auto state = task.member_states[index];
        return state == GroupMemberState::SUCCESS || state == GroupMemberState::FAILED ||
               state == GroupMemberState::SKIPPED;
    }
    return task.state == TaskState::COMPLETED || task.state == TaskState::CONSUMED ||
           task.state == TaskState::RETRY_PENDING;
}
}  // namespace

ReplayAssessment assess_replay_eligibility(const RecoveryRequest &request, const RecoveryGraphSnapshot &graph) {
    ReplayAssessment result;
    result.ticket = request.ticket;
    auto verdict = [&](ReplayStatus status, const std::string &reason) {
        result.status = status;
        result.reason = reason;
        return result;
    };
    if (!(request.ticket == graph.ticket) || request.ticket.run_id == INVALID_RUN_ID ||
        request.ticket.recovery_id == 0 || request.pipeline_lease.generation == 0 || graph.run_failed ||
        graph.active_recoveries == 0)
        return verdict(ReplayStatus::NACK_NOT_RETAINED, "run is not held by the current recovery episode");
    std::unordered_map<TaskSlot, const RecoveryTaskSnapshot *> tasks;
    const RecoveryTaskSnapshot *failed = nullptr;
    for (const auto &task : graph.tasks) {
        if (task.handle.run_id != request.ticket.run_id || !tasks.emplace(task.handle.task_slot, &task).second)
            return verdict(ReplayStatus::NACK_NOT_RETAINED, "inconsistent retained task identities");
        if (task.handle.task_slot == request.ticket.task_slot) failed = &task;
    }
    if (failed == nullptr || failed->state != TaskState::RETRY_PENDING ||
        failed->active_recovery_id != request.ticket.recovery_id ||
        failed->pipeline_generation != request.pipeline_lease.generation)
        return verdict(ReplayStatus::NACK_NOT_RETAINED, "failed task identity, generation or recovery hold changed");
    if (!graph.submission_closed) return verdict(ReplayStatus::WAITING_GRAPH, "graph construction is not closed");
    if (std::any_of(graph.tasks.begin(), graph.tasks.end(), [](const auto &task) {
            return task.state == TaskState::BUILDING;
        }))
        return verdict(ReplayStatus::WAITING_GRAPH, "a task's op identity or graph edges are not published yet");
    if (request.members.size() != failed->worker_ids.size())
        return verdict(ReplayStatus::NACK_METADATA, "recovery evidence has a different member count");
    if (failed->args.empty() || !failed->args.front().recovery_metadata)
        return verdict(ReplayStatus::NACK_METADATA, "failed task has no recovery metadata");
    const auto seed = *failed->args.front().recovery_metadata;
    try {
        seed.validate();
    } catch (const std::invalid_argument &error) {
        return verdict(ReplayStatus::NACK_METADATA, error.what());
    }
    DistributedOpRecord op;
    op.logical_run_id = seed.logical_run_id;
    op.domain_id = seed.domain_id;
    op.op_seq = seed.op_seq;
    op.participants = seed.participants;
    op.replay_safety = seed.replay_safety;
    std::map<int32_t, DistributedOpMember> members;
    std::unordered_set<int32_t> workers;
    std::unordered_set<TaskSlot> op_slots;
    bool consumed_member = false;
    for (const auto &task : graph.tasks) {
        bool belongs = false;
        for (const auto &args : task.args)
            belongs = belongs || (args.recovery_metadata && seed.same_op(*args.recovery_metadata));
        if (!belongs) continue;
        if (!task.local_chip || task.args.size() != task.worker_ids.size())
            return verdict(ReplayStatus::NACK_METADATA, "op metadata requires local chip member placement");
        if (task.state == TaskState::FREE || task.pipeline_generation != request.pipeline_lease.generation)
            return verdict(ReplayStatus::NACK_NOT_RETAINED, "an op member or its generation is no longer retained");
        consumed_member = consumed_member || task.state == TaskState::CONSUMED;
        op_slots.insert(task.handle.task_slot);
        for (size_t index = 0; index < task.args.size(); ++index) {
            const auto &metadata = task.args[index].recovery_metadata;
            if (!metadata || !seed.same_contract(*metadata))
                return verdict(ReplayStatus::NACK_METADATA, "op members disagree on identity, participants or safety");
            try {
                metadata->validate();
            } catch (const std::invalid_argument &error) {
                return verdict(ReplayStatus::NACK_METADATA, error.what());
            }
            DistributedOpMember member;
            member.rank = metadata->rank;
            member.worker_id = task.worker_ids[index];
            member.task = task.handle;
            member.state = task.state;
            member.terminal = terminal_member(task, index);
            if (task.handle.task_slot == request.ticket.task_slot) {
                if (index >= request.members.size() || request.members[index].worker_id != member.worker_id)
                    return verdict(ReplayStatus::NACK_METADATA, "recovery evidence does not match submitted placement");
                member.endpoint_generation = request.members[index].endpoint_generation;
                if (member.endpoint_generation == 0)
                    return verdict(ReplayStatus::NACK_NOT_RETAINED, "missing endpoint generation in failure evidence");
                member.terminal = request.members[index].terminal;
            }
            if (member.worker_id < 0 || !workers.insert(member.worker_id).second ||
                !members.emplace(member.rank, member).second)
                return verdict(ReplayStatus::NACK_METADATA, "duplicate rank or endpoint in logical op");
        }
    }
    for (const auto &[rank, member] : members)
        op.members.push_back(member);
    result.op = op;

    std::unordered_set<TaskSlot> upstream = op_slots;
    std::vector<TaskSlot> producer_work(op_slots.begin(), op_slots.end());
    while (!producer_work.empty()) {
        const auto slot = producer_work.back();
        producer_work.pop_back();
        for (const auto producer : tasks.at(slot)->producers) {
            if (!upstream.insert(producer).second) continue;
            if (tasks.find(producer) == tasks.end())
                return verdict(ReplayStatus::NACK_NOT_RETAINED, "a producer is no longer observable");
            producer_work.push_back(producer);
        }
    }

    // Includes ordering-only dependencies and unannotated transitive consumers.
    std::vector<TaskSlot> pending(op_slots.begin(), op_slots.end());
    std::unordered_set<TaskSlot> visited = op_slots;
    while (!pending.empty()) {
        const auto slot = pending.back();
        pending.pop_back();
        for (const auto consumer : tasks.at(slot)->consumers) {
            if (!visited.insert(consumer).second) continue;
            const auto found = tasks.find(consumer);
            if (found == tasks.end())
                return verdict(ReplayStatus::NACK_NOT_RETAINED, "a downstream task is no longer observable");
            const auto &task = *found->second;
            if (task.state == TaskState::FREE)
                return verdict(ReplayStatus::NACK_NOT_RETAINED, "a downstream slot is no longer retained");
            if (advanced(task.state)) result.blocking_tasks.push_back(task.handle);
            pending.push_back(consumer);
        }
    }
    // A fast rank can advance in its domain even without a cross-rank DAG edge.
    for (const auto &task : graph.tasks) {
        if (!advanced(task.state)) continue;
        for (const auto &args : task.args) {
            const auto &metadata = args.recovery_metadata;
            if (metadata && metadata->logical_run_id == seed.logical_run_id && metadata->domain_id == seed.domain_id &&
                metadata->op_seq > seed.op_seq) {
                if (std::none_of(result.blocking_tasks.begin(), result.blocking_tasks.end(), [&](const auto &handle) {
                        return handle.task_slot == task.handle.task_slot;
                    }))
                    result.blocking_tasks.push_back(task.handle);
                break;
            }
        }
    }
    if (!result.blocking_tasks.empty())
        return verdict(ReplayStatus::NACK_ADVANCED, "a downstream consumer or later domain op has advanced");
    for (const auto &task : graph.tasks) {
        if (!task.local_chip || !advanced(task.state) || upstream.count(task.handle.task_slot) != 0) continue;
        const bool overlapping_endpoint = std::any_of(task.worker_ids.begin(), task.worker_ids.end(), [&](int32_t id) {
            return workers.count(id) != 0;
        });
        if (overlapping_endpoint &&
            (task.args.empty() || std::any_of(task.args.begin(), task.args.end(), [](const auto &a) {
                 return !a.recovery_metadata;
             })))
            return verdict(
                ReplayStatus::NEEDS_COORDINATION,
                "advanced unannotated work on a participant endpoint has unknown failure-domain scope"
            );
    }
    if (consumed_member) return verdict(ReplayStatus::NACK_NOT_RETAINED, "an op member has already been consumed");
    if (op.members.size() != seed.participants.size())
        return verdict(ReplayStatus::NEEDS_COORDINATION, "not all declared ranks are present in this local graph");
    if (!request.all_members_terminal() || std::any_of(op.members.begin(), op.members.end(), [](const auto &member) {
            return !member.terminal;
        }))
        return verdict(ReplayStatus::WAITING_MEMBERS, "an op member still needs stop/terminal coordination");
    if (std::any_of(request.members.begin(), request.members.end(), [](const auto &member) {
            return member.completion.native_execution_fault.device_unusable != 0;
        }))
        return verdict(ReplayStatus::NACK_DEVICE_UNUSABLE, "an op endpoint is confirmed unusable");
    switch (seed.replay_safety) {
    case ReplaySafety::REPLAY_SAFE:
        return verdict(
            ReplayStatus::REPLAY_ALLOWED,
            "semantic and graph eligibility only; quiescence, rearm, binding and dispatch are still required"
        );
    case ReplaySafety::REQUIRES_RESTORE:
        return verdict(ReplayStatus::NEEDS_RESTORE, "input/output state restoration is required before replay");
    default:
        return verdict(ReplayStatus::NACK_UNSAFE, "replay safety is unknown or explicitly forbidden");
    }
}
