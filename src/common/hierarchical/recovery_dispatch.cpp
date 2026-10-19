/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "scheduler.h"
#include "recovery_dispatch.h"
#include "worker_manager.h"
#include "ring.h"
#include "host_log.h"

#include <cmath>
#include <cstring>
#include <future>

bool Scheduler::retry_recovery(
    const RecoveryTicket &ticket, uint32_t max_retries, double backoff_s, DomainRecoveryImpact impact,
    double control_timeout_s
) {
    if (!std::isfinite(backoff_s) || backoff_s < 0 || !std::isfinite(control_timeout_s) || control_timeout_s <= 0)
        throw std::invalid_argument("invalid recovery backoff or control timeout");
    // Policy delay belongs to the recovery owner; the ordinary scheduler keeps progressing.
    if (backoff_s > 0) {
        std::unique_lock<std::mutex> wait_lk(recovery_owner_mu_);
        recovery_owner_cv_.wait_for(wait_lk, std::chrono::duration<double>(backoff_s), [this, &ticket] {
            return stop_requested_.load(std::memory_order_acquire) || !recovery_coordinator_.is_pending(ticket);
        });
    }
    std::lock_guard<std::mutex> lk(loop_mu_);
    if (stop_requested_.load(std::memory_order_acquire)) throw std::runtime_error("worker stopping during recovery");
    if (!cfg_.recovery_graph_cb || !cfg_.start_task_replay_cb) throw std::logic_error("replay callbacks unavailable");
    auto assessment = recovery_coordinator_.assess(ticket, cfg_.recovery_graph_cb(ticket));
    if (assessment.status != ReplayStatus::REPLAY_ALLOWED)
        throw std::logic_error("replay refused: " + assessment.reason);
    auto pending = recovery_coordinator_.pending();
    if (pending.size() != 1 || cfg_.manager->any_busy())
        throw std::logic_error("local replay requires one held episode and no outstanding endpoint work");
    RecoveryRequest request = pending.front();
    if (ticket.attempt >= max_retries) {
        HostLogger::get_instance().log(
            simpler::log::LogLevel::WARN, __func__,
            "[PTO_RECOVERY] RETRY_BUDGET_GIVE_UP recovery_id=%llu attempt=%u max_retries=%u",
            static_cast<unsigned long long>(ticket.recovery_id), ticket.attempt, max_retries
        );
        give_up_recovery(ticket, "retry budget exhausted");
        notify_ready();
        return false;
    }
    for (const auto &member : assessment.op->members)
        if (member.task.task_slot != ticket.task_slot || member.endpoint_generation == 0)
            throw std::logic_error("redispatch requires one retained local group; separate rank slots need coordination"
            );
    if (impact != DomainRecoveryImpact::EXECUTION_ONLY && impact != DomainRecoveryImpact::DOMAIN_TRANSPORT)
        throw std::invalid_argument("unsupported soft operator recovery scope");
    TaskSlotState &task = *cfg_.ring->slot_state(ticket.task_slot);
    std::vector<RecoveryControlPacket> packets;
    for (size_t index = 0; index < request.members.size(); ++index) {
        const auto &member = request.members[index];
        auto *worker = cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, member.worker_id);
        if (worker == nullptr || worker->caps().kind != WorkerEndpointKind::LOCAL_MAILBOX ||
            worker->caps().max_inflight_tasks != 1 || worker->endpoint_generation() != member.endpoint_generation)
            throw std::logic_error("replay endpoint placement or generation changed");
        const auto &bindings = task.args(static_cast<int32_t>(index)).domain_bindings;
        if (bindings.size() > 1 || (impact == DomainRecoveryImpact::DOMAIN_TRANSPORT && bindings.empty()))
            throw std::logic_error("transport recovery requires one explicitly bound domain per member");
        RecoveryControlPacket packet;
        packet.recovery_id = ticket.recovery_id;
        packet.endpoint_generation = member.endpoint_generation;
        packet.impact = impact;
        packet.has_domain = bindings.empty() ? 0 : 1;
        if (!bindings.empty()) packet.before = bindings.front().domain;
        packets.push_back(packet);
    }
    if (impact == DomainRecoveryImpact::DOMAIN_TRANSPORT &&
        request.members.size() != cfg_.manager->next_level_worker_ids().size())
        throw std::logic_error("local transport recovery must own every communicator producer endpoint");

    auto phase = [&](RecoveryControlPhase value) {
        std::vector<std::future<RecoveryControlPacket>> futures;
        for (size_t index = 0; index < packets.size(); ++index) {
            auto packet = packets[index];
            packet.phase = value;
            const int32_t worker_id = request.members[index].worker_id;
            futures.push_back(std::async(std::launch::async, [this, packet, worker_id, control_timeout_s] {
                const auto bytes = cfg_.manager->control_payload(
                    WorkerType::NEXT_LEVEL, worker_id, CTRL_RECOVER_ENDPOINT, &packet, sizeof(packet), control_timeout_s
                );
                if (bytes.size() != sizeof(packet)) throw std::runtime_error("invalid recovery reply size");
                RecoveryControlPacket reply;
                std::memcpy(&reply, bytes.data(), sizeof(reply));
                if (reply.abi_version != 1 || reply.struct_bytes != sizeof(reply) || reply.phase != packet.phase ||
                    reply.recovery_id != packet.recovery_id ||
                    reply.endpoint_generation != packet.endpoint_generation || reply.impact != packet.impact ||
                    reply.has_domain != packet.has_domain || !(reply.before == packet.before))
                    throw std::runtime_error("stale or mismatched recovery control reply");
                return reply;
            }));
        }
        std::exception_ptr error;
        for (size_t index = 0; index < futures.size(); ++index) {
            try {
                packets[index] = futures[index].get();
            } catch (...) {
                if (!error) error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
    };
    bool taken = false;
    try {
        phase(RecoveryControlPhase::INSPECT);
        for (size_t index = 0; index < packets.size(); ++index) {
            const auto &fault = request.members[index].completion.native_execution_fault;
            const bool failed = fault.raw_rc != 0 || fault.runtime_status != 0;
            const auto &packet = packets[index];
            if (packet.faulted != static_cast<uint32_t>(failed) ||
                (failed && (packet.execution.status != ExecutionRecoveryStatus::RECOVERY_REQUIRED ||
                            packet.execution.quiesced != 1 || packet.execution.failed_run_epoch == 0)))
                throw std::runtime_error("endpoint failure evidence or cooperative stop proof changed");
            if (packet.has_domain && (packet.before.allocation_id != packets.front().before.allocation_id ||
                                      packet.before.generation != packets.front().before.generation))
                throw std::runtime_error("domain members disagree on allocation or generation");
        }
        phase(RecoveryControlPhase::PREPARE);
        for (const auto &packet : packets) {
            if (packet.has_domain && packet.domain.status != (impact == DomainRecoveryImpact::DOMAIN_TRANSPORT ?
                                                                  DomainRecoveryStatus::PREPARED :
                                                                  DomainRecoveryStatus::PRESERVED))
                throw std::runtime_error(
                    "domain preparation failed: recovery_rc=" + std::to_string(packet.domain.recovery_rc)
                );
        }
        if (impact == DomainRecoveryImpact::DOMAIN_TRANSPORT) phase(RecoveryControlPhase::COMMIT);
        RecoveryBindingPlan plan(ticket.recovery_id);
        bool bound = false;
        for (size_t index = 0; index < packets.size(); ++index) {
            if (!packets[index].has_domain) continue;
            plan.add_domain(request.members[index].worker_id, packets[index].before, packets[index].domain);
            bound = true;
        }
        phase(RecoveryControlPhase::REARM);
        for (const auto &packet : packets)
            if (packet.faulted && packet.execution.status != ExecutionRecoveryStatus::RECOVERED &&
                packet.execution.status != ExecutionRecoveryStatus::ALREADY_RECOVERED)
                throw std::runtime_error(
                    "execution recovery failed: recovery_rc=" + std::to_string(packet.execution.recovery_rc)
                );
        if (bound) {
            if (!cfg_.rebind_recovery_graph_cb) throw std::logic_error("graph rebinding unavailable");
            cfg_.rebind_recovery_graph_cb(ticket, plan);
        }
        assessment = recovery_coordinator_.assess(ticket, cfg_.recovery_graph_cb(ticket));
        if (assessment.status != ReplayStatus::REPLAY_ALLOWED || stop_requested_.load())
            throw std::runtime_error("replay eligibility changed during resource recovery");
        replaying_.emplace(ticket.task_slot, request);
        phase(RecoveryControlPhase::ARM);
        {
            std::lock_guard<std::mutex> owner_lk(recovery_owner_mu_);
            (void)recovery_coordinator_.take_for_replay(ticket);
        }
        recovery_owner_cv_.notify_all();
        taken = true;
        cfg_.start_task_replay_cb(ticket);
    } catch (const std::exception &error) {
        replaying_.erase(ticket.task_slot);
        try {
            phase(RecoveryControlPhase::ABORT);
        } catch (...) {}
        if (taken) recovery_coordinator_.submit(request);
        give_up_recovery(ticket, "resource recovery failed: " + std::string(error.what()));
        notify_ready();
        throw std::runtime_error(
            "original fault: " + request.original_failure.error_message + "; recovery fault: " + error.what()
        );
    }
    for (size_t index = 0; index < request.members.size(); ++index) {
        WorkerDispatch dispatch{ticket.task_slot, static_cast<int32_t>(index)};
        dispatch.recovery = true;
        dispatch_claimed(
            cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, request.members[index].worker_id), dispatch, false
        );
    }
    HostLogger::get_instance().log(
        simpler::log::LogLevel::WARN, __func__,
        "[PTO_RECOVERY] REPLAY_DISPATCH recovery_id=%llu task_slot=%d attempt=%u",
        static_cast<unsigned long long>(ticket.recovery_id), ticket.task_slot, ticket.attempt + 1
    );
    notify_ready();
    return true;
}

void Scheduler::coordinate_recovery() {
    uint64_t last_id = 0;
    while (auto observed = recovery_coordinator_.wait_next(last_id)) {
        const RecoveryRequest request = *observed;
        last_id = request.ticket.recovery_id;
        try {
            std::vector<std::pair<int32_t, RecoveryControlPacket>> stops;
            {
                std::lock_guard<std::mutex> lk(loop_mu_);
                if (stop_requested_.load()) break;
                for (size_t index = 0; index < request.members.size(); ++index) {
                    const auto &member = request.members[index];
                    auto *worker = cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, member.worker_id);
                    if (!worker || worker->caps().kind != WorkerEndpointKind::LOCAL_MAILBOX ||
                        worker->caps().max_inflight_tasks != 1 ||
                        worker->endpoint_generation() != member.endpoint_generation)
                        throw std::logic_error("coordinated stop requires the current local single-frame endpoint");
                    RecoveryControlPacket packet;
                    packet.phase = RecoveryControlPhase::STOP;
                    packet.recovery_id = request.ticket.recovery_id;
                    packet.endpoint_generation = member.endpoint_generation;
                    packet.run_id = request.ticket.run_id;
                    packet.task_slot = request.ticket.task_slot;
                    packet.group_index = static_cast<int32_t>(index);
                    packet.pipeline_generation = request.pipeline_lease.generation;
                    // Completion may race the initial event snapshot; obtain its current identity under loop_mu_.
                    const auto current = recovery_coordinator_.pending();
                    const auto episode = std::find_if(current.begin(), current.end(), [&](const auto &r) {
                        return r.ticket == request.ticket;
                    });
                    if (episode == current.end()) throw std::logic_error("recovery was cancelled before stop");
                    const auto &latest = episode->members[index];
                    packet.dispatch_id = latest.terminal ? latest.completion.dispatch_id :
                                                           worker->active_dispatch_id(packet.run_id, packet.task_slot);
                    if (packet.dispatch_id == 0)
                        throw std::logic_error("stop target lacks a physical dispatch identity");
                    stops.emplace_back(member.worker_id, packet);
                }
            }
            HostLogger::get_instance().log(
                simpler::log::LogLevel::WARN, __func__,
                "[PTO_RECOVERY] AUTO_RECOVERY_BEGIN recovery_id=%llu attempt=%u members=%zu",
                static_cast<unsigned long long>(request.ticket.recovery_id), request.ticket.attempt, stops.size()
            );
            std::vector<std::future<void>> replies;
            for (const auto &stop : stops)
                replies.push_back(std::async(std::launch::async, [this, worker_id = stop.first, packet = stop.second] {
                    const auto bytes = cfg_.manager->control_payload(
                        WorkerType::NEXT_LEVEL, worker_id, CTRL_RECOVER_ENDPOINT, &packet, sizeof(packet),
                        cfg_.recovery_timeout_s
                    );
                    if (bytes.size() != sizeof(packet)) throw std::runtime_error("invalid stop reply size");
                    RecoveryControlPacket reply;
                    std::memcpy(&reply, bytes.data(), sizeof(reply));
                    if (reply.abi_version != packet.abi_version || reply.struct_bytes != sizeof(reply) ||
                        reply.phase != packet.phase || reply.recovery_id != packet.recovery_id ||
                        reply.endpoint_generation != packet.endpoint_generation || reply.run_id != packet.run_id ||
                        reply.task_slot != packet.task_slot || reply.group_index != packet.group_index ||
                        reply.pipeline_generation != packet.pipeline_generation ||
                        reply.dispatch_id != packet.dispatch_id || reply.stop_accepted != 1)
                        throw std::runtime_error("stop delivery acknowledgement is stale or refused");
                }));
            std::exception_ptr failure;
            for (auto &reply : replies) {
                try {
                    reply.get();
                } catch (...) {
                    if (!failure) failure = std::current_exception();
                }
            }
            if (failure) std::rethrow_exception(failure);
            const auto terminal = recovery_coordinator_.wait_terminal(request.ticket, cfg_.recovery_timeout_s);
            if (!terminal) throw std::runtime_error("coordinated stop deadline expired or recovery cancelled");
            for (const auto &member : terminal->members)
                HostLogger::get_instance().log(
                    simpler::log::LogLevel::WARN, __func__,
                    "[PTO_RECOVERY] AUTO_MEMBER_TERMINAL recovery_id=%llu worker=%d raw_rc=%d runtime_status=%d",
                    static_cast<unsigned long long>(last_id), member.worker_id,
                    member.completion.native_execution_fault.raw_rc,
                    member.completion.native_execution_fault.runtime_status
                );
            HostLogger::get_instance().log(
                simpler::log::LogLevel::WARN, __func__, "[PTO_RECOVERY] ALL_MEMBERS_TERMINAL recovery_id=%llu",
                static_cast<unsigned long long>(last_id)
            );
            const auto assessment = assess_replay(request.ticket);
            if (assessment.status != ReplayStatus::REPLAY_ALLOWED) {
                give_up_recovery(request.ticket, "automatic replay refused: " + assessment.reason);
                continue;
            }
            (void)retry_recovery(
                request.ticket, cfg_.recovery_max_retries, cfg_.recovery_backoff_s,
                assessment.op->uses_communication ? DomainRecoveryImpact::DOMAIN_TRANSPORT :
                                                    DomainRecoveryImpact::EXECUTION_ONLY,
                cfg_.recovery_timeout_s
            );
        } catch (const std::exception &error) {
            if (stop_requested_.load()) break;
            const std::string reason = "automatic recovery failed: " + std::string(error.what());
            HostLogger::get_instance().log(
                simpler::log::LogLevel::WARN, __func__,
                "[PTO_RECOVERY] AUTO_RECOVERY_FAILED recovery_id=%llu original=%s recovery=%s",
                static_cast<unsigned long long>(last_id), request.original_failure.error_message.c_str(), error.what()
            );
            try {
                if (!give_up_recovery(request.ticket, reason)) continue;
            } catch (const std::logic_error &) {
                // Unproved live executions retain their slots/resources until default endpoint teardown.
                std::lock_guard<std::mutex> lk(loop_mu_);
                if (cfg_.on_task_failed_cb) cfg_.on_task_failed_cb(request.ticket.task_slot, reason);
                cfg_.manager->stop_workers();
            }
        }
    }
}
