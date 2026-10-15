/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "types.h"

// This command only releases a faulted child into terminal cleanup.
struct EndpointRecoveryRequest {
    int32_t worker_id{-1};
    uint64_t recovery_id{0};
    uint64_t expected_endpoint_generation{0};
};

struct RecoveryTicket {
    RunId run_id{INVALID_RUN_ID};
    TaskSlot task_slot{INVALID_SLOT};
    uint64_t recovery_id{0};
    uint32_t attempt{0};
    uint64_t owner_id{0};
};
inline bool operator==(const RecoveryTicket &a, const RecoveryTicket &b) {
    return a.run_id == b.run_id && a.task_slot == b.task_slot && a.recovery_id == b.recovery_id &&
           a.attempt == b.attempt && a.owner_id == b.owner_id;
}
struct RecoveryMember {
    int32_t worker_id{-1};
    uint64_t endpoint_generation{0};
    bool terminal{false};
    WorkerCompletion completion{};
};
struct RecoveryRequest {
    RecoveryTicket ticket{};
    PipelineSlotLease pipeline_lease{};
    CallableIdentity callable{};
    std::vector<RecoveryMember> members;
    WorkerCompletion failure{};
    bool all_members_terminal() const {
        return !members.empty() && std::all_of(members.begin(), members.end(), [](const auto &m) {
            return m.terminal;
        });
    }
};
struct RecoveryTaskSnapshot {
    TaskHandle handle{};
    TaskState state{TaskState::FREE};
    std::vector<int32_t> worker_ids;
    std::vector<TaskArgs> args;
    std::vector<TaskSlot> producers;
    std::vector<TaskSlot> consumers;
    uint64_t pipeline_generation{0};
    uint64_t active_recovery_id{0};
    int32_t fanin_count{0};
    int32_t fanin_released{0};
    int32_t fanout_total{0};
    int32_t fanout_released{0};
    size_t output_key_count{0};
};
struct RecoveryGraphSnapshot {
    RecoveryTicket ticket{};
    bool run_failed{false};
    bool submission_closed{false};
    uint32_t active_recoveries{0};
    std::vector<RecoveryTaskSnapshot> tasks;
};

// Keeps the logical episode pending until its owner explicitly gives up.
// Neither a missing policy nor queue emptiness resolves a retained graph.
class RecoveryCoordinator {
public:
    RecoveryCoordinator() :
        owner_id_(next_owner_id_.fetch_add(1, std::memory_order_relaxed) + 1) {
        if (owner_id_ == 0) throw std::overflow_error("recovery owner identity exhausted");
    }
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.clear();
        resolutions_.clear();
        cancelling_ = false;
    }
    RecoveryTicket submit(RecoveryRequest request) {
        if (request.ticket.recovery_id == 0 || request.members.empty())
            throw std::invalid_argument("invalid recovery episode");
        if (request.ticket.owner_id != 0 && request.ticket.owner_id != owner_id_)
            throw std::invalid_argument("foreign recovery owner");
        request.ticket.owner_id = owner_id_;
        const auto ticket = request.ticket;
        std::lock_guard<std::mutex> lk(mu_);
        auto [it, inserted] = pending_.emplace(request.ticket.recovery_id, Episode{std::move(request), {}});
        if (!inserted) throw std::logic_error("duplicate recovery episode");
        if (cancelling_) it->second.reason = "worker stopping during recovery";
        resolve_cancelled(it);
        cv_.notify_all();
        return ticket;
    }
    void complete_member(uint64_t id, int32_t index, const WorkerCompletion &completion) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pending_.find(id);
        if (it == pending_.end()) throw std::logic_error("unknown recovery episode");
        if (index < 0 || static_cast<size_t>(index) >= it->second.request.members.size())
            throw std::invalid_argument("invalid recovery member");
        auto &member = it->second.request.members[static_cast<size_t>(index)];
        if (member.terminal) return;
        member.completion = completion;
        member.terminal = true;
        resolve_cancelled(it);
        cv_.notify_all();
    }
    std::vector<RecoveryRequest> pending() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<RecoveryRequest> result;
        for (const auto &[id, episode] : pending_)
            result.push_back(episode.request);
        return result;
    }
    std::optional<RecoveryRequest> wait(double seconds) {
        if (!std::isfinite(seconds) || seconds < 0) throw std::invalid_argument("invalid recovery wait timeout");
        std::unique_lock<std::mutex> lk(mu_);
        auto ready = [this] {
            return cancelling_ || std::any_of(pending_.begin(), pending_.end(), [](const auto &e) {
                       return e.second.request.all_members_terminal();
                   });
        };
        if (!cv_.wait_for(lk, std::chrono::duration<double>(seconds), ready)) return std::nullopt;
        for (const auto &[id, episode] : pending_)
            if (episode.request.all_members_terminal()) return episode.request;
        return std::nullopt;
    }
    bool give_up(const RecoveryTicket &ticket, const std::string &reason) {
        if (reason.empty()) throw std::invalid_argument("give up requires a reason");
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pending_.find(ticket.recovery_id);
        if (it == pending_.end() || !(it->second.request.ticket == ticket)) return false;
        if (!it->second.request.all_members_terminal()) throw std::logic_error("recovery members still executing");
        it->second.reason = reason;
        resolve_cancelled(it);
        cv_.notify_all();
        return true;
    }
    void cancel_all() {
        std::lock_guard<std::mutex> lk(mu_);
        cancelling_ = true;
        for (auto it = pending_.begin(); it != pending_.end();) {
            auto next = std::next(it);
            it->second.reason = "worker stopping during recovery";
            resolve_cancelled(it);
            it = next;
        }
        cv_.notify_all();
    }
    bool try_pop_resolution(RecoveryRequest &request, std::string &reason) {
        std::lock_guard<std::mutex> lk(mu_);
        if (resolutions_.empty()) return false;
        request = std::move(resolutions_.front().request);
        reason = std::move(resolutions_.front().reason);
        resolutions_.pop_front();
        return true;
    }
    void cancel_run(RunId run_id, const std::string &reason) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = pending_.begin(); it != pending_.end();) {
            auto next = std::next(it);
            if (it->second.request.ticket.run_id == run_id) {
                it->second.reason = reason.empty() ? "run terminated" : reason;
                resolve_cancelled(it);
            }
            it = next;
        }
        cv_.notify_all();
    }
    bool has_work() const {
        std::lock_guard<std::mutex> lk(mu_);
        return !pending_.empty() || !resolutions_.empty();
    }

private:
    struct Episode {
        RecoveryRequest request;
        std::string reason;
    };
    using Episodes = std::unordered_map<uint64_t, Episode>;
    void resolve_cancelled(Episodes::iterator it) {
        if (!it->second.reason.empty() && it->second.request.all_members_terminal()) {
            resolutions_.push_back(std::move(it->second));
            pending_.erase(it);
        }
    }
    mutable std::mutex mu_;
    inline static std::atomic<uint64_t> next_owner_id_{0};
    const uint64_t owner_id_;
    std::condition_variable cv_;
    Episodes pending_;
    std::deque<Episode> resolutions_;
    bool cancelling_{false};
};
