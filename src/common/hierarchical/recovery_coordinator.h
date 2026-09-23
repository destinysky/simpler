/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "types.h"

// Recovery policy and transaction state belong to L3. The endpoint action is
// separate from replay eligibility and never dispatches the failed task.
enum class RecoveryStage : int32_t {
    DETECTED = 0,
    ELIGIBILITY_CHECK = 1,
    GIVE_UP = 2,
    ENDPOINT_REBUILD = 3,
    ENDPOINT_READY = 4,
};

enum class RecoveryResolutionKind : int32_t {
    GIVE_UP = 0,
    REBUILD_ENDPOINT = 1,
};

struct EndpointRecoveryRequest {
    int32_t worker_id{-1};
    uint64_t recovery_id{0};
    uint64_t expected_endpoint_generation{0};
};

struct EndpointRecoveryResult {
    int32_t worker_id{-1};
    uint64_t recovery_id{0};
    bool ok{false};
    // Execution-environment incarnation. It advances only after reset/probe,
    // new ChipWorker init, and callable preparation all succeed.
    uint64_t endpoint_generation{0};
    std::string error_message;
};

struct RecoveryTicket {
    RunId run_id{INVALID_RUN_ID};
    TaskSlot task_slot{INVALID_SLOT};
    uint64_t recovery_id{0};
    uint32_t attempt{0};
    // Snapshotted when L3 first intercepts the native fault. Every destructive
    // command in this episode is fenced by this immutable generation.
    uint64_t expected_endpoint_generation{0};
};

struct RecoveryRequest {
    RecoveryTicket ticket{};
    int32_t worker_id{-1};
    PipelineSlotLease pipeline_lease{};
    CallableIdentity callable{};
    WorkerCompletion failure{};
    RecoveryStage stage{RecoveryStage::DETECTED};
};

struct RecoveryResolution {
    RecoveryRequest request{};
    RecoveryResolutionKind kind{RecoveryResolutionKind::GIVE_UP};
    std::optional<EndpointRecoveryResult> endpoint_result;
};

class RecoveryCoordinator {
public:
    using EligibilityDecision = std::function<bool(const RecoveryRequest &)>;

    void set_eligibility_decision(EligibilityDecision decision) {
        std::lock_guard<std::mutex> lk(mu_);
        eligibility_decision_ = std::move(decision);
    }
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.clear();
        ready_.clear();
    }

    void submit(RecoveryRequest request) {
        std::lock_guard<std::mutex> lk(mu_);
        request.stage = RecoveryStage::DETECTED;
        pending_.push_back(std::move(request));
    }

    void submit_endpoint_result(RecoveryRequest request, EndpointRecoveryResult result) {
        std::lock_guard<std::mutex> lk(mu_);
        const bool identity_matches =
            result.worker_id == request.worker_id && result.recovery_id == request.ticket.recovery_id;
        const bool committed_generation_matches =
            !result.ok ||
            (request.ticket.expected_endpoint_generation != UINT64_MAX &&
             result.endpoint_generation == request.ticket.expected_endpoint_generation + 1);
        if (!identity_matches || !committed_generation_matches) {
            result.ok = false;
            result.endpoint_generation = request.ticket.expected_endpoint_generation;
            result.error_message = "endpoint action result identity or committed generation mismatch";
        }
        request.stage = result.ok ? RecoveryStage::ENDPOINT_READY : RecoveryStage::GIVE_UP;
        pending_.push_back(PendingTransition{std::move(request), std::move(result)});
    }

    // A missing eligibility decision gives up. An allowed request asks L3 to
    // rebuild its endpoint; replay remains a separate later decision.
    void progress() {
        std::lock_guard<std::mutex> lk(mu_);
        while (!pending_.empty()) {
            PendingTransition transition = std::move(pending_.front());
            pending_.pop_front();
            RecoveryRequest &request = transition.request;
            if (request.stage == RecoveryStage::ENDPOINT_READY) {
                // Step 4 establishes only the endpoint execution environment.
                // Domain/binding recovery and replay authorization are later
                // stages, so ENDPOINT_READY currently terminates as GIVE_UP.
                request.stage = RecoveryStage::GIVE_UP;
                ready_.push_back(
                    RecoveryResolution{std::move(request), RecoveryResolutionKind::GIVE_UP,
                                       std::move(transition.endpoint_result)}
                );
                continue;
            }
            if (request.stage == RecoveryStage::GIVE_UP) {
                ready_.push_back(
                    RecoveryResolution{std::move(request), RecoveryResolutionKind::GIVE_UP,
                                       std::move(transition.endpoint_result)}
                );
                continue;
            }

            request.stage = RecoveryStage::ELIGIBILITY_CHECK;
            bool allowed = false;
            try {
                allowed = eligibility_decision_ && eligibility_decision_(request);
            } catch (...) {
                allowed = false;
            }
            request.stage = allowed ? RecoveryStage::ENDPOINT_REBUILD : RecoveryStage::GIVE_UP;
            ready_.push_back(RecoveryResolution{
                std::move(request),
                allowed ? RecoveryResolutionKind::REBUILD_ENDPOINT : RecoveryResolutionKind::GIVE_UP,
                std::nullopt,
            });
        }
    }

    bool try_pop_resolution(RecoveryResolution &out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (ready_.empty()) return false;
        out = std::move(ready_.front());
        ready_.pop_front();
        return true;
    }

    bool has_work() const {
        std::lock_guard<std::mutex> lk(mu_);
        return !pending_.empty() || !ready_.empty();
    }

private:
    struct PendingTransition {
        RecoveryRequest request{};
        std::optional<EndpointRecoveryResult> endpoint_result;

        PendingTransition() = default;
        PendingTransition(RecoveryRequest value) : request(std::move(value)) {}
        PendingTransition(RecoveryRequest value, EndpointRecoveryResult result)
            : request(std::move(value)), endpoint_result(std::move(result)) {}
    };

    mutable std::mutex mu_;
    std::deque<PendingTransition> pending_;
    std::deque<RecoveryResolution> ready_;
    EligibilityDecision eligibility_decision_;
};
