/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

#include "types.h"

// Step 3 intentionally owns policy/state outside Scheduler. Later recovery
// work extends the stages and actions here without moving retry policy into L2.
enum class RecoveryStage : int32_t {
    DETECTED = 0,
    ELIGIBILITY_CHECK = 1,
    GIVE_UP = 2,
};

enum class RecoveryResolutionKind : int32_t {
    GIVE_UP = 0,
    REPLAY_READY = 1,
};

struct RecoveryTicket {
    RunId run_id{INVALID_RUN_ID};
    TaskSlot task_slot{INVALID_SLOT};
    uint64_t recovery_id{0};
    uint32_t attempt{0};
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
};

class RecoveryCoordinator {
public:
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

    // Step 3 skeleton: consume an asynchronous request, run the policy state
    // transition, and deliberately GIVE_UP. Step 4+ inserts recovery actions
    // between ELIGIBILITY_CHECK and the final resolution.
    void progress() {
        std::lock_guard<std::mutex> lk(mu_);
        while (!pending_.empty()) {
            RecoveryRequest request = std::move(pending_.front());
            pending_.pop_front();
            request.stage = RecoveryStage::ELIGIBILITY_CHECK;
            request.stage = RecoveryStage::GIVE_UP;
            RecoveryResolution resolution;
            resolution.request = std::move(request);
            resolution.kind = RecoveryResolutionKind::GIVE_UP;
            ready_.push_back(std::move(resolution));
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
    mutable std::mutex mu_;
    std::deque<RecoveryRequest> pending_;
    std::deque<RecoveryResolution> ready_;
};
