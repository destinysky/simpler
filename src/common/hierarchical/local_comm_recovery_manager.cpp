/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "local_comm_recovery_manager.h"

#include <limits>
#include <stdexcept>

#include "worker_manager.h"

LocalCommGroupState LocalCommRecoveryManager::state() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

LocalCommGroupState LocalCommRecoveryManager::commit_initial_ready() {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_.state == LocalCommState::READY) return state_;
    if (state_.state != LocalCommState::UNINITIALIZED || state_.generation != 0) {
        throw std::runtime_error("local communication scope cannot perform ordinary initialization");
    }
    state_ = {1, LocalCommState::READY};
    return state_;
}

std::optional<uint64_t> LocalCommRecoveryManager::mark_stale(uint64_t recovery_id) {
    if (recovery_id == 0) throw std::invalid_argument("local communication recovery id must be non-zero");
    std::lock_guard<std::mutex> lk(mu_);
    if (state_.state == LocalCommState::UNINITIALIZED && state_.generation == 0) return std::nullopt;
    if (state_.state == LocalCommState::BROKEN && active_recovery_id_ == 0) return std::nullopt;
    if (state_.state != LocalCommState::READY || state_.generation == 0 || active_recovery_id_ != 0) {
        throw std::runtime_error("local communication scope is not ready for invalidation");
    }
    state_.state = LocalCommState::STALE;
    active_recovery_id_ = recovery_id;
    return state_.generation;
}

LocalCommRecoveryResult
LocalCommRecoveryManager::rebuild(const LocalCommRecoveryRequest &request, WorkerManager &workers) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (request.recovery_id == 0 || request.recovery_id != active_recovery_id_ ||
            request.expected_local_comm_generation != state_.generation || state_.state != LocalCommState::STALE ||
            state_.generation == std::numeric_limits<uint64_t>::max()) {
            return {request.recovery_id, false, state_.generation, "stale local communication recovery request"};
        }
        state_.state = LocalCommState::REBUILDING;
    }

    LocalCommRecoveryResult result{
        request.recovery_id, false, request.expected_local_comm_generation, {}};
    try {
        result = workers.rebuild_local_comm(request);
    } catch (const std::exception &e) {
        result.error_message = e.what();
    } catch (...) {
        result.error_message = "unknown local communication rebuild failure";
    }
    std::lock_guard<std::mutex> lk(mu_);
    const bool identity_matches = request.recovery_id == active_recovery_id_ &&
                                  request.expected_local_comm_generation == state_.generation &&
                                  state_.state == LocalCommState::REBUILDING;
    const bool generation_matches =
        !result.ok || result.local_comm_generation == request.expected_local_comm_generation + 1;
    if (!identity_matches || !generation_matches) {
        result.ok = false;
        result.local_comm_generation = state_.generation;
        result.error_message = "local communication result identity or generation mismatch";
    }
    if (result.ok) {
        state_.generation = result.local_comm_generation;
        state_.state = LocalCommState::READY;
    } else {
        state_.state = LocalCommState::BROKEN;
        result.local_comm_generation = state_.generation;
    }
    active_recovery_id_ = 0;
    return result;
}

void LocalCommRecoveryManager::fail_stale_episode(uint64_t recovery_id, uint64_t expected_generation) {
    std::lock_guard<std::mutex> lk(mu_);
    if (recovery_id != active_recovery_id_ || expected_generation != state_.generation) return;
    if (state_.state == LocalCommState::STALE || state_.state == LocalCommState::REBUILDING) {
        state_.state = LocalCommState::BROKEN;
    }
    active_recovery_id_ = 0;
}

void LocalCommRecoveryManager::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    state_ = {};
    active_recovery_id_ = 0;
}
