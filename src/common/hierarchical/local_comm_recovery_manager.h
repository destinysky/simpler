/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

class WorkerManager;

enum class LocalCommState : uint8_t {
    UNINITIALIZED = 0,
    READY = 1,
    STALE = 2,
    REBUILDING = 3,
    BROKEN = 4,
};

struct LocalCommGroupState {
    uint64_t generation{0};
    LocalCommState state{LocalCommState::UNINITIALIZED};
};

struct LocalCommRecoveryRequest {
    uint64_t recovery_id{0};
    uint64_t expected_local_comm_generation{0};
};

struct LocalCommEndpointRequest {
    int32_t worker_id{-1};
    uint64_t recovery_id{0};
    uint64_t expected_endpoint_generation{0};
    uint64_t expected_local_comm_generation{0};
};

struct LocalCommEndpointResult {
    int32_t worker_id{-1};
    uint64_t recovery_id{0};
    bool ok{false};
    uint64_t local_comm_generation{0};
    std::string error_message;
};

struct LocalCommRecoveryResult {
    uint64_t recovery_id{0};
    bool ok{false};
    uint64_t local_comm_generation{0};
    std::string error_message;
};

class LocalCommRecoveryManager {
public:
    LocalCommGroupState state() const;
    LocalCommGroupState commit_initial_ready();
    std::optional<uint64_t> mark_stale(uint64_t recovery_id);
    LocalCommRecoveryResult rebuild(const LocalCommRecoveryRequest &request, WorkerManager &workers);
    void fail_stale_episode(uint64_t recovery_id, uint64_t expected_generation);
    void reset();

private:
    mutable std::mutex mu_;
    LocalCommGroupState state_{};
    uint64_t active_recovery_id_{0};
};
