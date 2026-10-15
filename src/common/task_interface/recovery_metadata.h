/*
 * Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

enum class ReplaySafety : int32_t {
    UNKNOWN = 0,
    REPLAY_SAFE = 1,
    REQUIRES_RESTORE = 2,
    FORBIDDEN = 3,
};

// Host-owned compiler/application declarations; neither addresses nor error codes identify an op.
struct RecoveryMetadata {
    uint64_t logical_run_id{0};
    uint64_t domain_id{0};
    uint64_t op_seq{0};
    int32_t rank{-1};
    std::vector<int32_t> participants;
    ReplaySafety replay_safety{ReplaySafety::UNKNOWN};

    bool same_op(const RecoveryMetadata &other) const {
        return logical_run_id == other.logical_run_id && domain_id == other.domain_id && op_seq == other.op_seq;
    }
    bool same_contract(const RecoveryMetadata &other) const {
        return same_op(other) && participants == other.participants && replay_safety == other.replay_safety;
    }
    void validate() const {
        if (logical_run_id == 0 || rank < 0 || participants.empty() ||
            !std::is_sorted(participants.begin(), participants.end()) || participants.front() < 0 ||
            std::adjacent_find(participants.begin(), participants.end()) != participants.end() ||
            !std::binary_search(participants.begin(), participants.end(), rank))
            throw std::invalid_argument("invalid recovery op identity or rank membership");
        switch (replay_safety) {
        case ReplaySafety::UNKNOWN:
        case ReplaySafety::REPLAY_SAFE:
        case ReplaySafety::REQUIRES_RESTORE:
        case ReplaySafety::FORBIDDEN:
            break;
        default:
            throw std::invalid_argument("invalid replay safety declaration");
        }
    }
};
