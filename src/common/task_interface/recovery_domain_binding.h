/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>
#include <vector>

enum class RecoveryArgKind : uint32_t { COMM_CONTEXT = 0, WINDOW_ADDRESS = 1, WINDOW_TENSOR = 2 };

struct RecoveryDomainBinding {
    uint64_t allocation_id{0};
    uint64_t generation{0};
    uint64_t device_ctx{0};
    uint64_t local_window_base{0};
    uint64_t window_size{0};
    uint64_t workspace{0};
};

inline bool operator==(const RecoveryDomainBinding &a, const RecoveryDomainBinding &b) {
    return a.allocation_id == b.allocation_id && a.generation == b.generation && a.device_ctx == b.device_ctx &&
           a.local_window_base == b.local_window_base && a.window_size == b.window_size && a.workspace == b.workspace;
}

struct RecoveryArgBinding {
    RecoveryArgKind kind{RecoveryArgKind::COMM_CONTEXT};
    uint32_t index{0};
    uint64_t offset{0};
};

struct TaskDomainBinding {
    RecoveryDomainBinding domain{};
    std::vector<RecoveryArgBinding> arguments;
};
