/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>

#define SIMPLER_ERROR_COORDINATED_STOP 12

// Dedicated DMA-written cache line; device polling never invalidates dirty runtime state.
struct alignas(64) ExecutionStopControl {
    uint64_t run_epoch{0};
    uint64_t requested_epoch{0};
};
static_assert(sizeof(ExecutionStopControl) == 64);

inline bool execution_stop_requested(const ExecutionStopControl &control) {
    const uint64_t epoch = __atomic_load_n(&control.run_epoch, __ATOMIC_ACQUIRE);
    return epoch != 0 && __atomic_load_n(&control.requested_epoch, __ATOMIC_ACQUIRE) == epoch;
}
