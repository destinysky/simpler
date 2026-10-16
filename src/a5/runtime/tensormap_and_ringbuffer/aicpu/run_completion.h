/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <atomic>
#include <cstdint>

// One native launch contains several AICPU entry calls. An early nonzero
// entry return can expose stream failure before the last participant has
// destroyed the PTO runtime and published its stop proof. Aggregate local
// errors here; only the last participant reports failure after deinit.
class AicpuRunCompletion {
public:
    struct Arrival {
        int32_t finished;
        bool last;
        int32_t report_rc;
    };

    Arrival arrive(int32_t participants, int32_t local_rc) {
        if (local_rc != 0) {
            int32_t expected = 0;
            first_error_.compare_exchange_strong(expected, local_rc, std::memory_order_acq_rel);
        }
        const int32_t finished = finished_.fetch_add(1, std::memory_order_acq_rel) + 1;
        const bool last = finished == participants;
        return {finished, last, last ? first_error_.load(std::memory_order_acquire) : 0};
    }

    int32_t finished() const { return finished_.load(std::memory_order_acquire); }

    // All run participants have left their work before the last one calls
    // deinit/reset. Arrivals retain their results locally, never via this state.
    void reset() {
        finished_.store(0, std::memory_order_release);
        first_error_.store(0, std::memory_order_release);
    }

private:
    std::atomic<int32_t> finished_{0};
    std::atomic<int32_t> first_error_{0};
};
