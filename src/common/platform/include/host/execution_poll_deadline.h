/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <chrono>
#include <cstdint>

// Bounds the pending phase of a nonblocking execution wait. Serialized by the
// native-run phase owner, just like poll/wait/finalize. Preparation never arms
// this clock: a prepared successor must not reset its predecessor's budget.
class ExecutionPollDeadline {
public:
    using Clock = std::chrono::steady_clock;

    void arm(int32_t timeout_ms, Clock::time_point now = Clock::now()) {
        enabled_ = timeout_ms > 0;
        deadline_ = enabled_ ? now + std::chrono::milliseconds(timeout_ms) : Clock::time_point::max();
        timed_out_ = false;
    }

    // Call only after querying both streams and observing NOT_READY. A real
    // completion/error observed by that query takes precedence over the clock.
    bool observe_pending(Clock::time_point now = Clock::now()) {
        timed_out_ = timed_out_ || (enabled_ && now >= deadline_);
        return timed_out_;
    }

    bool timed_out() const { return timed_out_; }

private:
    Clock::time_point deadline_{Clock::time_point::max()};
    bool enabled_{false};
    bool timed_out_{false};
};
