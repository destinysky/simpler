/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <cstdint>

namespace urma_drain {

enum class State : uint32_t { DRAINED = 0, PENDING = 1, CQE_ERROR = 2, INVALID = 3, READ_FAILED = 4 };

struct Queue {
    uint32_t submitted_bb{};
    uint32_t retired_bb{};
    uint32_t submitted_cqe{};
    uint32_t retired_cqe{};
    uint32_t sq_depth{};
    uint32_t cq_depth{};
};

struct Observation {
    State state{State::INVALID};
    uint32_t checked{};
    uint32_t sequence{};
    uint32_t dw0{};
    int read_rc{};
};

inline bool power_of_two(uint32_t n) { return n != 0 && (n & (n - 1)) == 0 && n < (1U << 31); }

// Producers and completion consumers are fenced throughout this read-only assessment.
// The runtime and PTO-ISA retire only successful CQE prefixes.
template <typename ReadCqe>
Observation inspect(const Queue &q, ReadCqe &&read_cqe) {
    Observation result;
    const uint32_t remaining_bb = q.submitted_bb - q.retired_bb;
    const uint32_t remaining_cqe = q.submitted_cqe - q.retired_cqe;
    if (!power_of_two(q.sq_depth) || !power_of_two(q.cq_depth) || remaining_bb > q.sq_depth ||
        remaining_cqe > q.cq_depth || (remaining_cqe == 0 && remaining_bb != 0))
        return result;
    if (q.submitted_bb == 0 && q.retired_bb == 0 && q.submitted_cqe == 0 && q.retired_cqe == 0) {
        result.state = State::DRAINED;
        return result;
    }
    // An already-retired target still needs its newest hardware CQE checked.
    uint32_t sequence = remaining_cqe == 0 ? q.submitted_cqe - 1 : q.retired_cqe;
    do {
        result.sequence = sequence;
        result.read_rc = read_cqe(sequence, result.dw0);
        if (result.read_rc != 0) {
            result.state = State::READ_FAILED;
            return result;
        }
        ++result.checked;
        const bool old_owner = ((sequence / q.cq_depth) & 1U) != 0;
        if (((result.dw0 >> 2) & 1U) == static_cast<uint32_t>(old_owner)) {
            result.state = State::PENDING;
            return result;
        }
        if ((result.dw0 & 0xFFFF0000U) != 0) {
            result.state = State::CQE_ERROR;
            return result;
        }
    } while (++sequence != q.submitted_cqe);
    result.state = State::DRAINED;
    return result;
}

}  // namespace urma_drain
