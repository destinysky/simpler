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
#include <type_traits>

struct UrmaCqeProbeArgs {
    uint64_t cqe_addr{};
    uint64_t result_addr{};
    uint64_t request_id{};
};

struct alignas(64) UrmaCqeProbeResult {
    uint64_t cqe_addr{};
    uint64_t request_id{};
    uint32_t dw0{};
    uint32_t reserved[11]{};
};

static_assert(sizeof(UrmaCqeProbeArgs) == 24 && sizeof(UrmaCqeProbeResult) == 64);
static_assert(std::is_trivially_copyable_v<UrmaCqeProbeArgs> && std::is_standard_layout_v<UrmaCqeProbeArgs>);
static_assert(std::is_trivially_copyable_v<UrmaCqeProbeResult> && std::is_standard_layout_v<UrmaCqeProbeResult>);

inline bool urma_cqe_probe_matches(const UrmaCqeProbeArgs &args, const UrmaCqeProbeResult &result) {
    return args.request_id != 0 && result.request_id == args.request_id && result.cqe_addr == args.cqe_addr;
}

inline constexpr uint32_t URMA_STOP_SAMPLED = 1U;
inline constexpr uint32_t URMA_STOP_PENDING = 2U;
inline constexpr uint32_t URMA_STOP_ERROR = 4U;
inline constexpr uint32_t URMA_STOP_BUSY = 8U;

struct UrmaStopEvidence {
    uint64_t run_epoch{};
    uint32_t peer{};
    uint32_t wqn{};
    uint32_t cqn{};
    uint32_t target_cqe{};
    uint32_t retired_cqe{};
    uint32_t sequence{};
    uint32_t dw0{};
    uint32_t flags{};
    uint32_t target_bb{};
    uint32_t retired_bb{};
};
static_assert(sizeof(UrmaStopEvidence) == 48);
static_assert(std::is_trivially_copyable_v<UrmaStopEvidence> && std::is_standard_layout_v<UrmaStopEvidence>);
