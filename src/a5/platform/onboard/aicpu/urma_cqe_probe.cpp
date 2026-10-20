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

#include "common/urma_cqe_probe.h"
#include "aicpu/cache_maintenance.h"

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_read_urma_cqe(void *arg) {
    if (arg == nullptr) return -1;
    const auto &args = *static_cast<const UrmaCqeProbeArgs *>(arg);
    if (args.cqe_addr == 0 || args.result_addr == 0 || args.request_id == 0 || args.cqe_addr % alignof(uint32_t) != 0 ||
        args.result_addr % alignof(UrmaCqeProbeResult) != 0)
        return -1;
    // HCOMM queue VAs belong to the device process, not the host address space.
    const auto *cqe = reinterpret_cast<const volatile uint32_t *>(args.cqe_addr);
    cache_invalidate_range(const_cast<const uint32_t *>(cqe), sizeof(uint32_t));
    auto *result = reinterpret_cast<UrmaCqeProbeResult *>(args.result_addr);
    result->dw0 = *cqe;
    result->cqe_addr = args.cqe_addr;
    result->request_id = args.request_id;
    cache_flush_range(result, sizeof(*result));
    return 0;
}
