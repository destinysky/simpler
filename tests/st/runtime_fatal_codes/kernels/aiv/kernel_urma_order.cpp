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

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif

#include <pto/pto-inst.hpp>
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *marker = reinterpret_cast<__gm__ uint32_t *>(marker_tensor->buffer.addr) + marker_tensor->start_offset;
    auto *order = reinterpret_cast<__gm__ uint32_t *>(args[2]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[5]);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(order), sizeof(uint32_t));
    const uint32_t previous = *order;
    marker[0] = 1;
    marker[1] = previous;
    marker[2] = previous == static_cast<uint32_t>(args[3]) ? 1 : 0;
    auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(comm->workSpace);
    const uint32_t peer = 1U - comm->rankId;
    auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + peer;
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + peer;
    marker[3] = wq->wqn;
    marker[4] = cq->cqn;
    marker[5] = static_cast<uint32_t>(comm->workSpace);
    marker[6] = static_cast<uint32_t>(comm->workSpace >> 32);
    marker[7] = static_cast<uint32_t>(info->sqPtr);
    *order = static_cast<uint32_t>(args[4]);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(marker), 8 * sizeof(uint32_t));
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(order), sizeof(uint32_t));
    pipe_barrier(PIPE_ALL);
}
