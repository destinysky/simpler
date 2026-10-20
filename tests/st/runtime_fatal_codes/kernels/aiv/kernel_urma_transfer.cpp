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
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    auto *source = reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset;
    auto *destination = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    const uint64_t offset = reinterpret_cast<uint64_t>(source) - comm->windowsIn[comm->rankId];
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm->workSpace);
    const uint32_t peer = 1U - comm->rankId;
    using Data = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>>;
    Data remote(urma_backend::peer_mr_ptr<float>(workspace, peer, offset));
    Data local(destination);
    AsyncCtx ctx = get_async_ctx(args);
    (void)send_request_entry(ctx, UrmaTget(local, remote, workspace, peer));
}
