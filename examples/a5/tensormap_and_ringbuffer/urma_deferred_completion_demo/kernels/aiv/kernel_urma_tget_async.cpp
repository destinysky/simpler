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

using namespace pto;

namespace {

constexpr int kElems = 128 * 128;

template <typename T>
static inline __aicore__ __gm__ T *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *evidence_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
    const bool notify = args[4] != 0;
    __gm__ int32_t *evidence = tensor_data<int32_t>(evidence_tensor);
    AsyncCtx async_ctx = get_async_ctx(args);
    const bool deferred = async_ctx_is_deferred(async_ctx) && async_ctx.completion_count != nullptr &&
                          async_ctx.completion_entries != nullptr;
    evidence[5] = deferred ? 1 : 0;
    if (!deferred || comm_ctx == nullptr || comm_ctx->rankNum != 2 || comm_ctx->rankId >= comm_ctx->rankNum ||
        comm_ctx->workSpace == 0 || comm_ctx->windowsIn[comm_ctx->rankId] == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    const uint32_t peer_rank = 1u - comm_ctx->rankId;
    const uint64_t peer_base = urma_backend::peer_mr_base_addr(workspace, peer_rank);
    __gm__ float *local_input = tensor_data<float>(input_tensor);
    __gm__ float *local_out = tensor_data<float>(out_tensor);
    const uint64_t input_offset = reinterpret_cast<uint64_t>(local_input) - comm_ctx->windowsIn[comm_ctx->rankId];
    __gm__ float *remote_input = reinterpret_cast<__gm__ float *>(peer_base + input_offset);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(workspace, peer_rank, session)) return;
    auto *info = reinterpret_cast<__gm__ comm::urma::UrmaInfo *>(workspace);
    auto *wq = reinterpret_cast<__gm__ comm::urma::UrmaWQCtx *>(info->sqPtr) + peer_rank;
    uint32_t previous_bb = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
    uint32_t previous_cqe = wq->submittedWqeCount;
    bool registered = true;
    if (notify) {
        using Payload = GlobalTensor<float, Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>>;
        Payload source(local_input);
        Payload destination(reinterpret_cast<__gm__ float *>(peer_base + 64));
        comm::Signal signal(reinterpret_cast<__gm__ int32_t *>(peer_base));
        auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
            destination, source, signal, 7, comm::NotifyOp::AtomicAdd, session, peer_rank
        );
        evidence[0] = static_cast<uint32_t>(event.handle) - previous_bb;
        evidence[1] = event.urmaTargetCqe - previous_cqe;
        previous_bb = static_cast<uint32_t>(event.handle);
        previous_cqe = event.urmaTargetCqe;
        registered = urma_backend::register_urma_async_event(async_ctx, event, session, workspace);
    }

    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = pto::Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    GlobalData remote_global(remote_input);
    GlobalData local_global(local_out);
    auto read = comm::TGET_ASYNC<comm::DmaEngine::URMA>(local_global, remote_global, session);
    evidence[2] = static_cast<uint32_t>(read.handle) - previous_bb;
    evidence[3] = read.urmaTargetCqe - previous_cqe;
    registered = urma_backend::register_urma_async_event(async_ctx, read, session, workspace) && registered;
    evidence[4] = *async_ctx.completion_count;
    evidence[7] = registered ? 1 : 0;
    pto2::detail::defer_flush(async_ctx);
    comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(evidence), 8 * sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}
