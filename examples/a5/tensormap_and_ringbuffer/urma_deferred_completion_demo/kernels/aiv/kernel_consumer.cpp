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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    __gm__ float *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    constexpr int kTotalRows = 128;
    constexpr int kRows = 64;
    constexpr int kCols = 128;
    constexpr int kIters = kTotalRows / kRows;
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileData src_tile(kRows, kCols);
    TileData result_tile(kRows, kCols);
    TASSIGN(src_tile, 0x0);
    TASSIGN(result_tile, 0x10000);

    constexpr int kChunkElems = kRows * kCols;
    for (int iter = 0; iter < kIters; ++iter) {
        GlobalData src_global(src + iter * kChunkElems);
        GlobalData result_global(result + iter * kChunkElems);
        TLOAD(src_tile, src_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TADDS(result_tile, src_tile, 1.0f);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(result_global, result_tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }

    if (args[4] != 0) {
        __gm__ Tensor *evidence_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
        __gm__ int32_t *evidence =
            reinterpret_cast<__gm__ int32_t *>(evidence_tensor->buffer.addr) + evidence_tensor->start_offset;
        auto *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);
        AsyncCtx ctx = get_async_ctx(args);
        if (!async_ctx_is_deferred(ctx) || ctx.completion_count == nullptr || ctx.completion_entries == nullptr ||
            comm_ctx == nullptr || comm_ctx->workSpace == 0)
            return;
        auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
        const uint32_t peer = 1u - comm_ctx->rankId;
        comm::AsyncSession session;
        if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(workspace, peer, session)) return;
        // The producer's deferred completions precede this signal readback.
        using SignalData = GlobalTensor<int32_t, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
        SignalData remote_signal(urma_backend::peer_mr_ptr<int32_t>(workspace, peer, 0));
        // Signal DMA writes a separate cache line from the evidence stores.
        SignalData local_signal(evidence + 32);
        auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(local_signal, remote_signal, session);
        const bool registered = urma_backend::register_urma_async_event(ctx, event, session, workspace);
        comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(evidence), 8 * sizeof(int32_t));
        evidence[6] = registered && *ctx.completion_count == 1 ? 1 : 0;
        pto2::detail::defer_flush(ctx);
        comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(evidence), 8 * sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
    }
}
