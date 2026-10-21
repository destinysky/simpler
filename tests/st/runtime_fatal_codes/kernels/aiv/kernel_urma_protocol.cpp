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

static inline __aicore__ void wait_counter(AsyncCtx &ctx, const __gm__ uint32_t *counter, uint32_t epoch) {
    CompletionToken wait{
        reinterpret_cast<uint64_t>(counter), epoch, COMPLETION_ENGINE_URMA, COMPLETION_TYPE_COUNTER, 0
    };
    (void)register_completion_condition(ctx, wait);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *flags = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    const uint32_t epoch = static_cast<uint32_t>(args[3]);
    const bool publish = args[4] != 0;
    const bool fail = (args[5] & 1) != 0;
    const bool burst = (args[5] & 4) != 0;
    auto *local = reinterpret_cast<__gm__ uint32_t *>(flags->buffer.addr) + flags->start_offset;
    AsyncCtx ctx = get_async_ctx(args);
    if (publish && fail) {
        // Malformed kernel completion injection; no hardware CQE is modified.
        CompletionToken invalid{0, epoch, COMPLETION_ENGINE_URMA, COMPLETION_TYPE_COUNTER, 0};
        (void)register_completion_condition(ctx, invalid);
        pto2::detail::defer_flush(ctx);
        return;
    }
    if (!publish && comm->rankId == 0) {
        wait_counter(ctx, local, epoch);
        pto2::detail::defer_flush(ctx);
        return;
    }
    for (uint32_t i = 0; i < 4; ++i)
        local[16 + i] = epoch;
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(local + 16), 4 * sizeof(uint32_t));
    pipe_barrier(PIPE_ALL);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm->workSpace);
    const uint32_t peer = 1U - comm->rankId;
    const uint64_t offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
    auto *remote = urma_backend::peer_mr_ptr<uint32_t>(workspace, peer, offset);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(workspace, peer, session)) return;
    using Payload = pto::GlobalTensor<uint32_t, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>>;
    Payload source(local + 16);
    Payload destination(remote + 8);
    pto::comm::Signal signal(reinterpret_cast<__gm__ int32_t *>(remote + (publish ? 32 : 0)));
    auto event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        destination, source, signal, static_cast<int32_t>(epoch), pto::comm::NotifyOp::Set, session, peer
    );
    if (burst && !publish) {
        if (!event.Wait(session)) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
            return;
        }
        constexpr uint32_t kWords = 256U * 1024U * 1024U / sizeof(uint32_t);
        auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
        auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + peer;
        auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + peer;
        if (wq->depth <= 2 || cq->depth <= 2) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
            return;
        }
        uint32_t requests = 128;
        if (requests > wq->depth - 1) requests = wq->depth - 1;
        if (requests > cq->depth - 1) requests = cq->depth - 1;
        if (requests < 2) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
            return;
        }
        using Traffic =
            pto::GlobalTensor<uint32_t, pto::Shape<1, 1, 1, 1, kWords>, pto::Stride<kWords, kWords, kWords, kWords, 1>>;
        Traffic source(urma_backend::peer_mr_ptr<uint32_t>(workspace, peer, offset) + 64);
        Traffic destination(local + 64 + kWords);
        for (uint32_t i = 0; i < requests; ++i)
            event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(destination, source, session);
        local[48] = requests;
        local[49] = event.urmaTargetCqe;
        pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(local + 48), 2 * sizeof(uint32_t));
    }
    (void)urma_backend::register_urma_async_event(ctx, event, session, workspace);
    if (!publish) wait_counter(ctx, local + 32, epoch);
    pto2::detail::defer_flush(ctx);
}
