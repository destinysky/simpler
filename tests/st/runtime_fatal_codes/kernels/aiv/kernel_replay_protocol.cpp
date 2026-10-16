/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#include <pto/pto-inst.hpp>
#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *flags = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    const uint32_t epoch = static_cast<uint32_t>(args[3]);
    const uint64_t publish = static_cast<uint64_t>(args[4]);
    const uint64_t fail = static_cast<uint64_t>(args[5]);
    AsyncCtx ctx = get_async_ctx(args);
    Tile<TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    TASSIGN(scratch, 0x0);
    pto::comm::sdma::detail::UbTmpBuf tmp{};
    pto::comm::sdma::detail::MakeTmpBufferFromTile(scratch, tmp);
    auto *local = reinterpret_cast<__gm__ uint8_t *>(flags->buffer.addr);

    if (publish && fail) {
        // The producer has observed the peer's entry notification before this kernel launches.
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    if (!publish && comm->rankId == 0) {
        save_expected_notification_counter(ctx, local, epoch);
        return;
    }

    // rank1 announces entry; rank0 publishes completion only after successful production.
    const uint64_t target_offset = publish ? 128 : 0;
    pto::comm::sdma::detail::SetValue<uint32_t>(local + 64, tmp, 0, epoch);
    const uint64_t offset = flags->buffer.addr - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ uint32_t *>(comm->windowsIn[1 - comm->rankId] + offset + target_offset);
    using ShapeT = Shape<1, 1, 1, 1, 8>;
    using StrideT = pto::Stride<8, 8, 8, 8, 1>;
    GlobalTensor<uint32_t, ShapeT, StrideT> source(reinterpret_cast<__gm__ uint32_t *>(local + 64));
    GlobalTensor<uint32_t, ShapeT, StrideT> destination(remote);
    send_request_entry(
        ctx, SdmaTput(destination, source, scratch, reinterpret_cast<__gm__ uint8_t *>(comm->workSpace))
    );
    if (!publish) save_expected_notification_counter(ctx, local + 128, epoch);
}
