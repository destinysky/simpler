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
    auto *src = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *dst = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    const uint64_t offset = src->buffer.addr - comm->windowsIn[comm->rankId];
    const uint64_t remote = comm->windowsIn[1 - comm->rankId] + offset;
    using ShapeT = Shape<1, 1, 1, 1, 16384>;
    using StrideT = pto::Stride<16384, 16384, 16384, 16384, 1>;
    GlobalTensor<float, ShapeT, StrideT> source(reinterpret_cast<__gm__ float *>(remote));
    GlobalTensor<float, ShapeT, StrideT> destination(reinterpret_cast<__gm__ float *>(dst->buffer.addr));
    Tile<TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    TASSIGN(scratch, 0x0);
    pto::comm::sdma::detail::UbTmpBuf tmp{};
    pto::comm::sdma::detail::MakeTmpBufferFromTile(scratch, tmp);
    AsyncCtx ctx = get_async_ctx(args);
    send_request_entry(
        ctx, SdmaTget(destination, source, scratch, reinterpret_cast<__gm__ uint8_t *>(comm->workSpace))
    );
}
