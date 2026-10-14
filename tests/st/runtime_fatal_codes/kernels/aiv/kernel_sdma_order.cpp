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
#include <pto/comm/async/sdma/sdma_async_intrin.hpp>
#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *marker = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *order = reinterpret_cast<__gm__ uint8_t *>(args[2]);
    const uint32_t expected = static_cast<uint32_t>(args[3]);
    const uint32_t next = static_cast<uint32_t>(args[4]);
    Tile<TileType::Vec, uint8_t, 1, 32> scratch;
    TASSIGN(scratch, 0x0);
    pto::comm::sdma::detail::UbTmpBuf tmp{};
    pto::comm::sdma::detail::MakeTmpBufferFromTile(scratch, tmp);
    const uint32_t previous = pto::comm::sdma::detail::GetValue<uint32_t>(order, tmp);
    auto *evidence = reinterpret_cast<__gm__ uint8_t *>(marker->buffer.addr);
    // This kernel cannot start until the real SDMA producer has completed.
    pto::comm::sdma::detail::SetValue<uint32_t>(evidence, tmp, 0, 1);
    pto::comm::sdma::detail::SetValue<uint32_t>(evidence + 4, tmp, 0, previous);
    pto::comm::sdma::detail::SetValue<uint32_t>(evidence + 8, tmp, 0, previous == expected ? 1 : 0);
    pto::comm::sdma::detail::SetValue<uint32_t>(order, tmp, 0, next);
    pipe_barrier(PIPE_ALL);
}
