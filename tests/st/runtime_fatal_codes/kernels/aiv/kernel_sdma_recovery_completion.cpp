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
#include "async_kernel_api.h"
#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *marker = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *order = reinterpret_cast<__gm__ uint8_t *>(args[2]);
    const uint32_t expected = static_cast<uint32_t>(args[3]);
    const uint32_t next = static_cast<uint32_t>(args[4]);
    const uint64_t mode = static_cast<uint64_t>(args[5]);
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
    if (mode == 1) {
        // No rank publishes this notification. The scheduler watchdog reports
        // the bounded protocol wait, after the SDMA data operation has ended.
        AsyncCtx ctx = get_async_ctx(args);
        save_expected_notification_counter(ctx, reinterpret_cast<volatile __gm__ void *>(order + 32), 1);
    } else {
        pto::comm::sdma::detail::SetValue<uint32_t>(order, tmp, 0, next);
    }
    pipe_barrier(PIPE_ALL);
}
