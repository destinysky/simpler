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

#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 13};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs transfer;
    transfer.add_input(args.tensor(0).ref());
    transfer.add_output(args.tensor(1).ref());
    transfer.add_scalar(args.scalar<uint64_t>(0));
    rt_submit_aiv_task(0, transfer);

    // Reading the produced tensor waits for the deferred transport completion.
    const uint32_t first[1] = {0};
    (void)get_tensor_data<float>(args.tensor(1).ref(), 1, first);
    const uint32_t index[1] = {static_cast<uint32_t>(args.scalar<uint64_t>(3) - 1)};
    const auto &counter = args.tensor(3).ref();
    const uint32_t attempt = get_tensor_data<uint32_t>(counter, 1, index) + 1;
    set_tensor_data(counter, 1, index, attempt);
    const uint64_t mode = args.scalar<uint64_t>(4);
    const uint64_t epoch = args.scalar<uint64_t>(3) * 16 + attempt;
    auto protocol = [&](uint64_t publish, uint64_t fail) {
        CoreTaskArgs task;
        task.add_inout(args.tensor(6).ref());
        task.add_output(args.tensor(2).ref());
        task.add_scalar(args.scalar<uint64_t>(0));
        task.add_scalar(epoch);
        task.add_scalar(publish);
        task.add_scalar(fail);
        rt_submit_aiv_task(1, task);
        (void)get_tensor_data<uint32_t>(args.tensor(2).ref(), 1, first);
    };
    protocol(0, (mode == 3 || mode == 4) && attempt == 1 ? 4 : 0);
    // Rank0 produces the ready notification. Rank1 cannot finish A without it.
    if (args.scalar<uint64_t>(5) == 0) protocol(1, mode == 2 || ((mode == 1 || mode == 3) && attempt == 1));

    CoreTaskArgs completion;
    completion.add_input(args.tensor(1).ref());
    completion.add_output(args.tensor(2).ref());
    completion.add_scalar(args.scalar<uint64_t>(1));
    completion.add_scalar(args.scalar<uint64_t>(2));
    completion.add_scalar(args.scalar<uint64_t>(3));
    completion.add_scalar(args.scalar<uint64_t>(0));
    // Protocol completion was awaited above. Keep the order kernel's six-slot
    // prefix and pass its diagnostic buffer address after the existing scalars.
    const auto &handshake = args.tensor(6).ref();
    completion.add_scalar(handshake.buffer.addr + handshake.start_offset * sizeof(uint32_t));
    completion.add_scalar(mode == 3 || mode == 4);
    rt_submit_aiv_task(2, completion);
}
}
