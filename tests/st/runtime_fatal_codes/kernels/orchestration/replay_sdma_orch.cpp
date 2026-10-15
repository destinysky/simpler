/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 11};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs transfer;
    transfer.add_input(args.tensor(0).ref());
    transfer.add_output(args.tensor(1).ref());
    transfer.add_scalar(args.scalar<uint64_t>(0));
    rt_submit_aiv_task(0, transfer);

    // Reading the produced tensor waits for the deferred SDMA completion.
    const uint32_t first[1] = {0};
    (void)get_tensor_data<float>(args.tensor(1).ref(), 1, first);
    const uint32_t index[1] = {static_cast<uint32_t>(args.scalar<uint64_t>(3) - 1)};
    const auto &counter = args.tensor(3).ref();
    const uint32_t attempt = get_tensor_data<uint32_t>(counter, 1, index) + 1;
    set_tensor_data(counter, 1, index, attempt);
    const uint64_t mode = args.scalar<uint64_t>(4);
    if (mode == 2 || (mode == 1 && attempt == 1)) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "replay regression fault after SDMA completion");
        return;
    }

    CoreTaskArgs completion;
    completion.add_input(args.tensor(1).ref());
    completion.add_output(args.tensor(2).ref());
    completion.add_scalar(args.scalar<uint64_t>(1));
    completion.add_scalar(args.scalar<uint64_t>(2));
    completion.add_scalar(args.scalar<uint64_t>(3));
    rt_submit_aiv_task(1, completion);
}
}
