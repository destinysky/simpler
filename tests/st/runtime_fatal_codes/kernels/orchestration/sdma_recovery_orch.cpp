/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs transfer;
    transfer.add_input(args.tensor(0).ref());
    transfer.add_output(args.tensor(1).ref());
    transfer.add_scalar(args.scalar<uint64_t>(0));          // device CommContext
    transfer.add_scalar(args.scalar<uint64_t>(1));          // fault injection
    transfer.add_scalar(args.tensor(2).ref().buffer.addr);  // submission evidence
    rt_submit_aiv_task(0, transfer);

    // The tensor dependency includes the producer's deferred SDMA completion.
    CoreTaskArgs completion;
    completion.add_input(args.tensor(1).ref());
    completion.add_output(args.tensor(2).ref());
    completion.add_scalar(args.scalar<uint64_t>(2));  // ordering word
    completion.add_scalar(args.scalar<uint64_t>(3));  // expected predecessor
    completion.add_scalar(args.scalar<uint64_t>(4));  // committed ordering word
    completion.add_scalar(args.scalar<uint64_t>(1));
    rt_submit_aiv_task(1, completion);
}
}
