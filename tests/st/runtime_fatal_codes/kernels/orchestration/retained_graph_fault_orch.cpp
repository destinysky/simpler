/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    // The tensor is a retained managed input; only the selected member faults.
    if (args.scalar<uint64_t>(0) != 0) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "retained graph selected member fault");
        return;
    }

    // Exercise the same nonempty healthy path as the same-stream recovery tests.
    // Its scratch output is separate from the retained managed input.
    uint32_t shape[1] = {1};
    TensorCreateInfo output(shape, 1, DataType::INT32);
    CoreTaskArgs healthy;
    healthy.add_output(output);
    rt_submit_aiv_task(0, healthy);
}
}
