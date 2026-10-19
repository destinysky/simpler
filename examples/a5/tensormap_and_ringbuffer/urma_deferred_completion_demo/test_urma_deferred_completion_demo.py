#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""URMA deferred-completion smoke test for onboard a5.

Each rank stages its input inside the communication window. The producer
fetches the peer rank's input into local ``out`` and registers the URMA async
event. The notify case submits an FAA notification before the read and reads
back the remote signal. The consumer depends on that output and writes ``result = out + 1``.
"""

import ctypes
import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

N = 128 * 128
NRANKS = 2
DTYPE_NBYTES = 4
EVIDENCE_ELEMS = 64
URMA_DATA_OFFSET_NBYTES = 64 * DTYPE_NBYTES
_URMA_WORKSPACE_ENV = "SIMPLER_ENABLE_PTO_URMA_WORKSPACE"
_WORKSPACE_TRUTHY = {"1", "ON", "TRUE", "YES"}


def _urma_workspace_enabled():
    return os.environ.get(_URMA_WORKSPACE_ENV, "").upper() in _WORKSPACE_TRUTHY


def _require_urma_workspace_enabled():
    if not _urma_workspace_enabled():
        raise RuntimeError(
            "urma_deferred_completion_demo requires host runtime built with "
            f"{_URMA_WORKSPACE_ENV}=ON; set it before rebuilding simpler."
        )


def urma_deferred_completion_orch_fn(orch, callables, task_args, config):
    _require_urma_workspace_enabled()
    input_nbytes = N * DTYPE_NBYTES
    with orch.allocate_domain(
        name="urma_deferred_completion",
        workers=list(range(NRANKS)),
        window_size=max(URMA_DATA_OFFSET_NBYTES + input_nbytes, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(
                name="urma_reserved",
                dtype="int32",
                count=URMA_DATA_OFFSET_NBYTES // DTYPE_NBYTES,
                nbytes=URMA_DATA_OFFSET_NBYTES,
            ),
            CommBufferSpec(name="input_window", dtype="float32", count=N, nbytes=input_nbytes),
        ],
    ) as handle:
        for rank in range(NRANKS):
            source = getattr(task_args, f"in_{rank}")
            orch.copy_to(handle[rank].buffers["input_window"], _rehosted_buffer_for(task_args, source))
            initial = getattr(task_args, f"reserved_{rank}")
            orch.copy_to(handle[rank].buffers["urma_reserved"], _rehosted_buffer_for(task_args, initial))
        for rank in range(NRANKS):
            domain = handle[rank]
            args = TaskArgs()
            args.add_tensor(domain.buffers["input_window"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"out_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"result_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_tensor(
                _rehosted_ref_for(task_args, getattr(task_args, f"evidence_{rank}")), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(domain.device_ctx)
            args.add_scalar(task_args.notify_mode)
            orch.submit_next_level(callables.urma_deferred_completion, args, config, worker=rank)


@pytest.mark.skipif(
    not _urma_workspace_enabled(),
    reason="URMA workspace overlay not enabled. "
    "Set SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON before rebuilding and running.",
)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaDeferredCompletionDemo(SceneTestCase):
    CALLABLE = {
        "orchestration": urma_deferred_completion_orch_fn,
        "callables": [
            {
                "name": "urma_deferred_completion",
                "orchestration": {
                    "source": "kernels/orchestration/urma_deferred_completion_orch.cpp",
                    "function_name": "urma_deferred_completion_orchestration",
                    "config_name": "urma_deferred_completion_orchestration_config",
                    "signature": [D.IN, D.OUT, D.OUT, D.OUT, D.IN, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_urma_tget_async.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.INOUT, D.IN, D.IN],
                    },
                ],
            }
        ],
    }
    CASES = [
        {
            "name": "peer_fetch",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"notify": False},
        },
        {
            "name": "notify_then_fetch",
            "platforms": ["a5"],
            "config": {"device_count": NRANKS, "num_sub_workers": 0},
            "params": {"notify": True},
        },
    ]
    RTOL = 0.0
    ATOL = 1e-3

    def generate_args(self, params):
        specs = []
        for rank in range(NRANKS):
            inp = torch.tensor([float(rank * 1000 + (i % 251)) / 10.0 for i in range(N)], dtype=torch.float32)
            specs.extend(
                [
                    TensorArg(f"in_{rank}", inp),
                    TensorArg(f"out_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"result_{rank}", torch.zeros(N, dtype=torch.float32)),
                    TensorArg(f"reserved_{rank}", torch.zeros(64, dtype=torch.int32)),
                    TensorArg(f"evidence_{rank}", torch.zeros(EVIDENCE_ELEMS, dtype=torch.int32)),
                ]
            )
        return TaskArgsBuilder(*specs, Scalar("notify_mode", ctypes.c_int64(int(params["notify"]))))

    def compute_golden(self, args, params):
        for rank in range(NRANKS):
            expected_out = getattr(args, f"in_{1 - rank}")
            getattr(args, f"out_{rank}").copy_(expected_out)
            getattr(args, f"result_{rank}").copy_(expected_out + 1.0)
            evidence = getattr(args, f"evidence_{rank}")
            evidence[:8] = torch.tensor(
                [3, 2, 1, 1, 2, 1, 1, 1] if params["notify"] else [0, 0, 1, 1, 1, 1, 0, 1],
                dtype=torch.int32,
            )
            if params["notify"]:
                evidence[32] = 7

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        mode = "notify_then_fetch" if params["notify"] else "peer_fetch"
        for rank in range(NRANKS):
            evidence = getattr(test_args, f"evidence_{rank}").tolist()
            print(
                f"[URMA_COMPLETION_TEST] EVIDENCE mode={mode} rank={rank} "
                f"notify_bb={evidence[0]} notify_cqe={evidence[1]} "
                f"read_bb={evidence[2]} read_cqe={evidence[3]} "
                f"registered={evidence[4]} deferred={evidence[5]} "
                f"readback_registered={evidence[6]} signal={evidence[32]}",
                flush=True,
            )
        print(f"[URMA_COMPLETION_TEST] URMA_COMPLETION_PASS mode={mode}", flush=True)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
