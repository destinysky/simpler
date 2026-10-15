#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real local L3 graph ownership after a selected chip member fails.

This regression deliberately ends with GIVE_UP. It verifies retention and
terminal cleanup without authorizing replay or simulating a network fault.
"""

import ctypes
import json
import os
import select
import sys
from pathlib import Path

import pytest
from simpler import Worker
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TaskState,
    TensorArgType,
)

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.timeout(180)
@pytest.mark.parametrize("close_held", [False, True], ids=["explicit-give-up", "close-held"])
def test_a5_retained_l3_graph(st_device_ids, close_held):  # noqa: PLR0915 -- one owned graph and its terminal cleanup
    assert len(st_device_ids) == 2
    compiler = KernelCompiler(platform="a5")
    core_binary = compiler.compile_incore(
        source_path=str(Path(__file__).parent / "kernels/aiv/kernel_noop.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=list(compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"))
        + [str(compiler.project_root / "src/common")],
    )
    healthy_core = CoreCallable.build(signature=[ArgDirection.OUT], binary=extract_text_section(core_binary))
    binary = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer",
        source_path=str(Path(__file__).parent / "kernels/orchestration/retained_graph_fault_orch.cpp"),
    )
    chip = ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN],
        func_name="aicpu_orchestration_entry",
        binary=binary,
        children=[(0, healthy_core)],
    )
    read_fd, write_fd = os.pipe()
    worker = Worker(
        level=3,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
        device_ids=list(st_device_ids),
        num_sub_workers=1,
        enable_operator_recovery=True,
    )

    def build_gate(_args):
        # The inherited pipe keeps A from launching until the entire graph exists.
        assert select.select([read_fd], [], [], 30)[0], "retained graph gate was not released within 30s"
        assert os.read(read_fd, 1) == b"G"

    gate_callable = worker.register(build_gate)
    chip_callable = worker.register(chip)
    handles = {}
    resources = []
    run = None
    expected_close_error = None
    try:
        worker.init()

        def graph(orch, _args, _config):
            try:
                with orch.scope():
                    buffer = orch.alloc((4,), DataType.UINT32)
                    resources.append(buffer)
                    ctypes.c_uint32.from_address(buffer.base).value = 0x1234ABCD
                    gate_args = TaskArgs()
                    gate_args.add_tensor(buffer.tensor((4,), DataType.UINT32), TensorArgType.OUTPUT)
                    orch.submit_sub(gate_callable, gate_args)
                    config = CallConfig()
                    config.aicpu_thread_num = 2

                    def args_for(fault, dependency=None):
                        args = TaskArgs()
                        args.add_tensor(buffer.tensor((4,), DataType.UINT32), TensorArgType.INPUT)
                        args.add_scalar(fault)
                        if dependency is not None:
                            args.add_dep(dependency)
                        return args

                    handles["A"] = orch.submit_next_level_group(
                        chip_callable, [args_for(1), args_for(0)], config, workers=[0, 1]
                    )
                    handles["B"] = orch.submit_next_level_group(
                        chip_callable, [args_for(0, handles["A"]), args_for(0, handles["A"])], config, workers=[0, 1]
                    )
                    handles["C"] = orch.submit_next_level_group(
                        chip_callable, [args_for(0, handles["B"]), args_for(0, handles["B"])], config, workers=[0, 1]
                    )
            finally:
                # Release before Worker.submit can wait for cancellation cleanup.
                os.write(write_fd, b"G")
                print("[PTO_RECOVERY_TEST] RETAINED_GRAPH_GATE_RELEASED", flush=True)

        run = worker.submit(graph)
        print("[PTO_RECOVERY_TEST] RETAINED_GRAPH_SUBMITTED", flush=True)
        episode = worker.wait_for_recovery(60)
        assert episode is not None and episode.all_members_terminal
        print(
            "[PTO_RECOVERY_TEST] RETAINED_GRAPH_MEMBERS "
            + json.dumps(
                [
                    {
                        "worker_id": member.worker_id,
                        "raw_rc": member.raw_rc,
                        "runtime_status": member.runtime_status,
                    }
                    for member in episode.members
                ],
                sort_keys=True,
            ),
            flush=True,
        )
        assert [member.worker_id for member in episode.members] == [0, 1]
        assert episode.members[0].runtime_status != 0 or episode.members[0].raw_rc != 0
        assert episode.members[1].raw_rc == episode.members[1].runtime_status == 0
        snapshot = worker.recovery_graph(episode.ticket)
        assert not snapshot.run_failed and snapshot.submission_closed and snapshot.active_recoveries == 1
        selected = {
            name: next(item for item in snapshot.tasks if item.is_task(handle)) for name, handle in handles.items()
        }
        assert selected["A"].state == TaskState.RETRY_PENDING
        assert selected["B"].state == selected["C"].state == TaskState.PENDING
        assert selected["B"].depends_on(handles["A"])
        assert selected["C"].depends_on(handles["B"])
        # The gate may have released its tensor edges; A and B still block successors.
        assert selected["B"].fanin_released < selected["B"].fanin_count
        assert selected["C"].fanin_released < selected["C"].fanin_count
        identity = resources[0].identity
        for name, item in selected.items():
            assert item.is_task(handles[name])
            assert len(item.args) == 2
            assert all(args.tensor(0).buffer.identity == identity for args in item.args)
            assert [args.scalar(0) for args in item.args] == ([1, 0] if name == "A" else [0, 0])
        assert ctypes.c_uint32.from_address(resources[0].base).value == 0x1234ABCD
        with pytest.raises(TimeoutError):
            run.wait(0.05)
        again = worker.recovery_graph(episode.ticket)
        assert not again.run_failed and len(worker.pending_recoveries()) == 1
        print(
            "[PTO_RECOVERY_TEST] RETAINED_GRAPH "
            + json.dumps(
                {
                    "recovery_id": episode.ticket.recovery_id,
                    "A": "RETRY_PENDING",
                    "B": "PENDING",
                    "C": "PENDING",
                    "buffer_preserved": True,
                    "rank1_healthy": True,
                    "run_failed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if close_held:
            # close cancels the episode before waiting for accepted run fences.
            with pytest.raises(
                RuntimeError,
                match=f"recovery id={episode.ticket.recovery_id} GIVE_UP: worker stopping during recovery",
            ) as closed:
                worker.close()
            expected_close_error = closed.value
            assert worker._close_completion is not None and worker._close_completion.done
            assert worker._close_completion.error is expected_close_error
            assert not worker._accepted_run_handles
            assert not worker._has_live_resources(), worker._describe_live_resources()
        else:
            assert worker.give_up_recovery(episode.ticket, "retention regression ends without replay")
            assert not worker.give_up_recovery(episode.ticket, "duplicate decision")
            with pytest.raises(RuntimeError):
                run.wait(30)
            assert not worker.pending_recoveries()
            with pytest.raises(ValueError):
                worker.recovery_graph(episode.ticket)
            worker.close()
    finally:
        primary_error = sys.exc_info()[1]
        try:
            # Retry close for teardown debt, preserving the first test failure.
            worker.close()
        except Exception as cleanup_error:  # noqa: BLE001 -- retain diagnostics without masking the primary error
            if cleanup_error is expected_close_error:
                # A terminal close replays the same outcome without redoing teardown.
                print("[PTO_RECOVERY_TEST] RETAINED_GRAPH_CLOSE_RESULT_REPLAYED", flush=True)
            elif primary_error is None:
                raise
            else:
                print(f"[PTO_RECOVERY_TEST] RETAINED_GRAPH_CLEANUP_ERROR {cleanup_error}", flush=True)
        finally:
            os.close(read_fd)
            os.close(write_fd)
    marker = "RETAINED_GRAPH_CLOSE_PASS" if close_held else "RETAINED_GRAPH_GIVE_UP_PASS"
    print(f"[PTO_RECOVERY_TEST] {marker}", flush=True)
