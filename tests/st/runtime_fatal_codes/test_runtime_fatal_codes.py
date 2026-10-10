#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative STs: deliberately trigger device error conditions and assert the host can tell them apart.

Closes the gap from issue #1180's comment — most ``orch_error_code`` /
``sched_error_code`` values had no end-to-end host coverage. Each case drives one
device error to its latch (orchestrator deadlocks via a tiny
``CallConfig.runtime_env``; the scheduler stall via a lowered scheduler timeout;
the async codes via a small AIV kernel that abuses the per-task completion slab).

Two host behaviours are exercised, because they genuinely differ:

* **sim** — ``worker.run`` returns the runtime's own status, so the host sees the
  real ``code -N`` directly (and the device error class on the failure line).
* **onboard** — for the cores-killed stalls the op-execute / stream-sync watchdog
  wins the race and masks the real code with a generic CANN ``507xxx``; the
  device-classified info (orchestrator code, the #1180 ``sub_class`` for the
  scheduler stall, or the async ``sched_error_code``) still reaches the host via
  the ``validate_runtime_impl`` log line. That race is the exact scenario #1180
  exists for and only hardware reproduces it; the onboard test therefore asserts
  on the host log rather than the masked exception code.
"""

import ctypes
import multiprocessing as mp
import os
import queue
import time

import pytest
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    ExecutionRecoveryScope,
    ExecutionRecoveryStatus,
    _flush_host_log,
    _host_log_dropped_records,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
KERNELS = os.path.join(HERE, "kernels")
ORCH_DIR = os.path.join(KERNELS, "orchestration")


def _wait_for_host_log(capfd, markers: tuple[str, ...], dropped_before: int, timeout_s: float = 5.0) -> str:
    """Poll for required records and report queue loss instead of trusting one flush deadline."""
    chunks: list[str] = []
    deadline = time.monotonic() + timeout_s
    last_flush = False
    while True:
        last_flush = _flush_host_log(100)
        captured = capfd.readouterr()
        chunks.extend((captured.err, captured.out))
        log = "".join(chunks)
        if all(marker in log for marker in markers):
            dropped_after = _host_log_dropped_records()
            assert dropped_after == dropped_before, (
                f"host-log drop counter changed while waiting for {markers}: "
                f"before={dropped_before}, after={dropped_after}"
            )
            return log
        if time.monotonic() >= deadline:
            dropped_after = _host_log_dropped_records()
            missing = [marker for marker in markers if marker not in log]
            raise AssertionError(
                f"host-log records did not arrive within {timeout_s:.1f}s: missing={missing}, "
                f"last_flush={last_flush}, dropped_delta={dropped_after - dropped_before}, tail={log[-2000:]!r}"
            )
        time.sleep(0.01)


# case -> dict(orch, code, runtime_env, kernel, marker, explain)
#   code       : runtime status the host reports in sim (orch_error_code or sched_error_code)
#   runtime_env: CallConfig.runtime_env overrides that pin the offending resource small
#   kernel     : AIV kernel (rel to kernels/) for the async cases, else None
#   marker     : substring of the validate_runtime_impl host-log line proving the
#                device error class reached the host (the assertion that holds on
#                both sim and onboard, even when onboard masks the code as 507xxx)
#   explain    : SIMPLER_ERROR_* name the "error detail:" annotation line must carry, so
#                the log says what the code *means* and not just what it is. Every reachable
#                code is checked here; the unreachable ones (10/11/103) are covered by the
#                cpput over the name table itself.
CASES = {
    "scope_deadlock": dict(
        orch="scope_deadlock_orch.cpp",
        code=1,
        runtime_env={"ring_task_window": 4},
        kernel=None,
        marker="orch_error_code=1",
        explain="SCOPE_DEADLOCK",
    ),
    "heap_ring_deadlock": dict(
        orch="heap_ring_deadlock_orch.cpp",
        code=2,
        runtime_env={"ring_heap": 1024},
        kernel=None,
        marker="orch_error_code=2",
        explain="HEAP_RING_DEADLOCK",
    ),
    "flow_control_deadlock": dict(
        orch="flow_control_deadlock_orch.cpp",
        code=3,
        runtime_env={"ring_task_window": 4},
        kernel=None,
        marker="orch_error_code=3",
        explain="FLOW_CONTROL_DEADLOCK",
    ),
    "dep_pool_overflow": dict(
        orch="dep_pool_overflow_orch.cpp",
        code=4,
        runtime_env={"ring_dep_pool": 4},
        kernel=None,
        marker="orch_error_code=4",
        explain="FANIN_CAPACITY_EXCEEDED",
    ),
    "invalid_args": dict(
        orch="invalid_args_orch.cpp",
        code=5,
        runtime_env={},
        kernel=None,
        marker="orch_error_code=5",
        explain="INVALID_ARGS",
    ),
    "require_sync_start_invalid": dict(
        orch="require_sync_start_orch.cpp",
        code=7,
        runtime_env={},
        kernel="aiv/kernel_noop.cpp",
        marker="orch_error_code=7",
        explain="REQUIRE_SYNC_START_INVALID",
    ),
    "aicore_hang": dict(
        orch="aicore_hang_orch.cpp",
        code=100,
        runtime_env={},
        kernel="aic/kernel_hang.cpp",
        kernel_core="aic",
        onboard_only=True,  # a while(true) kernel would hang the simulator
        # Scheduler watchdog (2 s) classifies the running stall before STARS (3 s)
        # and host stream sync (4 s) reap the device. Mirrors aicore_op_timeout.
        env={
            "SIMPLER_SCHEDULER_TIMEOUT_MS": 2000,
            "SIMPLER_OP_EXECUTE_TIMEOUT_US": 3000000,
            "SIMPLER_STREAM_SYNC_TIMEOUT_MS": 4000,
        },
        marker="sub_class=S1",
        explain="SCHEDULER_TIMEOUT",
    ),
    "tensor_wait_timeout": dict(
        orch="tensor_wait_timeout_orch.cpp",
        code=8,
        runtime_env={},
        kernel="aic/kernel_hang.cpp",
        kernel_core="aic",
        onboard_only=True,  # a while(true) kernel would hang the simulator
        # The data-wait timeout is 15 s on both arches now that it is frequency-
        # scaled (TENSOR_DATA_TIMEOUT_MS, #1189) -- before that it was 15 s on
        # a5 but 300 s on a2a3, so this case used to be a5-only. Raise every other
        # watchdog above 15 s so the tensor-data wait wins the race and latches
        # code 8 before they reap the hung core.
        env={
            "SIMPLER_SCHEDULER_TIMEOUT_MS": 30000,
            "SIMPLER_OP_EXECUTE_TIMEOUT_US": 30000000,
            "SIMPLER_STREAM_SYNC_TIMEOUT_MS": 40000,
        },
        marker="orch_error_code=8",
        explain="TENSOR_WAIT_TIMEOUT",
    ),
    "async_completion_invalid": dict(
        orch="async_error_orch.cpp",
        code=101,
        runtime_env={},
        kernel="aiv/kernel_async_completion_invalid.cpp",
        marker="sched_error_code=101",
        explain="ASYNC_COMPLETION_INVALID",
    ),
    "async_wait_overflow": dict(
        orch="async_error_orch.cpp",
        code=102,
        runtime_env={},
        kernel="aiv/kernel_async_wait_overflow.cpp",
        marker="sched_error_code=102",
        explain="ASYNC_WAIT_OVERFLOW",
    ),
    "explicit_fatal": dict(
        orch="explicit_fatal_orch.cpp",
        code=9,
        runtime_env={},
        kernel=None,
        marker="orch_error_code=9",
        explain="EXPLICIT_ORCH_FATAL",
    ),
    # Error codes still without an e2e case, and why. Device-side cpput keeps the
    # structural ones covered; the sub-classes have unit coverage of the classifier.
    #
    # -- Structurally pre-empted (a different code always fires first) --
    # * ASYNC_REGISTRATION_FAILED (103): the AICore-side register_completion_
    #   condition caps at MAX_COMPLETIONS_PER_TASK (64) and latches ASYNC_WAIT_
    #   OVERFLOW (102) on the 65th condition — so it never emits enough CONDITION
    #   messages for the scheduler-side count check (> 64 -> 103) to fire. 103 is
    #   only reachable by corrupting the slab past the cap (UB; segfaults a5) or by
    #   an internally-malformed message kind — the scheduler-side guard that the
    #   AICore cap (102) always pre-empts.
    # * SCOPE_TASKS_OVERFLOW (10): scope_tasks_cap is the in-flight slot budget (sum
    #   of the per-ring windows, since #1188), but each ring physically holds only
    #   window-1 tasks (the ring's full/empty distinction), so all rings together
    #   hold at most sum(window-1) = cap - CHIP_MAX_RING_DEPTH, strictly below the
    #   cap. scope_tasks therefore tops out below its own cap: the rings fill first
    #   and latch SCOPE_DEADLOCK (1, single scope) or FLOW_CONTROL_DEADLOCK (3,
    #   same-ring nesting). Verified: no (depth, tasks/level, window) combo reaches
    #   code 10 -- they all latch 1 or 3. Shrinking the window shrinks the cap and
    #   the ring capacity together, so the constant gap holds.
    #
    # -- Design-unreachable from the public API (defensive classifier branches) --
    # * SCHEDULER_TIMEOUT sub-classes S3/S4/S5/UNKNOWN: the pure classifier is unit-
    #   tested for all of them (classify_stall_detail priority table in
    #   tests/ut/cpp/.../test_shared_memory.cpp), but the live states cannot be
    #   produced stably through the public API after Orch-side wiring. S3 used to
    #   be reachable by under-sizing the fanout dep_pool for a ready dummy consumer,
    #   but completed-producer fanins now bypass dep_pool entirely, while live
    #   producers block in Orch-side prewire and either recover or latch the
    #   orchestrator dep_pool error before the scheduler has an orch-done total to
    #   classify as ready-but-idle. S4 (dep-deadlock, pure WAIT) needs a dependency
    #   that never resolves with nothing running/ready; set_dependencies only
    #   references already-submitted tasks, so cycles are inexpressible, and a stuck
    #   producer is RUNNING (-> S1), never pure WAIT. S5 (orch-starvation) needs
    #   completed < total with empty rings, but total_tasks_ is the count of
    #   *submitted* tasks (sum of current_task_index), so once they retire completed
    #   == total and the watchdog never fires — a while-loop in the orch does not
    #   help. UNKNOWN is a bookkeeping-invariant violation (corruption). These are
    #   kept as defensive labels: if a future bug ever produces the state, the
    #   classifier names it correctly instead of mislabeling. S1 remains the
    #   reproducible scheduler-timeout e2e case (aicore_hang_orch.cpp).
    #
    # -- Reachable only by exhausting a fixed compile-time cap --
    # * TENSORMAP_OVERFLOW (11): the tensormap entry pool (CHIP_TENSORMAP_POOL_SIZE
    #   == 65536) is compile-time and not shrinkable via runtime_env, so wedging it
    #   needs a 65536-entry flood plus a stalled producer holding entries. Its latch
    #   is also orchestrator-level (a reclaim deadlock), so even a device UT would be
    #   a runtime-integration test, not a ChipTensorMap unit test.
    #
    # TENSOR_WAIT_TIMEOUT (8) IS covered (onboard, both arches) — see the
    # tensor_wait_timeout case.
}

_PER_PR_SIM_CASES = {"async_completion_invalid", "explicit_fatal"}
assert _PER_PR_SIM_CASES <= CASES.keys(), f"unknown Per-PR Sim cases: {_PER_PR_SIM_CASES - CASES.keys()}"
_SIM_CASE_PARAMS = [
    case_name
    if case_name in _PER_PR_SIM_CASES
    else pytest.param(case_name, marks=pytest.mark.manual(["a2a3sim", "a5sim"]), id=case_name)
    for case_name in CASES
]


def _assert_annotated(log: str, case: dict) -> None:
    """The failure line must be followed by what the code *means*, not just the number.

    The machine-readable "simpler runtime failed: ..." line is deliberately left alone -- it is
    what ``marker`` matches and what conftest's device-poison regex reads -- so the meaning is
    carried on its own "error detail:" line, emitted by the LOG_RUNTIME_FAILURE macro
    (src/common/runtime_status/error_log.h).
    """
    assert "error detail:" in log, "host log carries the raw code but no explanation of it"
    assert case["explain"] in log, f"annotation does not name the code as '{case['explain']}'"
    assert "error hint:" in log, "annotation names the code but says nothing about what to do"


def _build_chip_callable(platform: str, case: dict) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    # AIV-kernel compilation needs the PTO-ISA headers (pto/pto-inst.hpp) on every
    # platform, sim included; onboard additionally needs them for orchestration.
    pto_isa_root = None
    if case["kernel"] is not None or not platform.endswith("sim"):
        pto_isa_root = ensure_pto_isa_root()
        # Path is passed explicitly to KernelCompiler; do not export (#1403).

    children = []
    if case["kernel"] is not None:
        core_type = case.get("kernel_core", "aiv")
        includes = list(kc.get_orchestration_include_dirs(RUNTIME)) + [str(kc.project_root / "src" / "common")]
        kernel = kc.compile_incore(
            source_path=os.path.join(KERNELS, case["kernel"]),
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        # The hang kernel binds no args (it just spins); the async kernels write output 0.
        if core_type == "aic":
            core_callable = CoreCallable.build(signature=[], binary=kernel)
        else:
            core_callable = CoreCallable.build(signature=[ArgDirection.OUT], binary=kernel)
        children.append((0, core_callable))

    orch_bytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=os.path.join(ORCH_DIR, case["orch"]))
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=children,
    )


def _make_worker(platform: str, device_id: int, case_name: str, monkeypatch):
    case = CASES[case_name]
    # Per-case timeout-chain overrides (which watchdog must fire, and when).
    for key, value in case.get("env", {}).items():
        monkeypatch.setenv(key, str(value))
    chip_callable = _build_chip_callable(platform, case)
    worker = Worker(level=2, platform=platform, runtime=RUNTIME, device_id=device_id)
    handle = worker.register(chip_callable)
    worker.init()
    config = CallConfig()
    config.aicpu_thread_num = 2
    for key, value in case["runtime_env"].items():
        setattr(config.runtime_env, key, value)
    return worker, handle, config


@pytest.mark.platforms(["a5sim", "a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("case_name", _SIM_CASE_PARAMS)
def test_fatal_code_surfaces_on_sim(st_platform, st_device_ids, case_name, monkeypatch, capfd):
    """sim: the runtime status (``code -N``) reaches the host directly."""
    configure_logging("error")
    case = CASES[case_name]
    if case.get("onboard_only"):
        pytest.skip("hang kernel would spin the simulator forever (no STARS watchdog on sim)")
    dropped_before = _host_log_dropped_records()
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), case_name, monkeypatch)
    try:
        with pytest.raises(RuntimeError, match=rf"(run_runtime|run) failed with code -{case['code']}\b"):
            worker.run(handle, None, config)
        log = _wait_for_host_log(
            capfd,
            (case["marker"], "error detail:", case["explain"], "error hint:"),
            dropped_before,
        )
        _assert_annotated(log, case)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(90)  # hang cases hold a core ~15-20 s; bound a wedge
@pytest.mark.parametrize("case_name", list(CASES))
def test_device_error_class_reaches_host_log(st_platform, st_device_ids, case_name, monkeypatch, capfd):
    """onboard: the watchdog may mask the code as 507xxx, but the device class still reaches the host log."""
    configure_logging("error")
    case = CASES[case_name]
    dropped_before = _host_log_dropped_records()
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), case_name, monkeypatch)
    try:
        # On hardware the op-execute / stream-sync watchdog can surface a generic
        # CANN 507xxx instead of the runtime's own -N; we only require that the
        # run fails. The point of the test is the device-classified host LOG.
        with pytest.raises(RuntimeError):
            worker.run(handle, None, config)
        log = _wait_for_host_log(
            capfd,
            (case["marker"], "error detail:", case["explain"], "error hint:"),
            dropped_before,
        )
        _assert_annotated(log, case)
    finally:
        worker.close()


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(90)
def test_a5_explicit_fatal_quiescence_probe(st_platform, st_device_ids, monkeypatch):
    """Manual Experiment 1: emit INFO-level PTO shutdown markers for one cooperative fatal run."""
    configure_logging("info")
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), "explicit_fatal", monkeypatch)
    try:
        with pytest.raises(RuntimeError):
            worker.run(handle, None, config)
        _flush_host_log(2000)
    finally:
        worker.close()
        _flush_host_log(2000)


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(90)
def test_a5_explicit_fatal_same_stream_reuse_probe(st_platform, st_device_ids, monkeypatch, capfd):
    """Explicit recovery preserves one Worker, both Contexts, and execution streams."""
    configure_logging("info")

    healthy_callable = _build_chip_callable(
        st_platform,
        {
            "orch": "async_error_orch.cpp",
            "kernel": "aiv/kernel_noop.cpp",
        },
    )
    fatal_callable = _build_chip_callable(st_platform, CASES["explicit_fatal"])

    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    healthy_handle = worker.register(healthy_callable)
    fatal_handle = worker.register(fatal_callable)
    worker.init()

    config = CallConfig()
    config.aicpu_thread_num = 2
    try:
        print("[PTO_RECOVERY_TEST] RUN1_HEALTHY_BEGIN", flush=True)
        worker.run(healthy_handle, None, config)
        print("[PTO_RECOVERY_TEST] RUN1_HEALTHY_END", flush=True)
        native_worker = worker._chip_worker
        before = native_worker.execution_recovery_state()

        dropped_before = _host_log_dropped_records()
        print("[PTO_RECOVERY_TEST] RUN2_EXPLICIT_FATAL_BEGIN", flush=True)
        with pytest.raises(RuntimeError):
            worker.run(fatal_handle, None, config)
        fatal_log = _wait_for_host_log(
            capfd,
            (
                "[PTO_RECOVERY] EXECUTION_RECOVERY_REQUIRED",
                CASES["explicit_fatal"]["marker"],
                CASES["explicit_fatal"]["explain"],
            ),
            dropped_before,
        )
        print("[PTO_RECOVERY_TEST] RUN2_EXPLICIT_FATAL_END", flush=True)

        with pytest.raises(RuntimeError, match="poisoned"):
            worker.run(healthy_handle, None, config)
        recovered = _recover_retired_generation(native_worker)
        _assert_execution_identities(before, recovered)
        print("[PTO_RECOVERY_TEST] RUN3_HEALTHY_SAME_STREAM_BEGIN", flush=True)
        worker.run(healthy_handle, None, config)
        print("[PTO_RECOVERY_TEST] RUN3_HEALTHY_SAME_STREAM_END", flush=True)

        _flush_host_log(2000)
        captured = capfd.readouterr()
        assert "507014" not in fatal_log + captured.out + captured.err
    finally:
        worker.close()
        _flush_host_log(2000)
    with capfd.disabled():
        print("[PTO_RECOVERY_TEST] SAME_CONTEXT_STREAM_RECOVERY_PASS", flush=True)


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(120)
def test_a5_persistent_failure_requires_external_dispatch(st_platform, st_device_ids, monkeypatch, capfd):
    """L2 rearms explicitly; a second external dispatch reports an independent fault."""
    configure_logging("info")
    fatal_callable = _build_chip_callable(st_platform, CASES["explicit_fatal"])
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    fatal_handle = worker.register(fatal_callable)
    worker.init()
    config = CallConfig()
    config.aicpu_thread_num = 2
    try:
        with pytest.raises(RuntimeError):
            worker.run(fatal_handle, None, config)
        native_worker = worker._chip_worker
        first = _recover_retired_generation(native_worker)
        with pytest.raises(RuntimeError):
            worker.run(fatal_handle, None, config)
        second = native_worker.execution_recovery_state()
        assert second.status == ExecutionRecoveryStatus.RECOVERY_REQUIRED
        assert second.failed_run_epoch > first.failed_run_epoch
        assert second.original_rc == 507018
        assert second.quiesced == 1
        stale = native_worker.recover_execution(first.failed_run_epoch)
        assert stale.status == ExecutionRecoveryStatus.STALE_FAILURE
        with pytest.raises(RuntimeError, match="poisoned"):
            worker.run(fatal_handle, None, config)
        print("[PTO_RECOVERY_TEST] SECOND_EXTERNAL_DISPATCH_FAILED", flush=True)
        # The caller ends this episode. close() invokes terminal fatal cleanup.
    finally:
        worker.close()
        _flush_host_log(2000)
    captured = capfd.readouterr()
    log = captured.out + captured.err
    assert "SECOND_EXTERNAL_DISPATCH_FAILED" in log
    assert "force_reset_device: aclrtResetDeviceForce" in log
    assert "507014" not in log
    assert "SOFT_RETRY" not in log
    with capfd.disabled():
        print("[PTO_RECOVERY_TEST] PERSISTENT_EXTERNAL_DISPATCH_PASS", flush=True)


def _assert_execution_identities(before, after):
    for field in ("communication_context", "execution_context", "aicpu_stream", "aicore_stream"):
        assert getattr(before, field) != 0
        assert getattr(before, field) == getattr(after, field), field
    assert after.communication_context != after.execution_context


def _recover_retired_generation(native_worker):
    failed = native_worker.execution_recovery_state()
    assert failed.status == ExecutionRecoveryStatus.RECOVERY_REQUIRED
    assert failed.failed_run_epoch != 0
    assert failed.original_rc == 507018
    assert failed.quiesced == 1
    refused_scope = native_worker.recover_execution(
        failed.failed_run_epoch, ExecutionRecoveryScope.COMMUNICATION_AFFECTED
    )
    assert refused_scope.status == ExecutionRecoveryStatus.UNSUPPORTED
    stale = native_worker.recover_execution(failed.failed_run_epoch + 1)
    assert stale.status == ExecutionRecoveryStatus.STALE_FAILURE
    recovered = native_worker.recover_execution(failed.failed_run_epoch)
    assert recovered.status == ExecutionRecoveryStatus.RECOVERED
    assert recovered.original_rc == failed.original_rc
    assert recovered.recovery_rc == 0
    assert recovered.communication_preserved == 1
    _assert_execution_identities(failed, recovered)
    duplicate = native_worker.recover_execution(failed.failed_run_epoch)
    assert duplicate.status == ExecutionRecoveryStatus.ALREADY_RECOVERED
    _assert_execution_identities(recovered, duplicate)
    print("[PTO_RECOVERY_TEST] EXPLICIT_RECOVERY_CONTRACT_PASS", flush=True)
    return recovered


def _exp3_plain_snapshot(native_worker, comm_handle: int, allocation_id: int) -> dict[str, int]:
    raw = native_worker._impl._experiment3_comm_recovery_probe(int(comm_handle), int(allocation_id))
    return {str(key): int(value) for key, value in dict(raw).items()}


def _exp3_rank_process(device_id: int, rank: int, rootinfo_path: str, result_queue) -> None:
    """One isolated rank: keep one dynamic CommDomain live across an execution-generation fault."""
    worker = None
    comm_handle = None
    allocation_live = False
    try:
        configure_logging("info")
        healthy_callable = _build_chip_callable(
            "a5",
            {
                "orch": "async_error_orch.cpp",
                "kernel": "aiv/kernel_noop.cpp",
            },
        )
        fatal_callable = _build_chip_callable("a5", CASES["explicit_fatal"])

        worker = Worker(level=2, platform="a5", runtime=RUNTIME, device_id=int(device_id))
        healthy_handle = worker.register(healthy_callable)
        fatal_handle = worker.register(fatal_callable)
        worker.init()
        native_worker = worker._chip_worker
        if native_worker is None:
            raise RuntimeError("Experiment 3 lost the L2 ChipWorker")

        comm_handle = native_worker.comm_init(rank, 2, rootinfo_path)
        allocation_id = 0xE3000001
        commit_flag = ctypes.c_uint64(0)
        native_worker._impl.comm_alloc_domain_windows(
            int(comm_handle), allocation_id, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(commit_flag)
        )
        if commit_flag.value != 1:
            raise RuntimeError("Experiment 3 dynamic domain did not publish its commit flag")
        allocation_live = True

        config = CallConfig()
        config.aicpu_thread_num = 2
        worker.run(healthy_handle, None, config)
        before = _exp3_plain_snapshot(native_worker, comm_handle, allocation_id)
        execution_before = native_worker.execution_recovery_state()

        try:
            worker.run(fatal_handle, None, config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Experiment 3 explicit-fatal run unexpectedly succeeded")
        after_fault = _exp3_plain_snapshot(native_worker, comm_handle, allocation_id)
        with pytest.raises(RuntimeError, match="poisoned"):
            worker.run(healthy_handle, None, config)
        recovered = _recover_retired_generation(native_worker)
        _assert_execution_identities(execution_before, recovered)
        after_recovery = _exp3_plain_snapshot(native_worker, comm_handle, allocation_id)

        worker.run(healthy_handle, None, config)
        after_retry = _exp3_plain_snapshot(native_worker, comm_handle, allocation_id)
        _assert_execution_identities(recovered, native_worker.execution_recovery_state())

        # Release is intentionally part of the experiment: a dynamic allocation
        # that survives recovery but cannot be collectively reclaimed is not a
        # usable recovery outcome.
        native_worker._impl.comm_release_domain_windows(int(comm_handle), allocation_id, 2, rank)
        allocation_live = False
        after_release = _exp3_plain_snapshot(native_worker, comm_handle, allocation_id)
        native_worker.comm_destroy(int(comm_handle))
        comm_handle = None
        worker.close()
        worker = None

        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "device_id": device_id,
                "before": before,
                "after_fault": after_fault,
                "after_recovery": after_recovery,
                "after_retry": after_retry,
                "after_release": after_release,
            }
        )
    except BaseException as exc:  # noqa: BLE001 -- child must return the first failure to the parent
        result_queue.put({"ok": False, "rank": rank, "device_id": device_id, "error": repr(exc)})
        # Best effort only. A peer may already have failed a collective; avoid
        # turning the diagnostic exception itself into an unbounded cleanup loop.
        try:
            if allocation_live and worker is not None and worker._chip_worker is not None and comm_handle is not None:
                worker._chip_worker._impl.comm_release_domain_windows(int(comm_handle), 0xE3000001, 2, rank)
        except BaseException:
            pass
        try:
            if worker is not None:
                worker.close()
        except BaseException:
            pass
        raise


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(240)
def test_a5_explicit_fatal_preserves_comm_domain(st_device_ids, tmp_path):
    """Communication ownership and empty queues survive explicit execution recovery."""
    assert len(st_device_ids) == 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    rootinfo_path = str(tmp_path / "exp3_hccl_rootinfo.bin")
    processes = [
        ctx.Process(
            target=_exp3_rank_process,
            args=(int(st_device_ids[rank]), rank, rootinfo_path, result_queue),
            name=f"pto-exp3-rank-{rank}",
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    join_deadline = time.monotonic() + 190.0
    for process in processes:
        process.join(timeout=max(0.0, join_deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    results = []
    while len(results) < 2:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            break
    assert len(results) == 2, f"Experiment 3 returned only {len(results)} rank result(s)"
    results.sort(key=lambda item: item["rank"])
    assert all(item["ok"] for item in results), results
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]

    stable_identity_fields = (
        "workspace_kind",
        "comm_handle_identity",
        "hccl_comm_identity",
        "comm_stream_identity",
        "local_window_base",
        "device_ctx",
        "workspace_addr",
        "workspace_size",
        "binding_signature",
        "queue_signature",
    )
    for result in results:
        snapshots = [result["before"], result["after_fault"], result["after_recovery"], result["after_retry"]]
        for snapshot in snapshots:
            assert snapshot["version"] == 1
            assert snapshot["workspace_kind"] in (1, 2), snapshot
            assert snapshot["domain_allocation_count"] == 1, snapshot
            assert snapshot["allocation_present"] == 1, snapshot
            assert snapshot["peer_window_count"] == 1, snapshot
            assert snapshot["active_queue_count"] == 48, snapshot
            assert snapshot["queue_read_errors"] == 0, snapshot
            assert snapshot["all_queues_empty"] == 1, snapshot
            assert snapshot["hccl_comm_identity"] != 0, snapshot
            assert snapshot["comm_stream_identity"] != 0, snapshot
            assert snapshot["workspace_addr"] != 0, snapshot
        for field in stable_identity_fields:
            assert all(snapshot[field] == snapshots[0][field] for snapshot in snapshots), (
                result["rank"],
                field,
                [snapshot[field] for snapshot in snapshots],
            )
        released = result["after_release"]
        assert released["domain_allocation_count"] == 0, released
        assert released["allocation_present"] == 0, released
    print("[PTO_RECOVERY_TEST] COMMUNICATION_PRESERVATION_PASS", flush=True)


# The orchestrator fatals before any task completes, so this case leaves the
# swimlane collector empty and only the two collectors asserted below hold
# anything. That is deliberate: the point is the teardown, not the volume.
_DFX_ON_FAILURE_CASE = "scope_deadlock"


@pytest.mark.platforms(["a5sim", "a2a3sim", "a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_failed_run_exports_its_diagnostics(st_platform, st_device_ids, monkeypatch, tmp_path):
    """A run that fails still exports the collectors it armed."""
    configure_logging("error")
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), _DFX_ON_FAILURE_CASE, monkeypatch)
    out = tmp_path / "fatal_dfx"
    config.output_prefix = str(out)
    # chip_swimlane is enabled but not asserted: it exports nothing when a run
    # holds no task records, so its artifact cannot witness the teardown. It is
    # on so the failure path drives its export too.
    config.enable_chip_swimlane = 1
    config.enable_scope_stats = True
    config.enable_dep_gen = True
    try:
        with pytest.raises(RuntimeError):
            worker.run(handle, None, config)
    finally:
        worker.close()

    produced = sorted(str(p.relative_to(out)) for p in out.rglob("*")) if out.is_dir() else []
    # scope_stats writes whenever its collector initialized, so it carries no
    # record-count precondition, and it is written last in the teardown: its
    # absence means the teardown did not run, or did not run to the end.
    assert (out / "scope_stats" / "scope_stats.jsonl").is_file(), (
        f"the failed run exported no scope_stats; {out} holds {produced}"
    )
    # The dep_gen emit follows the teardown and has its own completeness gate,
    # so it needs its own assertion. The device flushes its dep_gen buffers
    # during emergency shutdown, which is what lets the gate pass here.
    assert (out / "deps.json").is_file(), f"the failed run emitted no deps.json; {out} holds {produced}"
