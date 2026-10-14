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
import json
import multiprocessing as mp
import os
import queue
import time
import traceback

import pytest
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DomainRecoveryImpact,
    DomainRecoveryPhase,
    DomainRecoveryRequest,
    DomainRecoveryStatus,
    ExecutionRecoveryScope,
    ExecutionRecoveryStatus,
    RecoveryArgBinding,
    RecoveryArgKind,
    RecoveryBindingPlan,
    RecoveryDomainBinding,
    TaskDomainBinding,
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
def test_a5_explicit_fatal_same_stream_reuse_probe(st_platform, st_device_ids, capfd):
    """Explicit recovery preserves one Worker, both Contexts, and execution streams."""
    _check_a5_same_stream_recovery(st_platform, st_device_ids, capfd, "explicit_fatal")
    with capfd.disabled():
        print("[PTO_RECOVERY_TEST] SAME_CONTEXT_STREAM_RECOVERY_PASS", flush=True)


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(90)
def test_a5_kernel_completion_error_recovery(st_platform, st_device_ids, capfd):
    """An AIV-generated completion error is reported by the scheduler, then explicitly recovered."""
    _check_a5_same_stream_recovery(st_platform, st_device_ids, capfd, "async_completion_invalid")
    with capfd.disabled():
        print(
            "[PTO_RECOVERY_TEST] KERNEL_COMPLETION_ERROR_RECOVERY_PASS sched_error_code=101 ASYNC_COMPLETION_INVALID",
            flush=True,
        )


def _check_a5_same_stream_recovery(st_platform, st_device_ids, capfd, failure_case):
    configure_logging("info")

    healthy_callable = _build_chip_callable(
        st_platform,
        {
            "orch": "async_error_orch.cpp",
            "kernel": "aiv/kernel_noop.cpp",
        },
    )
    fatal_callable = _build_chip_callable(st_platform, CASES[failure_case])

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
        print("[PTO_RECOVERY_TEST] RUN2_FAULT_BEGIN", flush=True)
        with pytest.raises(RuntimeError):
            worker.run(fatal_handle, None, config)
        fatal_log = _wait_for_host_log(
            capfd,
            (
                "[PTO_RECOVERY] EXECUTION_RECOVERY_REQUIRED",
                CASES[failure_case]["marker"],
                CASES[failure_case]["explain"],
            ),
            dropped_before,
        )
        print("[PTO_RECOVERY_TEST] RUN2_FAULT_END", flush=True)

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


def _comm_snapshot(native_worker, comm_handle: int, allocation_id: int) -> dict[str, int]:
    raw = native_worker.comm_recovery_snapshot(int(comm_handle), int(allocation_id))
    return {str(key): int(value) for key, value in dict(raw).items()}


def _recover_rank_domain(native_worker, comm_handle, allocation_id, before):
    """Check execution-only preservation before execution admission reopens."""
    failed = native_worker.execution_recovery_state()
    preserve_request = DomainRecoveryRequest()
    preserve_request.recovery_id = 1
    preserve_request.expected_generation = 1
    preserve_request.failed_run_epoch = failed.failed_run_epoch
    preserve_request.original_rc = failed.original_rc
    preserved = native_worker.recover_comm_domain(comm_handle, allocation_id, preserve_request)
    assert preserved.status == DomainRecoveryStatus.PRESERVED
    assert preserved.generation == 1 and preserved.domain_preserved == 1
    assert preserved.rebind_required == 0
    assert preserved.device_ctx == before["device_ctx"]
    assert preserved.local_window_base == before["local_window_base"]
    assert preserved.workspace == before["workspace_addr"]
    duplicate = native_worker.recover_comm_domain(comm_handle, allocation_id, preserve_request)
    assert duplicate.status == DomainRecoveryStatus.ALREADY_RECOVERED


def _comm_recovery_rank_process(device_id: int, rank: int, rootinfo_path: str, result_queue) -> None:
    """One isolated rank: keep one dynamic CommDomain live across an execution-generation fault."""
    worker = None
    comm_handle = None
    allocation_live = False
    # L3's first dynamic allocation uses ID zero.
    allocation_id = 0
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
            raise RuntimeError("Communication recovery lost the L2 ChipWorker")

        comm_handle = native_worker.comm_init(rank, 2, rootinfo_path)
        commit_flag = ctypes.c_uint64(0)
        native_worker._impl.comm_alloc_domain_windows(
            int(comm_handle), allocation_id, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(commit_flag)
        )
        if commit_flag.value != 1:
            raise RuntimeError("Communication recovery dynamic domain did not publish its commit flag")
        allocation_live = True

        config = CallConfig()
        config.aicpu_thread_num = 2
        worker.run(healthy_handle, None, config)
        before = _comm_snapshot(native_worker, comm_handle, allocation_id)
        execution_before = native_worker.execution_recovery_state()

        try:
            worker.run(fatal_handle, None, config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Communication recovery explicit-fatal run unexpectedly succeeded")
        after_fault = _comm_snapshot(native_worker, comm_handle, allocation_id)
        with pytest.raises(RuntimeError, match="poisoned"):
            worker.run(healthy_handle, None, config)
        _recover_rank_domain(native_worker, comm_handle, allocation_id, before)
        recovered = _recover_retired_generation(native_worker)
        _assert_execution_identities(execution_before, recovered)
        after_recovery = _comm_snapshot(native_worker, comm_handle, allocation_id)

        worker.run(healthy_handle, None, config)
        after_retry = _comm_snapshot(native_worker, comm_handle, allocation_id)
        _assert_execution_identities(recovered, native_worker.execution_recovery_state())

        # Release is part of the preservation contract: a dynamic allocation
        # that survives recovery but cannot be collectively reclaimed is not a
        # usable recovery outcome.
        native_worker._impl.comm_release_domain_windows(int(comm_handle), allocation_id, 2, rank)
        allocation_live = False
        after_release = _comm_snapshot(native_worker, comm_handle, allocation_id)
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
                worker._chip_worker._impl.comm_release_domain_windows(int(comm_handle), allocation_id, 2, rank)
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
    """Communication ownership survives explicit execution recovery; SDMA drain is not inferred."""
    _check_comm_recovery(st_device_ids, tmp_path)


def _check_comm_recovery(st_device_ids, tmp_path):
    assert len(st_device_ids) == 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    rootinfo_path = str(tmp_path / "comm_recovery_hccl_rootinfo.bin")
    processes = [
        ctx.Process(
            target=_comm_recovery_rank_process,
            args=(int(st_device_ids[rank]), rank, rootinfo_path, result_queue),
            name=f"pto-comm-recovery-rank-{rank}",
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
    assert len(results) == 2, f"Communication recovery returned only {len(results)} rank result(s)"
    results.sort(key=lambda item: item["rank"])
    assert all(item["ok"] for item in results), results
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]

    # Queue progress may advance or reset; binding_signature covers queue resources.
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
    )
    for result in results:
        snapshots = [result["before"], result["after_fault"], result["after_recovery"], result["after_retry"]]
        for snapshot in snapshots:
            assert snapshot["version"] == 1
            assert snapshot["workspace_kind"] in (1, 2), snapshot
            assert snapshot["domain_allocation_count"] == 1, snapshot
            assert snapshot["allocation_present"] == 1, snapshot
            assert snapshot["peer_window_count"] == 1, snapshot
            expected_queues = 48 if snapshot["workspace_kind"] == 1 else 1
            assert snapshot["active_queue_count"] == expected_queues, snapshot
            assert snapshot["queue_read_errors"] == 0, snapshot
            # The SDMA snapshot copies workspace descriptions, not live SQ
            # registers. It cannot certify transport quiescence.
            if snapshot["workspace_kind"] == 2:
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


def _build_sdma_ordering_callable():
    compiler = KernelCompiler(platform="a5")
    root = ensure_pto_isa_root()
    includes = compiler.get_orchestration_include_dirs(RUNTIME)
    children = []
    for func_id, filename, signature in (
        (0, "kernel_sdma_transfer.cpp", [ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN]),
        (1, "kernel_sdma_order.cpp", [ArgDirection.IN, ArgDirection.OUT] + [ArgDirection.IN] * 3),
    ):
        binary = compiler.compile_incore(
            source_path=os.path.join(KERNELS, "aiv", filename),
            core_type="aiv",
            pto_isa_root=root,
            extra_include_dirs=includes,
        )
        children.append((func_id, CoreCallable.build(signature=signature, binary=extract_text_section(binary))))
    orch = compiler.compile_orchestration(
        runtime_name=RUNTIME, source_path=os.path.join(ORCH_DIR, "sdma_ordering_orch.cpp")
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT] + [ArgDirection.IN] * 4,
        func_name="aicpu_orchestration_entry",
        binary=orch,
        children=children,
    )


def _sdma_transaction_log(rank, phase, **fields):
    print(
        "[PTO_RECOVERY_TEST] SDMA_TRANSACTION " + json.dumps({"rank": rank, "phase": phase, **fields}, sort_keys=True),
        flush=True,
    )


def _sdma_domain_recovery_rank(device_id, rank, rootinfo, barrier, outcomes, result_queue, use_binding_plan=False):  # noqa: PLR0915
    """Formal local transaction and ordering regression; no transport-fault injection."""
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    worker = None

    def write(address, value):
        native._impl.copy_to(address, ctypes.addressof(value), ctypes.sizeof(value))

    def read_words(address, count=8):
        value = (ctypes.c_uint32 * count)()
        native._impl.copy_from(ctypes.addressof(value), address, ctypes.sizeof(value))
        return list(value)

    def check_data(address, expected):
        value = (ctypes.c_float * 16384)()
        native._impl.copy_from(ctypes.addressof(value), address, ctypes.sizeof(value))
        assert all(item == expected for item in value), (rank, list(value)[:8], expected)

    def args_for(output, marker, previous, next_value):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(base, (16384,), DataType.FLOAT32, True))
        args.add_tensor(ChipTensor.make(output, (16384,), DataType.FLOAT32, True))
        args.add_tensor(ChipTensor.make(marker, (8,), DataType.UINT32, True))
        for value in (device_ctx, base + 0x40000, previous, next_value):
            args.add_scalar(value)
        return args

    def submit(callable_id, args, slot, generation, dispatch):
        return native._impl._submit_chip_run_materialized(
            callable_id, args, config, slot, generation, dispatch, dispatch, 0, 0, True
        )

    def all_members_succeeded(local_ok):
        outcomes[rank] = int(local_ok)
        barrier.wait(timeout=45)
        agreed = all(outcomes)
        barrier.wait(timeout=45)  # no member overwrites its vote before all read
        return agreed

    def abort_transaction(request, local_rc):
        outcomes[rank] = int(local_rc)
        barrier.wait(timeout=45)
        peer_rc = next((int(rc) for rc in outcomes if rc != 0), -1)
        barrier.wait(timeout=45)
        request.phase = DomainRecoveryPhase.ABORT
        request.collective_rc = peer_rc
        aborted = native.recover_comm_domain(comm, 0, request)
        _sdma_transaction_log(rank, "ABORT", status=str(aborted.status), recovery_rc=aborted.recovery_rc)

    def snapshot(phase):
        current = _comm_snapshot(native, comm, 0)
        assert current["active_queue_count"] == 48 and current["queue_read_errors"] == 0, current
        for field in stable_fields:
            assert current[field] == before[field], (rank, field, before[field], current[field])
        _sdma_transaction_log(phase=phase, **current)
        return current

    def publish(report):
        result_queue.put(report)
        result_queue.close()
        result_queue.join_thread()
        _flush_host_log(1000)

    try:
        configure_logging("info")
        transfer = _build_sdma_ordering_callable()
        fatal = _build_chip_callable("a5", CASES["explicit_fatal"])
        worker = Worker(level=2, platform="a5", runtime=RUNTIME, device_id=int(device_id))
        worker.init()
        native = worker._chip_worker
        assert native is not None and native.runtime_slot_count >= 2
        transfer_id = native._resolve_handle(native.register_callable(transfer)).slot_id
        fatal_id = native._resolve_handle(native.register_callable(fatal)).slot_id
        comm = native.comm_init(rank, 2, rootinfo)
        commit = ctypes.c_uint64(0)
        native._impl.comm_alloc_domain_windows(int(comm), 0, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(commit))
        assert commit.value == 1
        before = _comm_snapshot(native, comm, 0)
        assert before["workspace_kind"] == 1, "requires the SDMA provider"
        # Queue progress may advance or reset; binding_signature covers queue resources.
        stable_fields = (
            "comm_handle_identity",
            "hccl_comm_identity",
            "comm_stream_identity",
            "workspace_addr",
            "workspace_size",
            "local_window_base",
            "device_ctx",
            "binding_signature",
        )
        base, device_ctx = before["local_window_base"], before["device_ctx"]
        a_out, b_out = base + 0x10000, base + 0x20000
        a_marker, b_marker = base + 0x30000, base + 0x30020
        config = CallConfig()
        config.aicpu_thread_num = 2
        execution_before = native.execution_recovery_state()
        write(base, (ctypes.c_float * 16384)(*([100.0 + rank] * 16384)))
        write(b_out, (ctypes.c_float * 16384)(*([-77.0] * 16384)))
        barrier.wait(timeout=60)
        warm = submit(transfer_id, args_for(a_out, a_marker, 0, 0), 0, 1, 1)
        assert warm.wait(30)
        check_data(a_out, 101.0 - rank)
        assert read_words(a_marker)[:3] == [1, 0, 1]
        snapshot("BEFORE_FAULT")
        barrier.wait(timeout=30)
        retry_args = args_for(a_out, a_marker, 0, 1)
        failed = submit(fatal_id, ChipStorageTaskArgs(), 0, 2, 2)
        b_args = args_for(b_out, b_marker, 1, 2)
        queued = submit(transfer_id, b_args, 1, 1, 3)
        assert not queued.launched
        with pytest.raises(RuntimeError):
            assert failed.wait(30), "failed A did not become terminal"
        state = native.execution_recovery_state()
        assert state.status == ExecutionRecoveryStatus.RECOVERY_REQUIRED and state.quiesced == 1, (
            "failed generation lacks cooperative-stop proof",
            str(state.status),
            state.failed_run_epoch,
            state.quiesced,
            state.original_rc,
            state.recovery_rc,
        )
        assert state.original_rc == 507018
        assert queued.done() and not queued.launched
        with pytest.raises(RuntimeError):
            queued.activate()
        check_data(b_out, -77.0)
        assert read_words(b_marker) == [0] * 8 and read_words(base + 0x40000, 1) == [0]
        snapshot("AFTER_FAULT")
        assert all_members_succeeded(state.quiesced == 1)

        request = DomainRecoveryRequest()
        request.recovery_id = 1
        request.expected_generation = 1
        request.failed_run_epoch = state.failed_run_epoch
        request.original_rc = state.original_rc
        request.impact = DomainRecoveryImpact.DOMAIN_TRANSPORT
        # These processes own all producers and have retired both A and old B.
        request.collective_quiescence = 1
        prepared = native.recover_comm_domain(comm, 0, request)
        _sdma_transaction_log(
            rank,
            "PREPARE",
            status=str(prepared.status),
            generation=prepared.generation,
            channels=prepared.transport_channel_count,
            stopped=prepared.stopped_channel_count,
            transport_stopped=prepared.transport_stopped,
            recovery_rc=prepared.recovery_rc,
        )
        if not all_members_succeeded(prepared.status == DomainRecoveryStatus.PREPARED):
            abort_transaction(
                request, prepared.recovery_rc or (-1 if prepared.status != DomainRecoveryStatus.PREPARED else 0)
            )
            raise AssertionError("an SDMA PREPARE failed; recovery cannot PASS")
        assert prepared.generation == 1 and prepared.previous_generation == 1
        assert (
            prepared.transport_stopped == 1 and prepared.stopped_channel_count == prepared.transport_channel_count == 48
        )
        assert prepared.device_ctx == device_ctx and prepared.local_window_base == base
        assert prepared.workspace == before["workspace_addr"]
        duplicate = native.recover_comm_domain(comm, 0, request)
        assert duplicate.status == DomainRecoveryStatus.PREPARED
        with pytest.raises(RuntimeError):
            native._impl.comm_release_domain_windows(int(comm), 0, 2, rank)
        with pytest.raises(RuntimeError):
            native._impl.comm_alloc_domain_windows(
                int(comm), 1, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(commit)
            )
        blocked = native.recover_execution(state.failed_run_epoch)
        assert blocked.status == ExecutionRecoveryStatus.BUSY
        with pytest.raises(RuntimeError):
            submit(transfer_id, args_for(a_out, a_marker, 0, 1), 0, 3, 4)
        snapshot("AFTER_PREPARE")

        request.phase = DomainRecoveryPhase.COMMIT
        committed = native.recover_comm_domain(comm, 0, request)
        _sdma_transaction_log(
            rank,
            "COMMIT",
            status=str(committed.status),
            generation=committed.generation,
            recovery_rc=committed.recovery_rc,
        )
        if not all_members_succeeded(committed.status == DomainRecoveryStatus.REARMED):
            abort_transaction(
                request, committed.recovery_rc or (-1 if committed.status != DomainRecoveryStatus.REARMED else 0)
            )
            raise AssertionError("an SDMA COMMIT failed; execution stays fenced")
        assert committed.previous_generation == 1 and committed.generation == 2
        assert committed.domain_preserved == 1 and committed.rebind_required == 1
        duplicate = native.recover_comm_domain(comm, 0, request)
        assert duplicate.status == DomainRecoveryStatus.ALREADY_RECOVERED and duplicate.generation == 2
        request.recovery_id = 2
        assert native.recover_comm_domain(comm, 0, request).status == DomainRecoveryStatus.STALE_GENERATION
        recovered = _recover_retired_generation(native)
        _assert_execution_identities(execution_before, recovered)
        snapshot("AFTER_RECOVERY")
        assert all_members_succeeded(recovered.status == ExecutionRecoveryStatus.RECOVERED)

        new_b_args = b_args
        if use_binding_plan:
            original = RecoveryDomainBinding()
            original.allocation_id = 0
            original.generation = 1
            original.device_ctx = device_ctx
            original.local_window_base = base
            original.window_size = 4 * 1024 * 1024
            original.workspace = before["workspace_addr"]
            plan = RecoveryBindingPlan(committed.recovery_id)
            plan.add_domain(rank, original, committed)

            def bindings_for(output, marker):
                binding = TaskDomainBinding()
                binding.domain = original
                arguments = []
                for kind, index, offset in (
                    (RecoveryArgKind.WINDOW_TENSOR, 0, 0),
                    (RecoveryArgKind.WINDOW_TENSOR, 1, output - base),
                    (RecoveryArgKind.WINDOW_TENSOR, 2, marker - base),
                    (RecoveryArgKind.COMM_CONTEXT, 0, 0),
                    (RecoveryArgKind.WINDOW_ADDRESS, 1, 0x40000),
                ):
                    argument = RecoveryArgBinding()
                    argument.kind, argument.index, argument.offset = kind, index, offset
                    arguments.append(argument)
                binding.arguments = arguments
                return [binding]

            a_bindings = bindings_for(a_out, a_marker)
            b_bindings = bindings_for(b_out, b_marker)
            bound_a = plan.rebind_materialized(retry_args, a_bindings, rank)
            bound_b = plan.rebind_materialized(b_args, b_bindings, rank)
            assert a_bindings[0].domain.generation == b_bindings[0].domain.generation == 1
            assert bound_a.domain_bindings[0].domain.generation == bound_b.domain_bindings[0].domain.generation == 2
            assert bound_a.args.scalar(0) == retry_args.scalar(0) == device_ctx
            assert bound_b.args.scalar(1) == b_args.scalar(1) == base + 0x40000
            assert (
                plan.rebind_materialized(bound_a.args, bound_a.domain_bindings, rank)
                .domain_bindings[0]
                .domain.generation
                == 2
            )
            retry_args, new_b_args = bound_a.args, bound_b.args
            _sdma_transaction_log(
                rank,
                "BINDING_PLAN_APPLIED",
                previous_generation=1,
                generation=2,
                recovery_id=plan.recovery_id,
                addresses_preserved=True,
            )

        write(base, (ctypes.c_float * 16384)(*([200.0 + rank] * 16384)))
        write(a_out, (ctypes.c_float * 16384)(*([-91.0] * 16384)))
        write(a_marker, (ctypes.c_uint32 * 8)())
        barrier.wait(timeout=30)
        retry = submit(transfer_id, retry_args, 0, 3, 4)
        assert retry.wait(30)
        check_data(a_out, 201.0 - rank)
        assert read_words(a_marker)[:3] == [1, 0, 1] and read_words(base + 0x40000, 1) == [1]
        barrier.wait(timeout=30)  # every rank finishes new A before publishing new B's source
        write(base, (ctypes.c_float * 16384)(*([300.0 + rank] * 16384)))
        barrier.wait(timeout=30)
        with pytest.raises(RuntimeError):
            queued.activate()
        assert not queued.launched
        successor = submit(transfer_id, new_b_args, 1, 2, 5)
        assert successor.wait(30)
        check_data(b_out, 301.0 - rank)
        assert read_words(b_marker)[:3] == [1, 1, 1] and read_words(base + 0x40000, 1) == [2]
        _assert_execution_identities(execution_before, native.execution_recovery_state())
        snapshot("AFTER_RETRY")
        barrier.wait(timeout=30)
        native._impl.comm_release_domain_windows(int(comm), 0, 2, rank)
        released = _comm_snapshot(native, comm, 0)
        assert released["allocation_present"] == released["domain_allocation_count"] == 0
        native.comm_destroy(comm)
        worker.close()
        worker = None
        publish({"rank": rank, "ok": True, "generation": committed.generation, "final_order": 2})
    except BaseException as exc:  # noqa: BLE001 -- child failure must reach the parent
        traceback.print_exc()
        publish({"rank": rank, "ok": False, "error": repr(exc)})
        # Failed recovery ends the episode; existing terminal cleanup owns reset.
        if worker is not None:
            try:
                worker.close()
            except BaseException:
                pass
        os._exit(1)


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(300)
def test_a5_sdma_domain_recovery(st_device_ids, tmp_path):
    """Explicit PREPARE/COMMIT, preserved resources, and real SDMA A-before-B replay."""
    _run_sdma_domain_recovery(st_device_ids, tmp_path)


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(300)
def test_a5_sdma_recovery_binding_plan(st_device_ids, tmp_path):
    """Rebind retained materialized args to the committed domain generation, then execute real SDMA."""
    _run_sdma_domain_recovery(st_device_ids, tmp_path, use_binding_plan=True)
    print("[PTO_RECOVERY_TEST] RECOVERY_BINDING_PLAN_SDMA_PASS", flush=True)


def _run_sdma_domain_recovery(st_device_ids, tmp_path, use_binding_plan=False):
    assert len(st_device_ids) == 2
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    outcomes = ctx.Array("i", [0, 0])
    results = ctx.Queue()
    ranks = [
        ctx.Process(
            target=_sdma_domain_recovery_rank,
            args=(int(st_device_ids[r]), r, str(tmp_path / "sdma.bin"), barrier, outcomes, results, use_binding_plan),
        )
        for r in range(2)
    ]
    for rank in ranks:
        rank.start()
    deadline = time.monotonic() + 250
    for rank in ranks:
        rank.join(max(0, deadline - time.monotonic()))
    timed_out = [rank.pid for rank in ranks if rank.is_alive()]
    for rank in ranks:
        if rank.is_alive():
            rank.terminate()
            rank.join(10)
    collected = []
    for _ in ranks:
        try:
            collected.append(results.get(timeout=5))
        except queue.Empty:
            break
    print(
        "[PTO_RECOVERY_TEST] SDMA_TRANSACTION_SUMMARY "
        + json.dumps({"timed_out": timed_out, "results": collected}, sort_keys=True),
        flush=True,
    )
    assert not timed_out, "rank watchdog fired; recovery did not PASS"
    assert len(collected) == 2 and all(item["ok"] for item in collected), collected
    assert all(rank.exitcode == 0 for rank in ranks)
    print("[PTO_RECOVERY_TEST] SDMA_DOMAIN_REARM_COMMIT_PASS", flush=True)
    print("[PTO_RECOVERY_TEST] SDMA_RECOVERY_ORDERING_PASS", flush=True)


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
