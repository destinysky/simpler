# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Child resource transaction gates without an NPU."""

import struct
from copy import deepcopy
from types import SimpleNamespace as Record

import pytest
from _task_interface import (
    DomainRecoveryDetail,
    DomainRecoveryImpact,
    DomainRecoveryPhase,
    DomainRecoveryStatus,
    ExecutionRecoveryStatus,
    _read_recovery_control,
)
from _task_interface import (
    _RecoveryControlPhase as Phase,
)
from simpler import worker as w
from simpler._recovery import EndpointRecoverySession, capture_binding, validate_stop_identity
from simpler.task_interface import TaskArgs


class Native:
    def __init__(self):
        self.execution = Record(
            status=ExecutionRecoveryStatus.RECOVERY_REQUIRED,
            quiesced=1,
            failed_run_epoch=17,
            original_rc=-98765,
            recovery_rc=0,
            communication_context=1,
            execution_context=2,
            aicpu_stream=3,
            aicore_stream=4,
        )
        self.snapshot = {
            "allocation_present": 1,
            "device_ctx": 10,
            "local_window_base": 20,
            "workspace_addr": 40,
            "comm_handle_identity": 5,
            "hccl_comm_identity": 6,
            "comm_stream_identity": 7,
            "binding_signature": 8,
        }
        self.calls = []
        self.prepare_status = DomainRecoveryStatus.PREPARED
        self.prepare_detail = DomainRecoveryDetail.NONE
        self.prepare_rc = 0
        self.rearm_status = ExecutionRecoveryStatus.RECOVERED
        self.replace_execution = False
        self.replace_communication = False

    def execution_recovery_state(self):
        return deepcopy(self.execution)

    def comm_recovery_snapshot(self, _comm, _allocation):
        return dict(self.snapshot)

    def recover_comm_domain(self, comm, allocation_id, request):
        assert comm == 9 and allocation_id == 0
        assert request.recovery_id == 1 and request.expected_generation == 1
        assert request.collective_quiescence == 1
        self.calls.append(request.phase)
        return Record(
            status=self.prepare_status
            if request.phase == DomainRecoveryPhase.PREPARE
            else DomainRecoveryStatus.REARMED,
            generation=1 if request.phase == DomainRecoveryPhase.PREPARE else 2,
            stopped_channel_count=48,
            recovery_rc=self.prepare_rc,
            transport_detail=self.prepare_detail,
        )

    def recover_execution(self, epoch):
        assert epoch == 17
        self.calls.append("execution")
        self.execution.status = self.rearm_status
        if self.replace_execution:
            self.execution.execution_context += 1
        if self.replace_communication:
            self.snapshot["workspace_addr"] += 1
        return deepcopy(self.execution)


def packet(phase):
    return Record(
        recovery_id=1,
        endpoint_generation=1,
        phase=phase,
        has_domain=1,
        impact=DomainRecoveryImpact.DOMAIN_TRANSPORT,
        before=Record(allocation_id=0, generation=1, device_ctx=10, local_window_base=20, window_size=16, workspace=40),
    )


def prepared(native):
    session = EndpointRecoverySession(native, lambda: 9)
    for phase in (Phase.INSPECT, Phase.PREPARE, Phase.COMMIT):
        session.handle(packet(phase), True, 1)
    return session


def test_full_transaction_preserves_identity_and_arms_only_after_rearm():
    native = Native()
    session = prepared(native)
    session.handle(packet(Phase.REARM), True, 1)
    execution, domain, faulted = session.handle(packet(Phase.ARM), True, 1)
    assert native.calls == [DomainRecoveryPhase.PREPARE, DomainRecoveryPhase.COMMIT, "execution"]
    assert execution.execution_context == 2 and domain.generation == 2 and faulted


@pytest.mark.parametrize("phase", [Phase.COMMIT, Phase.REARM, Phase.ARM])
def test_out_of_order_control_cannot_open_execution(phase):
    native = Native()
    session = EndpointRecoverySession(native, lambda: 9)
    session.handle(packet(Phase.INSPECT), True, 1)
    with pytest.raises(RuntimeError, match="out-of-order"):
        session.handle(packet(phase), True, 1)
    assert native.calls == []


@pytest.mark.parametrize(
    "field,value", [("quiesced", 0), ("failed_run_epoch", 0), ("status", ExecutionRecoveryStatus.DEVICE_UNUSABLE)]
)
def test_failed_generation_requires_structural_stop_proof(field, value):
    native = Native()
    setattr(native.execution, field, value)
    with pytest.raises(RuntimeError):
        EndpointRecoverySession(native, lambda: 9).handle(packet(Phase.INSPECT), True, 1)
    assert native.calls == []


@pytest.mark.parametrize(
    "status,detail,rc",
    [
        (DomainRecoveryStatus.GIVE_UP, DomainRecoveryDetail.NONE, 0),
        (DomainRecoveryStatus.RECOVERY_FAILED, DomainRecoveryDetail.READ_FAILED, -77),
    ],
)
def test_domain_prepare_failure_cannot_reach_execution_rearm(status, detail, rc):
    native = Native()
    native.prepare_status, native.prepare_detail, native.prepare_rc = status, detail, rc
    session = EndpointRecoverySession(native, lambda: 9)
    session.handle(packet(Phase.INSPECT), True, 1)
    with pytest.raises(RuntimeError, match="PREPARE failed") as raised:
        session.handle(packet(Phase.PREPARE), True, 1)
    assert f"rc={rc}" in str(raised.value) and str(detail) in str(raised.value)
    assert native.calls == [DomainRecoveryPhase.PREPARE]


@pytest.mark.parametrize("change", ["replace_execution", "replace_communication"])
def test_resource_migration_is_refused_in_place(change):
    native = Native()
    session = prepared(native)
    setattr(native, change, True)
    with pytest.raises(RuntimeError, match="changed"):
        session.handle(packet(Phase.REARM), True, 1)
    assert session.phase == Phase.COMMIT


def test_failed_execution_rearm_does_not_arm():
    native = Native()
    native.rearm_status = ExecutionRecoveryStatus.BUSY
    session = prepared(native)
    with pytest.raises(RuntimeError, match="execution rearm failed"):
        session.handle(packet(Phase.REARM), True, 1)
    with pytest.raises(RuntimeError, match="out-of-order"):
        session.handle(packet(Phase.ARM), True, 1)


def test_wrong_or_duplicate_resource_ticket_is_rejected():
    native = Native()
    session = EndpointRecoverySession(native, lambda: 9)
    session.handle(packet(Phase.INSPECT), True, 1)
    with pytest.raises(RuntimeError, match="duplicate or stale"):
        session.handle(packet(Phase.INSPECT), True, 1)
    with pytest.raises(RuntimeError, match="stale endpoint"):
        session.handle(packet(Phase.PREPARE), True, 2)
    wrong = packet(Phase.PREPARE)
    wrong.recovery_id = 2
    with pytest.raises(RuntimeError, match="wrong resource"):
        session.handle(wrong, True, 1)


@pytest.mark.parametrize("phase", [Phase.PREPARE, Phase.REARM, Phase.ABORT])
def test_resource_transaction_cannot_change_scope_or_binding(phase):
    native = Native()
    session = EndpointRecoverySession(native, lambda: 9)
    session.handle(packet(Phase.INSPECT), True, 1)
    if phase != Phase.PREPARE:
        session.handle(packet(Phase.PREPARE), True, 1)
        session.handle(packet(Phase.COMMIT), True, 1)
    before = (session.key, session.phase, list(native.calls))
    changes = [("has_domain", 0), ("impact", DomainRecoveryImpact.EXECUTION_ONLY)]
    changes.extend(
        (name, getattr(packet(phase).before, name) + 1)
        for name in ("allocation_id", "generation", "device_ctx", "local_window_base", "window_size", "workspace")
    )
    for name, value in changes:
        wrong = packet(phase)
        target = wrong if name in ("impact", "has_domain") else wrong.before
        setattr(target, name, value)
        with pytest.raises(RuntimeError, match="wrong resource recovery transaction"):
            session.handle(wrong, True, 1)
        assert (session.key, session.phase, native.calls) == before, name
    # Rejected packets neither advance the session nor reach the provider.
    session.handle(packet(phase), True, 1)


def test_failed_inspect_does_not_consume_transaction_identity():
    native = Native()
    session = EndpointRecoverySession(native, lambda: 9)
    wrong = packet(Phase.INSPECT)
    wrong.before.workspace += 1
    with pytest.raises(RuntimeError, match="binding no longer matches"):
        session.handle(wrong, True, 1)
    assert session.key is None and session.phase is None and not native.calls
    session.handle(packet(Phase.INSPECT), True, 1)
    session.handle(packet(Phase.PREPARE), True, 1)
    assert native.calls == [DomainRecoveryPhase.PREPARE]


def test_healthy_member_transport_recovery_does_not_invent_execution_failure():
    native = Native()
    native.execution.status = ExecutionRecoveryStatus.INVALID_STATE
    native.execution.failed_run_epoch = 0
    session = EndpointRecoverySession(native, lambda: 9)
    for phase in (Phase.INSPECT, Phase.PREPARE, Phase.COMMIT, Phase.REARM, Phase.ARM):
        session.handle(packet(phase), False, 1)
    assert "execution" not in native.calls


def test_capture_binding_has_real_addresses_and_requires_live_allocation():
    native = Native()
    binding = capture_binding(native, 9, 0, 16)
    assert binding.generation == 1 and binding.workspace == 40 and binding.device_ctx == 10
    native.snapshot["allocation_present"] = 0
    with pytest.raises(RuntimeError):
        capture_binding(native, 9, 0, 16)


def test_native_control_codec_rejects_invalid_size_before_dereference():
    with pytest.raises(ValueError):
        _read_recovery_control(0, 1)


def test_failed_communication_run_transfers_cleanup_to_native_teardown(monkeypatch):
    from simpler import Worker  # noqa: PLC0415
    from simpler.task_interface import CommDomainHandle  # noqa: PLC0415
    from simpler.worker import RunHandle, _RunResources  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    worker._effective_operator_recovery = True
    resources = _RunResources()
    domain = CommDomainHandle(name="held", workers=(0, 1), contexts={}, allocation_id=0, _release_fn=None)
    resources.live_domains["held"] = domain
    worker._live_domains["held"] = domain
    handle = RunHandle(worker, 7, (object(),), resources)
    worker._accepted_run_handles.add(handle)
    collective_calls = []
    monkeypatch.setattr(worker, "_release_domain_now", lambda value: collective_calls.append(value))
    original = RuntimeError("native fault; retry budget exhausted")
    result = worker._finalize_run_handle_unlocked(handle, 7, original)
    assert result.__cause__ is original
    assert collective_calls == []
    assert resources.terminal_recovery_cleanup and not domain.freed
    assert domain.released and resources.pending_release_domains == [domain]
    assert worker._ordered_cleanup_error is result
    assert handle in worker._abandoned_run_handles and handle._keepalive is not None
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        handle.wait(0)
    # This hook is the existing post-native-teardown ownership boundary.
    assert worker._drain_abandoned_run_keepalives() is None
    assert domain.freed and not resources.pending_release_domains
    assert handle._keepalive is None and not worker._live_domains
    worker.close()


@pytest.mark.parametrize(
    "field",
    ["run_id", "task_slot", "group_index", "pipeline_generation", "dispatch_id", "endpoint_generation", "recovery_id"],
)
def test_stop_rejects_stale_attempt_identity(field):
    frame = (1, 7, 0, 13, 31, 5, 1, 2)
    value = Record(
        run_id=7,
        task_slot=5,
        group_index=1,
        pipeline_generation=13,
        dispatch_id=31,
        endpoint_generation=2,
        recovery_id=3,
    )
    setattr(value, field, 0)
    with pytest.raises(RuntimeError, match="stale"):
        validate_stop_identity(value, frame, 2, 0)


def test_stop_duplicate_is_idempotent_only_for_the_same_episode():
    frame = (1, 7, 0, 13, 31, 5, 1, 2)
    value = Record(
        run_id=7,
        task_slot=5,
        group_index=1,
        pipeline_generation=13,
        dispatch_id=31,
        endpoint_generation=2,
        recovery_id=3,
    )
    validate_stop_identity(value, frame, 2, 0)
    validate_stop_identity(value, frame, 2, 3)
    with pytest.raises(RuntimeError, match="stale"):
        validate_stop_identity(value, frame, 2, 2)


def test_single_frame_mailbox_accepts_stop_before_active_task_finishes():
    buf = memoryview(bytearray(w.MAILBOX_SIZE))
    base = w._buffer_field_addr(buf, 0)
    task = buf[w.MAILBOX_FRAME_SIZE : 2 * w.MAILBOX_FRAME_SIZE]
    task_state = w._buffer_field_addr(task, w._OFF_STATE)
    w._mailbox_store_i32(task_state, w._TASK_READY)
    w._mailbox_store_i32(base + w._OFF_STATE, w._CONTROL_REQUEST)
    struct.pack_into("Q", buf, w._OFF_CALLABLE, w._CTRL_RECOVER_ENDPOINT)
    events = []

    def start(_task):
        events.append("started")

    def stop(command):
        assert command == w._CTRL_RECOVER_ENDPOINT
        assert events == ["started"]
        assert w._mailbox_load_i32(task_state) != w._TASK_DONE
        events.append("stop_delivered")
        return 0, ""

    def progress(task_buf):
        assert events == ["started", "stop_delivered"]
        assert w._mailbox_load_i32(base + w._OFF_STATE) == w._CONTROL_DONE
        events.append("terminal")
        struct.pack_into("<ii", task_buf, w._OFF_NATIVE_EXECUTION_FAULT, 507018, -12)
        w._mailbox_store_i32(base + w._OFF_SHUTDOWN, w._SHUTDOWN_REQUESTED)
        return 1, "coordinated stop"

    try:
        w._run_mailbox_loop(
            buf, base + w._OFF_STATE, handle_task=start, handle_control=stop, poll_task=progress, hold_native_fault=True
        )
        assert events == ["started", "stop_delivered", "terminal"]
        assert w._mailbox_load_i32(task_state) == w._TASK_DONE
        assert struct.unpack_from("<ii", task, w._OFF_NATIVE_EXECUTION_FAULT) == (507018, -12)
    finally:
        task.release()
        buf.release()


def test_active_child_rethrows_retired_native_fault_instead_of_reporting_success(monkeypatch):
    buf = memoryview(bytearray(w.MAILBOX_SIZE))
    base = w._buffer_field_addr(buf, 0)
    digest = bytes(w.CALLABLE_HASH_DIGEST_BYTES)
    calls = []

    class Run:
        def done(self):
            return True

        def wait(self, timeout):
            assert timeout == 0
            calls.append("fault observed")
            raise RuntimeError("structured native kernel fault")

    def serve(_buf, _state, *, handle_task, poll_task, **_kwargs):
        task = buf[w.MAILBOX_FRAME_SIZE : 2 * w.MAILBOX_FRAME_SIZE]
        try:
            assert handle_task(task) is None
            code, message = poll_task(task)
            assert code == 1 and "structured native kernel fault" in message
        finally:
            task.release()

    native = Record(_submit_chip_run_materialized=lambda *_args: Run())
    monkeypatch.setattr(w, "_read_task_digest", lambda _: digest)
    monkeypatch.setattr(w, "_read_task_frame_identity", lambda _: (1, 7, 0, 13, 31, 5, 1, 2))
    monkeypatch.setattr(w, "_read_config_from_mailbox", lambda *_args, **_kwargs: Record())
    monkeypatch.setattr(w, "read_args_from_blob", lambda *_args: TaskArgs())
    monkeypatch.setattr(w, "_run_mailbox_loop", serve)
    monkeypatch.setattr(w, "_teardown_chip_process_resources", lambda *_args: None)
    try:
        w._run_chip_main_loop(
            Record(_impl=native),
            buf,
            base,
            base + w._OFF_STATE,
            0,
            {1: object()},
            {digest: 1},
            {digest: 1},
            w.mint_owner_instance_id(),
            chip_platform="a5",
            chip_runtime="tensormap_and_ringbuffer",
            prepared={1},
            hold_native_fault=True,
        )
        assert calls == ["fault observed"]
    finally:
        buf.release()
