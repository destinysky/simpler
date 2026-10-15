# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Child resource transaction gates without an NPU."""

from copy import deepcopy
from types import SimpleNamespace as Record

import pytest
from _task_interface import (
    DomainRecoveryImpact,
    DomainRecoveryPhase,
    DomainRecoveryStatus,
    ExecutionRecoveryStatus,
    _read_recovery_control,
)
from _task_interface import (
    _RecoveryControlPhase as Phase,
)
from simpler._recovery import EndpointRecoverySession, capture_binding


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
            recovery_rc=0,
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


def test_domain_prepare_failure_cannot_reach_execution_rearm():
    native = Native()
    native.prepare_status = DomainRecoveryStatus.GIVE_UP
    session = EndpointRecoverySession(native, lambda: 9)
    session.handle(packet(Phase.INSPECT), True, 1)
    with pytest.raises(RuntimeError, match="PREPARE failed"):
        session.handle(packet(Phase.PREPARE), True, 1)
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
