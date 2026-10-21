# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local chip recovery controls. Policy and logical dispatch belong to L3."""

from _task_interface import (
    DomainRecoveryImpact,
    DomainRecoveryPhase,
    DomainRecoveryRequest,
    DomainRecoveryStatus,
    ExecutionRecoveryStatus,
    RecoveryDomainBinding,
)
from _task_interface import (
    _RecoveryControlPhase as Phase,
)


def capture_binding(native, comm, allocation_id, window_size, *, snapshot=None):
    if snapshot is None:
        snapshot = native.comm_recovery_snapshot(comm, allocation_id)
    if not snapshot["allocation_present"]:
        raise RuntimeError("recovery binding requires a live domain")
    binding = RecoveryDomainBinding()
    binding.allocation_id, binding.generation = allocation_id, 1
    binding.device_ctx = snapshot["device_ctx"]
    binding.local_window_base = snapshot["local_window_base"]
    binding.window_size, binding.workspace = window_size, snapshot["workspace_addr"]
    return binding


def _identity(execution):
    return (
        execution.communication_context,
        execution.execution_context,
        execution.aicpu_stream,
        execution.aicore_stream,
    )


def _transaction_key(packet):
    binding = packet.before
    return (
        packet.recovery_id,
        packet.endpoint_generation,
        packet.impact,
        packet.has_domain,
        binding.allocation_id,
        binding.generation,
        binding.device_ctx,
        binding.local_window_base,
        binding.window_size,
        binding.workspace,
    )


class EndpointRecoverySession:
    def __init__(self, native, comm_handle):
        self.native = native
        self.comm_handle = comm_handle
        self.key = None
        self.phase = None
        self.execution = None
        self.domain = None
        self.faulted = False
        self.snapshot = None

    def handle(self, packet, faulted, endpoint_generation):  # noqa: PLR0912, PLR0915
        if packet.endpoint_generation != endpoint_generation or packet.recovery_id == 0:
            raise RuntimeError("stale endpoint recovery identity")
        phase = packet.phase
        key = _transaction_key(packet)
        if phase == Phase.INSPECT:
            if self.key is not None and packet.recovery_id <= self.key[0]:
                raise RuntimeError("duplicate or stale resource recovery transaction")
            execution = self.native.execution_recovery_state()
            if execution.status == ExecutionRecoveryStatus.DEVICE_UNUSABLE:
                raise RuntimeError("device unusable")
            if faulted and (
                execution.status != ExecutionRecoveryStatus.RECOVERY_REQUIRED
                or not execution.quiesced
                or not execution.failed_run_epoch
            ):
                raise RuntimeError("failed execution lacks cooperative retirement proof")
            if not faulted and execution.status not in (
                ExecutionRecoveryStatus.INVALID_STATE,
                ExecutionRecoveryStatus.RECOVERED,
                ExecutionRecoveryStatus.ALREADY_RECOVERED,
            ):
                raise RuntimeError("healthy endpoint execution substrate is not idle and reusable")
            snapshot = None
            if packet.has_domain:
                snapshot = self.native.comm_recovery_snapshot(self.comm_handle(), packet.before.allocation_id)
                if (
                    not snapshot["allocation_present"]
                    or snapshot["device_ctx"] != packet.before.device_ctx
                    or snapshot["local_window_base"] != packet.before.local_window_base
                    or snapshot["workspace_addr"] != packet.before.workspace
                ):
                    raise RuntimeError("retained communication binding no longer matches its domain")
            self.key, self.execution, self.faulted = key, execution, faulted
            self.domain, self.snapshot = None, snapshot
        elif self.key != key:
            raise RuntimeError("wrong resource recovery transaction")
        elif phase == Phase.ABORT:
            if packet.has_domain and packet.impact == DomainRecoveryImpact.DOMAIN_TRANSPORT and self.phase is not None:
                self.domain = self.native.recover_comm_domain(
                    self.comm_handle(), packet.before.allocation_id, self._request(packet, DomainRecoveryPhase.ABORT)
                )
        elif phase == Phase.PREPARE:
            self._require_phase(Phase.INSPECT)
            if packet.has_domain:
                self.domain = self.native.recover_comm_domain(
                    self.comm_handle(), packet.before.allocation_id, self._request(packet, DomainRecoveryPhase.PREPARE)
                )
                expected = (
                    DomainRecoveryStatus.PREPARED
                    if packet.impact == DomainRecoveryImpact.DOMAIN_TRANSPORT
                    else DomainRecoveryStatus.PRESERVED
                )
                if self.domain.status != expected:
                    raise RuntimeError(
                        f"domain PREPARE failed: {self.domain.status}, rc={self.domain.recovery_rc}, "
                        f"detail={self.domain.transport_detail}"
                    )
        elif phase == Phase.COMMIT:
            self._require_phase(Phase.PREPARE)
            self.domain = self.native.recover_comm_domain(
                self.comm_handle(), packet.before.allocation_id, self._request(packet, DomainRecoveryPhase.COMMIT)
            )
            if self.domain.status != DomainRecoveryStatus.REARMED:
                raise RuntimeError(
                    f"domain COMMIT failed: {self.domain.status}, rc={self.domain.recovery_rc}, "
                    f"detail={self.domain.transport_detail}"
                )
        elif phase == Phase.REARM:
            self._require_phase(
                Phase.COMMIT if packet.impact == DomainRecoveryImpact.DOMAIN_TRANSPORT else Phase.PREPARE
            )
            before = self.execution
            if self.faulted:
                self.execution = self.native.recover_execution(before.failed_run_epoch)
                if self.execution.status not in (
                    ExecutionRecoveryStatus.RECOVERED,
                    ExecutionRecoveryStatus.ALREADY_RECOVERED,
                ):
                    raise RuntimeError(
                        f"execution rearm failed: {self.execution.status}, rc={self.execution.recovery_rc}"
                    )
            if _identity(before) != _identity(self.native.execution_recovery_state()):
                raise RuntimeError("execution recovery changed persistent Context/stream identity")
            if self.snapshot is not None:
                after = self.native.comm_recovery_snapshot(self.comm_handle(), packet.before.allocation_id)
                for name in (
                    "comm_handle_identity",
                    "hccl_comm_identity",
                    "comm_stream_identity",
                    "device_ctx",
                    "local_window_base",
                    "workspace_addr",
                    "binding_signature",
                ):
                    if self.snapshot[name] != after[name]:
                        raise RuntimeError(f"in-place recovery changed communication resource {name}")
        elif phase == Phase.ARM:
            self._require_phase(Phase.REARM)
        else:
            raise RuntimeError("unknown endpoint recovery phase")
        self.phase = phase
        print(
            f"[PTO_RECOVERY] ENDPOINT_RECOVERY phase={phase} recovery_id={packet.recovery_id} "
            f"rank={self.snapshot.get('rank', -1) if self.snapshot is not None else -1} "
            f"same_context_streams={int(phase in (Phase.REARM, Phase.ARM))} faulted={int(self.faulted)} "
            f"domain_generation={self.domain.generation if self.domain is not None else 0} "
            f"stopped_channels={self.domain.stopped_channel_count if self.domain is not None else 0}",
            flush=True,
        )
        return self.execution, self.domain, self.faulted

    def _require_phase(self, phase):
        if self.phase != phase:
            raise RuntimeError("out-of-order endpoint recovery phase")

    def _request(self, packet, phase):
        request = DomainRecoveryRequest()
        request.recovery_id = packet.recovery_id
        request.expected_generation = packet.before.generation
        request.failed_run_epoch = self.execution.failed_run_epoch if self.faulted else 0
        request.original_rc = self.execution.original_rc if self.faulted else 0
        request.impact = packet.impact
        request.collective_quiescence = 1
        request.phase = phase
        request.collective_rc = -1 if phase == DomainRecoveryPhase.ABORT else 0
        return request


def validate_stop_identity(packet, frame, endpoint_generation, previous_recovery_id):
    if frame is None or packet.endpoint_generation != endpoint_generation or packet.recovery_id == 0:
        raise RuntimeError("stale cooperative-stop endpoint")
    expected = (frame[1], frame[5], frame[6], frame[3], frame[4])
    actual = (packet.run_id, packet.task_slot, packet.group_index, packet.pipeline_generation, packet.dispatch_id)
    if actual != expected or previous_recovery_id not in (0, packet.recovery_id):
        raise RuntimeError("cooperative stop targets a stale task/attempt/generation")
