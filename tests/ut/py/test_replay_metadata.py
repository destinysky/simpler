# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Host metadata and real native Coordinator/Scheduler checks; no NPU is used."""

import ctypes
import struct
import threading
from contextlib import contextmanager

import pytest
import simpler.worker as worker_module
from _task_interface import _mailbox_load_i32, _mailbox_store_i32
from simpler.task_interface import (
    MAILBOX_FRAME_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_OFF_NATIVE_EXECUTION_FAULT,
    MAILBOX_SIZE,
    CallConfig,
    RecoveryMetadata,
    ReplaySafety,
    ReplayStatus,
    TaskArgs,
    TaskState,
    _Worker,
)


def _metadata(rank, safety=ReplaySafety.REPLAY_SAFE, seq=1):
    value = RecoveryMetadata()
    value.logical_run_id = 91
    value.domain_id = 0
    value.op_seq = seq
    value.rank = rank
    value.participants = [0, 1]
    value.replay_safety = safety
    return value


def _args(rank, safety=ReplaySafety.REPLAY_SAFE, seq=1):
    value = TaskArgs()
    value.add_scalar(42)
    value.set_recovery_metadata(_metadata(rank, safety, seq))
    return value


class _ChipMailbox:
    def __init__(self, fault):
        self.memory = ctypes.create_string_buffer(MAILBOX_SIZE)
        self.address = ctypes.addressof(self.memory)
        self.fault = fault
        self.tasks = 0
        self.enabled = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        control = self.address
        task = self.address + MAILBOX_FRAME_SIZE
        while not self.stop.is_set():
            if _mailbox_load_i32(control) == worker_module._CONTROL_REQUEST:
                ctypes.c_int32.from_address(control + worker_module._OFF_ERROR).value = 0
                _mailbox_store_i32(control, worker_module._CONTROL_DONE)
            if self.enabled.is_set() and _mailbox_load_i32(task) == worker_module._TASK_READY:
                self.tasks += 1
                _mailbox_store_i32(task + worker_module._OFF_ACCEPTED, 1)
                ctypes.c_int32.from_address(task + worker_module._OFF_ERROR).value = int(self.fault)
                fault = struct.pack("=iiii", -123456 if self.fault else 0, -9 if self.fault else 0, 0, 0)
                ctypes.memmove(task + MAILBOX_OFF_NATIVE_EXECUTION_FAULT, fault, len(fault))
                if self.fault:
                    message = b"CPU native fault fixture\0"
                    ctypes.memmove(task + MAILBOX_OFF_ERROR_MSG, message, len(message))
                _mailbox_store_i32(task, worker_module._TASK_DONE)


@contextmanager
def _native_pair():
    endpoints = [_ChipMailbox(True), _ChipMailbox(False)]
    worker = _Worker(3, 1 << 20)
    worker.configure_operator_recovery(True)
    for index, endpoint in enumerate(endpoints):
        worker.add_next_level_worker_at(index, endpoint.address)
    worker.init()
    try:
        yield worker, endpoints
    finally:
        for endpoint in endpoints:
            endpoint.enabled.set()
        worker.cancel_recoveries()
        worker.close()
        for endpoint in endpoints:
            endpoint.stop.set()
        for endpoint in endpoints:
            endpoint.thread.join(3)
            assert not endpoint.thread.is_alive()


def test_metadata_is_copied_and_cleared():
    metadata = _metadata(0)
    value = TaskArgs()
    value.add_scalar(42)
    value.set_recovery_metadata(metadata)
    metadata.op_seq = 99
    copied = value.recovery_metadata
    copied.rank = 1
    assert value.recovery_metadata.rank == 0
    assert value.recovery_metadata.op_seq == 1
    assert value.scalar(0) == 42
    value.clear()
    assert value.recovery_metadata is None


@pytest.mark.parametrize("participants,rank", [([1, 0], 0), ([0, 0], 0), ([0, 1], 2), ([], 0)])
def test_malformed_membership_is_rejected(participants, rank):
    metadata = _metadata(rank)
    metadata.participants = participants
    with pytest.raises(ValueError):
        TaskArgs().set_recovery_metadata(metadata)


@pytest.mark.parametrize(
    "safety,expected",
    [
        (ReplaySafety.REPLAY_SAFE, ReplayStatus.REPLAY_ALLOWED),
        (ReplaySafety.UNKNOWN, ReplayStatus.NACK_UNSAFE),
        (ReplaySafety.FORBIDDEN, ReplayStatus.NACK_UNSAFE),
        (ReplaySafety.REQUIRES_RESTORE, ReplayStatus.NEEDS_RESTORE),
    ],
)
def test_native_replay_assessment_keeps_original_graph_held(safety, expected):
    with _native_pair() as (worker, endpoints):
        orch = worker.get_orchestrator()
        run = orch._begin_run()
        config = CallConfig()
        digest = bytes([1]) * 32
        a = orch.submit_next_level_group(
            digest, "CHIP_CALLABLE", "LOCAL_CHIP", [_args(0, safety), _args(1, safety)], config, [0, 1]
        )
        successor = [_args(0, seq=2), _args(1, seq=2)]
        for args in successor:
            args.add_dep(a)
        b = orch.submit_next_level_group(digest, "CHIP_CALLABLE", "LOCAL_CHIP", successor, config, [0, 1])
        orch._close_run_submission(run)
        for endpoint in endpoints:
            endpoint.enabled.set()
        episode = worker.wait_for_recovery(3)
        assert episode is not None and episode.all_members_terminal
        result = worker.assess_replay(episode.ticket)
        assert result.status == expected
        assert result.op.logical_run_id == 91
        assert result.op.domain_id == 0
        assert [member.rank for member in result.op.members] == [0, 1]
        assert [member.worker_id for member in result.op.members] == [0, 1]
        assert all(member.terminal and member.endpoint_generation == 1 for member in result.op.members)
        assert worker.assess_replay(episode.ticket).status == expected
        snapshot = worker.recovery_graph(episode.ticket)
        assert next(item for item in snapshot.tasks if item.is_task(a)).state == TaskState.RETRY_PENDING
        assert next(item for item in snapshot.tasks if item.is_task(b)).state == TaskState.PENDING
        assert not snapshot.run_failed and snapshot.active_recoveries == 1
        assert [endpoint.tasks for endpoint in endpoints] == [1, 1]
        assert worker.give_up_recovery(episode.ticket, "CPU qualification regression ends without replay")
        with pytest.raises(RuntimeError):
            orch._wait_run_for(run, 3)
        with pytest.raises(ValueError):
            worker.assess_replay(episode.ticket)
        orch._release_run(run)
    print("[PTO_RECOVERY_TEST] REPLAY_METADATA_NATIVE_PASS", flush=True)


def test_partial_group_metadata_is_rejected_before_dispatch():
    with _native_pair() as (worker, endpoints):
        orch = worker.get_orchestrator()
        run = orch._begin_run()
        with pytest.raises(ValueError, match="partial recovery metadata"):
            orch.submit_next_level_group(
                bytes([1]) * 32, "CHIP_CALLABLE", "LOCAL_CHIP", [_args(0), TaskArgs()], CallConfig(), [0, 1]
            )
        orch._close_run_submission(run)
        assert orch._wait_run_for(run, 1)
        assert [endpoint.tasks for endpoint in endpoints] == [0, 0]
        orch._release_run(run)
