# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Automatic replay after a kernel error leaves the peer waiting for an SDMA notification."""

import ctypes
import json
import os
import re
import select
import sys
from pathlib import Path

import pytest
from simpler import Worker
from simpler.task_interface import (
    ArgDirection as D,
)
from simpler.task_interface import (
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    RecoveryArgBinding,
    RecoveryArgKind,
    RecoveryMetadata,
    ReplaySafety,
    TaskArgs,
    TaskDomainBinding,
    TensorArgType,
)

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

N = 16384
KERNELS = Path(__file__).parent / "kernels"


def _callable():
    compiler = KernelCompiler(platform="a5")
    root = ensure_pto_isa_root()
    includes = list(compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer")) + [
        str(compiler.project_root / "src/common")
    ]
    children = []
    for index, filename, signature in (
        (0, "kernel_sdma_transfer.cpp", [D.IN, D.OUT, D.IN]),
        (1, "kernel_replay_protocol.cpp", [D.INOUT, D.OUT] + [D.IN] * 4),
        (2, "kernel_sdma_order.cpp", [D.IN, D.OUT, D.IN, D.IN, D.IN]),
    ):
        binary = compiler.compile_incore(
            source_path=str(KERNELS / "aiv" / filename),
            core_type="aiv",
            pto_isa_root=root,
            extra_include_dirs=includes,
        )
        children.append((index, CoreCallable.build(signature=signature, binary=extract_text_section(binary))))
    binary = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer", source_path=str(KERNELS / "orchestration/replay_sdma_orch.cpp")
    )
    return ChipCallable.build(
        signature=[D.IN, D.OUT, D.OUT, D.INOUT, D.IN, D.INOUT, D.INOUT] + [D.IN] * 6,
        func_name="aicpu_orchestration_entry",
        binary=binary,
        children=children,
    )


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.timeout(240)
@pytest.mark.parametrize("mode", ["healthy", "transient", "persistent", "two-ops"])
def test_a5_original_graph_replay(st_device_ids, mode, capfd):  # noqa: PLR0912, PLR0915
    assert len(st_device_ids) == 2
    worker = Worker(
        level=3,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
        device_ids=list(st_device_ids),
        num_sub_workers=1,
        enable_operator_recovery=True,
        automatic_operator_recovery=True,
        operator_recovery_timeout_s=20.0,
    )
    read_fd, write_fd = os.pipe()

    def gate(_args):
        assert select.select([read_fd], [], [], 30)[0]
        assert os.read(read_fd, 1) == b"G"

    gate_handle = worker.register(gate)
    chip = worker.register(_callable())
    device_buffers, host_buffers = [], []
    counters, orders, outputs, markers, tasks = {}, {}, {}, {}, {}
    budget_exhausted = False
    outputs_verified = False

    def allocation(rank, count, dtype):
        value = worker.alloc_child_tensor(rank, (count,), dtype)
        device_buffers.append(value)
        zero = worker.create_buffer(count * 4)
        host_buffers.append(zero)
        zero.shm.buf[:] = bytes(count * 4)
        worker.copy_to(value, zero)
        return value

    def read(value, count, element_type):
        host = worker.create_buffer(count * 4)
        host_buffers.append(host)
        worker.copy_from(host, value)
        return list((element_type * count).from_buffer_copy(host.shm.buf))

    try:
        worker.init()
        for rank in range(2):
            counters[rank] = allocation(rank, 8, DataType.UINT32)
            orders[rank] = allocation(rank, 8, DataType.UINT32)
            for seq in range(1, 4):
                outputs[rank, seq] = allocation(rank, N, DataType.FLOAT32)
                markers[rank, seq] = allocation(rank, 8, DataType.UINT32)

        def graph(orch, _args, _config):
            try:
                with (
                    orch.scope(),
                    orch.allocate_domain(
                        name="replay",
                        workers=[0, 1],
                        window_size=4 * 1024 * 1024,
                        buffers=[
                            CommBufferSpec("source", "float32", N, N * 4),
                            CommBufferSpec("handshake", "uint32", 64, 256),
                        ],
                    ) as domain,
                ):
                    for rank in range(2):
                        host = worker.create_buffer(N * 4)
                        host_buffers.append(host)
                        host.shm.buf[:] = bytes((ctypes.c_float * N)(*([100.0 + rank] * N)))
                        orch.copy_to(domain[rank].buffers["source"], host)
                        zero = worker.create_buffer(256)
                        host_buffers.append(zero)
                        zero.shm.buf[:] = bytes(256)
                        orch.copy_to(domain[rank].buffers["handshake"], zero)
                    gate_buffer = orch.alloc((1,), DataType.UINT32)
                    gate_args = TaskArgs()
                    gate_args.add_tensor(gate_buffer.tensor((1,), DataType.UINT32), TensorArgType.OUTPUT)
                    orch.submit_sub(gate_handle, gate_args)
                    config = CallConfig()
                    config.aicpu_thread_num = 2
                    for seq in range(1, 4):
                        args_list = []
                        for rank in range(2):
                            context = domain[rank]
                            args = TaskArgs()
                            args.add_tensor(
                                context.buffers["source"].tensor((N,), DataType.FLOAT32), TensorArgType.INPUT
                            )
                            args.add_tensor(
                                outputs[rank, seq].tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING
                            )
                            args.add_tensor(
                                markers[rank, seq].tensor((8,), DataType.UINT32), TensorArgType.OUTPUT_EXISTING
                            )
                            args.add_tensor(counters[rank].tensor((8,), DataType.UINT32), TensorArgType.OUTPUT_EXISTING)
                            args.add_tensor(gate_buffer.tensor((1,), DataType.UINT32), TensorArgType.INPUT)
                            args.add_tensor(orders[rank].tensor((8,), DataType.UINT32), TensorArgType.OUTPUT_EXISTING)
                            args.add_tensor(
                                context.buffers["handshake"].tensor((64,), DataType.UINT32),
                                TensorArgType.OUTPUT_EXISTING,
                            )
                            fault = int(
                                rank == 0 and ((seq == 1 and mode != "healthy") or (seq == 2 and mode == "two-ops"))
                            )
                            if fault and mode == "persistent":
                                fault = 2
                            for scalar in (context.device_ctx, orders[rank].base, seq - 1, seq, fault, rank):
                                args.add_scalar(scalar)
                            binding = TaskDomainBinding()
                            binding.domain = context.recovery_binding
                            entries = []
                            for kind, index in ((RecoveryArgKind.WINDOW_TENSOR, 0), (RecoveryArgKind.COMM_CONTEXT, 0)):
                                entry = RecoveryArgBinding()
                                entry.kind, entry.index, entry.offset = kind, index, 0
                                entries.append(entry)
                            handshake = RecoveryArgBinding()
                            handshake.kind, handshake.index = RecoveryArgKind.WINDOW_TENSOR, 6
                            handshake.offset = (
                                context.buffers["handshake"].base - context.recovery_binding.local_window_base
                            )
                            entries.append(handshake)
                            binding.arguments = entries
                            args.add_domain_binding(binding)
                            metadata = RecoveryMetadata()
                            metadata.logical_run_id, metadata.domain_id, metadata.op_seq = (
                                801,
                                domain.allocation_id,
                                seq,
                            )
                            metadata.rank, metadata.participants = rank, [0, 1]
                            # Source is immutable. Outputs/order are private and overwritten idempotently.
                            metadata.replay_safety = ReplaySafety.REPLAY_SAFE
                            args.set_recovery_metadata(metadata)
                            if seq > 1:
                                args.add_dep(tasks[seq - 1])
                            args_list.append(args)
                        tasks[seq] = orch.submit_next_level_group(chip, args_list, config, workers=[0, 1])
            finally:
                os.write(write_fd, b"G")

        run = worker.submit(graph)
        if mode == "persistent":
            with pytest.raises(RuntimeError, match="retry budget exhausted"):
                run.wait(90)
            budget_exhausted = True
            print("[PTO_RECOVERY_TEST] ORIGINAL_GRAPH_BUDGET_GIVE_UP_PASS", flush=True)
            return
        run.wait(90)
        episode_count = 0 if mode == "healthy" else (2 if mode == "two-ops" else 1)
        for rank in range(2):
            expected_counts = [1, 1, 1, 0, 0, 0, 0, 0]
            if mode != "healthy":
                expected_counts[0] = 2
            if mode == "two-ops":
                expected_counts[1] = 2
            assert read(counters[rank], 8, ctypes.c_uint32) == expected_counts
            assert read(orders[rank], 8, ctypes.c_uint32)[0] == 3
            for seq in range(1, 4):
                assert all(value == 101.0 - rank for value in read(outputs[rank, seq], N, ctypes.c_float))
                marker = read(markers[rank, seq], 8, ctypes.c_uint32)
                assert marker[0] == 1 and marker[1] in (seq - 1, seq)
                if seq == 3:
                    assert marker[1:3] == [2, 1]
        assert not worker.pending_recoveries()
        for buffer in device_buffers:
            worker.free(buffer)
        outputs_verified = True
        print(
            "[PTO_RECOVERY_TEST] ORIGINAL_GRAPH_REPLAY_PASS "
            + json.dumps(
                {
                    "mode": mode,
                    "recovery_episodes": episode_count,
                    "original_tasks": "A,B,C",
                    "order": "A->B->C",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        try:
            worker.close()
        except RuntimeError as error:
            if not budget_exhausted or "retry budget exhausted" not in str(error):
                raise
            print(f"[PTO_RECOVERY_TEST] TERMINAL_CLOSE_FAULT {error}", flush=True)
        finally:
            for host in host_buffers:
                host.close()
            os.close(read_fd)
            os.close(write_fd)
            # Child exit and Worker.close flush their host writers before auditing the evidence.
            captured = capfd.readouterr()
            with capfd.disabled():
                print(captured.out, end="", flush=True)
                print(captured.err, end="", file=sys.stderr, flush=True)
            if outputs_verified or budget_exhausted:
                with capfd.disabled():
                    _assert_automatic_evidence(captured.out + captured.err, mode)


def _assert_automatic_evidence(log, mode):
    expected = 0 if mode == "healthy" else (2 if mode in ("persistent", "two-ops") else 1)
    outcomes = re.findall(
        r"AUTO_MEMBER_TERMINAL recovery_id=(\d+) worker=(\d+) raw_rc=(-?\d+) runtime_status=(-?\d+)", log
    )
    episodes = {}
    for episode, rank, _raw, status in outcomes:
        members = episodes.setdefault(episode, {})
        assert rank not in members, "duplicate terminal evidence"
        members[rank] = int(status)
    assert len(episodes) == expected, "automatic owner did not collect the expected episodes"
    for members in episodes.values():
        # Test-specific diagnosis, never a production recovery eligibility whitelist.
        assert members == {"0": -101, "1": -12}, "peer must be stopped by L3, without waiting for its watchdog"
    assert len(re.findall(r"STOP_DELIVERED[^\n]* active=1", log)) == expected
    dispatched = 0 if mode == "healthy" else (2 if mode == "two-ops" else 1)
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_DISPATCH ", log)) == dispatched
    committed = 0 if mode == "persistent" else dispatched
    assert len(re.findall(r"\[PTO_RECOVERY\] REPLAY_COMMIT ", log)) == committed
    if mode == "persistent":
        assert "RETRY_BUDGET_GIVE_UP" in log
    print(f"[PTO_RECOVERY_TEST] COORDINATED_PROTOCOL_RECOVERY_PASS mode={mode} episodes={expected}", flush=True)
