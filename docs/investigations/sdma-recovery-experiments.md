# SDMA failure and successor fencing experiments

Baseline: `opretry-soft` at `ad3fa870ad76564b5ef9f4721399a41fdb7b1e0f`.
These manual experiments require two A5 devices and the pinned PTO-ISA revision.
They do not implement automatic operator recovery dispatch.

## Ordering contract

A successor may be QUEUED or PREPARED in a native slot while its predecessor is
active, but it must not cross the device launch fence. A predecessor failure
poisons admission and terminalizes its unlaunched successor. The old successor
handle must remain terminal after recovery. The external driver redispatches A
with a new generation, waits for A on both ranks, then redispatches B with a new
generation. B's output must retain its sentinel until that redispatch.

The CPU test covers both QUEUED and PREPARED successor paths and stale-token
rejection. The device test records the actual preparation disposition. The two
persistent execution streams therefore never contain a launched B behind A.
This is a launch/admission contract, not a claim that arbitrary already-submitted
vendor stream tasks can be cancelled independently.

## Experiments

1. **Protocol timeout after completed SDMA:** one real peer TGET per rank completes.
   Its tensor-dependent consumer records completion, then registers a notification
   counter that neither rank publishes. The existing scheduler watchdog fails A.
   After local PTO retirement and both ranks' data-completion proof, the driver
   calls the explicit execution primitive and retries A before B. No other domain
   users or outstanding data operations exist in this isolated workload. This
   proves only this protocol-wait case; it does not prove transport fault repair.
2. **Transport observation:** an aligned, unmapped source (`0x1000`) is passed to
   the real SDMA TGET path. The checkpoint distinguishes entry (1), request-entry
   returned success (2), and request-entry rejected (3). Collect the device logs
   to establish whether an SDMA transport error or only completion timeout occurred.
   No proven transport quiescence is available. This case never rearms execution,
   retries, rematerializes windows, or explicitly releases communication resources.
   Each isolated rank reports its evidence and exits; default process cleanup is
   outside the soft-recovery primitive. An unclassified/failed injection is not
   evidence that transport recovery works or that resources cannot be reused.

The old SDMA snapshot comparison of workspace `sq_head/sq_tail` is removed.
`all_queues_empty = 0` means unknown for this provider. Snapshot-copy error counts
and stable identity/signature values cannot certify engine quiescence.

## Build and run

Run from the Linux repository root. Use separate invocations; run the protocol
case first. Retain device logs from the transport invocation before further tests.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-build-isolation -e .
cmake -S tests/ut/cpp -B build/sdma-recovery-ut -DCMAKE_BUILD_TYPE=Debug
cmake --build build/sdma-recovery-ut --target test_chip_run_lane -j 4
build/sdma-recovery-ut/test_chip_run_lane --gtest_filter=ChipRunLaneTest.FailedPredecessorRevokesSuccessorAndExternalReplayKeepsOrder
mkdir -p sdma-recovery-test-logs/protocol-device sdma-recovery-test-logs/transport-device
set -o pipefail
ASCEND_PROCESS_LOG_PATH="$PWD/sdma-recovery-test-logs/protocol-device" pytest tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py::test_a5_sdma_protocol_timeout_and_fifo_order --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-recovery-test-logs/protocol.log
ASCEND_PROCESS_LOG_PATH="$PWD/sdma-recovery-test-logs/transport-device" pytest tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py::test_a5_sdma_transport_fault_observation --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-recovery-test-logs/transport.log
grep -F '[SDMA_RECOVERY_EVIDENCE]' sdma-recovery-test-logs/protocol.log sdma-recovery-test-logs/transport.log
grep -E 'EXECUTION_RECOVER|EXPLICIT_ORCH_FATAL|sched_error_code|sub_class|shutdown|SHUTDOWN|507014|507018' sdma-recovery-test-logs/protocol.log sdma-recovery-test-logs/transport.log
grep -RniE 'sdma|smmu|exception|timeout|error|507014|507018' sdma-recovery-test-logs/transport-device
```

## Results to return

Return the CPU test summary, both pytest summaries, every
`[SDMA_RECOVERY_EVIDENCE]` record, and relevant device-error records with timestamps.
If a case fails, include its traceback and last evidence checkpoint.

Protocol recovery succeeds only if **both ranks** report
`SAME_RESOURCES_RETRY_A_THEN_B_PASS`, final ordering word 2, old B never launched,
full transfer data and output-sentinel assertions pass, and C/E/execution streams,
HCCL/comm stream/workspace/windows/CommContext bindings remain unchanged. There
must be no 507014. Failed or unproven recovery reports are observations, not success.

`OBSERVATION_COMPLETE` and pytest PASS alone mean the observation harness completed.
For the transport experiment, `TRANSPORT_QUIESCENCE_UNPROVEN_NO_RETRY` is the expected
conservative result. Submission checkpoint 2 and device logs are needed to classify
it; an unreadable checkpoint leaves injection evidence incomplete. A process timeout,
missing rank, B launch, stale-handle activation, or data/order mismatch is a failure.

The next decision is whether a protocol-only failure supports preservation on this
hardware, and which backend stop/health evidence is missing for actual transport
failure. Backend capability assessment, asymmetric-rank coordination, TaskArgs
rebinding, replay eligibility, retry budgets/backoff and distributed commit are
still pending. URMA and CCU are not changed by these experiments.
