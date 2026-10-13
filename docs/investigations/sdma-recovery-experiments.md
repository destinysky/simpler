# SDMA recovery method comparison

Preimage: `opretry-soft` at `83f6a6befb5d5245478df90ee61e0e6f52d9b323`.
Two A5 devices and the pinned PTO-ISA revision are required. URMA/CCU use no
SDMA assumptions or repair operations. These experiments do not implement
automatic distributed replay or select a recovery mode from a host error code.

## Scope and safety contract

The protocol regression completes a real peer TGET before withholding a
notification. Its explicit execution recovery preserves C/E, persistent execution
streams, HCCL, comm stream, SDMA workspace, VMM windows and device CommContext.

The transport comparison submits the same aligned unmapped source (`0x1000`) in
both independent runs. Checkpoint 2 proves request-entry success, not hardware
acceptance or an actual link/network outage. Local PTO retirement alone does not
stop DMA. Every producer is fenced; the backend calls `aclrtStreamAbort` on all 48
owned SDMA streams and collectively agrees on their results before workspace writes.
No device-wide synchronize, device Abort, device reset or Context recreation is used.

- `rearm`: keep all streams/workspace and requery live STARS queue metadata.
- `replace`: after the same successful collective Abort, provision new SDMA
  streams/workspace while the old resources still exist; update the sole domain's
  device workspace binding, then retire the old stopped transport. C/E, execution
  streams, HCCL, comm stream, VMM windows and device CommContext address stay fixed.

The current provider refuses other allocations, subsets, derived/base window contexts
and pending cleanup. Its request requires a full owner fence on all ranks. FIFO and
native-slot gates do not prove that external users are fenced. Abort failure causes
no requery/replacement/retry. Failure after mutation quarantines the transaction;
the lane rejects subsequent execution rearming and submission. Communication
cleanup transfers unproved DMA resources to the terminal device lifecycle,
without freeing their workspace/windows in the soft primitive.

The host-only PTO manager overlay supplies ownership-aware Abort and workspace
refresh APIs. CMake checks the complete pinned-header SHA256 and applies a checked-in
diff to a build-directory copy. The managed `build/pto-isa` checkout stays clean.
No private-member layout cast or duplicated workspace manager is used. The original
STOP_ON_FAILURE setter is retained. Once upstream supplies these APIs, the overlay
and build adaptation can be removed together.

## Ordering and fresh-data proof

A fault terminalizes the prepared/queued B without launching it. Ordinary submission
remains refused until explicit recovery. Old B cannot be activated after recovery.
Retry A uses a new generation/completion allocation, new input values (200 + rank),
zeroed completion markers and an output sentinel. All 16384 transferred floats must
equal the new peer data. Both ranks commit A before redispatching B with a new
generation and another input change (300 + rank). B remains untouched until then;
its completion marker records predecessor 1 and final ordering word 2.

Cached workspace head/tail and `all_queues_empty = 0` are diagnostic/unknown.
`queue_read_errors` counts snapshot-copy failures. Dynamic queue signatures are
progress, not transport identity. Refresh may update hardware descriptions while
retaining the Runtime stream objects; resource identity must be read with that scope.

## Commands

Run sequentially from the Linux repository root. Activate the existing project-local
venv. Reinstall before testing; runtime binaries are not rebuilt on import.

```bash
python -m pip install --no-build-isolation -e .
cmake -S tests/ut/cpp -B build/sdma-abort-ut -DCMAKE_BUILD_TYPE=Debug
cmake --build build/sdma-abort-ut --target test_chip_run_lane test_domain_recovery -j 4
ctest --test-dir build/sdma-abort-ut --output-on-failure -R '^(test_chip_run_lane|test_domain_recovery)$'
mkdir -p sdma-abort-test-logs/rearm-device sdma-abort-test-logs/replace-device
set -o pipefail
pytest examples/a5/tensormap_and_ringbuffer/vector_example/test_vector_example.py --platform a5 --runtime tensormap_and_ringbuffer --device 0 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-abort-test-logs/A1.log
pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-abort-test-logs/A2.log
pytest tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py::test_a5_sdma_protocol_timeout_and_fifo_order --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-abort-test-logs/protocol.log
ASCEND_PROCESS_LOG_PATH="$PWD/sdma-abort-test-logs/rearm-device" pytest 'tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py::test_a5_sdma_transport_recovery_compare[rearm]' --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-abort-test-logs/rearm.log
ASCEND_PROCESS_LOG_PATH="$PWD/sdma-abort-test-logs/replace-device" pytest 'tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py::test_a5_sdma_transport_recovery_compare[replace]' --platform a5 --runtime tensormap_and_ringbuffer --device 0,1 --manual include --require-pto-isa --pto-session-timeout 1200 -v -s 2>&1 | tee sdma-abort-test-logs/replace.log
grep -F '[SDMA_RECOVERY_EVIDENCE]' sdma-abort-test-logs/protocol.log sdma-abort-test-logs/rearm.log sdma-abort-test-logs/replace.log
grep -E 'SDMA_STREAM_ABORT|SDMA_ABORT_RESULT|SDMA_REQUERY_RESULT|SDMA_REPLACEMENT_READY|TRANSPORT_REARMED|DOMAIN_RECOVERY_REJECTED|EXECUTION_RECOVER' sdma-abort-test-logs/rearm.log sdma-abort-test-logs/replace.log
grep -RniE 'sdma|smmu|exception|timeout|error|507014|507018' sdma-abort-test-logs/rearm-device sdma-abort-test-logs/replace-device
```

When `task-submit` is installed on the hardware host, each pytest invocation must
hold its two-device lock for the whole run. Avoid broad G/H matrices for this question.

## Results and decision

Return CPU/pytest summaries, all evidence records, all Abort result records and
relevant device errors. Successful stop requires **channels = stopped = 48, rc = 0
on both ranks**. Successful recovery additionally requires the corresponding
`SDMA_REARM_RETRY_A_THEN_B_PASS` or `SDMA_REPLACE_RETRY_A_THEN_B_PASS` on both ranks,
full fresh-data assertions, old B never launched and final order 2. Protocol regression
requires `SAME_RESOURCES_RETRY_A_THEN_B_PASS`. A1/A2 require ordinary pytest PASS.
There must be no actual new 507014 error; numeric constants in tracebacks do not count.

The observation harness can complete with `TRANSPORT_RECOVERY_FAILED_NO_RETRY` or
`LEGAL_RETRY_FAILED_NO_B`. A green pytest in those cases means bounded evidence
collection, not recovered traffic. `failure_status` describes the original fault;
`current_execution_status` is refreshed after rearm/probe so the two are not confused.

| Original transport after Abort/requery | New transport after Abort | Decision |
| --- | --- | --- |
| Legal fresh transfer and A-before-B pass | Either | In-place recovery suffices for this injected failure |
| Fails, with established stop and matched injection | Legal fresh transfer and A-before-B pass | Transport replacement is required under these implemented recovery capabilities |
| Fails | Fails | Do not claim transport replacement solves it; inspect C/shared/device failure evidence |
| Stop unproven or injection invalid | Any | No method conclusion; no unsafe replay |

Replacement necessity is scoped to this fault and supported cleanup/requery method,
not permanent hardware damage or arbitrary network outages. Actual SDMA/SMMU/link
records are still required to classify the injected failure. Rank coordination,
logical-op replay eligibility, TaskArgs rebind planning and retry/backoff/commit remain
outside these tests. The obsolete observation-only transport entry is removed.
The comparison injection remains only while it is needed to select the mechanism;
after that decision, remove the injection branch and comparison harness together,
retaining the resulting production primitive and ordering regressions.
