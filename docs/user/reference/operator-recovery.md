# Operator recovery API

Opt-in local A5 recovery; see [Python API](python-api.md) for ordinary execution.

## L2 execution and communication primitives

The A5 native `simpler.task_interface.ChipWorker` exposes explicit recovery controls:
`execution_recovery_state()`, `recover_execution(failed_run_epoch)`,
`recover_comm_domain(comm_handle, allocation_id, request)`, and
`comm_recovery_snapshot(comm_handle, allocation_id)`. The snapshot reads identities
and queue diagnostics; it does not prove that async engines or peer ranks have drained.
For A5 SDMA, `all_queues_empty = 0` means **unproven**, not confirmed busy:
workspace channel descriptions do not read live SQ head/tail registers.
`queue_read_errors` counts snapshot-copy errors, not communication task failures.
Neither field authorizes recovery or window retirement.
`binding_signature` fingerprints resource addresses and queue configuration;
`queue_signature` fingerprints mutable queue progress. Its value may change
during normal SDMA work or rearm and is not a resource-preservation invariant.

`DomainRecoveryRequest` (ABI 3, 56 bytes) names a monotonically increasing
`recovery_id`, `expected_generation`, local `failed_run_epoch`, `original_rc`,
`DomainRecoveryImpact` and `DomainRecoveryPhase`. Original and recovery codes
are diagnostics; they do not select a recovery mode or authorize replay.

`EXECUTION_ONLY` with `PREPARE` returns `PRESERVED`: domain generation, Context C,
HCCL, comm stream, workspace, windows and device CommContext remain unchanged.
`DOMAIN_TRANSPORT` requests the backend's supported recovery mechanism, currently
SDMA in-place rearm. There is no caller choice of stream replacement or domain
rebuild. The owner fences all producers on every member before setting
`collective_quiescence = 1`; cooperative PTO retirement alone does not certify
outstanding async DMA stop. SDMA accepts one full-communicator domain with no
base/derived context users or pending allocation cleanup.

The transaction is local and explicit:

1. `PREPARE` stops every manager-owned SDMA stream through `aclrtStreamAbort`,
   validates the complete stop report, and requeries the existing workspace.
   `PREPARED` retains generation and resource addresses. Execution admission
   stays fenced, and `recover_execution` returns `BUSY` while a domain is prepared.
   Window allocation/release, context derivation and business comm barriers are
   refused until local COMMIT. Destruction of a pending transaction quarantines
   its resources for terminal device cleanup.
2. The owner collects matching successful preparation from **all** members,
   then issues `COMMIT` on all members. Local `REARMED` advances generation once
   and reports preserved addresses with `rebind_required = 1`; the owner refreshes
   generation metadata through `RecoveryBindingPlan` before any external dispatch.
3. Only after every COMMIT acknowledgment may the owner explicitly recover
   execution and decide whether to dispatch the logical task again. A failed or
   missing acknowledgment ends recovery; `ABORT` with nonzero `collective_rc`
   quarantines prepared or locally committed members and preserves both faults.
   A lost member requires the control plane's existing terminal handling.

All members agree on id, generation, impact and producer fence; failed execution
epochs are local. The cross-host control plane owns agreement, deadlines and
failure decisions. No filesystem marker or HCCL barrier implements recovery
agreement inside the backend. COMMIT is a local resource transaction acknowledgment,
not a distributed task/result commit. The owner must keep admissions fenced through
all-member acknowledgment; Step7 replay eligibility and Step8 dispatch remain separate.

`DomainRecoveryResult` (ABI 2, 136 bytes) includes `transport_channel_count`,
`stopped_channel_count`, and `transport_stopped`. These describe local provider
stop proof, unlike queue snapshots. Exact duplicate PREPARE is `PREPARED` before
commit and `ALREADY_RECOVERED` afterward; duplicate COMMIT never advances generation
twice. Changed/reused ids, stale generations and COMMIT without PREPARE are rejected.
Domain recovery never clears execution admission poison. Failures return no
dispatchable binding and permanently fence soft recovery. Unproved DMA/query
resources belong to terminal device cleanup; the primitive never resets a device.

## Retained local recovery graphs

Local A5 L3 workers may opt into `enable_operator_recovery=True` before `init`.
The default is false, and other platforms/endpoints keep their ordinary failure
handling. This mode intercepts a structured native TASK_FAILURE before terminal
propagation, places the original TaskSlot in RETRY_PENDING, and freezes new
dispatch/admission. The submitted graph, TaskArgs, dependencies, TensorMap keys
and managed-buffer references remain owned by the run. It is not a data snapshot.

For a local chip group, the first fault establishes one episode. Other members
continue reporting terminal outcomes while dispatch stays frozen, including a
healthy member that finishes before or after its peer's fault. This collects local
evidence for `assess_replay`; cross-host agreement remains the owner's responsibility.

The owner uses asynchronous `Worker.submit` and these explicit APIs:

- `pending_recoveries()` returns copied episode/member records, including groups
  with a member still executing.
- `wait_for_recovery(timeout)` returns a copied episode once all its members are
  terminal, or None on timeout/cancellation. Host completion is not a PTO/transport
  quiescence proof and does not authorize rearm or replay.
- `recovery_graph(ticket)` copies the retained run's task states, arguments,
  references and dependency metadata; a stale ticket is rejected. BUILDING slots
  expose only their identity/state until the submitter publishes their metadata.
  A task snapshot's `is_task(handle)` and `depends_on(handle)` preserve the opaque
  TaskHandle API while allowing the owner to inspect graph relationships.
- `give_up_recovery(ticket, reason)` explicitly ends an episode once every member
  is terminal. Wrong/stale/duplicate tickets return false; a live peer is refused.
  Original fault diagnostics are retained, held failed children enter terminal
  cleanup, and ordinary failure propagation reclaims the graph.
  Tickets also identify their local coordinator owner, so matching numeric run,
  slot and recovery ids from another Worker cannot resolve this episode.

Without an owner decision the graph remains held. Synchronous `run` therefore
requires a separate recovery owner; it cannot resolve its own wait. `close`
cancels held episodes before draining run fences and releases leased recovery
waiters. A remaining active peer still needs endpoint terminalization; timeout
does not justify freeing its resources. Terminal cleanup may reset a failed
device under the existing lifecycle, but no hold/rebind primitive resets one.
Graph-construction cancellation or another terminal run error also ends its
held episodes, so a failed submission cannot leave an orphaned recovery hold.

There is no implicit retry or endpoint recreation. The owner explicitly invokes
`retry_recovery` after eligibility succeeds. Unlaunched downstream tasks remain
in the graph during the hold; retired native execution tokens remain invalid.

## Logical-op metadata and replay eligibility

Before submitting a local chip task, the compiler/application may attach a copied
`RecoveryMetadata` declaration to its `TaskArgs`:

```python
from simpler.task_interface import RecoveryMetadata, ReplaySafety

metadata = RecoveryMetadata()
metadata.logical_run_id = 91
metadata.domain_id = 0
metadata.op_seq = 1
metadata.rank = 0
metadata.participants = [0, 1]
metadata.replay_safety = ReplaySafety.REPLAY_SAFE
args.set_recovery_metadata(metadata)
```

`(logical_run_id, domain_id, op_seq)` identifies one distributed logical op;
`rank` names its member, independently of the local worker id. Logical run ids
are nonzero, allocated by the compiler/application, and distinct from native
run ids. Domain ids and sequence numbers may be zero; sequence numbers increase
within each logical run/domain. All members use the same key, sorted unique
participant list and replay-safety contract. Each member has a distinct rank and
local chip endpoint. A task group has metadata on every member or none. Separate
per-rank TaskSlots with the same key are aggregated while they remain observable
in the held native run. Metadata is host-owned and survives argument rebinding;
it is absent from the ordinary device blob. Remote-endpoint annotations are
refused until the control-plane metadata transport exists.

`REPLAY_SAFE` is an explicit semantic promise: inputs remain valid for replay,
and effects already produced are either private/discardable or replay-safe.
Retaining TaskArgs or IN/OUT tags does not snapshot or restore tensor contents.
An operation needing restored data uses `REQUIRES_RESTORE`; unknown safety is
the default, and `FORBIDDEN` prohibits replay. Neither fault codes nor an empty
queue infer safety. SDMA, URMA and CCU share this declaration; it does not select
or implement a transport recovery backend.

`worker.assess_replay(ticket)` reads current native Scheduler/Coordinator evidence
and returns a copied `ReplayAssessment`: ticket, status, reason, optional
`DistributedOpRecord`, and blocking TaskHandles. The record contains common
identity, safety, participants, and rank-sorted member records with task/worker
placement, current state, terminal observation and endpoint generation. A zero
endpoint generation means that a separate rank TaskSlot was not part of the
queried native failure episode; it is not stop or dispatch evidence.

| Status | Meaning |
| --- | --- |
| `REPLAY_ALLOWED` | Current semantic and graph checks pass; recovery hold remains intact |
| `WAITING_GRAPH` | Submission is still open or relevant graph metadata is being published |
| `WAITING_MEMBERS` | A declared member still needs terminal/stop coordination |
| `NEEDS_COORDINATION` | Declared ranks are absent locally, or advanced unannotated participant work has unknown scope |
| `NEEDS_RESTORE` | The declared input/effect contract requires state restoration |
| `NACK_METADATA` | Missing/inconsistent identity, contract, membership or placement |
| `NACK_UNSAFE` | Replay semantics are unknown or forbidden |
| `NACK_ADVANCED` | A transitive consumer or later op in this logical domain has advanced |
| `NACK_NOT_RETAINED` | Required task, generation, failure evidence or recovery hold is unavailable |
| `NACK_DEVICE_UNUSABLE` | Structured evidence confirms an unusable participant endpoint |

The native graph must be closed and retained by the current ticket. Checks cover
ordering-only and unannotated transitive consumers, and later same-domain ops
even without a cross-rank DAG edge. RUNNING, COMPLETED, CONSUMED, FAILED and
RETRY_PENDING successors count as advanced. A completed upstream producer is
not a successor. Wrong-owner, stale, resolved or resolving tickets raise
`ValueError`; an assessment is not a reusable authorization token.

`REPLAY_ALLOWED` does **not** prove PTO quiescence, async transport stop, successful
resource recovery, argument rebinding or cross-host agreement. It performs none
of these actions and never releases A or dispatches B/C. The separate
`retry_recovery` transaction collects local resource proofs, revalidates the
current ticket/graph and commits only all-member success before releasing the
original successors. Rank-stop commands, cross-host evidence exchange, input
restoration and global result commit are not implemented by eligibility or
local dispatch. Unannotated ordinary execution keeps its existing behavior.

## Explicit local replay

```python
accepted = worker.retry_recovery(
    episode.ticket,
    max_retries=1,
    backoff_s=0.0,
    impact=DomainRecoveryImpact.DOMAIN_TRANSPORT,
    control_timeout_s=30.0,
)
```

The owner calls this after `wait_for_recovery` or equivalent terminal evidence.
`True` means a fresh physical attempt was dispatched on the original TaskSlot;
only the original RunHandle completing successfully proves logical success.
`False` means the caller's retry budget was exhausted and GIVE_UP was queued;
wait on the RunHandle for terminal failure, then close the Worker. A failed
communication run retains its domains and keepalives until native tree teardown;
it does not issue a collective free against an exiting peer. Domain claims become
freed only after that teardown succeeds. Ordinary admission remains closed during
this handoff. Attempt zero is the original execution. Policy is caller-supplied
at L3, with no L2 retry credit.
Backoff blocks the recovery owner on a cancellable wakeup, leaving scheduler
progress active. Close cancels the wait before draining operation leases.

Local replay requires one fully observed, terminal local chip group, no other
outstanding endpoint work, closed graph construction, and current replay-safe
metadata. Separate per-rank TaskSlots, missing/remote ranks, and still-running
peers require further coordination and are refused. The API does not stop a
live peer. Existing scope checks restrict the Python facade to A5 local L3 with
operator recovery enabled and single-frame endpoints.

Communication scope requires one explicitly bound full-communicator domain on
every participating endpoint, with all communicator producers owned locally.
Use a run-scoped `allocate_domain` context and leave it before requesting replay;
a live reusable Python domain handle is refused. Recovery-enabled allocation
captures `ChipDomainContext.recovery_binding` through a native, optional reply
trailer; ordinary allocation replies stay unchanged. The compiler/application
attaches that binding and exact argument indices before submission, alongside
logical-op metadata. No numeric address search infers bindings.

The transaction keeps ordinary dispatch frozen:

1. INSPECT every member and validate current endpoint identity, failed PTO
   retirement proof, and unchanged communication binding. Host TASK_DONE alone
   grants no DMA stop proof.
2. PREPARE all domains; only matching all-member acknowledgments permit COMMIT.
   The current SDMA provider aborts its private transport streams and rearms in
   place. Context C/E, execution streams, HCCL and windows are preserved.
3. Rearm failed execution substrates, verify persistent identities, and atomically
   rebind the original failed task plus every unlaunched bound consumer. A
   committed domain generation therefore reaches retained B/C before dispatch.
4. Revalidate eligibility and ARM all endpoints, then dispatch only original A
   with fresh dispatch/native-run identities. Original acceptance debt is not
   charged twice. No TaskSlot, dependency or canonical buffer identity changes.
5. All A members succeeding commits the logical attempt and releases original
   successors once. A renewed native failure establishes a fresh ticket and
   retains the same graph/hold; the owner may retry within its budget or GIVE_UP.

Malformed/stale tickets or unmet admission prerequisites raise without choosing
recovery policy. Once a resource transaction starts, any member/phase failure
aborts/quarantines the transaction, queues terminal GIVE_UP and reports original
plus recovery diagnostics. It never resets and retries inside the primitive.
The old ticket cannot dispatch twice or resolve a later attempt.

The default impact covers communication transport; `EXECUTION_ONLY` is available
when the owner has evidence for that narrower scope. Host error numbers do not
select it. URMA/CCU use the same binding/result contract but still need their
transport providers. Cross-host stop/commit, data restoration, compiler-side
automatic annotation and resource migration are outside this local API.

## Retained argument rebinding

`RecoveryBindingPlan` is a host-side mechanism. It consumes successful committed
domain results and returns new argument sets; it never decides replay eligibility,
dispatches a task, resets a device, or releases a recovery hold.

`TaskArgs.add_domain_binding(TaskDomainBinding)` annotates a domain's resources
and the exact argument indices that reference them. `RecoveryDomainBinding`
contains allocation id, generation, device CommContext, local window base/size
and workspace. `RecoveryArgBinding` names `COMM_CONTEXT` or `WINDOW_ADDRESS`
scalar slots, or a `WINDOW_TENSOR` backing with a domain-relative byte offset.
Ordinary scalar data is unchanged even if its bit pattern equals an address.
Annotations must be captured before submission. They remain on the retained
parent TaskSlot and are not serialized into the ordinary mailbox blob.

The owner constructs `RecoveryBindingPlan(recovery_id)` and calls
`add_domain(worker_id, before, committed_result)` for every bound domain/member.
PREPARED, failed, partial-stop, wrong-id and stale-generation results are refused.
PRESERVED retains both generation and resources; REARMED advances exactly one
generation, even if every address stays unchanged. `rebind(args, worker_id)`
returns a copy, preserving tensor views, argument tags and explicit dependencies.
Repeated application to the matching new generation is safe.

`Orchestrator.rebind_task_args(task_handle, plan)` applies these copies atomically
to the existing local chip TaskSlot in the current RETRY_PENDING episode. Every
group member validates before any member changes; executing peers are refused.
TaskSlot/run identity, callable, pipeline lease, TensorMap keys, fanin/fanout and
the recovery hold stay intact. A new scalar address does not relabel the DAG.
This API is for the recovery owner; it is not callable on an ordinary READY or
RUNNING task and does not connect automatic Scheduler recovery dispatch.

If a window address changes, wire tensors require an owner-registered replacement
descriptor supplied through `add_tensor_replacement`. The descriptor must retain
the view and access permissions but carry a fresh canonical buffer identity, so
an old import-cache mapping cannot silently stand for a new address. The owner
owns registry publication and old-buffer lifetime; the plan does neither.
The retained-TaskSlot API refuses a changed canonical backing identity: publishing
its graph aliases and rebinding downstream consumers is not implemented yet.
Standalone copy/materialization can describe relocated resources, but must not
be mistaken for a complete retained-graph migration transaction.
The current SDMA provider preserves addresses and needs no replacement descriptor.

`rebind_materialized(chip_args, domain_bindings, worker_id)` is the same mechanism
for already-materialized local L2 arguments. It returns `MaterializedRecoveryBinding`
with copied arguments and updated binding metadata. The original retired native
token remains invalid; a later external dispatch acquires a fresh native lease.
The CPU tests exercise a real retained L3 TaskSlot; the A5 SDMA regression tests
this materialized boundary. The local `retry_recovery` API applies an atomic graph-wide plan to the
failed task and its unlaunched consumers. L4+ binding transport and canonical
backing migration remain unsupported.

The SDMA provider targets the validated A5 CANN runtime. The
[CANN 9.2.0-beta.2 StreamAbort contract](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_1777.html)
does not exclude `DEVICE_USE_ONLY` streams; the A5 restriction concerns
`ACL_STREAM_PERSISTENT`, which this manager does not use. Older SDK contracts
must be checked separately; absence of the Abort symbol returns failure.

SDMA transport recovery preserves window contents. The regression separately
proves fresh SDMA data movement and failed-A / blocked-old-B / new-A / new-B ordering.
Tests use the existing explicit execution-fatal fixture; they do not simulate a
network outage or certify recovery from every physical transport fault.
URMA supports execution-only preservation; its transport recovery is unsupported.
CCU retains an engine/capability boundary. The former binding-rebuild and transport
replacement APIs are removed; shared communication or device scope returns `GIVE_UP`. Other
hardware backends retain their existing comm ABI and return `UNSUPPORTED`.
