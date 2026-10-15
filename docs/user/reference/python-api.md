# Python API reference

The surface you write against, hand-maintained: what `**config` keys `Worker`
accepts, `CallConfig` defaults, and the argument-order footguns a generator
cannot state. For the complete generated listing of every public symbol and
signature, see the API pages on the
[documentation site](https://www.pypto.ai/simpler/user/reference/api/worker/).
Treat the source as authoritative when the two disagree, and fix this page in the
same change.

`Worker` is available from the package root; the remaining task and callable
types live in `simpler.task_interface`, and the tracing helpers in
`simpler.trace`. All of them resolve on first access, so `import simpler` alone
stays cheap and does not require the `_task_interface` extension.

```python
from simpler import Worker, trace     # or: from simpler.worker import Worker
from simpler.task_interface import (
    ArgDirection, CallConfig, ChipCallable, CoreCallable,
    DataType, TaskArgs, TaskHandle, TensorArgType,
)
from simpler_setup import KernelCompiler, SceneTestCase, scene_test
```

## `Worker`

```python
Worker(level: int, **config)
```

`level` is the only declared parameter; everything else is a keyword collected
into `**config` and validated later. The recognized keys:

| Key | Applies to | Meaning |
| --- | ---------- | ------- |
| `platform` | all | `a2a3`, `a2a3sim`, `a5`, `a5sim` |
| `runtime` | all | `tensormap_and_ringbuffer` or `host_build_graph` |
| `device_id` | L2 | the single chip this worker drives |
| `device_ids` | L3+ | one chip child process per entry |
| `num_sub_workers` | L3+ | host-side Python callables to fork |
| `enable_sdma` | a2a3 | provisions the SDMA workspace; defaults to `False` |
| `heap_ring_size` | all | heap ring sizing |
| `remote_heap_ring_size`, `remote_session_timeout_s` | L4 | remote-session sizing and timeout |

`level` selects the topology: `2` is one chip, `>= 3` is hierarchical. Anything
else raises. Remote-worker and remote-memory calls require `level >= 4`.

### Lifecycle

| Method | Notes |
| ------ | ----- |
| `register(target, *, workers=None) -> CallableHandle` | Accepts a `ChipCallable` or, at L3+, a Python callable. Pre-init registrations enter the startup snapshot; post-init registration installs the callable on eligible live targets before returning |
| `unregister(handle_or_slot)` | Releases a registration |
| `add_worker(worker) -> int` | Attaches a child worker; returns its id |
| `add_remote_worker(spec: RemoteWorkerSpec) -> int` | L4; see the remote-L3 design doc |
| `init(prewarm_config=None)` | Resolves runtime binaries, opens the device, forks children. First place setup errors appear |
| `close()` | Releases the device and reaps children. Put it in a `finally` — a skipped `close()` leaves the device held |

### Memory

| Method | Notes |
| ------ | ----- |
| `malloc(size) -> Buffer` | L2 only; allocates on this worker's chip |
| `alloc_child_tensor(worker_id, shapes, dtype) -> Buffer` | Allocates on an L3 worker's chip child; use `handle.tensor(shapes, dtype)` to name a task argument |
| `free(handle)` | Releases a device `Buffer` returned by either allocation method |
| `copy_to(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | H2D; `dst` is a device `Buffer`, `src` a host `Buffer` from `create_buffer` (at L2, also any torch tensor or writable buffer). `nbytes` defaults to the rest of the host side after `src_offset`, so `copy_to(dst, src)` transfers the whole host backing |
| `copy_from(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | D2H; `dst` is the host `Buffer` (at L2, also any writable buffer). Same defaulting, measured from `dst_offset` on the host side |
| `create_buffer(nbytes) -> Buffer` / `Buffer.close()` | Shared host backing this Worker owns; build a view over `handle.shm.buf`, name it on the wire with `handle.tensor(shapes, dtype)` |
| `remote_malloc` / `remote_free` / `remote_copy_to` / `remote_copy_from` / `remote_export` / `remote_import` / `remote_release_import` | L4 only |

A partial update names the **whole allocation plus an offset** — `copy_to(dev, src,
dst_offset=32, nbytes=16)`. A handle rebuilt at `base + 32` is not an interior view of that
allocation; it is a different canonical identity that names no allocation at all, and is refused.
Both offsets are bounded together with the length against the *registered* extent, so an offset
cannot walk a legal-looking length past the end.

### Execution

| Method | Notes |
| ------ | ----- |
| `run(callable, args=None, config=None) -> None` | Blocks until the run completes |
| `submit(callable, args=None, config=None) -> RunHandle` | Non-blocking |
| `live_domains() -> dict[str, CommDomainHandle]` | Currently allocated communication domains |

`RunHandle` exposes `done() -> bool`, `wait(timeout=None)`, and
`result(timeout=None)`.

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

### Retained local recovery graphs

Local A5 L3 workers may opt into `enable_operator_recovery=True` before `init`.
The default is false, and other platforms/endpoints keep their ordinary failure
handling. This mode intercepts a structured native TASK_FAILURE before terminal
propagation, places the original TaskSlot in RETRY_PENDING, and freezes new
dispatch/admission. The submitted graph, TaskArgs, dependencies, TensorMap keys
and managed-buffer references remain owned by the run. It is not a data snapshot.

For a local chip group, the first fault establishes one episode. Other members
continue reporting terminal outcomes while dispatch stays frozen, including a
healthy member that finishes before or after its peer's fault. This is local
evidence collection, not distributed replay eligibility or L4+ coordination.

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

There is no automatic retry, successful replay resolution or endpoint recreation
in this mode. The old endpoint-reset/rebuild control is retired. Connecting
explicit L2 recovery, rank authorization and fresh dispatch is a later recovery
owner responsibility. Unlaunched downstream tasks remain in the graph during
the hold, and must not be confused with already-retired native execution tokens.

### Retained argument rebinding

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
this materialized boundary. They do not claim automatic L3 replay or L4+ binding
transport is implemented. Step7/8 must coordinate rank eligibility and validate
the attempt envelope when actually dispatching the rebound task.

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


At L2 the `callable` is a registered `ChipCallable` handle. At L3+ the top-level
callable is a **Python orchestration function** `f(orch, args, cfg)`, where
`orch` is the `Orchestrator`:

| Method | Notes |
| ------ | ----- |
| `submit_next_level(callable_handle, args, config=None, *, worker: int) -> TaskHandle` | Hands a callable to one exact NEXT_LEVEL child and returns an opaque handle for explicit task dependencies |
| `submit_next_level_group(callable_handle, args_list, config=None, *, workers: list[int]) -> TaskHandle` | Submits one group DAG node and returns its opaque handle |
| `submit_sub(callable_handle, args=None)` | Schedules a registered host-side Python callable |
| `allocate_domain(*, name, workers, window_size, buffers=())` | Context manager returning a handle indexed by domain-local rank |
| `alloc_child_tensor(worker_id, shapes, dtype) -> Buffer` | Delegates allocation to the owning Worker; target chip memory is named by the returned handle |

## Callables and task args

```python
CoreCallable.build(signature=[ArgDirection...], binary=kernel_bytes)

ChipCallable.build(
    signature=[ArgDirection...],
    func_name="my_orchestration",     # the exported orchestration symbol
    binary=orch_bytes,
    children=[(func_id, core_callable), ...],
)
```

`ArgDirection` is `SCALAR`, `IN`, `OUT`, or `INOUT`. The signature list is
positional and defines the task-arg order. `func_id` must match the id the
orchestration submits. `ChipCallable` exposes `binary_size` and
`scalar_count`. `scalar_count` counts the signature's `SCALAR` entries,
including for callables loaded from cached bytes. The tensor count is
`sig_count - scalar_count`.

Public `Worker` calls use `TaskArgs` containing address-free `Tensor` views at
every level. The L2 leaf resolves those views into the internal
`ChipStorageTaskArgs` / `ChipTensor` representation; callers do not pass that
internal representation to `Worker.run()`.

For L3+ graph construction, `TaskArgs.add_dep(*handles)` adds `WAIT | RETAIN`
edges: each consumer waits for its producers and keeps their task-owned
resources alive until it completes. `TaskArgs.add_dep_wait(*handles)` adds
ordering-only `WAIT` edges. Handles must come from a NEXT_LEVEL submit in the
same orchestration run; they are opaque and cannot be constructed by the
caller.

```python
args = TaskArgs()
args.add_tensor(device_buffer.tensor((rows, cols), DataType.FLOAT32), TensorArgType.INPUT)
```

`device_buffer` is a `Buffer` returned by `malloc` or `alloc_child_tensor`.
Add tensors **in signature order**, before any scalars. Use `INPUT` for an
input, `OUTPUT_EXISTING` for a caller-allocated output, and `INOUT` for an
input/output view. `DataType` carries the element types.

## `CallConfig`

`CallConfig()` defaults are fine for an ordinary run.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `aicpu_thread_num` | `0` | AICPU threads for this run; `0` selects the architecture default |
| `enable_chip_swimlane` | `0` | `0` off; `1`–`4` select detail. L2 and same-host L3; a lower level simply records fewer device streams, and cross-Rank merging works at any of them |
| `enable_dump_args` | `0` | Capture per-task arguments |
| `enable_pmu` | `0` | `0` off; `>0` selects the event type |
| `enable_dep_gen` | `0` | Emit the dependency graph |
| `enable_scope_stats` | `0` | Writes `<output_prefix>/scope_stats/scope_stats.jsonl` |
| `capture_clock_anchors` | `False` | Samples the Host and Device clocks around a run and serializes the pairs into the capture. Set by the ChipWorker child for an L3 chip-swimlane capture; not a caller knob. The offline tools place device records by span containment and no longer read these |
| `output_prefix` | `""` | **Required whenever any diagnostic is enabled** |
| `runtime_env` | — | TRB uses `ring_task_window`, `ring_heap`, and `ring_dep_pool`. HBG uses `ring_task_window[0]` for graph task capacity; its graph heap is sized after orchestration |

`validate()` runs at every submit/run entry point and throws if a diagnostic is
on without `output_prefix`, or if a ring override breaks the ring's constraints.

## `simpler.trace`

Puts your own phases on the same host timeline the runtime's spans land on, on
the same clock and in the same log. One producer, one span, one gate:

| Name | Use |
| ---- | --- |
| `span(name, **attributes)` | A timed region, as a `with` block or a decorator. Emits even when the block raises, and does not swallow the exception. A marker is a span whose body does no work — there is no separate zero-duration call |
| `producer(name)` | A `Producer` with the same `span` / `enabled`, for a library that wants its own name instead of the default `app` |
| `enabled()` | Whether spans are being emitted right now — ask before building attributes that cost more than the span |

```python
from simpler import trace

with trace.span("my_phase", batch=16, layer=3):
    ...
```

Names are prefixed `ext.<producer>.`, so your spans can never be mistaken for
the runtime's own. See
[host trace markers](../../dfx/host-trace.md#emitting-your-own-spans-with-simplertrace)
for what the namespace guarantees in each view, the cost of one attempt, and how
to read the records back.

## `simpler_setup`

Compilation and test scaffolding. Exported from the package root:

| Name | Use |
| ---- | --- |
| `KernelCompiler` | `compile_incore(source_path, core_type, pto_isa_root, extra_include_dirs)`, `compile_orchestration(runtime_name, source_path)`, `get_orchestration_include_dirs(runtime)` |
| `ensure_pto_isa_root()` | Clones/updates the pinned pto-isa checkout and returns its path |
| `extract_text_section(binary)` | Required on hardware platforms before wrapping a kernel `.o` |
| `scene_test(level, runtime)` / `SceneTestCase` | The decorator and base class for declarative examples and tests |
| `TensorArg`, `Scalar`, `TaskArgsBuilder`, `CallableNamespace` | Scene-test arg construction; `TensorArg` is separate from the runtime `Tensor` type |
| `make_chip_tensor_arg`, `torch_dtype_to_datatype` | torch interop |
| `parse_platform` | Platform string parsing |
| `RuntimeBuilder` | Runtime build orchestration |

Analysis CLIs live under `simpler_setup.tools`; see
[command-line flags and tools](cli.md).

## See also

- [How-to: write and run a kernel](../how-to/write-and-run-a-kernel.md)
- [Task Flow](../../task-flow.md) — how these handles travel through the runtime
- [Communication Domains](../../comm-domain.md) — `allocate_domain` semantics
