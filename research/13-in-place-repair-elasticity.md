# 13 — In-place repair and elasticity for a running training job

Second-sweep file. Question: when a node dies, can the survivors keep running, and can a node be
added, without relaunching every Python process? Builds on `03-rank-configuration.md`,
`04-failure-taxonomy-recovery.md`, `05-networking-interconnects.md`, `01-orchestrators-schedulers.md`.

All findings below were read from source on 2026-11-09 unless marked `(unverified)`.

| Repo | What was read | Revision |
|---|---|---|
| `pytorch/pytorch` | `torch/distributed/distributed_c10d.py`, `torch/csrc/distributed/c10d/{ProcessGroupNCCL.cpp,.hpp,NCCLUtils.*,Backend.hpp}`, `torch/csrc/distributed/c10d/nccl2/*`, `torch/distributed/{run.py,launcher/api.py,elastic/**}` | `main` @ `780bf8ee` (`version.txt` = `2.16.0a0`) and tags `v2.8.0` … `v2.15.0` (`v2.15.1` does not exist) |
| `NVIDIA/nccl` | `src/nccl.h.in`, `src/misc/utils.cc`, `src/init.cc`, `src/transport/*.cc`, `src/graph/paths.cc`, `src/ras/ras_internal.h`, `docs/userguide/source/**` | `master` @ `ad2d192c` ("Merge v2.32.6-1 release into master"), plus `src/nccl.h.in` at 11 tags |
| `NVIDIA/nvidia-resiliency-ext` | `docs/source/**`, `src/nvidia_resiliency_ext/fault_tolerance/config.py`, in-process sources at tags | `main` @ `3e712dd8` (2026-11-05); tags `v0.7.0`, `v0.6.0`, `v0.4.1` |
| `meta-pytorch/torchft` | `README.md`, `torchft/manager.py`, `torchft/process_group.py`, `src/lighthouse.rs` | `main` @ `43e3b6d` (crate version 0.3.1) |
| `pytorch/torchtitan`, `NVIDIA/Megatron-LM`, `meta-pytorch/torchcomms` | greps for FT hooks | `8a33dbf`, `d2b0e35`, `8ecc348` (0.3.0) |

Raw URLs follow the pattern `https://raw.githubusercontent.com/<repo>/<rev>/<path>`.

## Summary

1. **In-place repair exists in stock PyTorch today, in two different forms.**
   `torch.distributed.shrink_group()` (public, since **v2.10.0**) wraps `ncclCommShrink`.
   A second, stronger mechanism, `init_process_group(..., enable_reconfigure=True)` plus the private
   `dist._reconfigure(uuid, handles)`, shipped in **v2.14.0** and gained an NCCL shrink/grow fast
   path in **v2.15.0**. Reconfigure mutates the existing `ProcessGroup` object in place, can both
   remove and add ranks, and never aborts the process on a collective error.
2. **The default `"nccl"` backend changed in v2.15.0.** It now resolves to `ProcessGroupNCCL2`
   (`torch/csrc/distributed/c10d/nccl2/`, "built on the torchcomms engine") unless
   `TORCH_DIST_USE_NCCL2=0`. In v2.14.0 it was opt-in (`TORCH_DIST_USE_NCCL2=1`). The legacy backend
   rejects `enable_reconfigure`. Most `TORCH_NCCL_*` watchdog variables documented in the first
   sweep are not referenced by the new backend.
3. **NCCL primitives by version** (probed from `src/nccl.h.in` per tag): `ncclCommShrink` 2.27,
   `ncclCommRevoke` 2.28.9, `ncclCommGrow` + `ncclCommGetUniqueId` 2.29.2, `ncclCommSuspend/Resume`
   2.29.7. PyTorch 2.15.0 wheels pin `nvidia-nccl-cu13==2.30.7`, so all of them are present.
4. **TorchElastic's static rendezvous is verified and supports unequal `--nproc-per-node`**, but it
   has no membership signalling (`num_nodes_waiting()` returns 0) and no coordinated restart.
5. **NVIDIA's in-process restart is gone**: deprecated in NVRx v0.7.0, removed on `main`. NVRx is now
   "all workers restart when any worker fails" with hot spares; it does not need Slurm but takes
   its ordering hints from Slurm variables.
6. **The closest thing to "replace one rank group without full restart" is Meta's `torchft`**
   (per-step quorum through a "lighthouse", live weight recovery from a healthy peer, already wired
   into torchtitan). Its unit of failure is a replica group, which is exactly the DP dimension.
7. **NCCL host identity** is `hash(hostname + /proc/sys/kernel/random/boot_id)`, overridable with
   the undocumented `NCCL_HOSTID`; shared memory is additionally gated on `/dev/shm` having the same
   `st_dev`. Two pods on one physical host are treated as two nodes unless they share a hostname.

## 1. NCCL: shrink, grow, revoke, abort, RAS

### 1.1 API surface (`src/nccl.h.in`, master = 2.32.6)

```c
#define NCCL_SHRINK_DEFAULT 0x00 /* shrink the parent communicator */
#define NCCL_SHRINK_ABORT   0x01 /* First, terminate ongoing parent operations, and then shrink */
#define NCCL_REVOKE_DEFAULT 0x00 /* reserved for future use; must be 0 */
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags);
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount,
                            ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank,
                          ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
```

Result codes include `ncclRemoteError = 6`, `ncclInProgress = 7`, `ncclTimeout = 8`.
`ncclConfig_t` carries `blocking`, `splitShare` and `shrinkShare` among 22 user fields.

Presence of each symbol in `src/nccl.h.in` by tag (fetched individually):

| Tag | `ncclCommShrink` | `ncclCommRevoke` | `ncclCommGrow` | `ncclCommSuspend` |
|---|---|---|---|---|
| v2.26.6-1 | no | no | no | no |
| v2.27.3-1, v2.27.7-1, v2.28.3-1 | yes | no | no | no |
| v2.28.9-1 | yes | yes | no | no |
| v2.29.2-1 | yes | yes | yes | no |
| v2.29.7-1 and later (2.30.3, 2.30.7, 2.31.2, 2.32.3) | yes | yes | yes | yes |

### 1.2 Semantics and limits

From the header comments, `docs/userguide/source/usage/communicators.rst` and `api/comms.rst`:

- **Abort**: "Frees resources associated with communicator object and aborts any operations that
  might still be running on the device." The Fault Tolerance section requires `config.blocking = 0`
  and "no thread is calling any NCCL operations while calling ncclCommAbort"; with blocking
  communicators "the thread can possibly get stuck inside NCCL calls due to network errors … might
  hang forever". "When any rank in a communicator fails … all other ranks need to call
  ncclCommAbort." Detection is left to the application (`checkTimeout()` and
  `reportErrorGlobally()` in the doc example are user-supplied).
- **Revoke**: "stops all in-flight operations and marks communicator state as ncclInProgress … then
  management operations (destroy, split, shrink) can proceed safely." New collectives on a revoked
  communicator return `ncclInvalidUsage`; `ncclCommFinalize` after revoke is invalid;
  `splitShare`/`shrinkShare` resource sharing is disabled while revoked.
- **Shrink**: only survivors call it; excluded ranks must not; "ranks will be re-ordered to fill the
  gap of removed ones" (contiguous renumbering, survivors keep relative order).
  `NCCL_SHRINK_DEFAULT` is for a healthy parent; `NCCL_SHRINK_ABORT` is "when shrinking after an
  error … also useful when there might be ongoing operations on the parent". `config == NULL`
  inherits the parent's configuration.
- **Grow**: three call shapes: existing non-root `(comm, NULL, -1)`, existing root
  `(comm, &id, -1)`, new rank `(NULL, &id, assignedRank)`. "Existing ranks retain their original
  rank numbers … New ranks must be assigned ranks starting from the size of the parent." The grow id
  comes from `ncclCommGetUniqueId` on one existing rank and travels out of band; it is single-use and
  a new one cannot be generated "while a previous UID is unconsumed". "There should not be any
  outstanding NCCL operations on the parent communicator when calling ncclCommGrow to avoid
  potential deadlocks." Existing ranks destroy the parent afterwards.
- Nothing in the docs or header restricts shrink/grow to IB; nothing states that they were tested on
  the socket transport either.

What NCCL does **not** do: decide which rank is dead, restore tensors corrupted by a half-finished
collective, or re-create child communicators (every `ncclCommSplit` child must be rebuilt).

### 1.3 RAS

`docs/userguide/source/troubleshooting/ras.rst` and `src/ras/ras_internal.h`:

- One RAS thread per process, plain TCP keep-alives; clients connect to `localhost:28028`
  (`NCCL_RAS_ADDR`); `ncclras` takes `-h -p -v -t`, `-f json` (since 2.28.7), `-m` monitor mode
  (since 2.29, wire command `MONITOR`), `-D` diagnostics. Monitor mode emits events such as
  `PEER_DEAD: Process 1524345 on node 172.16.64.245 managing GPU 1`, also as JSON.
- Timeouts, all multiplied by `NCCL_RAS_TIMEOUT_FACTOR`: `RAS_KEEPALIVE_INTERVAL` 1 s,
  `RAS_KEEPALIVE_TIMEOUT_WARN` 5 s, `RAS_STUCK_TIMEOUT` 20 s (= `RAS_KEEPALIVE_TIMEOUT_ERROR`),
  `RAS_PEER_DEAD_TIMEOUT` 60 s, `RAS_IDLE_TIMEOUT` 60 s.
- There is still **no RAS C API**: `src/nccl.h.in` at 2.32.6 contains no `ncclRas*` symbol. The
  text/JSON socket is the only interface. Because the listener is on localhost, a per-node agent has
  to query it; with defaults a dead peer is declared after about 60 s, which is slower than a
  control-plane heartbeat.

## 2. PyTorch

### 2.1 `shrink_group` (public, v2.10.0+)

`def shrink_group(ranks_to_exclude, group=None, shrink_flags=SHRINK_DEFAULT, pg_options=None) -> ProcessGroup`
is in `__all__`; `SHRINK_DEFAULT = 0x00`, `SHRINK_ABORT = 0x01`. Present at tag `v2.10.0`, absent at
`v2.9.0`. Behaviour read from the function and its helpers:

- Backend must report `supports_shrinking`; `ProcessGroupNCCL::supportsShrinking()` is true when
  compiled with `NCCL_HAS_COMM_SHRINK` (`NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)`), and
  `ProcessGroupNCCL::shrink` re-checks the runtime version.
- It returns a **new** `ProcessGroup`. If the default group is shrunk the new one becomes the
  default and "Shrinking the default group destroys all other process groups since rank reassignment
  makes them inconsistent" (`_destroy_all_other_groups`). Any object holding the old group (DDP,
  FSDP, a DeviceMesh, Megatron's group globals) is left with a dead handle.
- Group size must be at least 2; an excluded rank calling it raises `RuntimeError`.
- With `SHRINK_DEFAULT` the nccl2 implementation requires an empty work queue ("NCCL communicator
  cannot shrink while work is pending"). With `SHRINK_ABORT` the parent is aborted through
  `_abort_process_group`.
- Docstring hazard: "For NCCL, destruction is an intra-node collective; mixing abort and destroy on
  the same communicator can hang with NCCL >= 2.31.2." Excluded ranks that are still alive must also
  abort.

### 2.2 Reconfigure (private, v2.14.0+; shrink/grow fast path v2.15.0+)

Python API (`distributed_c10d.py`): `init_process_group(..., enable_reconfigure=True)` ("The
communicator is not initialized until `ProcessGroup.reconfigure` is called. Backends that do not
support reconfigure ignore this flag"), `_supports_reconfigure(group)`,
`_get_reconfigure_handle(group) -> str`, and
`_reconfigure(uuid: int, handles: set[str] | list[str], group=None, timeout=None, hints=None) -> Work`.
All three are marked "experimental and subject to change or removal". "A list assigns ranks by
position; a set lets the backend choose the rank assignment." `uuid` must be fresh on every call.
`Backend.hpp` defines `C10D_BACKEND_HAS_RECONFIGURE 1`.

Backend support: `ProcessGroupGloo` (`gloo/ProcessGroupGlooReconfigure.cpp`) and the nccl2 backend
(`nccl2/ReconfigureNCCL.cpp`, 303 lines at v2.14.0, 582 at v2.15.0). The legacy backend has
`TORCH_CHECK(!options_->enable_reconfigure, "ProcessGroupNCCL does not support enable_reconfigure (reconfigure-based fault tolerance).")`.

What `c10d::nccl2::ProcessGroupNCCL::reconfigure` does (main):

1. The handle is `"<backend>:<rank>:<reconfigure_uuid>:<instance_id>:<tcpstore host:port>"`. The
   local handle must be in the list; its index becomes the new rank; list length is the new size.
2. `stopWatchdog()`, then `commRevoke` if work is pending, drain the work queue, and `commAbort` the
   old communicator (unless it will be shrunk).
3. Default path ("init"): rank 0 claims `nccl2_reconfigure/<uuid>/claimed` with `compareSet`,
   publishes a fresh `ncclUniqueId` in the store; everyone calls `commInitRankConfig` with
   `config.blocking = 0` and polls until `timeout`. This is a full communicator re-bootstrap, but
   inside the existing process: CUDA context, model, optimizer state and Python stay up.
4. Fast path, only with **`TORCH_NCCL2_RECONFIGURE_SHRINK_GROW=1` "set consistently on every
   rank"** and compiled against NCCL >= 2.29.0: `findReconfigureQuorum` picks the largest set of
   handles sharing a previous communicator uuid. It is used only if that quorum has at least 2 ranks
   and forms the leading entries of the handle list in ascending old-rank order. Quorum members run
   `commShrink(..., NCCL_SHRINK_ABORT)` then `commGrow`; joiners wait for `grow_unique_id/<rank-0
   handle>` in the store and call `commGrow(nullptr, newSize, &id, newRank, …)`. Log line:
   `path=init | shrink_grow | grow_join`.
5. In reconfigurable mode the process is never aborted: `abortProcess()` returns early when
   `options_c10d_->enable_reconfigure` is set; timeouts and async errors call `revokeNcclComm()` and
   throw (`"NCCL operation timed out"` / `NCCLException`) to the caller.

The contract for the caller (that is, grid): on an exception from a collective, collect one handle
per surviving or joining process out of band, agree on one ordered list and one fresh uuid, call
`_reconfigure` on every member. No in-tree code does this; the only caller found is `torchft`
(`Manager._reconfigure_pg`, §5).

### 2.3 Watchdog and timeouts

Legacy backend (`ProcessGroupNCCL.cpp/.hpp`, identical defaults at v2.15.0 and main):

| Variable | Default | Effect |
|---|---|---|
| `kProcessGroupNCCLDefaultTimeout` | 10 min | per-collective timeout (the `timeout=` of `init_process_group`) |
| `kWatchdogThreadSleepMillis` | 100 ms | watchdog poll |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `3` (`SkipCleanUp`) | `0 NoHandling`, `1 TearDown`, `2 CleanUpOnly`, `3 SkipCleanUp` |
| `TORCH_NCCL_ENABLE_MONITORING` | `true` | heartbeat monitor thread |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `480` | watchdog stuck this long: `terminateProcess()` (`LOG(FATAL)` then `std::abort()`) |
| `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` | `15000` | watchdog sleeps 4x this (60 s) for flight-recorder dumps before acting |
| `TORCH_NCCL_COORD_CHECK_MILSEC` | `1000` | poll for cross-rank dump signal |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | `true` | |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | `2000` | |
| `TORCH_NCCL_RETHROW_CUDA_ERRORS` | `true` | rethrow from watchdog thread (kills the process) |
| `TORCH_NCCL_TEARDOWN_ON_TIMEOUT` | `false` | timeouts tear down even when rethrow is off |
| `TORCH_NCCL_PROPAGATE_ERROR` | `false` | broadcast `remote_error:<pg_uid>` through the TCPStore |
| `TORCH_NCCL_BLOCKING_WAIT` | `false` | |
| `TORCH_NCCL_USE_COMM_NONBLOCKING` | unset (blocking) | auto-nonblocking was disabled in 2.7.1 for an NCCL 2.26 hang |
| `TORCH_NCCL_NONBLOCKING_TIMEOUT` | `1800` s | wait bound for nonblocking init |

On timeout or error the legacy watchdog broadcasts the dump signal, sleeps 60 s, aborts
communicators only if `SHOULD_CLEAN_UP` (modes 1, 2; not the default 3), then
`work.handleException()` rethrows when `SHOULD_TEAR_DOWN` (modes 1, 3): "To avoid data
inconsistency, we are taking the entire process down." So with defaults a peer death surfaces as a
process abort up to 10 minutes later if no NCCL error fires first. `_abort_process_group` documents
that it "should be used with `TORCH_NCCL_ASYNC_ERROR_HANDLING` turned off (i.e. set to 0)".

nccl2 backend (`nccl2/ProcessGroupNCCLUtils.cpp`, `ProcessGroupNCCL.cpp`): one `timeoutWatchdog`
thread polling every 1 s; it checks the work queue and `commGetAsyncError`. On failure,
`handleWatchdogFailure`: reconfigurable groups revoke and return; otherwise abort the communicator
if `SHOULD_CLEAN_UP`, then `abortProcess()`, which sleeps `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`
(15 s) and calls `::abort()` if `SHOULD_TEAR_DOWN`. The only `TORCH_NCCL_*` names referenced in
`nccl2/` are `ASYNC_ERROR_HANDLING`, `BLOCKING_WAIT`, `CUDA_EVENT_CACHE`, `ENABLE_TIMING`,
`HIGH_PRIORITY`, `RANKS_PER_ROOT`, `USE_COMM_NONBLOCKING`, `WAIT_TIMEOUT_DUMP_MILSEC`, plus
`TORCH_NCCL2_RECONFIGURE_SHRINK_GROW`. There is no heartbeat monitor and no
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `ENABLE_MONITORING`, `RETHROW_CUDA_ERRORS` or `PROPAGATE_ERROR`
there. Python's `_get_default_timeout` applies `default_pg_nccl_timeout` to `nccl`, `nccl-legacy`,
`nccl2` and `nccl-lazy` (binding to the 10-minute constant not re-read, `(unverified)`).

### 2.4 Which NCCL the wheels bundle

`.ci/docker/ci_commit_pins/nccl-cu12.txt` (to 2.10) and `nccl.txt` (2.11+):

| PyTorch tag | NCCL pin | `shrink_group` | reconfigure |
|---|---|---|---|
| v2.8.0 | v2.27.3-1 | no | no |
| v2.9.0 | v2.27.5-1 | no | no |
| v2.10.0 | v2.27.5-1 | yes | no |
| v2.11.0 | v2.28.9-1 | yes | no |
| v2.12.0, v2.13.0 | v2.29.7-1 | yes | no |
| v2.14.0 | v2.30.7-1 | yes | yes (nccl2 opt-in, init path only) |
| v2.15.0 | v2.30.7-1 (`nvidia-nccl-cu13==2.30.7`) | yes | yes (nccl2 default, shrink/grow opt-in) |
| main (2.16.0a0) | v2.32.3-1 (`nvidia-nccl-cu13==2.32.3`) | yes | yes |

## 3. TorchElastic (v2.15.0 sources)

- Flags, all with `PET_<DEST>` env fallbacks (`argparse_util.env`): `--rdzv-backend` default
  **`static`**; `--node-rank` default 0, "only used for static rendezvous"; `--master-addr` default
  `127.0.0.1`; `--master-port` default `None`, resolved to `29500`; `--max-restarts` default 0;
  `--monitor-interval` default 0.1. With `static`, no endpoint, no master port and `max_nodes == 1`,
  `run()` silently switches to `c10d` on `localhost:0`.
- `StaticTCPRendezvous.next_rendezvous()`: creates one `TCPStore(master_addr, master_port,
  world_size, is_master=(rank == 0), timeout, multi_tenant=True)` with `world_size =
  params.max_nodes`, returns `RendezvousInfo(PrefixStore(run_id, store), rank, world_size, …)`.
  `num_nodes_waiting()` returns 0, `is_closed()` False, `use_agent_store` True. So there is no
  negotiation and no way to signal joins or departures.
- Unequal `--nproc-per-node`: `SimpleElasticAgent._assign_worker_ranks` is backend-agnostic. Unless
  `TORCH_ELASTIC_WORKER_IDENTICAL=1`, each agent writes `torchelastic/role_info/<group_rank>`; the
  `group_rank == 0` agent `multi_get`s all of them, computes cumulative base ranks and writes
  `torchelastic/assigned_ranks/<i>` as `[base_global_rank, global_size, role_rank, role_size]`. The
  static path therefore supports heterogeneous GPU counts by construction (confirmed by reading; not
  run).
- Dynamic rank assignment at **v2.15.0** is `for rank, node in enumerate(sorted(state.participants))`
  over `_NodeDesc(addr, pid, local_id)` with dataclass ordering, i.e. plain lexicographic on the
  address. The `_natural_sort_key` variant exists only on `main`. Defaults: `join` 600 s,
  `last_call` 30 s, `close` 30 s, `heartbeat` 5 s; `keep_alive_interval=5`,
  `keep_alive_max_attempt=3`.
- Restart flow (`_invoke_run`): poll `_monitor_workers` every `monitor_interval`. On
  `UNHEALTHY`/`FAILED` with budget left: `_remaining_restarts -= 1`, then `_restart_workers` =
  `_stop_workers` (SIGTERM, then SIGKILL after `shutdown_timeout`, default
  `TORCH_ELASTIC_SHUTDOWN_TIMEOUT` = 30 s), `_rendezvous`, `_start_workers`. On `HEALTHY` with
  `num_nodes_waiting() > 0`: `_restart_workers` **without** decrementing ("membership changes do
  not count as retries"). `_exit_barrier` default timeout 300 s.
- Consequence for static mode (inference from the code, not tested): a restart is purely local.
  Other nodes only restart when their own workers die of the resulting NCCL error, which with the
  defaults in §2.3 can take minutes; the rank-0 agent re-reads stale `role_info` keys without
  waiting for peers; and if node rank 0 dies the store is gone for everyone.

## 4. NVIDIA Resiliency Extension

### 4.1 In-process restart: how it worked, and its status

Status: v0.7.0 notes say "In-process restart remains deprecated"; `main` (Unreleased) says
"**In-process restart has been removed**, including the `nvidia_resiliency_ext.inprocess` package,
its documentation, examples, and tests. Use InJob restart through `ft_launcher`." Megatron-LM `main`
still has `megatron/training/inprocess_restart.py` and `--inprocess-*` flags that import that
package, so it only works with NVRx <= 0.7.0.

Mechanism (v0.7.0 `docs/source/inprocess/usage_guide.rst`, `inprocess/wrap.py`, `inprocess/abort.py`):

- `inprocess.Wrapper(...)(train)` re-invokes the training function in the same process. Loop:
  `initial_barrier → rank_assignment → [initialize → health_check → fn() → completion_barrier]`; on
  exception `abort → finalize → health_check → iteration_barrier → rank_assignment` and loop.
- "Re-forming the world" = `AbortTorchDistributed`: `backend.abort()` on every NCCL process group
  (`_shutdown()` before torch 2.6) in threads, then `destroy_process_group()`. The wrapper rewrites
  `RANK`/`WORLD_SIZE`, and the user function calls `init_process_group()` again. Dead ranks are
  dropped; inactive (reserve) ranks sleep inside the wrapper and are promoted by `rank_assignment`
  policies (`ActivateAllRanks`, `ShiftRanks`, `MaxActiveWorldSize`, `ActiveWorldSizeDivisibleBy`,
  `FilterCountGroupedByKey`, `Layer(..., LayerFlag.RESERVE)`).
- Defaults: `soft_timeout` 60 s (async exception raised in the main thread), `hard_timeout` 90 s
  (SIGKILL; the rank does not rejoin), `heartbeat_timeout` 30 s, `barrier_timeout` 120 s,
  `completion_timeout` 120 s, `last_call_wait` 1 s, `termination_grace_time` 5 s, monitor intervals
  1 s. Published latency: `destroy_process_group` "~0.5s + 0.01s * num pending NCCL kernels",
  TCPStore barrier "0.5s @ 16k ranks", rank assignment ~0.5 s.
- Requirements: PyTorch >= 2.5.1 and NCCL >= 2.26.2; the function must be re-entrant and must not
  swallow `BaseException`; code waiting on NCCL must release the GIL; state must be reloaded from a
  checkpoint; the PyTorch watchdog must not kill the process (`TORCH_NCCL_RETHROW_CUDA_ERRORS=0`,
  process-group `timeout` longer than `hard_timeout`); the scheduler must not kill the job when one
  rank exits (`srun --kill-on-bad-exit=0`, `wait_daemon.py`); "not fully compatible with
  `torch.distributed.run`" because torchrun kills all local workers when one fails; rank 0's node
  hosts the TCPStore and its loss ends the job. Slurm is one supported launcher, not a requirement.

### 4.2 `ft_launcher` (InJob restart), `main`

- One `ft_launcher` per node; it spawns the workers and one rank-monitor process per worker (Unix
  socket IPC). "All workers restart when any worker fails." Flow: terminate group → health checks
  and barrier rendezvous → "assign healthy active nodes (promote spares if available)" → relaunch →
  workload reloads the checkpoint.
- Slurm is not required: "`ft_launcher` does not allocate or cancel Slurm jobs." The example uses
  `--rdzv-backend=c10d --rdzv-endpoint=…`. Ordering sources in precedence order:
  `NVRX_INFRA_RANK_FROM_NODENAME`, `CROSS_SLURM_PROCID`, `SLURM_TOPOLOGY_ADDR_PATTERN=block.node`,
  `SLURM_PROCID`, then "`GROUP_RANK`, then deterministic sorted node descriptors as fallback".
  Replacement groups default to `SLURM_ARRAY_TASK_ID` but accept `NVRX_REPLACEMENT_GROUP_ID`.
- Hot spares: `--nnodes MIN:MAX` means `MIN` active nodes and up to `MAX - MIN` joined standbys
  with no workers ("`6:8` means 48 active workers, not a variable 48-to-64-worker model"). "Spares
  replace nodes without changing the active world size; they do not implement elastic Megatron
  parallelism." `--ft-segment N` needs GPU ClusterUUID (NVLink domains); omit it elsewhere.
- Timeouts: `--max-restarts` defaults to 0. "Neither heartbeat nor section hang detection has
  finite timeouts by default" (`rank_heartbeat_timeout`, `initial_rank_heartbeat_timeout`,
  `rank_out_of_section_timeout` default `None`). Other defaults in `FaultToleranceConfig`:
  `workload_check_interval` 5.0, `node_health_check_interval` 5.0, `restart_check_interval` 60.0,
  `safety_factor` 5.0, `rank_termination_signal` SIGKILL, `gpu_memory_reclaim_timeout` 50.0.
- Store host: "Loss of the colocated rendezvous host ends that generation." An experimental
  `nvrx-control` command runs "the long-lived TCPStore and control services outside compute
  launchers" (`is_host=false` on compute nodes); "not yet QA-qualified".
- Packaging: wheels target `manylinux_2_39` (not stock Ubuntu 22.04) and Python 3.12/3.14.

## 5. Other elastic precedents

- **torchft** (`meta-pytorch/torchft`, "Easy Per Step Fault Tolerance for PyTorch", torch >= 2.7).
  A Rust "lighthouse" (`--bind [::]:29510`, `--min_replicas`, `--join_timeout_ms 60000`,
  `--quorum_tick_ms 100`, `--heartbeat_timeout_ms 5000`) computes a quorum each step from manager
  heartbeats (`heartbeat_interval` 100 ms). `Manager(pg, load_state_dict, state_dict,
  min_replica_size, timeout=60s, quorum_timeout=60s, connect_timeout=60s,
  world_size_mode=WorldSizeMode.DYNAMIC, checkpoint_transport=…)`; env overrides
  `TORCHFT_TIMEOUT_SEC`, `TORCHFT_QUORUM_TIMEOUT_SEC`, `TORCHFT_CONNECT_TIMEOUT_SEC`,
  `TORCHFT_QUORUM_RETRIES`, `TORCHFT_LIGHTHOUSE`. `WorldSizeMode.DYNAMIC` normalises gradients by
  the live replica count; `FIXED_WITH_SPARES` keeps `min_replica_size` active and spares "contribute
  zero gradients". A step is applied only if `should_commit()` succeeds. A recovering or new replica
  pulls weights from a healthy peer through a checkpoint transport (`http_transport.py`,
  `pg_transport.py`). Cross-replica communicators are either `torchft.ProcessGroupNCCL` (abort and
  re-create, "NCCL >= 2.26, torch >= 2.7"; "may result in deadlocks"), `ProcessGroupBabyNCCL` (NCCL
  in a subprocess, about 1 GB extra CUDA context, no overlap of compute and network kernels), or a
  native group via `Manager._reconfigure_pg` → `pg.reconfigure(opts)`. Each replica group is its own
  job with its own inner world, so failure of one group restarts only that group.
- **torchtitan**: `torchtitan/experiments/torchft/` (trainer, manager, optimizer, DiLoCo). The
  README runs each HSDP replica group as a separate torchtitan instance with
  `fault_tolerance.group_size` and `fault_tolerance.replica_id`; "the alive replica group with the
  smallest replica ID will perform checkpointing saving". It is under `experiments/`. No use of
  `shrink_group` or `_reconfigure` in the main trainer.
- **Megatron-LM**: `ft_integration` hooks for NVRx sections plus the (now orphaned) in-process
  wrapper. No shrink, reconfigure or world-size change support found by grep.
- **Ray Train** ([elastic training guide](https://docs.ray.io/en/latest/train/user-guides/elastic-training.html)):
  `ScalingConfig(num_workers=(min, max), elastic_resize_monitor_interval_s=…)` (default 60 s) with
  `FailureConfig(max_failures=…)`. The page says the controller will "resize the worker group and
  restart with the new workers added"; it shows `TorchTrainer` and TPU `JaxTrainer` examples. Still
  a worker-group restart.
- **Horovod elastic** ([docs](https://horovod.readthedocs.io/en/stable/elastic_include.html),
  requires Horovod >= 0.20 with Gloo): `@hvd.elastic.run`, `state.commit()` keeps an in-memory
  copy, `HorovodInternalError` rolls back to it, `HostsUpdatedInterrupt` handles graceful changes
  without rollback; reset "reinitialize[s] Horovod context performing a new round of rendezvous" and
  syncs state from the new worker 0. Flags `--min-np`, `--max-np`, `--host-discovery-script`,
  `--slots`, `--blacklist-cooldown-range`; `HOROVOD_ELASTIC_TIMEOUT` default 600 s. This is the
  oldest in-process reset design and the model for torchft's commit/rollback. Whether surviving
  processes are always kept alive is not stated explicitly on the page.
- **MPI ULFM**: not re-read this sweep; see `04-failure-taxonomy-recovery.md` §4.8. NCCL's
  revoke/shrink pair now mirrors `MPIX_Comm_revoke`/`MPIX_Comm_shrink`, but there is no NCCL
  equivalent of `MPIX_Comm_agree`; agreement must come from the control plane.
- **NCCL checkpoint shim** (`contrib/nccl_checkpoint`): `LD_PRELOAD` library that destroys and
  replays all communicators around CUDA Checkpoint + CRIU, using a temporary Redis for rendezvous
  "because it is useful to restore on different hardware". A whole-job migrate path, not repair.

## 6. NCCL host identity in containers

`src/misc/utils.cc`, `getHostHashOnce()`: the id string is `gethostname()` concatenated with the
first token of `/proc/sys/kernel/random/boot_id`; if `NCCL_HOSTID` is set, it **replaces the whole
string** (`INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", …)`). `NCCL_HOSTID` does not
appear in `docs/userguide/source/env.rst`. `getPidHash()` hashes the pid plus
`readlink("/proc/self/ns/pid")`. In `src/init.cc`, `info->hostHash = getHostHash() + commHash`, and
`info->shmDev = stat("/dev/shm").st_dev`, commented "so we can … decide whether we can use SHM for
inter-process communication in a container environment".

How it is used:

- `transport/shm.cc` `canConnect`: SHM only if `hostHash` equal **and** `shmDev` equal, and
  `NCCL_SHM_DISABLE` is not set.
- `graph/paths.cc` `ncclTopoCheckP2p`: different `hostHash` → no P2P (unless MNNVL); same hash but
  different `shmDev` → no P2P. Comment: "Rule out different nodes / isolated containers".
- `transport/net.cc`: the network transport is always allowed across hosts; on the same host only
  if intra-node net is permitted.
- `init.cc`: `hostHash` equality defines local ranks, node count, and the check that returns
  `ncclInvalidUsage` with "Multiple Ranks are using the same GPU/Partition. Set
  NCCL_MULTI_RANK_GPU_ENABLE=1" when two ranks on the "same host" have the same GPU UUID.

Consequences:

- Two pods on one physical host with different hostnames get different hashes and are treated as
  two nodes: traffic goes over the net transport even though NVLink/PCIe P2P is physically there.
  Safe but slow.
- Two pods on one host that share a hostname but have private `/dev/shm` mounts: same hash,
  different `shmDev`, so neither SHM nor P2P is used, but NCCL still counts them as one node for
  topology. Whether the net transport is then used correctly is untested.
- Pods on different machines with identical hostnames are distinguished by `boot_id` (assuming the
  container sees the host kernel's value, `(unverified)`).
- Setting `NCCL_HOSTID` to one value for the whole job makes every rank "same host". `st_dev` of a
  tmpfs is a small anonymous device number that can coincide across machines, so NCCL may then
  attempt SHM/P2P between machines. `NCCL_HOSTID` must be unique per physical host (or per pod).

## Corrections to first sweep

- **`04-failure-taxonomy-recovery.md` §4.8**: "No evidence was found that PyTorch `ProcessGroupNCCL`
  uses it yet (unverified), so in practice PyTorch jobs still restart." Wrong.
  `torch.distributed.shrink_group` has wrapped `ncclCommShrink` since v2.10.0, and v2.14/2.15 add
  reconfigure with `ncclCommRevoke`/`ncclCommGrow`.
- **`04` §4.3**: "Every node must run the same `LOCAL_WORLD_SIZE`, which is a hard limit for
  heterogeneous clusters." Wrong (and contradicted by `03` §2.2): the default slow path of
  `_assign_worker_ranks` supports unequal local world sizes on every backend including `static`.
- **`04` §4.3**: "Scale events and failures share the same restart budget." Wrong: `_invoke_run`
  decrements `_remaining_restarts` only for `UNHEALTHY`/`FAILED`; membership-change restarts do not
  count.
- **`04` §4.4 and Implication 12**: in-process restart is no longer merely deprecated; it is removed
  on NVRx `main`. "Every mainstream stack restarts all workers on membership change" is outdated:
  PyTorch reconfigure and torchft do not.
- **`04` §3 and Implication 4**: `TORCH_NCCL_ENABLE_MONITORING`, `TORCH_NCCL_DUMP_ON_TIMEOUT` and
  `TORCH_NCCL_TRACE_BUFFER_SIZE` already default to on/2000 in the legacy backend, and none of the
  three is referenced in the nccl2 backend that is default from v2.15.0. "Minutes with defaults" is
  precisely: 10 min collective timeout, 480 s heartbeat, 60 s dump wait.
- **`03-rank-configuration.md` §2.2**: the `_natural_sort_key` rank ordering is `main` only; released
  v2.15.0 uses plain `sorted(state.participants)`. `node10` sorts before `node2` there.
- **`03` §2.1 and Open questions**: `--node-rank`, `--master-addr`, `--master-port` and the `PET_*`
  aliases are now verified (§3). `--rdzv-backend` defaults to `static`, not `c10d`.
- **`03` §5**: `NCCL_HOSTID` is confirmed in source and confirmed absent from the env docs; it
  replaces hostname + boot id rather than adding to them.
- **`05-networking-interconnects.md` §6**: the table omits `ncclCommRevoke` (2.28.9) and
  `ncclCommGrow`/`ncclCommGetUniqueId` (2.29.2). The RAS C API on the Q2 2026 roadmap is not in the
  public header at 2.32.6. NCCL `master` is 2.32.6, not 2.32.3.
- **`05` Open questions / `04` Implication 4**: PyTorch 2.15.0 bundles NCCL 2.30.7, so
  `ncclSetEncryption` and the 2.32 RAS diagnostics are not available in released wheels;
  2.32.3 is pinned only on `main`.

## Implications for grid

Ranked recovery modes grid can realistically offer, most dependable first.

**1. Whole-group restart with fast re-rendezvous (baseline; build first).**
Prerequisites: grid owns rank assignment and the store, not torchrun's dynamic rendezvous. Use
static-style launch (explicit `RANK`/`WORLD_SIZE`/`MASTER_ADDR`, or `torchrun --rdzv-backend=static
--node-rank`) with the store on the grid control node, not on training rank 0. Fail fast: on a
control-plane node-loss signal, SIGTERM every rank immediately instead of waiting for the 10-minute
collective timeout; set a short `timeout=` on `init_process_group`. Pin `TORCH_DIST_USE_NCCL2`
explicitly so the watchdog behaviour is known. Works with any framework, any PyTorch, any transport.
Cost per failure: process start, imports, NCCL init, checkpoint reload.

**2. Hot-spare swap at fixed world size (layer on mode 1).**
Prerequisites: spare nodes preflighted, image pulled, checkpoint reachable, agent joined and idle;
grid's rank map substitutes the spare for the dead node and keeps all other ranks stable (which
neither TorchElastic nor NVRx guarantees). TP/PP/EP shapes are unchanged, so Megatron and torchtitan
work unmodified. NVRx `ft_launcher --nnodes MIN:MAX` is a ready implementation if grid accepts its
packaging limits (glibc 2.39, Python 3.12/3.14) and puts the store off-node (`nvrx-control`,
experimental); otherwise reimplement the launcher-with-no-workers design. Removes provisioning time
from the recovery path, nothing else.

**3. In-process re-form with `enable_reconfigure` (experimental tier; biggest win).**
Prerequisites: PyTorch >= 2.14 (2.15 for it to be the default backend), `backend="nccl"` resolving to
nccl2, `init_process_group(enable_reconfigure=True)`; a grid sidecar that collects
`_get_reconfigure_handle()` from each process, decides membership, and distributes one ordered
handle list plus a fresh uuid; a training loop that catches the collective exception, calls
`_reconfigure(...).wait()`, rebuilds any child groups, reloads or re-syncs state, and re-derives
batch/LR. With the handle list kept at the same length and a spare's handle substituted for the dead
rank, this is a hot-spare swap without relaunching survivors. Private API "subject to change or
removal": pin the PyTorch version and gate behind a flag. The NCCL fast path
(`TORCH_NCCL2_RECONFIGURE_SHRINK_GROW=1`) additionally needs survivors listed first in ascending old
rank order; without it the communicator is re-bootstrapped from scratch, which is still far cheaper
than a process restart. Only useful when model state on survivors is intact, so it needs either a
replica to copy from or a step-level commit protocol.

**4. Shrink-and-continue on the DP dimension.**
Two routes. (a) torchft: run each DP replica group as an independent inner job (own TP/PP/FSDP
world), with grid running the lighthouse and supplying `replica_id`; survivors continue on the next
step, a replacement joins and pulls weights from a peer. Prerequisites: torch >= 2.7, a torchft-aware
training loop (torchtitan's experimental trainer, or the DDP example), global batch semantics chosen
via `WorldSizeMode`. This is the only route found that keeps training through a failure today.
(b) `shrink_group(SHRINK_ABORT)`: PyTorch >= 2.10, NCCL >= 2.27, all groups rebuilt by the framework
afterwards because shrinking the default group destroys every other group. No LLM framework does
this, so route (b) is only for grid's own reference trainer.

Other points:

- **Unit of elasticity is the replica group**, as `04` concluded; the mechanisms now exist to act on
  it. grid's job spec should name the DP dimension and group membership explicitly.
- **Agreement is grid's job.** NCCL and PyTorch provide revoke/shrink/grow but no consensus on who
  is dead. grid's coordinator must issue the membership decision; RAS (`PEER_DEAD`, about 60 s) is a
  secondary signal read by the node agent from `localhost:28028`.
- **Never run with the legacy defaults unattended**: `SkipCleanUp` plus a 10-minute timeout means a
  silent peer death costs up to 10 minutes before the process even aborts.
- **Host identity**: emit `NCCL_HOSTID` only when hostnames are not unique, and make it unique per
  physical host. Record whether co-located pods share `/dev/shm`; if a provider packs several grid
  pods on one host, prefer one pod per host with all GPUs.
- **Version matrix is a product surface.** grid must read `torch.__version__`,
  `torch.cuda.nccl.version()` and the resolved backend at preflight and choose the recovery mode
  from them.

## Still open (needs hands-on experiment)

- Does `ncclCommShrink(NCCL_SHRINK_ABORT)` / `ncclCommGrow` complete over the **socket transport**
  when the excluded peer vanished without closing its TCP connections (the RunPod case)? How long do
  revoke and abort take then?
- End-to-end timing of `_reconfigure` at 64–512 ranks on TCP, init path versus shrink/grow path,
  compared with a full restart; and whether it is stable when the dead rank was rank 0 or hosted the
  TCPStore.
- State validity after an aborted collective: can survivors safely continue from in-memory weights
  if the failed step is discarded (torchft's `should_commit` model), for FSDP/TP where shards are not
  replicated?
- Behaviour of DDP, FSDP2 and `DeviceMesh` after `reconfigure` (same `ProcessGroup` object, changed
  size) and after `shrink_group` (new object); which caches of world size need invalidating.
- nccl2 backend maturity in 2.15.0: flight-recorder coverage, hang detection without the heartbeat
  monitor, and the "mixing abort and destroy … can hang with NCCL >= 2.31.2" case on `main`.
- TorchElastic static mode with unequal `--nproc-per-node`, and the stale `role_info` behaviour on a
  local restart, need an actual run.
- torchft at scale on TCP: lighthouse quorum latency, HTTP/PG weight transfer time for multi-GB
  replicas, `ProcessGroupBabyNCCL` overhead, and whether torchtitan's experimental trainer tracks
  current torchtitan `main`.
- What a RunPod/Vast container actually sees for hostname, `boot_id` and `/dev/shm` `st_dev`, and
  NCCL's behaviour when two pods share a host (net transport over the overlay versus failure).
- `ft_launcher` on bare pods without Slurm variables: rank ordering fallback, standby promotion
  time, and `nvrx-control` reliability.
- PyTorch release dates and whether a 2.15.x patch changes the nccl2 default were not checked (the
  GitHub releases API was not reachable from this session).
