# 03 — Rank configuration and process launch

Research sweep of 2026-11-09. Versions seen in the docs and source read for this file: PyTorch 2.15 docs
(`docs.pytorch.org/docs/2.15`), PyTorch `main` source, NCCL 2.32.3 user guide, Slurm 26.05 man pages,
Open MPI `main` docs (v6.x), Kubeflow Trainer v2 `master`, Ray `master`, Megatron-LM `main`.
Anything not confirmed against a primary source in this sweep is marked `(unverified)`.

## Summary

- Every launcher ends up giving each training process the same four facts: **my global rank, the world
  size, where the rendezvous point is, and which GPU is mine**. They differ only in who computes them
  (a central launcher, a per-node agent, or the process itself) and in the names of the variables.
- In PyTorch the minimum contract is `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` (the `env://`
  init method) plus `LOCAL_RANK` for device selection. Anything that can set those five variables and
  start one process per GPU can drive DDP, FSDP, Accelerate, Lightning, DeepSpeed, Megatron-LM and
  torchtitan. `torchrun` is one way to produce them, not a requirement.
- LLM frameworks derive all parallelism groups (TP, CP, EP, DP, PP, FSDP shard/replicate) from the
  global rank by **pure arithmetic**. No framework looks at hostnames or topology. The only link
  between "rank" and "hardware" is the order in which the launcher numbers processes, so **rank
  ordering is the placement policy**. Megatron-LM says this outright: "the caller should make sure
  adjacent ranks are on the same DGX box".
- TorchElastic ranks are explicitly unstable across restarts, and they are assigned by sorting node
  addresses, so a replacement node with a different hostname or IP can renumber the whole job.
  Checkpoint formats, data shards and RNG streams that are keyed by rank then break or silently change.
- Nodes with unequal GPU counts are legal at the launcher level (torchrun, MPI hostfiles, Slurm) but
  violate the assumption behind every mesh-style group layout. The orchestrator has to fix this up
  before ranks are numbered.

---

## 1. PyTorch: `init_process_group`, backends, and what a rank is

### 1.1 The contract

`torch.distributed.init_process_group(backend=None, init_method=None, timeout=None, world_size=-1,
rank=-1, store=None, group_name='', pg_options=None, device_id=None, ...)`
([docs](https://docs.pytorch.org/docs/2.15/distributed.html)).

| Init method | What must be supplied | Notes |
|---|---|---|
| `env://` (default) | `MASTER_ADDR` (not needed on rank 0), `MASTER_PORT` (free port on rank 0), `WORLD_SIZE`, `RANK` | `WORLD_SIZE` and `RANK` may be passed as arguments instead |
| `tcp://addr:port` | address of rank 0; `rank` and `world_size` as arguments | "Ranks must be set manually" |
| `file:///path` | shared filesystem with `fcntl` locking; fresh file every run | no automatic rank assignment |
| `store=` | an existing `TCPStore`, plus `rank` and `world_size` | mutually exclusive with `init_method` |

Source for the table: [torch.distributed docs](https://docs.pytorch.org/docs/2.15/distributed.html).

Mechanism: rank 0 hosts a `TCPStore` key-value server on `MASTER_ADDR:MASTER_PORT`; all other ranks
connect to it as clients. The store is used only for bootstrap (exchanging the NCCL unique id, or Gloo
addresses) and barrier-style synchronisation. It is not on the data path. Consequences:

- Rank 0's host must be reachable by every other rank on one TCP port over the control network.
- If the process holding the store dies, every later `new_group` or re-init fails. With torchrun the
  store can instead live in the agent on the rank-0 node (section 2.3), which survives worker restarts.

### 1.2 Backends

| Backend | Use | Rank-relevant behaviour |
|---|---|---|
| `nccl` | CUDA GPUs; default when the device is CUDA (since 2.6 the backend is inferred from `device_id`) | one rank = one CUDA device. Default timeout 10 min; other backends 30 min. Communicator is created lazily at first collective unless `device_id` is passed, in which case it is formed eagerly and subgroups can use `ncclCommSplit` |
| `gloo` | CPU tensors, control-plane collectives | full TCP mesh unless `TORCH_GLOO_LAZY_INIT` is set. Interface chosen with `GLOO_SOCKET_IFNAME` (comma-separated list; every process must list the same count) |
| `mpi` | only when PyTorch is built from source against MPI | rank and size come from `MPI_COMM_WORLD`. No `reduce_scatter`, no `set_timeout`, no `use_local_synchronization` |
| `xccl`, `ucc`, custom (`"cpu:gloo,cuda:custom"`) | Intel XPU and plugins | — |

Source: [torch.distributed docs](https://docs.pytorch.org/docs/2.15/distributed.html).

The same page documents newer fault-tolerance entry points that matter for rank handling:
`torch.distributed.shrink_group(ranks_to_exclude, group=None, shrink_flags=0, ...)` (flags
`SHRINK_DEFAULT`, `SHRINK_ABORT`; "shrinking the default group destroys all other groups"), an
`enable_reconfigure=True` argument to `init_process_group`, and experimental
`_reconfigure(uuid, handles, ...)`. These map onto NCCL's `ncclCommShrink` and `ncclCommGrow`
(section 5). See `04-failure-taxonomy-recovery.md` for the recovery side.

### 1.3 Subgroups

`torch.distributed.new_group(ranks=None, ..., use_local_synchronization=False, sort_ranks=True)`:

- "Every process in the main group must call it, including ranks that won't be members", and groups
  "must be created in the same order on all processes". Group creation is itself a collective. A
  framework that builds 10 to 30 groups (Megatron-LM does) needs all ranks alive and in step at init.
- `sort_ranks=True` by default: the rank inside the group follows ascending global rank unless turned off.
- `use_local_synchronization=True` lets only members participate, but "can deadlock if ranks create
  overlapping groups in different orders" and does not work with MPI.
- `split_group(parent_pg, split_ranks=[[...],[...]])` is the newer form backed by `ncclCommSplit`:
  non-overlapping rank lists, all parent ranks must call it.

Source: [torch.distributed docs](https://docs.pytorch.org/docs/2.15/distributed.html).

---

## 2. torchrun / TorchElastic

### 2.1 Variables handed to each worker

From [torchrun docs](https://docs.pytorch.org/docs/2.15/elastic/run.html):

| Variable | Meaning |
|---|---|
| `LOCAL_RANK` | rank within the local worker group (one node) |
| `RANK` | global rank |
| `GROUP_RANK` | rank of the worker group, 0..`max_nnodes`; with one agent per node this is the node rank |
| `ROLE_RANK` | rank among workers with the same role |
| `LOCAL_WORLD_SIZE` | number of local workers, equals `--nproc-per-node` |
| `WORLD_SIZE` | total workers in the job |
| `ROLE_WORLD_SIZE` | total workers with the same role |
| `MASTER_ADDR` | FQDN of the host running rank 0 |
| `MASTER_PORT` | port on `MASTER_ADDR` hosting the C10d TCP store |
| `TORCHELASTIC_RESTART_COUNT` | number of worker-group restarts so far |
| `TORCHELASTIC_MAX_RESTARTS` | configured maximum |
| `TORCHELASTIC_RUN_ID` | the rendezvous `run_id` (job id) |
| `PYTHON_EXEC` | optional interpreter override |

Flags: `--nnodes N` or `--nnodes MIN:MAX`; `--nproc-per-node {int|gpu|cpu|xpu|auto}`;
`--rdzv-backend`, `--rdzv-endpoint host[:port]` (port defaults to 29400), `--rdzv-id`, `--rdzv-conf`;
`--max-restarts`; `--standalone`; `--local-ranks-filter`; `--numa-binding`; `--redirects`, `--tee`,
`--log-dir`. The static path also takes `--node-rank`, `--master-addr`, `--master-port` (flag names
from `torch/distributed/run.py`; not re-read in this sweep, `(unverified)`).

### 2.2 Rendezvous: how ranks are actually decided

One `LocalElasticAgent` runs per node. It "does not communicate to other local agents deployed on
other hosts"; coordination goes through the rendezvous backend only
([agent docs](https://docs.pytorch.org/docs/2.15/elastic/agent.html)).

`DynamicRendezvousHandler`
([source](https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py)):

- Shared state is a pickled object written with a compare-and-set on a fencing token.
- Default timeouts: `join` 600 s, `last_call` 30 s, `close` 30 s, `heartbeat` 5 s. Keep-alive interval
  5 s with `keep_alive_max_attempt = 3`, so a silent agent is dropped after roughly 15 s. All are
  settable through `--rdzv-conf` (`join_timeout=`, `last_call_timeout=`, `close_timeout=`,
  `keep_alive_interval=`).
- Once `min_nodes` have joined, the rendezvous waits `last_call` for stragglers up to `max_nodes`,
  then completes.
- **Rank assignment**:
  `for rank, node in enumerate(sorted(state.participants, key=_natural_sort_key))`. The key is the
  node address split into digit and non-digit chunks (so `node10` sorts after `node2`), then
  `(addr, pid, local_id)`. Group rank is therefore a function of the *set of hostnames or addresses*,
  not of join order and not of any orchestrator-chosen id.

Backends: `c10d` (state stored in a `TCPStore` on the `--rdzv-endpoint` host; no extra service),
`etcd` and `etcd-v2` (external etcd), `static` (no negotiation; rank from `--node-rank`).

From group rank to global rank, `_assign_worker_ranks`
([source](https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/agent/server/api.py)):

- Fast path, only when `TORCH_ELASTIC_WORKER_IDENTICAL=1`:
  `global_rank = group_rank * local_world_size + local_rank`, no store traffic.
- Slow path (default): each agent publishes its `local_world_size` and role to the store; the
  `group_rank == 0` agent computes each agent's base rank as "the cumulative sum of the
  local_world_size for all workers in front of it" and writes it back. This is what makes
  **unequal `--nproc-per-node` across nodes legal**: nodes with 2 and 4 workers get ranks 0-1 and 2-5.

### 2.3 Restart and membership change

- "If k of n workers fail, all workers stop and restart, up to `max_restarts`." A worker failure is a
  failure of the whole group ([run docs](https://docs.pytorch.org/docs/2.15/elastic/run.html)).
- The agent polls every `monitor_interval` (default 0.1 s). On `HEALTHY` it also checks
  `rdzv_handler.num_nodes_waiting()`; if positive it restarts the workers *without* decrementing the
  restart budget, since "membership changes do not count as retries"
  ([source](https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/agent/server/api.py)).
- Node departure or arrival: "all workers stop, a new worker group forms, and all workers restart with
  new RANK and WORLD_SIZE".
- "RANK is NOT stable." Scripts must not assume a stable `RANK` to `LOCAL_RANK` relationship, and must
  not hardcode `WORLD_SIZE` when `min != max`.
- `MASTER_ADDR` defaults to the hostname of agent rank 0; `MASTER_PORT` to a random free port. The
  rendezvous returns a `bootstrap_store_info`, so the agent-hosted store can be reused by workers
  (disabled by `TORCH_DISABLE_SHARE_RDZV_TCP_STORE`).
- Watchdog hooks: `TORCHELASTIC_ENABLE_FILE_TIMER=1` and `TORCHELASTIC_TIMER_FILE` (named-pipe
  watchdog), `TORCH_ELASTIC_EXIT_BARRIER_TIMEOUT`.
- `virtual_local_rank` option: `LOCAL_RANK` is set to 0 and `CUDA_VISIBLE_DEVICES` is rewritten so each
  worker sees its GPU as index 0.

Elasticity here means "kill everyone and re-rendezvous with a different world size". It does not
resize a running job.

---

## 3. Other launchers

### 3.1 MPI (Open MPI / PRRTE / PMIx, MPICH Hydra)

`mpirun` in Open MPI 5 and later is "a thin layer around the PRRTE `prun`"; remote daemons are started
over non-interactive ssh, with Open MPI on `PATH` and `LD_LIBRARY_PATH` on every node
([quickstart](https://docs.open-mpi.org/en/main/launching-apps/quickstart.html)).

| Concept | Detail | Source |
|---|---|---|
| Slot | how many processes may be launched on a host; "actually unrelated" to core count | [scheduling](https://docs.open-mpi.org/en/main/launching-apps/scheduling.html) |
| Hostfile | `node0 slots=2 max_slots=20`; default slots = core count | same |
| `--host` | `a,b:2,c:3`; without `:N` a host gets 1 slot | same |
| Mapping | default `--map-by slot` (fill a node before the next, so adjacent ranks share a host); `--map-by node` is round-robin; also `ppr:N:<object>`, `seq`, `rankfile`, `pe-list` | same, [mpirun(1)](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html) |
| Ranking | `--rank-by slot|node|fill|span` | mpirun(1) |
| Binding | default core for np <= 2, package for np > 2; none if oversubscribed | mpirun(1) |
| Rankfile | `rank 0=aa slot=1:0-2`; relative hosts `rank 0=+n0 slot=...` | mpirun(1) |
| Oversubscription | more processes than slots aborts unless `:OVERSUBSCRIBE` | scheduling |
| Env forwarding | `-x VAR`, `-x VAR=val` | mpirun(1) |
| Failure | any abnormal exit: mpirun "will print out an error message and kill the rest of the MPI application" | mpirun(1), flagged there as older text |

Variables visible in each process
([Open MPI docs](https://docs.open-mpi.org/en/main/tuning-apps/environment-var.html)):
`OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE`, `OMPI_COMM_WORLD_LOCAL_RANK`,
`OMPI_COMM_WORLD_LOCAL_SIZE`, `OMPI_COMM_WORLD_NODE_RANK`, `OMPI_UNIVERSE_SIZE`. PMIx additionally
exports `PMIX_RANK` and `PMIX_NAMESPACE` `(unverified in this sweep)`. MPICH and Intel MPI Hydra set
`PMI_RANK` and `PMI_SIZE`; these are non-standard
([mpich-discuss](https://lists.mpich.org/pipermail/discuss/2019-February/005645.html)).

Open MPI 5 replaced ORTE with PRTE: the old `OMPI_MCA_orte_*` keys are ignored and the equivalents are
`PRTE_MCA_prte_default_hostfile`, `PRTE_MCA_plm_ssh_args`, `PRTE_MCA_prte_keep_fqdn_hostnames`,
`PRTE_MCA_prte_set_default_slots`
([Kubeflow constants](https://github.com/kubeflow/trainer/blob/master/pkg/constants/constants.go)).

For PyTorch LLM training MPI is normally used only as a *process launcher and rank oracle*. DeepSpeed
states it "will still use the torch distributed NCCL backend and not the MPI backend"
([DeepSpeed docs](https://www.deepspeed.ai/getting-started/)).

### 3.2 Slurm `srun`

Per-task variables ([srun(1), 26.05](https://slurm.schedmd.com/srun.html)):

| Variable | Definition |
|---|---|
| `SLURM_PROCID` | "The MPI rank (or relative process ID) of the current process" |
| `SLURM_NODEID` | relative node id |
| `SLURM_LOCALID` | node-local task id |
| `SLURM_NTASKS` / `SLURM_STEP_NUM_TASKS` | total tasks in the job or step |
| `SLURM_TASKS_PER_NODE` / `SLURM_STEP_TASKS_PER_NODE` | per-node counts in nodelist order, e.g. `2(x3),1` |
| `SLURM_JOB_NODELIST` / `SLURM_STEP_NODELIST` | compressed node list |
| `SLURM_GTIDS` | global task ids running on this node |
| `SLURM_STEP_GPUS`, `SLURM_JOB_GPUS` | global GPU ids allocated |
| `SLURM_TOPOLOGY_ADDR` | switch path to the node, only with the topology/tree plugin |

`-m/--distribution` sets rank-to-node order: `block` (default; consecutive tasks share a node),
`cyclic` (round-robin), `plane=<size>`, `arbitrary` (order from `SLURM_HOSTFILE`). `--gpus-per-task`
implies `--tres-bind=gres/gpu:per_task:<n>`. `--mpi=pmix|pmi2|cray_shasta|none`. `-K/--kill-on-bad-exit`
ends the step on any non-zero exit. `--multi-prog` config files take `%t` (task number) and `%o`
(offset in range).

Common mapping: `RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`, `LOCAL_RANK=$SLURM_LOCALID`,
`MASTER_ADDR=` first host of `SLURM_JOB_NODELIST`. Lightning's `SLURMEnvironment` and JAX's
auto-detection do this internally.

### 3.3 Kubeflow Trainer v2 and TF_CONFIG

The Torch plugin launches `torchrun` and injects
([plugin source](https://github.com/kubeflow/trainer/blob/master/pkg/runtime/framework/plugins/torch/torch.go),
[constants](https://github.com/kubeflow/trainer/blob/master/pkg/constants/constants.go),
[ML policy docs](https://trainer.kubeflow.org/en/latest/operator-guides/ml-policy.html)):

| Variable | Value |
|---|---|
| `PET_NNODES` | trainer PodSet count (`numNodes`) |
| `PET_NPROC_PER_NODE` | `numProcPerNode`; `auto` stays `auto` when GPUs are requested, otherwise the CPU count |
| `PET_NODE_RANK` | field reference to the Job completion index annotation |
| `PET_MASTER_ADDR` | `<trainjob>-node-0-0.<trainjob>` (headless-service DNS of pod 0) |
| `PET_MASTER_PORT` | `29500` (`ContainerTrainerPort`) |

`PET_*` are torchrun's own environment aliases for its flags (argparse-with-env-default convention,
`(unverified)` here). Node rank comes from the indexed-Job completion index, so it is stable across
pod restarts: pod index 3 is always node rank 3. The MPI policy writes a hostfile at
`/etc/mpi/hostfile` with `numProcPerNode` slots and launches `mpirun` over ssh.

TensorFlow uses one JSON variable: `TF_CONFIG = {"cluster": {"worker": ["host1:port", ...],
"ps": [...]}, "task": {"type": "worker", "index": 1}}`. The `cluster` part is identical everywhere, the
`task` part differs per process, and worker index 0 is the chief by convention, taking on "saving
checkpoint and writing summary file"
([TF guide](https://www.tensorflow.org/guide/distributed_training#TF_CONFIG)). It is the only
mainstream scheme where every process is told every peer's address up front.

### 3.4 Hugging Face Accelerate

`accelerate launch` wraps `torch.distributed.run`: `multi_gpu_launcher` filters its arguments down to
those torchrun accepts and calls `distrib_run.run(args)`
([source](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/launch.py)).
Multi-node flags: `--num_processes` (total), `--num_machines`, `--machine_rank`, `--main_process_ip`,
`--main_process_port`, `--same_network`, `--gpu_ids`, `--rdzv_backend` (default `static`),
`--rdzv_conf`, `--max_restarts` (default 0), `--monitor_interval` (0.1). The same keys live in
`default_config.yaml`. "It is required that the command be run on all nodes"
([docs](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch)).
Trap for heterogeneous clusters: if `check_cuda_p2p_ib_support()` fails the launcher sets
`NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` unless they are already set.

### 3.5 Lightning Fabric

`fabric run --node-rank=<i> --main-address=<ip> --accelerator=cuda --devices=8 --num-nodes=2 script.py`
on every node; "the only difference between the two commands is the `--node-rank` setting"
([docs](https://lightning.ai/docs/fabric/stable/guide/multi_node/barebones.html)).
`LightningEnvironment` reads `MASTER_ADDR` (default `127.0.0.1`), `MASTER_PORT` (else a free port),
`NODE_RANK` falling back to `GROUP_RANK`, and `LOCAL_RANK`; `creates_processes_externally` is simply
`"LOCAL_RANK" in os.environ`, so when an external launcher sets `LOCAL_RANK` Lightning does not spawn
its own per-GPU children
([source](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/plugins/environments/lightning.py)).
Other environment plugins: `SLURMEnvironment`, `TorchElasticEnvironment`, `KubeflowEnvironment`,
`LSFEnvironment`, `MPIEnvironment`. Global rank is `node_rank * devices + local_rank` `(unverified)`,
which assumes the same device count on every node.

### 3.6 DeepSpeed launcher

`deepspeed --hostfile=hf train.py ...`; hostfile lines are `hostname slots=N`, default path
`/job/hostfile`; filters `-i "worker-0@worker-1:0,2"`, `-e`, `--num_nodes`, `--num_gpus`; backends
`--launcher pdsh|openmpi|mvapich|mpich|impi|slurm`; `--no_ssh` with `--node_rank` for externally
started nodes; `--master_addr` defaults to `ssh <first_host> hostname -I`
([runner.py](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/launcher/runner.py)).
The resource map is passed to each node as base64 JSON `--world_info`. `launch.py` walks nodes in
hostfile order and GPUs in slot order assigning consecutive ranks, then exports `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, `CROSS_RANK` (node rank), `CROSS_SIZE` (node count), `LOCAL_SIZE`, `RANK`,
`LOCAL_RANK`, and pins the GPU through the accelerator's visible-devices variable
([launch.py](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/launcher/launch.py)).
Unequal slots per host are handled. A non-zero child exit triggers SIGTERM of the process tree, a 30 s
wait, then kill; there is no restart. Extra environment is forwarded from `.deepspeed_env` (`DS_ENV_FILE`).

### 3.7 Horovod

`horovodrun -np 16 -H server1:4,server2:4,...` or `-hostfile` (same `slots=N` format as mpirun), over
passwordless ssh, with MPI or Gloo as controller
([docs](https://horovod.readthedocs.io/en/stable/running_include.html)). Scripts pin GPUs with
`hvd.local_rank()`. Elastic mode (`--min-np`, `--max-np`, `--host-discovery-script` printing
`host:slots` lines, `HOROVOD_ELASTIC_TIMEOUT` default 600 s, `--blacklist-cooldown-range`) is the one
launcher that resizes *without* a full process restart: state is committed in memory
(`state.commit()`), a `HorovodInternalError` rolls back and a `HostsUpdatedInterrupt` does not, then
Horovod re-rendezvouses and "older workers are preferred for the worker-0 role" so the broadcast state
is current ([elastic docs](https://horovod.readthedocs.io/en/stable/elastic_include.html)). Its data
guidance is worth copying: keep the partition count fixed at `hvd.max_size()` and redistribute
partitions, rather than re-sharding by the current size.

### 3.8 JAX

`jax.distributed.initialize(coordinator_address=None, num_processes=None, process_id=None,
local_device_ids=None, cluster_detection_method=None, initialization_timeout=300,
heartbeat_timeout_seconds=100, shutdown_timeout_seconds=300, coordinator_bind_address=None,
partition_index=None, ...)`
([docs](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html)). Process 0
starts a coordinator service on `coordinator_address`; ids "must form a dense range from 0 to
`num_processes - 1`". Auto-detection covers Slurm, Open MPI (mpi4py needed for Open MPI 5+) and TPU.
Any dead process (no heartbeat for 100 s) shuts everything down. Unlike PyTorch, the usual JAX
arrangement is one process per *host* owning all local GPUs.

### 3.9 Ray Train

The driver creates worker actors, asks worker 0 for an address and port
(`worker_group.execute_single(0, get_address_and_port)`), exports `MASTER_ADDR`/`MASTER_PORT`, sets
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `NODE_RANK` per worker, then calls
`init_process_group(backend, init_method, rank=i, world_size=len(worker_group),
timeout=timedelta(seconds=timeout_s))` with `timeout_s = 1800` and backend `nccl` when workers have
GPUs. It also sets `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
([source](https://github.com/ray-project/ray/blob/master/python/ray/train/torch/config.py)).
`ray.train.get_context()` exposes `get_world_rank()`, `get_local_rank()`, `get_node_rank()`,
`get_world_size()`, `get_local_world_size()`
([API](https://docs.ray.io/en/latest/train/api/doc/ray.train.context.TrainContext.html)). This is the
"central driver assigns ranks directly, no torchrun" model and is closest to what grid would do.

### 3.10 Cross-launcher vocabulary

| Concept | torchrun | Open MPI | Slurm | DeepSpeed | Kubeflow | Ray Train | JAX |
|---|---|---|---|---|---|---|---|
| global rank | `RANK` | `OMPI_COMM_WORLD_RANK` | `SLURM_PROCID` | `RANK` | derived by torchrun | `get_world_rank()` | `process_id` |
| world size | `WORLD_SIZE` | `OMPI_COMM_WORLD_SIZE` | `SLURM_NTASKS` | `WORLD_SIZE` | `PET_NNODES` x nproc | `get_world_size()` | `num_processes` |
| local rank | `LOCAL_RANK` | `OMPI_COMM_WORLD_LOCAL_RANK` | `SLURM_LOCALID` | `LOCAL_RANK` | via torchrun | `get_local_rank()` | `local_device_ids` |
| node rank | `GROUP_RANK` | — | `SLURM_NODEID` | `CROSS_RANK` | `PET_NODE_RANK` | `get_node_rank()` | — |
| rendezvous | `MASTER_ADDR:MASTER_PORT` | PMIx server URI | PMIx/PMI2 via srun | `MASTER_ADDR:PORT` | `PET_MASTER_ADDR:29500` | worker 0 addr | `coordinator_address` |
| unequal GPUs per node | yes (slow path) | yes (slots) | yes | yes | no (one `numProcPerNode`) | yes | n/a |

---

## 4. From global rank to parallelism groups

### 4.1 Megatron-LM (and NeMo, which builds on it)

`megatron/core/parallel_state.py`
([source](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)):

- `RankGenerator(tp, ep, dp, pp, cp, order, rank_offset)` treats each parallelism type as an axis of a
  grid over `range(world_size)`. The `order` string lists axes from **smallest stride (leftmost) to
  largest (rightmost)**. The default is `order="tp-cp-ep-dp-pp"`.
- Dense grid: `model_size = tp * pp * cp`, `data_parallel_size = world_size // model_size`, and
  `world_size` must be divisible by `model_size`. DP is never specified; it is whatever is left over.
- Expert (MoE) grid: built by a second generator with `cp=1` over the *same* ranks, using
  `expert_tp * ep * pp`; EP and CP cannot both exceed 1 in one generator. The code asserts that the
  pipeline groups of the two grids match.
- Worked example, `tp=2, pp=2, dp=2`, 8 ranks, order `tp-dp-pp`: TP groups `[0,1] [2,3] [4,5] [6,7]`;
  DP groups `[0,2] [1,3] [4,6] [5,7]`; PP groups `[0,4] [1,5] [2,6] [3,7]`. The docstring's 16-GPU
  example (TP=2, PP=4, so DP=2) gives PP groups like `[g0, g4, g8, g12]`.
- Placement rule from the docstring: "for efficiency, the caller should make sure adjacent ranks are
  on the same DGX box."
- Each group is a `torch.distributed.new_group`. A rank ends up holding tensor, pipeline, model
  (`tp-pp`), embedding, data, data-with-CP, context, hierarchical context, tensor-and-data, several
  expert groups, plus Gloo twins of the DP groups (`create_gloo_process_groups=True` by default).
- Per-group NCCL tuning through `nccl_communicator_config_path`, a YAML keyed by group name with
  `is_high_priority_stream`, `cga_cluster_size`, `min_ctas`, `max_ctas`, and `net_name` (`ib` or
  `socket`). This is a ready-made hook for "TP on the fast transport, DP on sockets".

Implication of the default order with 8-GPU nodes: TP groups are consecutive ranks (inside a node if
`tp` divides 8), CP next, DP groups stride by `tp*cp`, and PP stages stride by `tp*cp*dp`, which makes
them the farthest apart. That fits the traffic: PP is point-to-point activations once per micro-batch
and tolerates slow links best; TP is all-reduce inside every layer.

### 4.2 torchtitan / PyTorch DeviceMesh (FSDP2, HSDP)

`init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))` builds an N-D array of
ranks. `mesh["tp"]` slices a child mesh and `mesh.get_group("dp_shard")` returns the process group, so
users create groups "without worrying about how to set up ranks correctly"
([recipe](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)). The mesh is
`arange(world_size)` reshaped row-major, so the **last dimension varies fastest** `(unverified in this
sweep; from memory of the DeviceMesh source)`. The recipe itself says the mesh connects each device
"with its counterpart on the other hosts in a homogeneous setup".

torchtitan's dimension order is `["pp", "dp_replicate", "dp_shard", "cp", "tp"]`, keeping only
dimensions of size > 1; with MoE it becomes
`["pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]`. FSDP uses a flattened
`dp_shard_cp` dimension
([Parallelism Mesh Zoo, E. Yang](https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/); the
upstream `parallel_dims.py` could not be fetched in this sweep, so exact current names are
`(unverified)`). Same source on intent:

- "GPUs are first organized into nodes on the inner-most dimension, and then the nodes are collected
  together in the outer-most dimension."
- TP is "the inner mesh dimension, since it benefits the most from the high bandwidth network"; "you
  typically want to keep the communications for it within a single node".
- HSDP: "the inner dimension shards, while the outer dimension replicates": reduce-scatter inside the
  shard group, all-reduce across replicas. This is the natural shape for marketplace clusters: shard
  inside a node or fast island, replicate across slow links.
- torchtitan puts PP outermost and Megatron puts DP outside PP; "the order matters little" for PP.

### 4.3 DeepSpeed

ZeRO stages 1 to 3 shard optimizer state, gradients and parameters over the single world group. There
is no second mesh axis, so rank order has no bandwidth meaning beyond what NCCL's ring or tree does
with it. `save_checkpoint` must be called by every rank ("all processes must call this method and not
just the process with rank 0"), and each process saves its own optimizer partition
([docs](https://www.deepspeed.ai/getting-started/)). A ZeRO checkpoint is therefore keyed by rank and
world size.

### 4.4 Summary of stride conventions

| Framework | Fastest-varying (adjacent ranks) → slowest | DP specified? |
|---|---|---|
| Megatron-LM default | TP → CP → EP → DP → PP | derived: `world / (tp*cp*pp)` |
| torchtitan | TP → CP → DP shard → DP replicate → PP | `dp_shard` derived, others given |
| DeepSpeed ZeRO, plain DDP/FSDP1 | single axis | world size |

All of them assume "consecutive ranks = same node". The orchestrator has to guarantee it.

---

## 5. How NCCL communicators are built from ranks

From the [NCCL 2.32.3 user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html):

- One process calls `ncclGetUniqueId()`; the id is shared "through a CPU channel such as MPI or
  sockets" (in PyTorch, the `TCPStore`); every member calls `ncclCommInitRank(comm, nranks, id, rank)`.
  Ranks are 0..n-1 and each is bound to one CUDA device. Reusing a device for two ranks of the same
  communicator "is not supported and may lead to hangs".
- `ncclCommInitRankConfig` takes an `ncclConfig_t`: `blocking`, `minCTAs`/`maxCTAs`, `cgaClusterSize`,
  `netName`, `trafficClass`. These are the fields Megatron's YAML exposes.
- `ncclCommSplit(comm, color, key)`: same color forms a child communicator, key orders ranks, all
  parent ranks must call, and the child inherits the parent's configuration.
- `ncclCommInitRankScalable` accepts several unique ids to spread bootstrap load.
  `NCCL_UID_STAGGER_THRESHOLD` (default 256 ranks per id) and `NCCL_UID_STAGGER_RATE` (default 7000
  msg/s) throttle bootstrap storms
  ([env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).
- `ncclCommShrink(comm, excludeRanks, count, &newcomm, config, flags)`: only survivors call it and
  remaining ranks are **renumbered contiguously**. `ncclCommGrow`: existing ranks keep their numbers
  and new ranks start at the parent's size.
- Failure guidance: use non-blocking communicators; on any rank failure "all healthy ranks should
  abort their communicators and restart", or shrink with `NCCL_SHRINK_ABORT`.

Rank-relevant details:

- A PyTorch process group's NCCL rank is the position in the (sorted) `ranks` list, not the global
  rank. NCCL discovers topology per host and builds rings and trees itself; rank order only seeds it.
- NCCL groups ranks by a host hash. In containers on different machines that share a hostname, or on
  the same machine with different hostnames, the hash can be wrong and transport selection (shared
  memory or P2P versus network) follows it. An `NCCL_HOSTID` override exists in the NCCL source but was
  not found in the documentation pages read `(unverified)`. Worth testing explicitly on RunPod-style
  pods; see `05-networking-interconnects.md`.
- Interface selection is per process: `NCCL_SOCKET_IFNAME` (prefix match, `^` excludes, `=` exact),
  `NCCL_IB_HCA`, and `NCCL_NET` override the per-communicator `netName`. On heterogeneous nodes these
  must be emitted per node, not globally.

---

## 6. Rank-to-hardware mapping concerns

### 6.1 What rank 0 is responsible for

| Duty | Where it comes from |
|---|---|
| Hosts the `TCPStore` (`MASTER_ADDR:MASTER_PORT`) | `env://` init; the torchrun agent on group rank 0 |
| Computes global ranks for all agents | TorchElastic slow path |
| Hosts the JAX coordinator service | `jax.distributed.initialize` |
| Writes tracker file, run config, train state | Megatron: global rank 0 writes `latest_checkpointed_iteration.txt` ([checkpointing.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py)) |
| Logging, metrics, evaluation, uploads | convention (`rank_zero_only` in Lightning, `is_main_process` in Accelerate, chief in TF) |
| Source of truth for state broadcast after resize | Horovod elastic prefers the oldest worker |

So the rank-0 node should be the most reliable node, on-demand rather than spot, with the best path to
checkpoint storage and a control address reachable from every other node. Losing it is the most
expensive single failure. A stable DNS name or virtual IP for `MASTER_ADDR` avoids re-templating every
node when rank 0 is replaced.

### 6.2 Deterministic ordering across restarts

- TorchElastic dynamic rendezvous sorts by address, so replacing `10.0.0.7` with `10.0.0.31` moves that
  node's ranks to the end and shifts everyone in between. Static rendezvous with an explicit
  `--node-rank`, or setting `RANK` directly, gives the orchestrator full control.
- Kubernetes indexed Jobs give stable node ranks through the completion index. Slurm requeue keeps
  task order only if the node list is unchanged.
- The design to copy: a **logical slot id** that is stable for the life of the run and independent of
  hostname or IP, with rank derived from the slot. A replacement node inherits the dead node's slot.

### 6.3 `CUDA_VISIBLE_DEVICES` and device order

From the [CUDA environment variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html):

- `CUDA_DEVICE_ORDER` defaults to `FASTEST_FIRST` (a heuristic); `PCI_BUS_ID` orders by bus id and
  matches `nvidia-smi`. On mixed-GPU nodes the default order differs from `nvidia-smi` indices.
- `CUDA_VISIBLE_DEVICES` accepts indices, `GPU-<uuid>` strings (unique prefixes allowed), or one MIG
  instance. Visible devices are renumbered from 0 in list order, and parsing stops silently at the
  first invalid entry.
- Recommendation: always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and pin by UUID. Then `LOCAL_RANK` to GPU
  is reproducible and a preflight-excluded GPU is removed by identity, not by index shift.
- Two valid styles: (a) all GPUs visible and the script uses `cuda:{LOCAL_RANK}`; (b) one GPU visible
  per process and the script uses `cuda:0` (torchrun's `virtual_local_rank`, DeepSpeed's launcher).
  Style (b) is more robust to scripts that ignore `LOCAL_RANK`, but a script that does
  `cuda:{LOCAL_RANK}` under (b) crashes. The orchestrator should pick one per job and make it explicit.

### 6.4 Placing groups on the topology

- TP (and FSDP shard groups in HSDP) should sit inside one NVLink or shared-memory domain; DP
  replicate and PP go across the slower links. With Megatron's default order this happens when the
  per-node GPU count is a multiple of `tp*cp` and ranks are numbered node by node. If `tp=8` and one
  node exposes only 6 healthy GPUs, every TP group after that node straddles two machines and all TP
  all-reduces go over the network, with no error or warning.
- When the fast domain is a group of nodes (an InfiniBand island, a multi-node NVLink domain,
  `NCCL_MNNVL_ENABLE`), number ranks island by island and make the group size a multiple of the island.
- PP stages can sit on different GPU models (stages are independent programs), while TP and FSDP
  groups assume equal memory and speed per member. See `06-gpu-hardware-nuances.md`.

### 6.5 Unequal GPU counts per node

Launch is fine (torchrun slow path, MPI slots, DeepSpeed hostfile). The group arithmetic is not.
Options for the orchestrator:

1. **Truncate**: use `g = gcd` or `min` of per-node counts, or the largest multiple of the inner group
   size on each node, and leave the rest idle.
2. **Bucket**: split the fleet into homogeneous buckets and assign each a contiguous rank range
   aligned to the inner group size.
3. **Flat DP only**: with pure DDP/FSDP/ZeRO any counts work; only the ring or tree shape changes.
4. **Separate worlds**: independent jobs joined by a looser algorithm (see
   `08-heterogeneous-distributed-training.md`).

### 6.6 What breaks when rank assignment changes on restart

| Thing keyed by rank | Failure mode |
|---|---|
| Legacy Megatron checkpoints: `iter_NNNNNNN/mp_rank_XX[_YYY]/model_optim_rng.pt`, written by DP-rank-0 of each model-parallel shard | file `XX_YYY` holds one TP/PP shard. If the same world size is renumbered so that a process's TP/PP coordinates change, it loads a different but valid-looking shard. Changing TP or PP size is rejected by `check_checkpoint_args` for non-distributed formats; `torch_dist`, `torch_dcp` and `fsdp_dtensor` skip that check, which suggests resharding is possible ([checkpointing.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py)) |
| DeepSpeed ZeRO optimizer partitions (one per rank) | world size change needs an elastic or universal checkpoint path `(unverified)` |
| `DistributedSampler(dataset, num_replicas, rank, seed=0)` | shard = f(rank, world size, epoch). The seed must be "identical across all processes" and `set_epoch()` called ([docs](https://docs.pytorch.org/docs/2.15/data.html)). A mid-epoch restart with a new world size re-partitions the data: some samples repeat, some are skipped |
| Per-rank RNG (`seed + rank`, dataloader worker seeds, dropout) | reproducibility lost. The Megatron file name `model_optim_rng.pt` shows RNG state is saved per shard |
| Sharded or streaming datasets (`shard_id = rank % num_shards`) | duplicate or missing shards unless the shard count is fixed and independent of world size (the Horovod `max_size` trick) |
| LR and batch scaling by `WORLD_SIZE` | effective batch size changes after elastic resize |
| Per-rank logs, profiler traces, W&B groups | history discontinuity |
| NCCL communicators after `ncclCommShrink` | ranks renumbered contiguously inside the communicator while framework-level ranks are unchanged, so any group index cached before the shrink is stale |

General rule: **same world size and same slot-to-rank map is a cheap restart; anything else is a
reconfiguration** that needs a resharding-capable checkpoint and a data pipeline keyed by global
sample index rather than by rank.

---

## Implications for grid

1. **One canonical artefact: a rank table.** Per process: `rank`, `world_size`, `slot_id` (stable
   logical id), `node_id`, `node_rank`, `local_rank`, `local_world_size`, GPU UUID and PCI bus id, GPU
   model, control address, data-plane interface names per network, and a topology path (island, node,
   NVLink domain). Per job: rendezvous host and port, run id, restart count, and the parallel layout
   (`tp, cp, ep, pp, dp_shard, dp_replicate`, order string). Every launcher format is a projection of
   this table.

2. **Emit adapters, not one launcher.**

   | Target | Emit |
   |---|---|
   | Direct env (default, most universal) | per process: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<uuid>`, `TORCHELASTIC_RESTART_COUNT`, `TORCHELASTIC_RUN_ID` |
   | torchrun static | per node: `torchrun --nnodes N --nproc-per-node k_i --node-rank i --master-addr A --master-port P --rdzv-backend static` |
   | torchrun elastic | per node: `--nnodes MIN:MAX --rdzv-backend c10d --rdzv-endpoint A:29400 --rdzv-id RUN --max-restarts R --rdzv-conf join_timeout=...,last_call_timeout=...` plus hostnames that sort in the intended order |
   | MPI | hostfile `host slots=k`, optional rankfile `rank r=host slot=...`, `--map-by slot`, `-x` list |
   | DeepSpeed | hostfile `host slots=k` in rank order; or `--no_ssh --node_rank i` per node; `.deepspeed_env` |
   | Accelerate | per-node config: `num_machines`, `machine_rank`, `main_process_ip`, `main_process_port`, `num_processes` |
   | Lightning Fabric | `--node-rank i --main-address A --num-nodes N --devices k`, or direct env with `LOCAL_RANK` set |
   | Slurm-style scripts | aliases: `SLURM_PROCID`, `SLURM_NODEID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_NODELIST` |
   | Kubeflow-style | `PET_NNODES`, `PET_NPROC_PER_NODE`, `PET_NODE_RANK`, `PET_MASTER_ADDR`, `PET_MASTER_PORT` |
   | JAX | `coordinator_address`, `num_processes`, `process_id`, `local_device_ids` (one process per host) |
   | TF | `TF_CONFIG` JSON |
   | Megatron-LM | the env above plus a generated `nccl_communicator_config_path` YAML |

3. **grid should own rank numbering and prefer direct env or static rendezvous.** TorchElastic's
   address-sorted ranks are not controllable. If c10d is used, synthesize zero-padded, ordered
   hostnames (`grid-0007`) via `/etc/hosts` so natural sort equals slot order.

4. **Topology-aware numbering as a first-class step.** Sort nodes by (island, GPU model, node) and
   GPUs by PCI bus id, then validate against the layout before launch: every inner group (TP, CP,
   FSDP shard) must fall inside one node or declared fast domain. Fail loudly, or truncate or bucket
   (section 6.5), rather than letting TP run over TCP unnoticed.

5. **Stable slots.** A replacement node takes the dead node's `slot_id` and therefore its ranks,
   checkpoint shard and data shard. World-size changes are a separate, explicit reconfigure operation
   gated on a resharding-capable checkpoint format.

6. **Rank 0 is special infrastructure.** Place it on the most reliable, non-preemptible node with the
   best storage path. Consider hosting the rendezvous store in a grid agent (as torchrun's agent does)
   so it survives worker restarts, and give it a stable name.

7. **Per-node environment, not per-job.** `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `GLOO_SOCKET_IFNAME`
   differ per node on heterogeneous fleets; note that Gloo needs the same *count* of interfaces
   everywhere. Guard against helpers that silently set `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE`
   (Accelerate) by setting them explicitly.

8. **Init-time fragility scales with group count.** Megatron creates dozens of process groups
   collectively at start; a single slow or dead rank stalls everything until the timeout (10 min for
   NCCL, 600 s rendezvous join, 300 s JAX init). Preflight should prove all-pairs control connectivity
   before the launcher starts, and grid should expose these timeouts.

9. **Emit restart metadata.** `TORCHELASTIC_RESTART_COUNT`-style counters and a run id let user code
   distinguish first start from resume even when grid, not torchrun, does the restarting.

## Open questions

- **torchtitan current code**: `parallel_dims.py` could not be fetched (404; the file may have moved to
  `parallelism_context.py`). Confirm exact mesh dimension names and order, and whether DeviceMesh still
  maps ranks with a row-major `arange`. Also whether DeviceMesh has any support for non-rectangular
  (unequal GPU count) layouts.
- **torchrun static rendezvous with unequal `--nproc-per-node`**: the slow rank-assignment path should
  make this work; needs an actual test, along with confirmation of the `--node-rank`, `--master-addr`
  and `--master-port` flag behaviour and `PET_*` aliases in 2.15.
- **`shrink_group`, `enable_reconfigure`, `_reconfigure` in PyTorch and `ncclCommShrink`/`ncclCommGrow`
  in NCCL 2.32**: which PyTorch and NCCL versions ship them, how framework ranks map after a shrink,
  and whether any LLM framework (Megatron, torchtitan) uses them yet. This could turn "restart
  everything" into in-place repair and deserves its own sweep with `04-failure-taxonomy-recovery.md`.
- **NCCL host identity in containers** (`NCCL_HOSTID`, hostname plus boot id hashing): verify in the
  NCCL source and test on RunPod pods, where hostnames and boot ids may collide or differ unexpectedly.
- **Megatron-LM**: why the Gloo twin groups exist and when they can be disabled; the resharding
  capability and limits of `torch_dist` / `torch_dcp` / `fsdp_dtensor` checkpoints when TP, PP or DP
  change; the new `ProcessGroupCollection` API that replaces the global state.
- **DeepSpeed**: ZeRO checkpoint file naming per rank and the universal-checkpoint path for world-size
  changes; current status of `--elastic_training`.
- **Slurm**: how `CUDA_VISIBLE_DEVICES` is set per task under `--gpus-per-task` and cgroup device
  constraints; heterogeneous-job rank offsets (`SLURM_HET_SIZE`). Not covered in the pages read.
- **PMIx variables** (`PMIX_RANK`, `PMIX_NAMESPACE`, `PMIX_SERVER_URI*`) and whether grid could act as
  a minimal PMIx server to drive MPI-launched codes without ssh.
- **Data pipelines**: how common LLM data loaders (Megatron's, HF `datasets` streaming, WebDataset,
  Mosaic streaming) key shards by rank, and which support world-size-independent resume.
- **Rank count limits**: TCPStore and NCCL bootstrap behaviour at thousands of ranks
  (`NCCL_UID_STAGGER_*`, `ncclCommInitRankScalable`); whether PyTorch uses the scalable init.
