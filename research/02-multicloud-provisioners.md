# 02 - Multi-cloud GPU provisioners and cluster launchers (prior art)

Research date: 2026-11-09. Sources are cited inline. Claims marked `(unverified)` come from search
snippets, third-party pages or mirrors rather than a primary page I could open. Two doc hosts could
not be read from this session: `docs.tensorpool.dev` (not resolvable from the sandbox) and the Prime
Intellect multi-node tutorial pages (HTTP 404 on direct fetch).

## Summary

- The two serious open-source "launch anywhere" tools are **SkyPilot** and **dstack**. Both treat a
  multi-node cluster as *N identical nodes from one offer, in one region of one provider*, provisioned
  all-or-nothing, and both recover from a node failure by **tearing the whole cluster down and
  relaunching**. Neither does in-place replacement of a single failed node, neither spans providers
  inside one training job, and neither mixes GPU types inside one job.
- On the marketplace providers grid cares about, multi-node support is thin. SkyPilot's `RunPod` and
  `Vast` cloud classes both declare `MULTI_NODE` unsupported ("the interconnection among nodes are
  non-trivial"). dstack supports multi-node on RunPod only by calling RunPod's own cluster API
  (`create_cluster`, `cluster_type="TRAINING"`), which is capped at 2-8 nodes self-serve; dstack does
  not support clusters on Vast.ai at all.
- Rank wiring is the same everywhere and is shallow: the launcher exports node count, node rank, the
  list of IPs and GPUs-per-node as env vars (`SKYPILOT_*`, `DSTACK_*`, RunPod's `NODE_RANK` /
  `PRIMARY_ADDR` / `WORLD_SIZE`), and the user writes the `torchrun` line. Only dstack also writes an
  MPI hostfile. Nobody validates that the mesh actually works (NCCL all-reduce, MTU, RDMA) before
  running user code, except Modal, which does it as a managed service.
- The newest entrant that matters is **Modal Clusters** (GA 2026-10-01): gang-scheduled, RDMA-wired,
  per-second-billed clusters that start "within seconds", up to 32 nodes / 256 GPUs by default. It
  sets the bar for time-to-first-step but is a closed platform on Modal's own capacity.
- Provider-native "instant cluster" products (RunPod, Together, Mithril, TensorPool, Prime Intellect,
  Shadeform clusters) give a homogeneous, single-fabric cluster in minutes, usually with Slurm or
  Kubernetes preinstalled, and usually cap self-serve size at 8-64 nodes.
- The gap: nothing today assembles a **large** (hundreds of GPUs) training cluster **fast** from
  **many small offers** across hosts, regions and providers, accepts **partial fills**, tolerates
  **mixed hardware and no private network**, and **repairs** the cluster in place when a node dies.

## Comparison at a glance

| Tool | Kind | Multi-node on RunPod / Vast | Unit of provisioning | Partial capacity | Rank wiring | Node failure |
|---|---|---|---|---|---|---|
| SkyPilot | OSS client + API server | No / No (declared unsupported) | N identical nodes, one zone | All-or-nothing, then next zone/region/cloud | `SKYPILOT_NODE_RANK`, `SKYPILOT_NODE_IPS`, `SKYPILOT_NUM_NODES`, `SKYPILOT_NUM_GPUS_PER_NODE` | Managed jobs: tear down, relaunch elsewhere, restart from user checkpoint |
| dstack | OSS server + CLI | Yes via RunPod cluster API (2-8 nodes) / No | Fleet with `placement: cluster` | Fleet `nodes: min..max`; a run needs all its nodes | `DSTACK_NODE_RANK`, `DSTACK_NODES_IPS`, `DSTACK_MASTER_NODE_IP`, `DSTACK_MPI_HOSTFILE` | With `retry`: stop all jobs, resubmit the run |
| Modal Clusters | Closed serverless platform | n/a (own capacity) | `@modal.clustered(size=N)` | Gang scheduled; code runs only when all nodes are held | `cluster.container_rank()`, `cluster.container_ips()` | Any failure or preemption kills all containers; retry policy reruns |
| Ray cluster launcher | OSS, `ray up` | No provider plugin for either | Head + autoscaled worker types | Autoscaler adds nodes as they come; no gang semantics | None for torchrun; Ray Train owns it | Autoscaler replaces dead workers; training recovery is Ray Train's job |
| Prime Intellect | Aggregator CLI + dashboard | n/a (aggregates providers) | Pod, or dashboard "Multi-Node Cluster" | Not documented | Not documented | Not documented |
| Shadeform | Aggregator REST API | n/a | Instance; clusters object exists | Not documented | None (public IP + SSH) | Spot: instance goes `deleting` / "preempted" |
| Brev (NVIDIA) | Dev-instance aggregator | n/a | Single instance | n/a | n/a | n/a |
| Terraform `runpod/runpod` 2.x | IaC | Pods only, no rank wiring | `runpod_pod` | Apply fails per resource | None | None |
| Provider instant clusters (RunPod, Together, Mithril, TensorPool) | Provider-native | RunPod: yes | Cluster of identical nodes on one fabric | Whole cluster or nothing | RunPod env vars; Slurm/K8s elsewhere | Provider repairs or user rebuilds |

## SkyPilot

### Architecture

A client talks to an API server (local or remote) that keeps state in SQLite, or optionally Postgres,
under `~/.sky/`. On each cluster SkyPilot still runs its own Ray cluster for orchestration: the head
runs `ray start --head` on port 6380, workers join by the head's internal IP, the dashboard is on port
8266, and the pinned Ray version is 2.56.1 inside a `skypilot-runtime` conda env. Users are told not to
call `ray.init(address="auto")` or `ray stop`, and to run their own Ray on 6379 if they need Ray
(https://docs.skypilot.ai/en/latest/reference/architecture/internals.html).

Recent releases, from mirrors and aggregators rather than the release page itself `(unverified)`:
v0.11 made managed-jobs "consolidation mode" GA and added Pools; v0.11.2 added a Slurm backend in beta
with multi-node support; v0.12 added Job Groups; v0.13 added Sky Batch, lifecycle hooks and cost caps.

### Provisioning N nodes

The launch pipeline for a new cluster is a sequence of barriers: (1) provision nodes, (2) sync workdir
and `file_mounts` to all nodes, (3) run `setup` on all nodes, (4) run `run` on all nodes
(https://docs.skypilot.ai/en/latest/running-jobs/distributed-jobs.html). Every barrier waits for the
slowest node.

From `sky/provision/provisioner.py`
(https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/provision/provisioner.py):

- `wait_for_ssh` probes the first IP in the main thread and the rest with
  `subprocess_utils.run_in_parallel`. The overall timeout is `60 * 10` seconds, each probe uses
  `provision.ssh_timeout` (default 10 s), and the raw socket connect timeout is 1 s.
- After SSH, setup is sequential in stages: internal file mounts, runtime setup, start Ray on head,
  start Ray on workers, start skylet on the head. Ray health requires the alive-node count to equal
  `num_instances` exactly.
- On any provisioning exception `bulk_provision` tears the cluster down (terminate if it was never up,
  stop otherwise), retrying teardown up to `_MAX_RETRY = 3` times with 5 s sleeps, then re-raises so the
  caller can fail over. If teardown itself fails it raises `StopFailoverError` and warns of "resource
  leakage".

So a multi-node launch is **all-or-nothing per location**: if node 7 of 8 fails, the other seven are
terminated and the whole request moves on. I found no published launch-time numbers for large
clusters. The only scale hint in the docs is that launching about 100 nodes can hit
`[Errno 24] Too many open files`, fixed with `ulimit -n 65535`
(https://docs.skypilot.ai/en/latest/running-jobs/distributed-jobs.html).

### Stockouts and failover

The optimizer ranks candidates by price. The provisioner tries every zone and region of the cheapest
cloud, then the next cheapest cloud. It handles "out of capacity" and "out of quota" errors, surfaced
as `ResourcesUnavailableError` (https://docs.skypilot.ai/en/latest/examples/auto-failover.html).
Candidate sets are expressed in the task YAML:

```yaml
resources:
  accelerators: {A10:1, L4:1, A10g:1}   # set: any, cheapest first
  # accelerators: [A10:1, A10g:1, L4:1] # list: strict order
  any_of:                               # or `ordered:`
    - infra: gcp
      accelerators: A100-80GB
    - instance_type: g5.xlarge
```

The choice applies to the whole cluster: `resources` is "per-node resource requirements" and the spec
has no way to give different nodes different instance types
(https://docs.skypilot.ai/en/latest/reference/yaml-spec.html). `network_tier: best` asks for the
fastest fabric the infra has (EFA on AWS, InfiniBand on Nebius), same page.

### Multi-node wiring

Four env vars are exported on every node: `SKYPILOT_NODE_RANK` (0..N-1), `SKYPILOT_NODE_IPS` (one IP
per line), `SKYPILOT_NUM_NODES`, `SKYPILOT_NUM_GPUS_PER_NODE`. The documented pattern is:

```bash
MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
torchrun --nnodes=$SKYPILOT_NUM_NODES --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  --node_rank=${SKYPILOT_NODE_RANK} --master_addr=$MASTER_ADDR --master_port=8008 main.py
```

For MPI the docs only say to run `mpirun` on rank 0; there is no generated hostfile. SSH aliases
`<cluster>` and `<cluster>-worker<n>` are written to the client's `~/.ssh/config`
(https://docs.skypilot.ai/en/latest/running-jobs/distributed-jobs.html).

### Node failure: managed jobs

`sky jobs launch` puts a controller in charge of an ephemeral cluster. On preemption or hardware
failure it tears the cluster down, provisions a new one (possibly in another region or cloud) and
reruns the task from the top; the user must checkpoint to a bucket or volume and resume themselves
(https://docs.skypilot.ai/en/latest/examples/managed-jobs.html). Controls:

| Key | Meaning |
|---|---|
| `job_recovery.strategy` | `EAGER_NEXT_REGION` (default: move region immediately) or `FAILOVER` (retry same region first) |
| `job_recovery.max_restarts_on_errors` | Restarts allowed on non-zero user exit (default 0) |
| `job_recovery.recover_on_exit_codes` | Exit codes that always trigger recovery; any node exiting with one triggers it. 137 is reserved |
| `job.wait_for_scheduling_timeout` | Bound on the otherwise indefinite capacity retry before first start |

Sources: managed-jobs page above and https://docs.skypilot.ai/en/latest/reference/yaml-spec.html.
Controller capacity is memory-bound: launching slots = 8 x floor((mem - 2 GiB) / 3.59 GiB), capped at
512; running jobs = 200 x the same floor, capped at 2000; the default 16 GiB controller gives 24
concurrent launches and 600 running jobs (managed-jobs page).

### Heterogeneity: Job Groups

Job Groups are a multi-document YAML with `execution: parallel`, where each task has its own
`resources` and `num_nodes`. Tasks find each other by hostname `{task_name}-{node_index}.{job_group_name}`
and the single env var `SKYPILOT_JOBGROUP_NAME`. Hostname discovery works only inside one Kubernetes
cluster ("Hostname-based service discovery across clusters is not yet supported"); with
`inter_connection: false` tasks may land anywhere but get no networking. Each task recovers from
preemption independently (https://docs.skypilot.ai/en/latest/examples/job-groups.html). This is aimed
at trainer-plus-CPU-services layouts, not at one DDP job over mixed GPUs.

### Marketplace providers

| Cloud class | Declared unsupported | Source |
|---|---|---|
| RunPod | `MULTI_NODE` ("Multi-node not supported yet, as the interconnection among nodes are non-trivial on RunPod."), `STOP`, `STORAGE_MOUNTING` (use `mode: COPY`), `CUSTOM_NETWORK_TIER`, `CUSTOM_MULTI_NETWORK`, `LOCAL_DISK`; one network volume max | https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/runpod.py |
| Vast | `MULTI_NODE` (same wording), `STORAGE_MOUNTING`, `CUSTOM_NETWORK_TIER`, `CUSTOM_MULTI_NETWORK`; no zones; `vast.datacenter_only` filter | https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/vast.py |
| Lambda | `SPOT_INSTANCE`, `STOP`, `IMAGE_ID`, `CUSTOM_NETWORK_TIER`; multi-node is not listed as unsupported | https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/clouds/lambda_cloud.py |

The RunPod provisioner (`sky/provision/runpod/instance.py`) shows why this is hard. It launches pods
**one at a time** in a `for` loop, then polls every `POLL_INTERVAL = 5` s until all are `RUNNING` with
an `ssh_port`, with **no timeout** in that loop; any pod in `EXITED`, `ERROR` or `TERMINATED` raises.
`get_cluster_info` sets `internal_ip` equal to `external_ip` with the comment "RunPod pods have no
private network; the public IP is the only address other nodes can reach", and SSH is on a mapped
port (https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/provision/runpod/instance.py).
The file has head/worker logic even though the cloud class refuses `num_nodes > 1`; I did not resolve
whether a gate lifts this in some configuration. SkyPilot does not use RunPod Instant Clusters as far
as I could find.

## dstack

### Architecture

A server (SQLite, or Postgres for more than one replica) holds backends, fleets and runs; a CLI applies
`.dstack.yml` files. Sizing guidance is about 8 MB RAM per active instance, and one replica is stated to
support at least 1,000 active instances, 1,000 runs and 1,000 jobs
(https://dstack.ai/docs/guides/server-deployment/). Since 0.20 fleets are explicit, first-class
resources: a run no longer creates a fleet implicitly (https://dstack.ai/blog/0_20/).

Backends split into VM-based (AWS, Azure, GCP, Lambda, Nebius, Crusoe, Verda, OCI, Vultr, Hot Aisle,
JarvisLabs, CloudRift and others) and container-based (Kubernetes, Slurm, Daytona, Lium, **Runpod**,
**Vast.ai**). RunPod is Secure Cloud only unless `community_cloud: true`; Vast includes community hosts
unless `community_cloud: false` (https://dstack.ai/docs/concepts/backends/).

### Fleets and capacity

```yaml
type: fleet
name: train
placement: cluster      # required for multi-node tasks
nodes:
  min: 2
  max: 10
  target: 6             # provision to target, shrink to min after idle_duration
```

Pre-provisioning (`min > 0`) works only on VM-based backends; container backends release instances
as soon as they go idle. Default `idle_duration` is 3 days for pre-provisioned instances and 5 minutes
for instances created at submit time. Fast interconnect is wired automatically on `aws`, `gcp`,
`nebius`, `crusoe` and `kubernetes` (EFA, RoCE VPCs, InfiniBand)
(https://dstack.ai/docs/concepts/fleets/).

When capacity is short, dstack walks the offer list with an attempt limit; the documented error is
"tried 5 of 12 offers (attempt limit reached), all failed". Runs fail on no capacity unless `retry` is
set (https://dstack.ai/docs/guides/troubleshooting/). Clusters and multi-node tasks are supported on
`aws, azure, crusoe, gcp, jarvislabs, kubernetes, nebius, oci, runpod, seeweb, slurm, vultr` and SSH
fleets; **not on vastai** (same page).

### Multi-node wiring

`nodes: N` on a task requires a `placement: cluster` fleet. Jobs talk over private IPs, and inter-node
SSH is preconfigured on port 10022 with key `~/.ssh/dstack_job`. Exported env vars:
`DSTACK_NODES_NUM`, `DSTACK_GPUS_PER_NODE`, `DSTACK_GPUS_NUM`, `DSTACK_NODE_RANK`,
`DSTACK_MASTER_NODE_IP`, `DSTACK_NODES_IPS` (newline-delimited) and `DSTACK_MPI_HOSTFILE` (a ready
hostfile with `<ip> slots=<gpus>` lines). The docs tell users to set `NCCL_SOCKET_IFNAME` themselves if
the framework picks the wrong interface (https://dstack.ai/docs/concepts/tasks/).

```bash
torchrun --nproc-per-node=$DSTACK_GPUS_PER_NODE --node-rank=$DSTACK_NODE_RANK \
  --nnodes=$DSTACK_NODES_NUM --master-addr=$DSTACK_MASTER_NODE_IP --master-port=12345 multinode.py
```

Tasks can also define `groups`, each with its own `nodes`, `resources`, `commands` and `ports`,
addressed as `${{ groups[i].nodes[j].IP_ADDRESS }}`, with `startup_order: workers-first`. That is the
closest thing in this survey to a declared heterogeneous job, but all groups still live in one cluster
fleet (same page).

### Node failure

`retry: {on_events: [no-capacity, error, interruption], duration: 1h}`. The rule for multi-node is
explicit: "If one job of a multi-node task fails with retry enabled, `dstack` will stop all the jobs
and resubmit the run." Spot interruption surfaces as `INTERRUPTED_BY_NO_CAPACITY`
(https://dstack.ai/docs/concepts/tasks/, https://dstack.ai/docs/guides/troubleshooting/).
`utilization_policy` (`min_gpu_utilization`, `time_window`) kills runs with idle GPUs.

### RunPod backend internals

From `src/dstack/_internal/core/backends/runpod/compute.py`
(https://raw.githubusercontent.com/dstackai/dstack/master/src/dstack/_internal/core/backends/runpod/compute.py):

- The class mixes in `ComputeWithMultinodeSupport` and `ComputeWithGroupProvisioningSupport`.
  Multi-node requests are served only by **cluster offers**: `run_jobs` calls `create_cluster` with
  `cluster_type="TRAINING"`, and the requested node count must be in the offer's `pod_counts`.
- After creation it lists the cluster's pods and checks the count; on mismatch it **deletes the
  cluster** (all-or-nothing). Private IPs come from the pod's `cluster.ip`, commented "May not be
  assigned yet". Single pods get `internal_ip=None`.
- Spot is rejected ("Runpod spot offers are not supported"). Clusters need a secure-cloud data center,
  allow at most one network volume, and do not support private registries.
- Community-cloud pods still go through the deprecated GraphQL call because "API v2 does not support
  country codes".

So dstack on RunPod inherits RunPod's cluster limits: 2-8 nodes (16-64 GPUs) self-serve, larger "up to
512 GPUs" through sales, account spending limits, and inter-node traffic on `ens1`..`ens8` rather than
`eth0` (https://docs.runpod.io/instant-clusters.md, https://docs.runpod.io/instant-clusters/configuration).
Details of RunPod itself belong in `07-provider-landscape.md`.

## Modal Clusters

`@modal.clustered(size=N, rdma=True)` on a function with `gpu="H100:8"` gang-schedules N containers.
Limits: up to 32 nodes and 256 GPUs per cluster, more via support; each node must request the full GPU
count (`H100:4` is invalid); no CPU-only clusters. RDMA is available on H100, H200, B200, B300 and
GB200, at 3,200 Gbps (6,400 Gbps on B300), over RoCE, InfiniBand or EFA depending on hardware
(https://modal.com/docs/guide/multi-node-clusters.md).

- **Wiring**: `cluster.container_rank()` (rank 0 is leader) and `cluster.container_ips()` ordered by
  rank, on Modal's i6pn private network, IPv6 by default with `family="ipv4"` for frameworks such as
  Ray. NCCL env vars are set automatically; the image needs `libibverbs1`.
- **Scheduling**: a scheduler places all N nodes in one cycle, grouping by availability zone and
  network ID, and code starts only when all hardware is held. The GA post (2026-10-01) claims code runs
  "within seconds", per-second billing, and one customer at "hundreds of B300s"
  (https://modal.com/blog/modal-clusters-generally-available).
- **Failure**: if any container fails, Modal terminates the rest and fails the call; preemption kills
  the whole cluster; `retries=modal.Retries(...)` reruns with the same input. Health checks and burn-in
  cover GPUs and RDMA (docs page above).

Modal is the reference for user experience (decorator, seconds to start, mesh pre-validated) and shows
the same whole-cluster restart model as everyone else. It is not a provisioner for third-party
capacity.

## Ray cluster launcher

`ray up cluster.yaml` starts a head node and an autoscaler. Built-in `provider.type` values are `aws`,
`azure`, `gcp`, `vsphere` and `local`; there is no RunPod, Lambda or Vast provider in the reference.
Heterogeneity is first-class at the provisioning layer: `available_node_types` maps names to
`node_config`, `resources`, `min_workers`, `max_workers`. Defaults are `max_workers: 2`,
`upscaling_speed: 1.0` (pending launches as a multiple of current nodes, with a floor of 5) and
`idle_timeout_minutes: 5`. Setup is SSH-driven (`initialization_commands`, `setup_commands`,
`worker_start_ray_commands`)
(https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html).

There are no gang semantics: workers join as they arrive, and whether a training job can start or
survive is left to Ray Train (see `01-orchestrators-schedulers.md`). The `local` provider with
`head_ip` and `worker_ips` is the only path for marketplace nodes, which means grid would still have to
do the renting, the SSH bring-up and the network mesh itself.

## Aggregators and marketplaces

| Tool | What it does | Multi-node | Notes | Source |
|---|---|---|---|---|
| Prime Intellect | `prime availability list --gpu-type H100_80GB`, `prime pods create`, `prime pods ssh`; shows availability "across providers" | Dashboard "Multi-Node Cluster" tab, 16-64+ H100 in 8xH100 nodes, one public IP per node, email when ready, Slurm guide linked `(unverified, search snippet; tutorial pages returned 404)` | No documented CLI path for clusters, no documented failover or rank wiring | https://docs.primeintellect.ai/cli-reference/introduction.md |
| Shadeform | REST at `api.shadeform.ai/v1`, `POST /instances/create` with `cloud`, `region`, `shade_instance_type`; resells provider on-demand pools at provider price, or uses your own accounts with `"shade_cloud": false` | `GET /clusters` returns clusters with an `instances` array; FAQ says clusters and spot "recently shipped"; create parameters not documented on the pages I read | Instance states `creating, pending_provider, pending, active, error, deleting, deleted`; preempted spot shows `status_details: "preempted"`; only public IP + SSH documented | https://docs.shadeform.ai/api-reference/clusters/clusters.md, https://docs.shadeform.ai/getting-started/faq |
| NVIDIA Brev | Dev instances, Environments and shareable Launchables | None found; a third-party skill page describes Brev as "instance-based ... no cross-instance coordination" `(unverified)` | Not a training-cluster tool | https://docs.nvidia.com/brev/concepts/overview.md |
| Mithril (ex-Foundry) | Auction market; `flow instance create -i 8xh100 -N 20`, SDK `num_instances=4`, `max_price_per_hour` cap `(unverified, PyPI page snippets)` | Managed Kubernetes control plane; workers join when a reservation or spot bid is placed with the cluster selected | Spot preemption runs `kubectl drain` with a 5-minute grace period, then powers off; cluster provisioning "up to 10 minutes"; no autoscaling by Mithril | https://docs.mithril.ai/compute-and-storage/mithril-compute-clusters.md |
| Together Instant Clusters | `tg clusters create`; Kubernetes, or Slurm on Kubernetes via Slinky; HGX H100/H200/B200 on 3.2 Tb/s InfiniBand | Yes, "8 to 512+ GPUs in minutes" on the marketing page `(unverified)` | On-demand, reserved (3-180 days), scheduled, and preemptible (preview) capacity; automated node health checks and repair | https://docs.together.ai/docs/gpu-clusters-overview.md |
| TensorPool | `tp cluster create -i <pubkey> -t 8xH200 -n 2`; Slurm preinstalled; jumphost with public IP, workers `{cluster_id}-N` on private IPs; multi-node only for 8xH200 and 8xB200 `(unverified, third-party skill page; official docs unreachable)` | Yes | Closest in spirit to grid's CLI | https://pypi.org/project/tensorpool/ |
| SF Compute | Order-book market for node-hours: `sf buy -d "1hr" -n 8 -p "2.50"`, `sf nodes create -n 8 -s 'tomorrow at 10am' -d '1d'`; newer preview CLI uses `sf capacities` / `sf orders` `(unverified, search snippets)` | Kubernetes H100 nodes with InfiniBand; VM nodes historically without it | Time-boxed purchases rather than open-ended rental | https://docs.sfcompute.com/docs/instance-types |
| Covalent (Agnostiq, now DataRobot) | Workflow SDK with pluggable executors; acquired by DataRobot in 2025 `(unverified)` | Marketing claims multi-node GPU training | Not a marketplace cluster launcher; not investigated further | https://www.datarobot.com/product/covalent/ |

### Terraform and Pulumi

RunPod publishes an official Terraform provider; 2.0.0 is a breaking release generated from the REST
v2 API, with `runpod_pod`, `runpod_pod_action` and `runpod_serverless_endpoint`, and needs Terraform
1.6+ (https://docs.runpod.io/integrations/terraform/overview.md). The Pulumi registry package I found
is v1.9.99 from 2024-07-03 and looks stale (https://www.pulumi.com/registry/packages/runpod/api-docs/pod).
I did not confirm whether the Terraform provider exposes Instant Clusters. IaC of this kind is a poor
fit for grid's problem: it is declarative per resource, has no notion of "any N of these M offers",
no stockout fallback, no rank wiring and no health gating. It is useful only for long-lived fixtures
such as network volumes.

## Cross-cutting observations

**1. Everyone's cluster is homogeneous and single-fabric.** SkyPilot `resources` are per-node and
identical; dstack needs a `placement: cluster` fleet; Modal requires full identical nodes; provider
clusters are one SKU on one fabric. Heterogeneity exists only as side-car task groups (SkyPilot Job
Groups, dstack `groups`), not as mixed GPUs inside one data-parallel job.

**2. Provisioning is all-or-nothing, then move on.** SkyPilot tears down a partial cluster and tries
the next zone; dstack deletes a RunPod cluster whose pod count is short; Modal waits for the whole
gang. None of them keeps the nodes it got and tops up from another region or provider, which is the
natural strategy on a marketplace where each host has a handful of GPUs.

**3. Recovery is restart, not repair.** SkyPilot managed jobs, dstack `retry` and Modal `retries` all
kill every node and rerun from the top, leaving resume to the user's checkpoint code. There is no
"replace rank 5 and rejoin" path, no hot spare, and no notion of shrinking the world size.

**4. Wiring stops at env vars.** All tools export count, rank and IPs and leave `torchrun` to the
user. None probes the mesh first: no NCCL smoke test, no interface selection (dstack and RunPod both
just tell the user to set `NCCL_SOCKET_IFNAME`), no check that every node sees every other node. On
RunPod pods outside a cluster there is no private network at all, and SSH is on a mapped port.

**5. Bring-up is serial where it matters.** SkyPilot's RunPod loop creates pods one by one, has no
timeout on the ready wait, and gates each phase on the slowest node. A layered Ray runtime plus skylet
is installed before user code. This is acceptable for 2-8 nodes and painful at 50+.

**6. Self-serve scale caps are low.** RunPod Instant Clusters: 8 nodes. Modal: 32 nodes / 256 GPUs.
Prime Intellect dashboard: 16-64+ GPUs. SkyPilot documents a file-descriptor problem at about 100
nodes. dstack states "at least 1,000 instances" per server replica, but that is control-plane sizing,
not a demonstrated single training job.

**7. User friction.** I could not retrieve specific GitHub issue threads in this sweep (search engines
returned docs, not issues), so complaints here are inferred from code and docs: SkyPilot's dual-Ray
port collision warnings, "resource leakage" on failed teardown, object-store mounting unsupported on
RunPod and Vast; dstack's no-capacity attempt limit, no privileged containers or instance volumes on
RunPod and Vast, and no spot on RunPod.

## Gap analysis: what is not served today

| Need for fast, large, heterogeneous training clusters | SkyPilot | dstack | Modal | Provider clusters |
|---|---|---|---|---|
| More than 8 nodes on RunPod-class marketplaces, self-serve | No | No (capped by RunPod cluster API) | n/a | No (sales) |
| Assemble a cluster from many hosts, regions or providers | No | No | No | No |
| Keep partial fills and top up elsewhere | No | No | No | No |
| Mixed GPU types in one training job | No | No | No | No |
| Work without a private L2/L3 network (public IP + mapped ports) | No | No | n/a | n/a |
| Pre-flight mesh and GPU validation before user code | No | No | Yes (managed) | Partly |
| Replace one failed node in place, or continue with fewer | No | No | No | Provider-dependent |
| Parallel bring-up designed for 100+ nodes | Partly | Unknown | Yes | Yes |
| Seconds-to-minutes time to first step | Minutes or more | Minutes | Seconds (claimed) | Minutes |

The unserved combination is: **over-provision in parallel across many offers, admit nodes as they pass
preflight, build an overlay or pick the best transport per node pair, start when a quorum is healthy,
and repair or shrink on failure instead of restarting everything.**

## Implications for grid

1. **Do not model a cluster as "N of one offer".** Model it as a target (GPU-count or throughput) filled
   from a ranked, heterogeneous offer list. Keep whatever arrives, and keep bidding in parallel on other
   hosts, regions and providers until the target or a deadline is reached.
2. **Race, do not wait.** Request more nodes than needed (for example 1.2x), admit the first healthy N,
   and release stragglers. Every existing tool gates on the slowest node; SkyPilot's RunPod loop has no
   timeout at all. Give every phase a per-node deadline.
3. **Launch API calls concurrently** with per-provider rate limiting, rather than SkyPilot's sequential
   `for` loop.
4. **Own the network layer.** RunPod pods outside Instant Clusters have "no private network", so grid
   needs either an overlay (WireGuard or similar) or direct use of public IP plus mapped ports, and
   must set interface selection (`NCCL_SOCKET_IFNAME` and friends) itself. See
   `05-networking-interconnects.md`. Where a provider cluster API exists (RunPod `create_cluster`),
   use it as one building block: several 8-node RDMA islands joined by a slower inter-island link.
5. **Export a superset of the existing env-var conventions** so user scripts port unchanged: node rank,
   node count, GPUs per node, world size, master address and port, newline IP list, and an MPI hostfile
   (dstack's `DSTACK_MPI_HOSTFILE` format `<ip> slots=<gpus>`). Add what nobody has: per-node GPU model
   and memory, and an island or fabric ID per node, so heterogeneity-aware launchers can use them
   (`03-rank-configuration.md`, `08-heterogeneous-distributed-training.md`).
6. **Gate on preflight, not on SSH reachability.** Admission should require GPU health, a pairwise or
   sampled connectivity test and a short NCCL all-reduce, which only Modal does today
   (`09-fast-bootstrap-preflight.md`).
7. **Make repair a first-class operation.** Keep warm spares or a live bidding loop, and define what
   happens on node loss: replace-and-rejoin at the next checkpoint boundary, or shrink. Whole-cluster
   restart should be the fallback, not the only mode (`04-failure-taxonomy-recovery.md`).
8. **Skip the heavyweight runtime.** No Ray-plus-skylet layer on every node; a small static agent or
   plain SSH fan-out keeps bring-up short and avoids port collisions with user Ray.
9. **Always clean up.** SkyPilot's `StopFailoverError` "resource leakage" path shows the risk. Tag every
   rented resource with a cluster ID and run an idempotent reaper; on marketplaces a leaked 8-GPU pod is
   expensive.
10. **Consider being a backend, not only a competitor.** dstack SSH fleets and SkyPilot's existing-machine
    and Slurm paths can consume a cluster grid has already assembled, which gives users a job layer
    without grid having to build one.

## Open questions

- **GitHub issue evidence.** I could not pull concrete SkyPilot or dstack issue threads on large
  multi-node launches, partial provisioning or RunPod/Vast pain. A follow-up should query the issue
  trackers directly (`runpod`, `vast`, `num_nodes`, `placement: cluster`, `ulimit`, `teardown`).
- **Measured launch times.** No tool publishes time-to-ready versus node count. Worth benchmarking
  SkyPilot and dstack at 2, 8 and 32 nodes on Lambda, Nebius and RunPod, and Modal's "seconds" claim.
- **SkyPilot RunPod multi-node.** `clouds/runpod.py` declares `MULTI_NODE` unsupported while
  `provision/runpod/instance.py` contains head/worker logic and a public-IP-as-internal-IP comment. Is
  there a flag or branch that enables it, and does it use RunPod global networking?
- **RunPod cluster API surface.** dstack calls `create_cluster` with `cluster_type="TRAINING"`. What are
  the REST/GraphQL shapes, the allowed `pod_counts`, creation latency, and can several clusters be
  joined? RunPod's changelog suggests cluster expansion is admin-only `(unverified)`.
- **Prime Intellect and TensorPool.** Their multi-node docs were unreachable here. Need: whether the
  `prime` CLI can create clusters, which providers sit underneath, and TensorPool's actual limits and
  provisioning times. `docs.tensorpool.dev` must be allowed in the session's network settings to read it.
- **Shadeform cluster creation.** The create endpoint, supported clouds and inter-node networking were
  not on the pages read.
- **Vast.ai.** Neither SkyPilot nor dstack supports multi-node there. What exactly blocks it (port
  mapping, NAT, no host-to-host reachability), and does Vast have any cluster or private-network
  feature in 2026?
- **Terraform provider coverage** of RunPod Instant Clusters and network volumes in 2.x.
- **Scale ceilings in practice.** Has anyone run a single SkyPilot or dstack training job at 50+ nodes,
  and what broke first?
