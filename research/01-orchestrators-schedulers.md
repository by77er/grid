# 01 - Orchestrators and schedulers for distributed training

Research date: 2026-11-09. Scope: how Slurm, Kubernetes-based stacks, Ray, Nomad and
Slurm-on-Kubernetes model a multi-node training job, gang-schedule it, inject worker identity,
express heterogeneity and topology, and recover from failure, and how heavy each is to stand
up. Sibling files cover adjacent ground: `02-multicloud-provisioners.md` (SkyPilot, dstack),
`03-rank-configuration.md` (launcher-level rank/rendezvous), `04-failure-taxonomy-recovery.md`,
`05-networking-interconnects.md`, `07-provider-landscape.md`, `09-fast-bootstrap-preflight.md`.

Claims carry a source URL. Anything marked "(unverified)" comes from prior knowledge and was not
re-confirmed against a primary source in this sweep.

## Summary

- Every system reduces to the same five primitives: (1) an **atomic group** of N workers,
  (2) a **stable identity** per worker (index plus resolvable address), (3) a **rendezvous
  endpoint** (rank 0 address and port, or a hostfile), (4) a **placement constraint**
  (node class plus network domain), and (5) a **restart scope** (process, worker, group, whole
  job). They differ mostly in who owns the node agent and how much privilege that agent needs.
- Restart scope for tightly coupled training is almost universally **"tear down the whole worker
  group and restart from checkpoint"**: Slurm kills the allocation on node failure by default,
  JobSet's default action is `RestartJobSet`, Ray Train shuts down all workers and starts a new
  group. Partial restart exists (JobSet `RestartJob`, Slurm `--no-kill`, Kueue TAS hot swap of a
  single node) but only helps if the training framework can re-form its world.
- Kubernetes gang scheduling is consolidating into the core scheduler: the Workload/PodGroup
  API went alpha in v1.35, was reworked in v1.36 and gang scheduling is beta (still disabled by
  default) in v1.37. Until that lands widely, gang semantics come from Kueue, Volcano, KAI or
  the coscheduling plugin.
- Everything Kubernetes-shaped (kubelet, GPU Operator, Slinky, soperator, KubeRay) assumes you
  own the node: kubelet needs real mounts, device access and a container runtime. On a
  marketplace that hands out **unprivileged containers** (RunPod pods), the one public account
  of an attempt, K3s inside a RunPod pod, failed at the kubelet.
- What does run in an unprivileged container: user-space agents that only need outbound or
  intra-cluster TCP and the ability to fork processes: `sshd` plus `mpirun`/`torchrun`,
  `ray start`, a single-user Flux instance, and Slurm with cgroup enforcement switched off
  (RunPod's own managed Slurm cluster product runs inside its PyTorch container image).
- Recommended pattern for `grid`: an **external, provider-API-driven control plane** that does
  gang acquisition itself, plus a **thin in-container agent** that implements identity
  injection, rendezvous, health and group restart. Borrow the *contracts* (JobSet's failure
  rules, Kueue's `waitForPodsReady` timeout plus requeue, Slurm's feature/GRES vocabulary and
  env var names) and not the daemons. Offer a Slurm or Ray "personality" on top as an optional
  layer for users who want `srun`/`sbatch` or Ray Train.

## 1. Comparison at a glance

| System | Job unit | Gang / all-or-nothing | Identity and peer discovery | Default restart scope | Needs on each node |
|---|---|---|---|---|---|
| Slurm | Job (allocation) -> steps -> tasks; het job = several components | Native: an allocation is granted whole; het job components "are either started together or deferred" | `SLURM_PROCID`, `SLURM_NODEID`, `SLURM_JOB_NODELIST`, PMIx/PMI2 via `srun` | Whole allocation killed on node failure unless `--no-kill`; batch job requeued if `--requeue` | `slurmd` (normally root), munge key, cgroup v2 for enforcement |
| Kubeflow Trainer v2 | `TrainJob` -> `JobSet` -> indexed `Job`s -> Pods | Via `podGroupPolicy` (coscheduling/Volcano) or Kueue | `PET_*` env vars, headless-service DNS names | JobSet `failurePolicy` (default `RestartJobSet`) | kubelet + runtime + GPU device plugin |
| JobSet | `replicatedJobs[]` of Job templates | None itself; pairs with Kueue or a gang scheduler | `<jobset>-<rjob>-<jobIdx>-<podIdx>.<subdomain>` DNS, labels | Recreate all child Jobs, up to `maxRestarts` | as above |
| Volcano | `vcjob` with `tasks[]`, `minAvailable` | Native `gang` plugin on PodGroup | job plugins `ssh`, `svc`, `env` | `policies` event -> action (e.g. `PodEvicted` -> `RestartJob`), `maxRetry` | as above + Volcano scheduler |
| Kueue | `Workload` wrapping any job CRD | Quota admission + TAS + `waitForPodsReady` | none (delegates to job CRD) | evict and requeue with backoff | as above + Kueue controller |
| KAI Scheduler | PodGroup (hierarchical) created by podgrouper | Native | none | scheduler only | as above |
| MPI Operator | `MPIJob` (launcher + workers) | none itself | hostfile + SSH | `runPolicy` | as above |
| LeaderWorkerSet | replica = 1 leader + N workers | none itself | `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX` | `RecreateGroupOnPodRestart` | as above |
| Ray Train (+KubeRay) | Trainer run on a Ray cluster; `RayJob` on K8s | Ray waits for all workers; K8s-level gang via Kueue/Volcano/YuniKorn/KAI | Ray actors; Train sets up the torch process group | Whole worker group restarted, `FailureConfig(max_failures=...)` | `raylet` (unprivileged user process) |
| Nomad | job -> group (`count`) -> task | No documented gang semantics found | Nomad runtime env / service discovery | per-allocation restart/reschedule | Nomad client (root for most drivers) (unverified) |
| Slinky / soperator | Slurm cluster as K8s CRs | Slurm's | Slurm's | Slurm's | K8s >= 1.35 plus everything Slurm needs |

## 2. Slurm

### Job model

Jobs own an allocation of nodes; `srun` launches steps of tasks inside it. Each task gets
`SLURM_PROCID` ("the MPI rank (or relative process ID) of the current process"), `SLURM_NODEID`,
`SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_JOB_NODELIST`, `SLURM_STEP_NODELIST`,
`SLURM_TASKS_PER_NODE`, `SLURM_GPUS_ON_NODE`, and `SLURM_LAUNCH_NODE_IPADDR`
(https://slurm.schedmd.com/srun.html). In direct-launch mode Slurm itself bootstraps MPI through
the PMI2 or PMIx APIs (same page), so no SSH or hostfile is needed. For PyTorch the usual idiom
is one task per GPU with rank derived from `SLURM_PROCID` and the master address from the first
host in `SLURM_JOB_NODELIST`; details in `03-rank-configuration.md`.

Task-to-node layout is controlled with `--distribution=block|cyclic|plane=<n>|arbitrary`;
`arbitrary` follows a file named by `SLURM_HOSTFILE`, and with a topology plugin the default
order follows the topology (https://slurm.schedmd.com/sbatch.html).

### Gang scheduling

Slurm's allocation is inherently all-or-nothing: a job stays `PENDING` until its full node
request can be met; `--nodes=min-max` relaxes it to "as many nodes as possible within the range
without delaying its start" (https://slurm.schedmd.com/sbatch.html). (Slurm's own term "gang
scheduling" means time-slicing, a different thing.)

### GRES and heterogeneity

- `gres.conf` keys: `Name`, `Type`, `File`, `Cores`, `Count`, `Flags`, `MultipleFiles`,
  `AutoDetect=nvml|nvidia|rsmi|nrt|oneapi`; `slurm.conf` needs `GresTypes=gpu` and per-node
  `Gres=gpu:<type>:<n>` (https://slurm.schedmd.com/gres.html).
- Requests: `--gres=gpu:<type>:<n>`, `--gpus`, `--gpus-per-node`, `--gpus-per-task`,
  `--cpus-per-gpu`, `--mem-per-gpu`, `--gpu-bind`. Constraint: "Requests for typed vs non-typed
  generic resources must be consistent within a job" (same page).
- `CUDA_VISIBLE_DEVICES` is set per step. When cgroups constrain device files, the value differs
  between Prolog/Epilog and the job (same page). Without device cgroups, isolation is advisory
  only.
- Node features and `--constraint` give a small algebra over heterogeneous pools: `a&b`, `a|b`,
  matching-OR `[a|b]` (all nodes share one choice), and counted AND `[a*2&b*4]` (2 nodes with
  `a`, 4 with `b`); `--prefer` is the soft form. Only one bracketed set per job, and "multiple
  counts may produce a non-optimal network layout" (https://slurm.schedmd.com/sbatch.html).
- **Heterogeneous jobs**: components separated by `:` on the command line or by `#SBATCH hetjob`
  lines. Env: `SLURM_HET_SIZE`, `SLURM_JOB_ID_HET_GROUP_<n>`, `SLURM_JOB_NODELIST_HET_GROUP_<n>`.
  One `srun` across components yields a single `MPI_COMM_WORLD` with non-overlapping task IDs and
  a global `SLURM_PROCID`. Only `sched/backfill` schedules them, and all components start
  together or are deferred. Limitations: no job arrays, one step per component per `srun`,
  overlapping partitions can starve later components
  (https://slurm.schedmd.com/heterogeneous_jobs.html).

### Topology

- Plugins: `topology/tree` (`SwitchName=... Nodes=...`/`Switches=...`, job option
  `--switches=<count>[@<max-wait>]`), `topology/block` (`BlockSizes`, job option
  `--segment=<size>`, total node count must divide by segment size), `topology/ring` (added in
  26.05), `topology/torus3d` (https://slurm.schedmd.com/topology.html).
- 25.05 introduced `topology.yaml` with multiple named topologies, selected per partition
  (same page). 25.11 added `--consolidate-segments`, `--spread-segments` and
  `TopologyParam=BlockAsNodeRank`
  (https://slurm.schedmd.com/news.html, https://slurm.schedmd.com/SC25/Slurm_BoF_SC25.pdf).
- Dynamic nodes can declare their place at registration:
  `slurmd -Z --conf "... Topology=topo-switch:s1,topo-block:b1"` or `scontrol update`
  (https://slurm.schedmd.com/topology.html).

### Dynamic nodes (the relevant mode for ephemeral clusters)

`slurmd -Z --conf "RealMemory=80000 Gres=gpu:2 Feature=f1"` self-registers a node; the
controller learns `NodeAddr`/`NodeHostname` from the registration. Requires
`SelectType=select/cons_tres` and `MaxNodeCount=<n>`; partitions pick dynamic nodes up via
`Nodes=ALL` or a `NodeSet` keyed on a `Feature`. GPU GRES is auto-detected if `--conf` and
`gres.conf` give none. Removal: `scontrol delete nodename=...`, only when idle. Caveat: dynamic
nodes are not sorted, which "can cause suboptimal allocations if node names reflect physical
topology" (https://slurm.schedmd.com/dynamic_nodes.html).

### Failure and requeue

- Default: if any allocated node fails the whole allocation is terminated. `--no-kill` keeps the
  allocation so new steps can be launched on survivors
  (https://slurm.schedmd.com/sbatch.html).
- `--requeue` makes a batch job eligible for requeue after node failure, preemption or admin
  action; it "restarts from the beginning with the same job ID". Cluster default is `JobRequeue`
  (same page). The requeue count is exposed as `SLURM_RESTART_COUNT` (unverified in this sweep).
- 25.11 added **expedited requeue**: `SlurmctldParameters=enable_expedited_requeue` plus
  `--requeue=expedite`; the job is requeued immediately at top priority and its previous nodes
  are held for it; it auto-requeues on node failure or non-zero batch exit
  (https://slurm.schedmd.com/news.html).
- `--signal=[B:]<sig>[@<secs>]` delivers a warning signal before the time limit (default 60 s,
  may be up to 60 s early), which is the standard hook for "checkpoint now"
  (https://slurm.schedmd.com/sbatch.html).

### Operational weight and privilege

Minimum: `slurmctld` on one node, `slurmd` on every node, a shared munge key, consistent
`slurm.conf` (or configless). `slurmd` in a container has been supported since 23.02 and
"requires cgroup v2", with only limited cgroup v1 support in privileged mode
(https://slurm.schedmd.com/SC24/Containers.pdf). Where cgroup delegation is unavailable, users
report falling back to `ProctrackType=proctrack/linuxproc`, at the price of no enforcement ("you
can submit a job for 1 core, but then use 20") (slurm-users threads, e.g.
https://groups.google.com/g/slurm-users/c/JI3UkbCtj3U). Existence proof on the target platform:
RunPod's managed "Slurm Cluster" product has "Slurm and munge ... pre-installed and fully
configured", designates controller and agent nodes automatically, writes
`/etc/slurm/slurm.conf` and `/etc/slurm/gres.conf`, and only works with the official RunPod
PyTorch images (https://docs.runpod.io/instant-clusters/slurm-clusters). Which proctrack/cgroup
settings RunPod uses was not found.

## 3. Kubernetes stacks

### 3.1 Native building blocks

- **Indexed Job + headless Service** gives stable per-pod index and DNS; everything below builds
  on it.
- **Core gang scheduling (KEP-4671)**: Workload API alpha in v1.35 (`scheduling.k8s.io/v1alpha1`),
  replaced by `v1alpha2` in v1.36 with topology-aware workload scheduling in alpha
  (https://kubernetes.io/blog/2026/05/13/kubernetes-v1-36-advancing-workload-aware-scheduling/).
  In v1.37 gang scheduling is "Beta ... disabled by default" behind `GenericWorkload`, with
  `PodGroup` at `scheduling.k8s.io/v1beta1` and a `gang` policy field `minCount`: pods are held
  at PreEnqueue until `minCount` exist, then bound atomically or not at all. Hierarchical
  `CompositePodGroup` (`minGroupCount`) is alpha. The scheduler "never admits fewer Pods than
  the configured `minCount` during initial placement", but the running count may later drop
  below it (https://kubernetes.io/docs/concepts/scheduling-eviction/gang-scheduling/).
- **DRA**: enabled by default from v1.34; NVIDIA's DRA driver needs Kubernetes >= 1.32 and
  provides `ComputeDomain`s (IMEX channels for multi-node NVLink)
  (https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/,
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3/dra-cds.html). An older
  release note says that if one node of an IMEX domain goes down the whole workload must be
  rescheduled; current status not confirmed.
- **NVIDIA GPU Operator** 26.7.x (latest listed v26.7.2) supports Kubernetes 1.33-1.37 with
  containerd 2.0-2.3 or CRI-O, and deploys: GPU driver, driver manager, container toolkit,
  device plugin, DRA driver, DCGM + exporter, NFD, GFD, MIG manager, validator
  (https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/platform-support.html).
  Driver and toolkit containers load kernel modules and rewrite runtime config on the host, so
  they need privileged host access (unverified on that page, but inherent to the function).

### 3.2 JobSet

`jobset.x-k8s.io/v1alpha2`. A JobSet holds `spec.replicatedJobs[]` (name, `replicas`, Job
`template`); child Jobs are named `<jobSetName>-<replicatedJobName>-<jobIndex>`; pods get
hostname `<jobSetName>-<replicatedJobName>-<jobIndex>-<podIndex>` under a headless service whose
name is `spec.network.subdomain` (default: the JobSet name). Labels:
`jobset.sigs.k8s.io/jobset-name`, `.../replicatedjob-name`, `.../replicatedjob-replicas`,
`.../job-index`. `spec.coordinator` (`replicatedJob`, `jobIndex`, `podIndex`) stamps a
`jobset.sigs.k8s.io/coordinator` label/annotation with the coordinator's stable endpoint on all
pods. `alpha.jobset.sigs.k8s.io/exclusive-topology` places each child Job in its own topology
domain (https://jobset.sigs.k8s.io/docs/concepts/).

Failure policy (https://jobset.sigs.k8s.io/docs/tasks/failure_policy/): `maxRestarts`, plus
ordered `rules[]` with `action`, `targetReplicatedJobs`, and `onJobFailureReasons`
(`BackoffLimitExceeded`, `DeadlineExceeded`, `PodFailurePolicy`); first matching rule wins,
default is `RestartJobSet`.

| Action | Effect | Counts toward `maxRestarts` |
|---|---|---|
| `FailJobSet` | fail immediately | n/a |
| `RestartJobSet` | recreate all child Jobs | yes |
| `RestartJobSetAndIgnoreMaxRestarts` | same | no |
| `RestartJob` | recreate only the failed Job (max 1024 replicas per ReplicatedJob) | yes |
| `RestartJobAndIgnoreMaxRestarts` | same | no |

This rule table is the cleanest published design for "classify the failure, then pick restart
scope and whether it burns retry budget" (for example: preemption restarts without consuming
budget, application crash consumes it). An in-place restart design (restart containers without
recreating pods) is reported to cut restart time from about 2 min 10 s to about 10 s in a
prototype, depending on Kubernetes 1.34+/1.35+ kubelet features; a third-party claim, not
confirmed against a JobSet release (https://docs.daocloud.io/en/blogs/2026/jobset-ai-restart/).

### 3.3 Kubeflow Training Operator v1 and Trainer v2

v1 had one CRD per framework (`PyTorchJob`, `MPIJob`, `TFJob`, ...) with `replicaSpecs`
(Master/Worker), `runPolicy`, and per-replica `restartPolicy` (unverified this sweep). v2
replaces them with one `TrainJob` that references a `ClusterTrainingRuntime` (cluster-scoped) or
`TrainingRuntime` (namespaced) through `spec.runtimeRef`; all are
`trainer.kubeflow.org/v1alpha1`, and each TrainJob is translated into a JobSet
(https://trainer.kubeflow.org/en/latest/operator-guides/runtime.html,
https://developers.redhat.com/articles/2026/09/29/distributed-training-openshift-ai-34-kubeflow-trainer-v2).
v2.2.0 went GA on 2026-03-24 (https://blog.kubeflow.org/kubeflow-trainer-v2.2-release/).

- Runtime template: `replicatedJobs[].name: node`, pod label
  `trainer.kubeflow.org/trainjob-ancestor-step: trainer`, `mlPolicy.numNodes`
  (runtime guide above).
- **Torch**: launched via `torchrun`; the plugin injects `PET_NNODES`, `PET_NPROC_PER_NODE`,
  `PET_NODE_RANK`, `PET_MASTER_ADDR`, `PET_MASTER_PORT` into the trainer container only, and
  `torch.envInjection.targets` extends that to init/sidecar containers (their example target is a
  container literally named `preflight-check`)
  (https://trainer.kubeflow.org/en/latest/operator-guides/ml-policy.html).
- **MPI**: `mlPolicy.mpi` with `numProcPerNode` (slots in the hostfile), `mpiImplementation:
  OpenMPI`, `sshAuthMountPath: /home/mpiuser/.ssh`, `runLauncherAsNode: true`; command/args go to
  the launcher only, workers keep the runtime command that starts `sshd` (same page).
- **Flux** (new in v2.2): `mlPolicy: flux` bootstraps MPI over ZeroMQ instead of SSH, with PMIx;
  an init container installs Flux into `/mnt/flux`, and the plugin generates broker config and
  CURVE certificates. **JAX**: injects `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`,
  `JAX_COORDINATOR_ADDRESS` (v2.2 release blog above).
- v2.2 breaking changes relevant to design: `PodTemplateOverrides` replaced by manager-keyed
  `RuntimePatches`; `ElasticPolicy` removed ("Elastic training is not yet available in this
  release"). Roadmap: MNNVL, DRA, Workload-API gang scheduling after Kubernetes v1.36,
  redesigned elastic training (same blog).
- Gang scheduling is delegated: `podGroupPolicy` (coscheduling plugin or Volcano) or Kueue
  (`podGroupPolicy` field details unverified; the job-scheduling doc page 404ed).

### 3.4 MPI Operator and LeaderWorkerSet

- `MPIJob` (`kubeflow.org/v2beta1`): `mpiReplicaSpecs` with one `Launcher` running `mpirun` and
  N `Worker`s, `slotsPerWorker`, `runPolicy.cleanPodPolicy`
  (https://github.com/kubeflow/mpi-operator). v2 uses SSH between launcher and workers with an
  operator-generated hostfile in a ConfigMap (unverified this sweep). In Trainer v2 this role is
  absorbed by the MPI and Flux runtimes.
- `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`, v1.0.0 stable): a replica is 1 leader + N
  workers; restart policies `RecreateGroupOnPodRestart`, `None`, `RecreateGroupAfterStart`; one
  headless service per replica (`UniquePerReplica`) (https://lws.sigs.k8s.io/docs/concepts/).
  Injects `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE`, `LWS_WORKER_INDEX`; annotation
  `leaderworkerset.sigs.k8s.io/exclusive-topology`; a per-group restart budget
  (`maxGroupRestarts`)
  (https://lws.sigs.k8s.io/docs/reference/labels-annotations-and-environment-variables/).
  LWS targets multi-node inference, but its three env vars are the minimal identity contract.

### 3.5 Kueue

Kueue (v0.19.0 announced 2026-07-22; docs list v0.20) queues and admits workloads; it does not
place pods except through TAS. All-or-nothing is layered
(https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/):

1. Quota admission: a Workload is unsuspended only when quota for all pod sets is reserved.
2. **Topology Aware Scheduling**: tracks free capacity per node and domain and admits only if
   all pods fit.
3. `waitForPodsReady`: timeout-based safety net; if pods are not ready in time the workload is
   evicted, quota released and it is requeued with backoff. Enabled by default with a 30-minute
   timeout since v0.19 (https://groups.google.com/a/kubernetes.io/g/wg-batch/c/U4S5Nmh7puc).
   `recoveryTimeout` bounds how long a running workload may wait for a replacement pod.
4. `ProvisioningRequest` admission checks (`check-capacity.autoscaling.x-k8s.io`,
   `best-effort-atomic-scale-up.autoscaling.x-k8s.io`): admit only once the autoscaler reports
   capacity; atomicity "depends on the provider".

TAS specifics (https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/): a
`Topology` object lists levels as node labels and is referenced from a `ResourceFlavor`
(`spec.topologyName`). Pod-template annotations: `kueue.x-k8s.io/podset-required-topology`,
`...podset-preferred-topology`, `...podset-unconstrained-topology`,
`...podset-slice-required-topology` + `...podset-slice-size`. Node failure: "hot swap" replaces
one failed node while keeping the rest of the assignment (`TASFailedNodeReplacement`, beta since
v0.14); multiple failures cause eviction; with `TASFailedNodeReplacementFailFast` Kueue tries
once and then evicts and requeues. If the workload fills its whole domain there is no
replacement, hence the advice to use FailFast or `recoveryTimeout`. Heterogeneous pools are
expressed as ordered `ResourceFlavor`s (node labels + tolerations) in a `ClusterQueue`.

### 3.6 Volcano

`vcjob` has job-level `minAvailable`, `tasks[].replicas`, `policies` (event -> action, e.g.
`PodEvicted` -> `RestartJob`, `TaskCompleted` -> `CompleteJob`), `maxRetry`, `queue`, and job
`plugins` such as `ssh`, `svc`, `env`; the TensorFlow example reads `/etc/volcano/ps.host` and
`/etc/volcano/worker.host` and `VK_TASK_INDEX` (https://volcano.sh/en/docs/vcjob/). So Volcano
ships its own identity injection (host files, passwordless SSH, task index env).

Network topology (https://volcano.sh/en/docs/network_topology_aware_scheduling/): `HyperNode`
CRD (`topology.volcano.sh/v1alpha1`) with `spec.tier` and `members[]` selected by `exactMatch`,
`regexMatch` or `labelMatch`. Jobs set `spec.networkTopology.mode: hard|soft` and
`highestTierAllowed`; `hard` keeps the job Pending until it fits in one HyperNode at or below
that tier. HyperNodes can be discovered from UFM (InfiniBand) or node labels; RoCE discovery is
"currently not supported".

### 3.7 KAI Scheduler

NVIDIA's open-sourced Run:ai scheduler, a CNCF sandbox project: gang scheduling of PodGroups,
hierarchical queues with DRF and time-based fairshare, TAS and hierarchical PodGroups (v0.10.0,
2025-10), DRA support, elastic workloads with min/max members, KubeRay integration. Helm install
into `kai-scheduler`; GPU Operator needed only for GPU workloads
(https://github.com/NVIDIA/KAI-Scheduler). Latest version seen is v0.13.0
(https://pkg.go.dev/github.com/NVIDIA/KAI-scheduler@v0.13.0). Which workload CRDs its
podgrouper recognises was not confirmed.

### 3.8 Operational weight of the Kubernetes path

A working training stack is: control plane, CNI, kubelet + containerd on every node, GPU
Operator (or pre-baked drivers + device plugin), cert-manager, JobSet, Trainer, and a
queue/gang layer (Kueue or Volcano or KAI). All of it presumes root on the node. The one
practitioner report of trying K3s inside an unprivileged RunPod pod concluded that the platform
blocks filesystem binds, kernel modules and device file operations, and that the kubelet "needs
real disk bind-mounts, only possible on VMs or privileged clusters"
(https://davidgao7.github.io/posts/wrong-vllm-production-stack-setup; a single personal
account). Docker-in-pod without privileges is likewise generally not feasible without a special
runtime such as Sysbox, which a tenant cannot install.

## 4. Ray

- **Job model**: a Ray cluster is a head (GCS, dashboard, driver) plus worker `raylet`s. Ray
  Train's trainer requests a worker group (`ScalingConfig`) and runs the training function on
  each worker, setting up the framework's process group itself; users do not handle ranks.
- **Standing up without privileges**: `ray start --head --port=6379` on one node,
  `ray start --address=<head>:6379` on the rest, as an ordinary user; `--num-gpus` overrides
  detection. `ray up` with a `head_ip`/`worker_ips`/`ssh_user` config does the same over SSH
  (https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
  Ray 2.59 turned token authentication on by default for local clusters and 2.61 for all
  clusters (banner on docs.ray.io pages).
- **Failure handling** (https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html):
  on a worker process or node failure Ray Train "shuts down all workers, adds nodes if needed,
  and starts a new worker group", resuming from the latest reported checkpoint
  (`ray.train.get_checkpoint()`). Driver/head loss is recovered by relaunching with the same
  `RunConfig(storage_path=..., name=...)`; driver interruptions do not count against
  `max_failures`. Train V2 is available from Ray 2.43 (`RAY_TRAIN_V2_ENABLED=1`).
- `FailureConfig` signature
  (https://docs.ray.io/en/latest/train/api/doc/ray.train.FailureConfig.html):
  `max_failures=0` (0 disables, -1 unlimited), `max_preemption_failures=-1` (counted separately
  from `max_failures`), `controller_failure_limit=-1`, `relax_collectives_on_preemption=False`,
  `preemption_grace_s=0.0`. Separate budgets for preemption and for real failures mirror JobSet's
  "ignore max restarts" actions; `relax_collectives_on_preemption` has no effect if rank 0 is
  the preempted node.
- **KubeRay `RayJob`**
  (https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html):
  `submissionMode` = `K8sJobMode` (default), `HTTPMode`, `InteractiveMode`, `SidecarMode`;
  top-level `backoffLimit` (default 0) where each retry creates a new RayCluster;
  `submitterConfig.backoffLimit` (default 2); `activeDeadlineSeconds`;
  `preRunningDeadlineSeconds` (fail if not Running in time, a provisioning deadline);
  `shutdownAfterJobFinishes`, `ttlSecondsAfterFinished`; `suspend` (owned by Kueue);
  rule-based `deletionStrategy` (`DeleteCluster`, `DeleteWorkers`, `DeleteSelf`, `DeleteNone`).
- **Gang scheduling on K8s** is external: operator flag `--batch-scheduler=volcano`
  (https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html); KubeRay 1.5
  added native YuniKorn and Volcano support for RayJob (previously the submitter was scheduled
  separately), plus KAI integration (https://www.anyscale.com/blog/kuberay-v1-5); Kueue admits
  RayJob/RayCluster all-or-nothing
  (https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kueue.html).
- **Heterogeneity**: worker groups with different resource shapes and custom resources;
  Autoscaler V2 worker-group priorities arrived in KubeRay 1.6/1.7 (secondary source:
  https://www.plushcap.com/content/anyscale/blog/anyscale-introducing-kuberay-v16-and-v17).
  No network-topology concept in Ray itself was found; it relies on the underlying scheduler.

## 5. Nomad

A job contains groups (`count` instances) of tasks. GPUs are requested with a `device` block
under `resources` (docs are for Nomad v2.1.x): name `"nvidia/gpu"` or
`"nvidia/gpu/<model>"`, `count`, hard `constraint` and weighted `affinity` over
`${device.attr.memory}`, `${device.model}`, `${device.ids}`
(https://developer.hashicorp.com/nomad/docs/job-specification/device). That attribute-based
constraint/affinity model is a neat way to express "at least 40 GiB, prefer 80 GiB". The NVIDIA
plugin fingerprints devices per node
(https://www.hashicorp.com/blog/using-hashicorp-nomad-to-schedule-gpu-workloads).

What was not found: any documented all-or-nothing semantics for a group, an ML job abstraction,
rank/world-size injection, or network-topology placement. Nomad looks like a capable generic
placer with no distributed-training layer. Client privilege needs (root for the docker/exec
drivers) are unverified here.

## 6. Slurm on Kubernetes

| | Slinky slurm-operator (SchedMD/NVIDIA) | soperator (Nebius) |
|---|---|---|
| Version seen | 1.2.0 announced 2026-07-02; compatibility matrix lists 1.3.x | version not found; ships Slurm `26.05.4-nebius-1` |
| Requires | Kubernetes >= 1.35, cert-manager, Helm; Slurm >= 25.11 | Kubernetes >= 1.35, Helm >= 3.14, Cilium recommended, GPU Operator, Ubuntu 24.04, CUDA 13.0.3 |
| Model | CRDs: `NodeSet`, `LoginSet`, `Accounting`, `RestApi`, `Token`; NodeSet as StatefulSet (fixed replicas) or DaemonSet (one slurmd pod per matching node) | `SlurmCluster` CR; shared-root "jail" PVC "mounted into each login and worker node as its root" |
| Elasticity | NodeSets scale to zero, HPA-compatible; nodes set to DRAIN before pod termination | workers ephemeral; scale-down leaves stale nodes in `sinfo` until removed with `scontrol` |
| Health | Slurm node state mirrored into pod conditions | passive checks plus scheduled `ActiveCheck` GPU/system/storage/network probes; failed nodes drained and replaced |
| Limits | controller HA needs RWX PVC; cgroups were disabled in early versions because Kubernetes "does not natively allow delegation of a cgroup sub-tree to a pod" (current status unconfirmed) | single partition; GPU nodes only; `scontrol`/file edits overwritten on reconcile |
| Sources | https://github.com/SlinkyProject/slurm-operator, https://groups.google.com/g/slurm-users/c/k03FWLE-0g4, https://slurm.schedmd.com/SLUG24/Slinky-Slurm-Operator.pdf | https://github.com/nebius/soperator, https://github.com/nebius/soperator/blob/main/docs/limitations.md |

Slinky 1.1.0 added dynamic topology: worker pods register with topology derived from their
Kubernetes node (search summary of Slinky release notes; unverified at source). Community
deployments commonly grant privileged security contexts (an OpenShift walkthrough uses a
privileged SCC and notes a more restrictive one is planned:
https://github.com/redhat-hpc/slinky-on-openshift). Both projects are strictly heavier than
Kubernetes alone, so neither fits an unprivileged-container provider. Their value to `grid` is
as design references: the shared-root jail (identical userland on all nodes without rebuilding
images), drain-before-terminate, and scheduled active health checks feeding node state.

## 7. Cross-cutting mechanics

### How identity and peer discovery get injected

| Mechanism | Used by | Notes |
|---|---|---|
| Env vars with rank, world size, master addr/port | Trainer v2 (`PET_*`), LWS (`LWS_*`), JAX runtime, RunPod Instant Clusters, dstack | Simplest; computed by the control plane before start, so replacing a node means restarting the others or re-rendering |
| Stable DNS from index | JobSet, StatefulSet-based systems | Survives pod replacement because name follows index and not IP; needs cluster DNS |
| Hostfile + SSH | MPI Operator, Trainer MPI runtime, Volcano `ssh`/`svc` plugins | Needs `sshd` and key distribution; works in unprivileged containers |
| Launcher-owned wire-up (PMIx/PMI2, ZeroMQ) | Slurm `srun`, Flux | No SSH; requires the resource manager's daemon on every node |
| Actor runtime | Ray | Identity is hidden; framework wires itself |

The provider-native contract on RunPod Instant Clusters is already the env-var form:
`PRIMARY_ADDR`/`MASTER_ADDR`, `PRIMARY_PORT`/`MASTER_PORT`, `NODE_ADDR`, `NODE_RANK` (0 on the
primary), `NUM_NODES`, `NUM_TRAINERS` (GPUs per node), `HOST_NODE_ADDR`, `WORLD_SIZE`; inter-node
traffic must use `ens1`..`ens8` (`NCCL_SOCKET_IFNAME=ens1`), not `eth0`
(https://docs.runpod.io/instant-clusters/configuration). Standard clusters are 2-8 nodes (16-64
GPUs), up to 512 GPUs via sales (https://docs.runpod.io/instant-clusters). dstack exposes
`DSTACK_MASTER_NODE_IP`, `DSTACK_NODES_IPS`, `DSTACK_NODE_RANK` and inter-node SSH on port
10022 (https://dstack.ai/docs/concepts/tasks/).

### How heterogeneity and topology are expressed

| System | Node class | Network locality |
|---|---|---|
| Slurm | `Feature=` + `--constraint` algebra, typed GRES, het-job components | `--switches`, `--segment`, per-partition topology, dynamic `Topology=` |
| Kueue | ordered `ResourceFlavor`s per ClusterQueue | `Topology` levels + required/preferred/slice annotations |
| Volcano | node selectors / queues | `HyperNode` tiers, `networkTopology.mode` hard/soft |
| JobSet / LWS | one ReplicatedJob per class | exclusive-topology annotation (1 job or group per domain) |
| Core K8s | DRA device attributes | Workload API topology-aware scheduling (alpha, v1.36) |
| Nomad | `device` constraint/affinity on attributes | none found |
| Ray | worker groups, custom resources | none found |

Common shape: a hard level ("must fit within X"), a soft level ("prefer X, widen if needed"),
and a slice size ("groups of k must be co-located").

### Restart policy patterns worth copying

1. Classify, then choose scope and budget (JobSet rules; Ray's separate
   `max_preemption_failures`).
2. Provisioning deadline distinct from run deadline (Kueue `waitForPodsReady.timeout`, RayJob
   `preRunningDeadlineSeconds` vs `activeDeadlineSeconds`).
3. Recovery deadline for replacing a lost member (Kueue `recoveryTimeout`, TAS FailFast).
4. Hold the surviving nodes while the job restarts (Slurm expedited requeue) so a restart does
   not lose the hard-won allocation.
5. Warning signal before termination (Slurm `--signal`, Ray `preemption_grace_s`).
6. Same job identity across restarts with a visible restart counter (Slurm requeue keeps the job
   ID; resume keyed on `storage_path` + `name` in Ray).

## Implications for grid

1. **Do not put Kubernetes on the nodes.** On unprivileged-container providers the kubelet
   cannot run, and the whole K8s family (GPU Operator, Kueue, Volcano, KAI, Trainer, Slinky,
   soperator) is unreachable. Keep K8s as a possible *backend* (emit a JobSet/TrainJob when the
   provider sells a managed cluster), not as the substrate.
2. **External control plane, thin agent.** The scheduler role collapses to: acquire N pods via
   the provider API, treat readiness of all N as the gang condition, then start. Implement the
   Kueue pattern directly: a provisioning timeout, release-and-retry with backoff if the gang
   does not form, and optionally accept `min..max` nodes like Slurm `--nodes=min-max`.
3. **Agent requirements**: runs as an ordinary process in the user's image, dials out to the
   control plane (inbound ports are not guaranteed), needs no cgroup, mount or device
   privileges, and provides: identity injection, peer table, process-group launch, log/exit
   relay, heartbeat, and warning-signal delivery.
4. **Adopt an existing identity contract.** Emit a superset so existing scripts work unchanged:
   torchrun's `PET_*`, RunPod's `NODE_RANK`/`WORLD_SIZE`/`PRIMARY_ADDR`, and Slurm-style
   `SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_JOB_NODELIST`, plus a generated hostfile and SSH keys for
   MPI-style launchers. Give nodes index-derived stable names (JobSet style) so a replacement
   node can take over an index.
5. **Failure policy as data.** Copy JobSet's rule table: match on failure class (preemption,
   node lost, GPU fault, app non-zero exit, provisioning timeout) -> action
   (`restart-group`, `replace-node-then-restart-group`, `fail`) with a flag for whether it
   consumes the retry budget. Default scope is whole-group restart from checkpoint, since that
   is what every system does for tightly coupled training.
6. **Hold survivors during restart** (Slurm expedited-requeue idea): on a single-node loss keep
   the other N-1 pods warm (image, dataset cache, checkpoint already local), replace one, and
   relaunch. Add a recovery timeout after which the whole gang is released.
7. **Heterogeneity vocabulary**: node "features" plus typed GPU counts, with Slurm-like
   constraint forms (uniform choice `[a|b]`, counted mixes `[a*2&b*4]`) and Nomad-like attribute
   predicates (`gpu.memory >= 40GiB`, soft preference weight). Het-job-style components map to
   rank ranges so mixed pools still get one contiguous world.
8. **Topology as measured labels.** Providers rarely publish fabric topology, so derive levels
   (same host, same cluster network, same datacenter, WAN) from provider metadata and preflight
   measurements, then expose hard/soft/slice constraints in the Kueue/Volcano shape.
9. **Optional personalities, layered on top.** (a) Ray: `ray start` works unprivileged and gives
   Ray Train users group-restart for free. (b) Slurm: dynamic nodes (`slurmd -Z --conf ...`)
   with cgroup enforcement off, for users who want `sbatch`/`srun`; RunPod's own product shows
   it is viable in their containers. (c) Flux single-user instance as an SSH-free PMIx launcher.
   None should be on the critical path to first training step.
10. **Health as scheduling input**: mirror soperator's active checks and Slinky's
    drain-before-terminate: nodes failing preflight never join the gang, and a node flagged
    mid-run is drained at the next checkpoint boundary.

## Open questions

- **RunPod container capabilities, precisely**: which Linux capabilities, cgroup v2 delegation,
  `/dev` access and memlock limits a pod gets, and what Slurm settings (`ProctrackType`,
  `TaskPlugin`, cgroup config) RunPod's managed Slurm uses. Needs a hands-on probe; overlaps
  `07-provider-landscape.md`.
- **Slurm fully unprivileged**: can `slurmd` run as non-root with munge in a container, and what
  breaks (multi-user job launch, device constraint, accounting)? No authoritative source found.
- **Slinky current security context and cgroup status** in 1.2/1.3; the release notes and Helm
  values were not reachable in this sweep.
- **Partial-restart feasibility**: JobSet `RestartJob`, Kueue hot swap and Slurm `--no-kill`
  only help if the framework can re-form its world without restarting survivors. Which launchers
  (torchrun elastic agent, PMIx, Ray) support that today belongs with
  `03-rank-configuration.md` and `04-failure-taxonomy-recovery.md`.
- **JobSet in-place restart**: release status, exact API, and the measured 10x restart speedup
  need confirming from the JobSet repo and KEPs.
- **Trainer v2 `podGroupPolicy` and current Kueue/Volcano topology integration**: doc page
  404ed; also what replaces the removed `ElasticPolicy`.
- **Nomad**: gang semantics, client privilege requirements, and whether anyone runs multi-node
  training on it at scale; no evidence found either way.
- **Flux as a launcher inside containers**: bootstrap details for a single-user instance across
  nodes without a shared filesystem, and real startup times at hundreds of nodes.
- **Scale behaviour**: none of the sources gave time-to-ready numbers for these control planes
  at 100-1000 nodes (Slurm dynamic registration rate, JobSet pod creation rate, Ray head
  limits). Worth a dedicated measurement-oriented sweep with `09-fast-bootstrap-preflight.md`.
- **Volcano job plugin env vars** (`VC_TASK_INDEX`, `VC_<task>_HOSTS`) and KAI's supported
  workload CRDs were not confirmed from primary docs.
