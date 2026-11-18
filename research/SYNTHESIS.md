# Synthesis

What the research in this directory says about how `grid` should be built. Each claim points
at the file that carries the evidence and sources. Both sweeps are incorporated. The
second sweep (13–17) read source code and API specs and corrected several first-sweep claims;
where the two disagree, the second-sweep file wins and lists the correction.

Nothing here was run against real hardware or a live provider account. Section 4 lists what
only an experiment can settle.

## 1. The gap grid fills

No existing tool builds a large multi-node job out of marketplace pods (02).

- SkyPilot's RunPod and Vast backends declare multi-node unsupported. dstack does multi-node
  on RunPod only through RunPod's own cluster product, which is 2–8 nodes self-serve, one GPU
  type, one data center (02, 07).
- Every tool treats a cluster as N identical nodes on one fabric, provisioned all-or-nothing,
  and recovers from any failure by restarting the whole cluster (01, 02, 04).
- Modal Clusters set the experience bar (gang scheduling, automatic NCCL/RDMA setup, start in
  seconds) but are closed and run only on Modal's capacity (02).

The unserved combination: acquire capacity in parallel across many offers and providers, admit
nodes as they pass preflight, start on a healthy quorum, and repair or shrink instead of
restarting from scratch.

## 2. Constraints that shape everything

| Constraint | Consequence | File |
|---|---|---|
| Marketplace pods are unprivileged containers | No Kubernetes on the nodes; no lazy-pull snapshotters; FUSE, TUN, DCGM diagnostics and GPU reset must each be probed, not assumed | 01, 09 |
| Plain RunPod pods have TCP/HTTP only, NAT'd ports that change on reset, 100 Mb/s private networking, no UDP | NCCL cannot span plain pods; WireGuard-style meshes are not possible there | 05, 07 |
| TCP instead of RDMA costs 10–57x on FSDP throughput | Synchronous training is only viable inside a provider-built fast fabric | 05 |
| NCCL falls back to sockets silently | Transport must be pinned and verified from logs before a job is admitted | 05 |
| Job MTBF = node MTBF / N; a 1,024-GPU job at Meta had a 7.9 h MTTF | Recovery speed and checkpoint cost matter more than avoiding failure | 04 |
| A few bad nodes cause most failures | A persistent strike list keyed on GPU and host identity is high leverage | 04 |
| SKU name does not determine hardware (SXM vs PCIe, power caps, CPU per GPU, driver version) | Every node gets fingerprinted and benchmarked; scheduling keys on the measured class | 06 |
| Host driver decides which CUDA / PyTorch image runs | Image is chosen per node from the fingerprint | 06 |
| RunPod default spend limit is $80/hour | Quota and spend caps are first-class errors, distinct from stockout | 07, 16 |
| RunPod stockout is HTTP 400 with free text, the same status as a rule violation | The driver must classify errors by comparing outcomes across candidates | 16 |
| RunPod spot is retired (re-confirmed 2026-11-18); REST v1 was retired on 2026-11-15 | Model RunPod as on-demand only, through REST v2. Spot comes from other providers, whose notice runs from none (Verda, Hyperstack) to 2 minutes (AWS, CoreWeave) | 16, 19 |
| List prices for the same 8x H100 node differ 7x across providers and drift within days | Rank on live prices and expected total cost, never on prices copied into config | 19 |
| A RunPod Instant Cluster cannot be resized and its pods cannot be deleted individually | It is a fixed-size island; repair inside it means replacing the island or shrinking the job | 16 |

## 3. Architecture the research points to

### Control plane

An external controller drives provider APIs and does gang acquisition itself. Each node runs a
thin agent, injected through the pod's start command and environment, that dials out to the
controller (01, 09). Slurm, Ray or MPI are optional layers on top, not on the path to the
first step.

### Cluster model: islands, not a flat node list

The central abstraction is the **island**: a set of nodes that share a measured fast fabric
and one hardware class.

- Inside an island: synchronous training (FSDP/DDP, with tensor parallelism kept inside a
  node's NVLink domain) (03, 05, 06).
- Across islands: a low-communication outer loop in the DiLoCo family, exchanging quantised
  parameter deltas every H steps over TCP or object storage (08). This is the only approach
  with LLM-scale evidence on slow links, and its quality cost is contested by independent
  replications, so it has to be opt-in and measurable.
  No surveyed library works as this transport on RunPod-style pods without patching. The
  recommendation is to build a small coordinator with an object-store data plane and an HTTP
  control interface, reusing `torchft`'s DiLoCo hooks and an existing delta codec (14).
  Object-store relay was used at 72B parameters, and needs zero-egress storage to be
  affordable (14).
- Provider objects map onto this: a `Fabric` (RunPod Instant Cluster, Nebius GPU cluster,
  Crusoe IB partition) is created first and nodes are created into it (07).

Heterogeneity is handled by keeping each synchronous group homogeneous and mixing only across
islands (06, 08). The same rule holds for serving: mix GPU types across replicas, never
inside one (10).

### Acquisition

- Fan out creates in parallel across offers, over-provision by 10–20%, and cull (09, Imbue's
  figures). Each RunPod create places one GPU type with no fallback and is not idempotent, so
  the driver needs a candidate ladder and list-before-retry (07).
- Separate a provisioning deadline from a run deadline (01).
- Capability flags are per offer, not per provider (07).
- On RunPod, IPs and port mappings come from a per-pod GET, not the list call; ownership has
  to be encoded in the pod name because tags cannot be set and creates have no idempotency
  key (16).

### Rank plan

- grid assigns ranks centrally and sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`,
  `MASTER_PORT` directly, or uses `torchrun`'s static mode (`--node-rank`, `--master-addr`),
  which supports unequal process counts per node (13). Dynamic rendezvous is avoided because
  its ranks follow sorted addresses and are documented as unstable (03).
- Ranks belong to stable logical slots. A replacement node inherits the slot, so data
  sharding and per-rank checkpoint files stay valid (03).
- Rank order encodes placement: adjacent ranks share a node, because Megatron and torchtitan
  derive TP/PP/DP groups by arithmetic on the global rank (03).
- Rank 0 goes on the most reliable, non-preemptible node (03).
- One canonical rank table is projected onto each launcher's dialect: torchrun static, MPI
  hostfile, DeepSpeed, `PET_*`, Slurm-style, JAX, and the vLLM/SGLang multi-node flags (03, 10).

### Preflight and admission

Preflight runs concurrently with image and data staging (09):

1. Fingerprint: GPU model and UUIDs, driver, P2P islands, CPU/RAM from cgroup limits, disk.
2. Short GPU burn and a per-node throughput benchmark.
3. Pairwise network measurement and a small NCCL all-reduce; classify each link into a tier
   and form islands from the result (05).
4. Verify the NCCL transport actually selected.

Nodes that fail go on the strike list and are replaced from the over-provisioned pool.

### Failure handling

A policy table maps a detected failure class to an action and says whether it consumes the
retry budget (04, modelled on JobSet failure rules and Ray's separate preemption budget):

| Class | Detection | Action |
|---|---|---|
| Xid 13, 31, 94 (application-level) | NVML / logs | Restart process group in place |
| Xid 48, 64, 79, 95, 119 and similar | NVML / logs | Replace node (no GPU reset in an unprivileged pod) |
| Hang (NCCL, dataloader) | Phase-aware step-time watchdog in the agent | Capture stacks, kill all ranks, restart |
| Node gone / preempted | Agent heartbeat, provider API | Tear down ranks proactively, replace or shrink |
| Fail-slow node | Per-rank step-time breakdown | Swap at next checkpoint boundary |
| Stockout on replace | Provider error taxonomy | Walk SKU / data-center / provider ladder; shrink DP |
| Suspected silent corruption | Deterministic step replay | Roll back to an older checkpoint generation |

Recovery modes, in the order grid can realistically offer them (13):

1. Whole-group restart with grid-owned rendezvous and fail-fast kill. PyTorch's legacy
   watchdog can take up to 10 minutes to notice a silent peer death, so the agent must kill
   ranks itself.
2. Hot-spare swap at fixed world size.
3. In-process re-form of the communicator. Stock PyTorch has `shrink_group()` (2.10+) and
   `init_process_group(enable_reconfigure=True)` (2.14+, private API). This corrects the first
   sweep, which said no mainstream stack keeps survivors alive.
4. Shrink-and-continue on the data-parallel dimension, with `torchft` replica groups as the
   working precedent.

Modes 3 and 4 are unproven over TCP and at scale.

Resuming on a different cluster shape depends on the job's checkpoint format (15). Only
formats that store tensors by name with a global shape can be resharded (PyTorch Distributed
Checkpoint, Megatron `torch_dist`); anything keyed by rank is locked to its shape, and
optimizer state is the binding constraint. File 15 proposes a portable checkpoint contract:
logical addressing, progress in global units, a data cursor independent of world size, and a
hashed manifest as commit marker. Jobs that do not meet it get same-shape restore only.

Checkpoints live in object storage, not provider volumes (RunPod network volumes are pinned to
a data center and attach only at creation) (04, 07). Cadence follows Young/Daly; asynchronous
sharded checkpointing is the main lever on overhead (04).

### Serving

Serving reuses provider drivers, acquisition, staging, preflight, agent and inventory. It adds
a router, autoscaler, drain protocol and rollout controller (11).

- Engines: vLLM and SGLang as first-class (10).
- Router: grid's own, on a stable host, with replicas registering themselves. RunPod's HTTP
  proxy cuts connections at 100 s and mapped TCP ports change on reset (11).
- Routing on queue depth plus prefix-cache affinity; precise cache-aware routing was worth
  about 2x in llm-d's benchmark (11).
- Autoscale on engine queue depth and KV-cache saturation, not GPU utilisation (11).
- Readiness is a real generation with a tokens/s floor, not `/health` (10, 11). vLLM exits on
  engine death and reports a dying stream as an error chunk under HTTP 200, so the router has
  to parse stream bodies (17).
- Among routers that run without Kubernetes, only SGLang Model Gateway has an HTTP API for
  adding and removing workers plus active health checks (17).
- Mid-stream failover by replaying prompt plus emitted tokens on another replica makes
  preemptible capacity usable for interactive traffic (11). No existing router does this for
  plain vLLM or SGLang backends; grid would have to write it (17).

### RL

RL needs typed pools: a reliable learner island, preemptible heterogeneous actors, a CPU
sandbox fleet, and a control node (12). On top of training and serving it needs:

- a weight plane with two tiers: engine-native push inside a fabric, object-store deltas
  across providers;
- per-replica policy-version state, because a failed weight update can leave a server unusable
  while `/health` still passes;
- a private control network, since the engines' weight-update endpoints are unauthenticated;
- metrics for tuning the actor-to-learner ratio.

### State movement as a selectable axis

How model state moves and persists is not one mechanism. It is chosen per purpose and tunable
between methods (18).

- **Seven purposes, each with its own policy:** durable recovery, fast local restart, joining
  or replacement node, cross-island sync, RL weight updates, serving rollout, and migration to
  a different shape or provider. The cheapest method for one can be unsafe for another.
- **Payload and codec matter more than transport.** At 70B parameters, full training state is
  about 945 GB, bf16 weights 140 GB, an int8 delta 70 GB, and a sparse top-k delta about
  1.9 GB: a 500x spread (derived in 18).
- **Six orthogonal fields describe every method:** `payload`, `codec`, `transport`, `trigger`,
  `commit`, `retention`, plus an optional `fallback` ladder. Picking a method is choosing a
  named preset; compromising is overriding one field.
- **Dials between methods:** checkpoint interval, mix of local / buddy / durable tiers,
  full-copy interval for delta chains, optimizer cadence relative to weights, bit-width and
  density of lossy deltas, sync interval and quorum across islands, and a staleness bound for
  RL.
- **A cost model picks defaults.** From parameter count, measured link and storage rates,
  failure-rate estimate and step time, grid predicts overhead, bytes and dollars for the chosen
  policy and its alternatives.
- **Static validation refuses unsafe combinations**, for example a lossy codec as the only
  durable copy, a delta chain with no full copy, a rank-keyed format used for migration, or a
  local-only copy on a pod's container disk.
- **Two derived results revise earlier advice.** On pods whose disk dies with the node, a
  local checkpoint tier without replication to a second node is worse than asynchronous save
  straight to durable storage when most failures lose the node. And the Young/Daly interval in
  file 04 holds only for blocking saves; asynchronous saves add a term for background persist
  time.
- **Job cooperation.** Transport, tiering, retention, hashing and commit work with only a
  directory protocol. Payload splitting, lossy codecs, live peer pull and collective push to
  engines need hooks in the training job.

### Cost-optimal provider selection, including spot

How grid picks the cheapest offer for a rental (19).

- **Live prices are readable by program almost everywhere.** AWS bulk files, Azure Retail Prices,
  Nebius spot prices, Verda instance types and Shadeform instance types answer without an
  account. RunPod, Vast, Lambda, Prime Intellect, TensorDock, Hyperstack and GCP need the user's
  key. CoreWeave and Crusoe publish only web pages, so grid keeps a dated, hand-maintained file
  for them.
- **Terms of use shape the cache.** Vast forbids caching and price indexes built from its data.
  RunPod, TensorDock and Verda forbid scraping their sites. grid queries APIs with the user's own
  credentials, caches marketplace offers for about a minute, re-checks just before each create,
  and never pools prices across users.
- **Spot exists today** at AWS, GCP, Azure, Nebius (dynamic, uncapped price), CoreWeave (fixed
  list), Vast (bid), Verda (fixed 50%, no notice), Hyperstack (no notice, nothing survives),
  Shadeform and Prime Intellect. It does not exist at RunPod or Lambda, and goes through sales at
  Crusoe. Only the hyperscalers and Vast publish any interruption data, so grid learns preemption
  rates from its own telemetry, starting from a pessimistic prior.
- **Rank on expected cost per useful work, not sticker price.** The expected cost adds:
  - billed boot time per start;
  - billing increments;
  - failed-preflight waste;
  - storage billed while stopped or waiting;
  - transfer charges;
  - checkpoint and rework overhead from the extended Young/Daly formula of 18, with notice
    salvage when a save fits in the notice.

  Offers are first filtered on hard constraints (interconnect tier, form factor, memory, region).
  They are then scaled by a measured speed factor from preflight. An island is priced at its
  slowest node.
- **Worked example (one 8x H100 node, 24 h of work, Nebius):**
  - on-demand costs $878;
  - spot costs $415-484 for per-node preemption intervals of 24 h to 2 h;
  - the cost of spot is dominated by restart time, not checkpoint cost.

  On AWS with an external checkpoint store, spot's 12x more upload bytes turn a $550 saving into
  a $531 loss. Checkpoint store placement belongs in the cost model.
- **Spot belongs where a preemption loses one unit, not the job.** That means:
  - whole DiLoCo islands;
  - RL rollout workers;
  - serving replicas behind failover;
  - single-node jobs.

  Synchronous islands stay on-demand. Mixing spot and on-demand inside one island is refused.
  Rank 0, the coordinator, the learner and the checkpoint store are never on spot.
- **Proposed components:**
  - a `catalogue` with public, account-scoped and hand-maintained sources;
  - an extended `Offer` record (capacity class, price kind, billing increment, storage and
    transfer rates, preemption terms, hardware form, measured speed, quote time);
  - `grid plan` output that shows sticker price, expected cost breakdown, wall-clock and
    sensitivity for the top plans, which then become the acquisition candidate ladder.

Earlier statements this corrects (details in 19): GCP spot notice is 0 s by default or 120 s if
configured, not 30 s (07). CoreWeave and Hyperstack spot are documented, not hearsay (07). The
spot capability row of 16 gains Nebius's dynamic pricing and Lambda's absence of spot. Prices in
06 and 07 have already drifted.

## 4. What only an experiment can settle

These recur across files and cannot be resolved by reading:

1. What a RunPod pod and an Instant Cluster node actually permit: capabilities, `/dev/fuse`,
   `/dev/net/tun`, `dmesg`, DCGM, memlock, cgroup limits.
2. Instant Cluster fabric in practice: how often RDMA is present, measured all-reduce
   bandwidth per data center and GPU type.
3. Time-to-ready distributions (p50/p95/p99) per provider, and real API rate limits and pod
   count caps.
4. Whether `podCount` above 8 is reachable without sales, and what a partial fill looks like.
5. The exact error strings RunPod returns for stockout, spend cap and rule violations, and
   whether cluster creation is all-or-nothing.
6. DiLoCo outer-loop quality at the scales grid targets, against a synchronous baseline on
   downstream evals.
7. Whether communicator shrink and re-form complete over the socket transport when a peer
   vanishes, and how long they take at 64–512 ranks.
8. Whether a save-at-shape-A, load-at-shape-B resume actually works in the target frameworks.
9. Cross-pod TCP throughput and object-store throughput from marketplace hosts, which set the
   default sync interval and quantisation.

10. Pod-to-object-store upload throughput and lossless delta ratios between nearby
    checkpoints, which set every checkpoint-interval number in file 18.
11. Spot preemption rates per provider and GPU pool, outbid rates on Vast at a given margin over
    `min_bid`, and measured speed gaps between SXM, PCIe and power-capped hosts. These set the
    priors of file 19's cost model.

A probe harness that answers 1–5 is the natural first piece of code.
