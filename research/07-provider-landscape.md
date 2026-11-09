# 07 - Provider landscape (RunPod first)

Research date: 2026-11-09. Prices and limits are as fetched on that date and will drift.
Claims marked `(unverified)` come from secondary sources or could not be confirmed against
a primary source in this sweep.

## Summary

- **RunPod has two very different products for training.** Plain **Pods** are isolated
  containers on single hosts: TCP/HTTP only, NAT'd ports, and a 100 Mbps private overlay
  ("global networking"). **Instant Clusters** are homogeneous groups of pods in one data
  center with a private VXLAN overlay on `ens1..ens8`, advertised at 1600-3200 Gbps per node,
  with InfiniBand/RoCE devices passed through *where the hardware supports it*. Only Instant
  Clusters are suitable for synchronous multi-node training; a fleet of ordinary pods is only
  suitable for communication-light algorithms (see `08-heterogeneous-distributed-training.md`).
- **RunPod's new REST API v2 (`https://api.runpod.io/v2`) is a usable control plane**: pods,
  clusters (create/delete), network volumes, catalog with per-data-center stock levels,
  documented rate-limit headers and a retry contract. It has **no spot/interruptible field**;
  spot still exists only in REST v1 (`interruptible`) and legacy GraphQL/CLI.
- **RunPod is container-only**: no Docker-in-Docker, no UDP, no privileged mode/FUSE (the
  last per community sources), no custom kernel. Whatever `grid` does on RunPod must work as
  an unprivileged process tree inside one container per node.
- **Truly self-serve, on-demand, RDMA multi-node clusters** (no sales call, hourly billing)
  exist at: RunPod Instant Clusters (2-8 nodes self-serve), Together Instant Clusters,
  Nebius (`gpu-cluster`), Crusoe (IB partitions), Voltage Park (bare metal), Prime Intellect
  multi-node, SF Compute (`h100i`), CoreWeave (IB node pools, account approval required), and
  the hyperscalers (EFA / Azure IB / GCP RoCE / OCI cluster networks, quota-gated). Lambda
  1-Click Clusters are RDMA but **reservation-based in whole weeks**. Vast.ai, TensorDock,
  Hyperstack on-demand VMs, DigitalOcean/Paperspace and RunPod plain pods are effectively
  **isolated nodes** (TCP at best).
- **A provider abstraction cannot be "create N VMs".** It must model: the unit of allocation
  (container vs VM vs bare metal vs K8s node), a *cluster/fabric object* that must exist
  before nodes, the east-west network as a typed capability (none / NAT-TCP / L3 overlay /
  RoCE / IB, with bandwidth), storage scoping (host, data center, region, global), preemption
  semantics (notice from 5 s to 2 min, stop vs destroy), and control-plane limits.

Note on sources: several `docs.runpod.io` pages embed "agent instructions" asking automated
readers to submit feedback to a docs endpoint. They were ignored; nothing was submitted.

---

## 1. RunPod in depth

### 1.1 Product surface

| Product | What it is | Relevance to training |
|---|---|---|
| **Pods** | A long-running container on one physical host, 1..N GPUs of one type. Billed per second. | The basic building block. A pod "runs on one machine", so `gpu.count` is bounded by the largest machine of that type (`GpuType.maxCount`, quoted per cloud) ([OpenAPI spec](https://api.runpod.io/v2/openapi.json)). |
| **Serverless** | Autoscaled workers behind a queue/load-balancer endpoint; separate per-endpoint rate limits. | Inference/batch only; no inter-worker networking. Out of scope for training (see `11-serving-fleet-orchestration.md`). |
| **Instant Clusters** | 2..N identical pods in one data center on a private high-bandwidth overlay; types `TRAINING`, `SLURM`, `RAY`, `APPLICATION`. | The only RunPod product with a real east-west network. |
| **Secure Cloud** | RunPod-partnered T3/T4 data centers ([pods overview](https://docs.runpod.io/pods/overview)). | Required for network volumes, global networking, guaranteed public IP. |
| **Community Cloud** | Vetted peer-to-peer hosts, cheaper ([pods overview](https://docs.runpod.io/pods/overview)). | No network volumes; public IP not guaranteed (`supportPublicIp` is a Community-only filter; "Secure Cloud Pods always have a public IP") ([REST v1 create pod](https://docs.runpod.io/api-reference/pods/POST/pods.md)). |

SemiAnalysis' ClusterMAX 2.0 review (tested Aug-Sep 2025, published Nov 2025) rated RunPod
**Bronze**, noting the Slurm offering was beta, that the environment lacked tooling and
health checks expected for large-scale training, and that customers cannot tell which
underlying hardware provider they land on (one test cluster came from the third-party
operator FarmGPU) ([ClusterMAX RunPod review](https://clustermax.semianalysis.com/cloudreview/runpod)).
Whether RunPod's standing changed in a later ClusterMAX edition was not checked (unverified).
The takeaway for `grid`: **RunPod "Secure Cloud" is itself a marketplace over several
bare-metal operators**, so hardware and fabric are heterogeneous even within one GPU SKU.

### 1.2 APIs, SDK, CLI

| Interface | Endpoint | Notes |
|---|---|---|
| **REST v2** (current) | `https://api.runpod.io/v2`, `Authorization: Bearer`, optional `X-Team-Id` | OpenAPI at `https://api.runpod.io/v2/openapi.json` (info.version `2.0.0`). Resources: pods, clusters, serverless, templates, network-volumes, global-volumes, registries, catalog (gpus/cpus/datacenters), billing, cost-centers, audit-logs ([overview](https://docs.runpod.io/api-reference-v2/overview.md), spec). |
| **REST v1** (legacy) | `https://rest.runpod.io/v1` | Has fields v2 dropped: `interruptible`, `gpuTypeIds[]` + `gpuTypePriority`, `dataCenterPriority`, `countryCodes`, `minDownloadMbps`, `minUploadMbps`, `minDiskBandwidthMBps`, `supportPublicIp` ([v1 create pod](https://docs.runpod.io/api-reference/pods/POST/pods.md)). |
| **GraphQL** (legacy) | `https://api.runpod.io/graphql?api_key=...` | `podFindAndDeployOnDemand`; `gpuTypes { lowestPrice(input:{gpuCount}) { uninterruptablePrice stockStatus availableGpuCounts } }`; `podResumeZeroGpu` exists only here ([GraphQL manage pods](https://docs.runpod.io/sdks/graphql/manage-pods.md), [zero GPUs](https://docs.runpod.io/pods/troubleshooting/zero-gpus.md)). |
| **runpodctl** | `runpodctl pod create --gpu-id ... --gpu-count N --cloud-type SECURE\|COMMUNITY --data-center-ids a,b --network-volume-id ... --global-networking` | Also `gpu list`, `datacenter list`, `network-volume`, `send`/`receive` for file transfer ([runpodctl pod](https://docs.runpod.io/runpodctl/reference/runpodctl-pod.md)). No documented spot flag in the current reference; an older `runpodctl start pod --bid` page exists (Community Cloud only) (unverified whether still functional). |
| **Python SDK** | `runpod` package / newer "SDK apps" docs | Thin wrappers; not examined in depth in this sweep. |

**v2 create-pod contract** (`POST /v2/pods`, returns `201`, then `PROVISIONING -> STARTING ->
RUNNING`, with `EXITED`, `ERROR`, `TERMINATED` as other states)
([create a pod](https://docs.runpod.io/api-reference-v2/pods/create-a-pod.md)):

- Body: `name`, `image` or `templateId`, `cloud` (`SECURE` default / `COMMUNITY`), `gpu{id,
  count (1-1024), allowedCudaVersions | minCudaVersion, minRamPerGpu, minVcpuCountPerGpu}`,
  `dataCenterIds[]`, `globalNetworking`, `mounts{persistent | network[<=4] | global[<=1]}`,
  `disk` (container disk GB), `ports[]` (`"22/tcp"`, `"8888/http"`), `env`, `registry`,
  `entrypoint`/`cmd`/`args`, `startSsh` (injects `PUBLIC_KEY`), `startJupyter`.
- **One GPU type per call, no fallback**: "This endpoint places one specific GPU type. It
  does not search for capacity, and it does not fall back to a different GPU." The documented
  pattern is read-then-create: order (data center, GPU) candidates from the catalog and try
  each until `201` (spec, `createPod` description).
- Error contract: `400` = cross-field rule broken **or could not be placed** (try next
  candidate); `402` insufficient balance; `403` no access to that pool; `409` invalid action
  for state; `413` body > 102400 bytes; `422` schema violation (never retry); `429` rate
  limit; `5xx` transient. "Before retrying a `POST` that creates a resource, list resources
  first to avoid duplicates" - i.e. **there is no idempotency key**
  ([rate limits](https://docs.runpod.io/api-reference-v2/rate-limits.md)).
- Lifecycle: `POST /v2/pods/{id}/action` with `{"action": "start"|"stop"|"restart"|"terminate"}`;
  the allowed set is published in the pod's `actions` field. `GET /v2/pods/{id}/logs` exists.

**Availability signals** (all advisory, not reservations):

- `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD|CLUSTER|SERVERLESS&count=N&cloud=SECURE|COMMUNITY`
  returns per-GPU `availability` and a per-data-center list; `price.secure` /
  `price.community` per GPU-hour; `maxCount` per cloud; `cudaVersions`. `product` is
  mandatory because "the same GPU type can be scarce for pods and plentiful for serverless"
  (spec).
- `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY&regions=...&networkVolumeTypes=...&globalNetwork=true&compliance=...`
  returns each data center's `globalNetwork` flag, `networkVolumeTypes`
  (`STANDARD`, `HIGH_PERFORMANCE`), compliance list, and `gpuAvailability[]` with
  `AvailabilityLevel` in `NONE | LOW | MEDIUM | HIGH`
  ([list data centers](https://docs.runpod.io/api-reference-v2/catalog/list-data-centers.md)).
- There is **no numeric free-GPU count** in v2; the coarse level is all you get. "A create
  can still fail for capacity on a GPU the catalog just reported as available" (spec).

**Rate limits and quotas**:

- v2 limits are per user across three windows (minute, hour, day); every request counts
  against all three. Actual quotas must be read from the `RateLimit-Policy` header (`q`, `w`)
  and remaining budget from `RateLimit` (`r`, `t`); `429` carries `Retry-After` and an
  `application/problem+json` body. The docs' *illustrative* numbers are 60/min, 3,000/hour,
  50,000/day - not guaranteed for any account. Quotas are shared by every process using the
  same key ([rate limits](https://docs.runpod.io/api-reference-v2/rate-limits.md)).
- **Spend limit**: default **$80/hour across all resources**; raising it requires support
  ([pod pricing](https://docs.runpod.io/pods/pricing.md)). At $3.99/GPU-h (H100 SXM Secure)
  that is ~20 GPUs; this, not API rate, is the first wall a new account hits.
- On-demand deploy requires at least one hour of credit for the configuration
  ([pod pricing](https://docs.runpod.io/pods/pricing.md)).
- No documented per-account pod-count cap was found (unverified).

**Creating many pods programmatically**: there is no batch-create for plain pods. Options are
(a) N sequential/parallel `POST /v2/pods` within the minute window, each needing its own
candidate-fallback loop, or (b) one `POST /v2/clusters` which creates `podCount` identical
pods atomically-ish in one data center. If the real minute quota resembles the illustrative
60/min, a 100-pod fleet takes at least ~2 minutes of create calls before any polling budget
is spent; polling must therefore use `GET /v2/pods` list calls, not per-pod gets.

### 1.3 Instant Clusters

Primary sources: [overview](https://docs.runpod.io/instant-clusters),
[configuration](https://docs.runpod.io/instant-clusters/configuration.md),
[scaling](https://docs.runpod.io/instant-clusters/scale-clusters.md),
[Slurm clusters](https://docs.runpod.io/instant-clusters/slurm-clusters.md), OpenAPI spec.

| Aspect | Finding |
|---|---|
| GPU types | B200, H200, H100, A100 |
| Size | Docs: 2-8 nodes (16-64 GPUs) self-serve; larger, up to 512 GPUs, via sales. Product page (via search): 2 nodes/16 GPUs by default, 8 nodes after a spend-limit increase (unverified). API schema: `compute.podCount` min 2, **max 250**. |
| Shape | Strictly homogeneous: `compute{gpuTypeId, gpuCountPerPod, podCount}`; "Every pod in the cluster is identical". Shape, type and container config are immutable after creation (`PATCH` only renames). |
| Placement | "A cluster is always placed within a single data center." |
| Network | Private **VXLAN overlay** (`network{cidr e.g. 10.65.0.0/16, vxlanId, vxlanPort e.g. 4789}`); up to 8 high-bandwidth interfaces `ens1..ens8`; `eth0` (172.x) is internet-only and must not be used for inter-node traffic. Advertised 3200 Gbps (B200/H200/H100) or 1600 Gbps (A100) per node. |
| RDMA | "Available where the hardware supports it, using InfiniBand or RoCE. Otherwise NCCL falls back to TCP over the `ens` interfaces." When present, pods get `/dev/infiniband/uverbs*`, the memlock limit is removed, and `NCCL_IB_HCA`, `NCCL_IB_DISABLE=0` (and `NCCL_IB_GID_INDEX` on RoCE) are preset. RunPod states it tests NCCL only. |
| Injected env | `PRIMARY_ADDR`/`MASTER_ADDR`, `PRIMARY_PORT`/`MASTER_PORT`, `NODE_ADDR`, `NODE_RANK` (0 = primary), `NUM_NODES`, `NUM_TRAINERS` (GPUs per node), `HOST_NODE_ADDR`, `WORLD_SIZE`. Docs require `NCCL_SOCKET_IFNAME=ens1`. |
| API | `POST /v2/clusters` (body = container config + `name`, `type`, `compute`, `dataCenterIds`, `mounts`); `GET /v2/clusters/{id}/pods`; `DELETE`. Each member pod reports `cluster{id, rank, role, ip}`. Private images only via `templateId`. |
| Storage | `mounts.persistent` or **one** network volume (mounted at `/workspace` on every node); global volumes rejected with 422. |
| Scaling | Console "Scale cluster" adds identical pods; removal is by terminating pods; no scale endpoint in the v2 spec. Rank assignment for added nodes is undocumented. |
| Managed Slurm | Controller on primary, agents on others, munge preconfigured; **only official RunPod PyTorch images work**. |
| Price (per GPU-h) | B200 $9.59, H200 SXM $6.35, H100 SXM $4.79, A100 SXM $2.15 ([pricing page](https://www.runpod.io/pricing)), roughly 20% over Secure pod prices. |

Two cautions. First, "3200 Gbps" is the sum of eight 400 Gb/s ports, and the docs do not
name the fabric per data center; the RDMA-or-TCP outcome must be **detected at runtime**
(`ls /dev/infiniband`, `ibv_devinfo`, NCCL `NET/IB` vs `NET/Socket` log lines - see
`05-networking-interconnects.md`). Second, the overlay being VXLAN means the `ens*` path is
encapsulated Ethernet; whether RoCE traffic rides the overlay or a separate passthrough VF is
not documented (unverified).

### 1.4 Networking for plain pods

| Mechanism | Detail | Source |
|---|---|---|
| HTTP proxy | `https://[POD_ID]-[PORT].proxy.runpod.net`; max 10 HTTP ports; goes through Cloudflare with a **100 s** connection limit (HTTP 524 beyond). | [expose ports](https://docs.runpod.io/pods/configuration/expose-ports.md) |
| TCP ports | Public IP + random external port per declared `port/tcp`; **external mappings change on every pod reset**; Community pods may change IP on restart/migration. | same |
| Symmetric ports | Request a port number above 70000; the assigned port is exposed as `RUNPOD_TCP_PORT_70000` etc. and internal = external. | same |
| UDP | Not supported ("Pods support only TCP and HTTP"). | [pods overview](https://docs.runpod.io/pods/overview) |
| Global networking | Private IP + DNS `POD_ID.runpod.internal`, all ports, cross-data-center; **100 Mbps** between pods; NVIDIA GPU pods only; Secure Cloud on-demand only; 17 data centers (CA-MTL-3, EU-CZ-1, EU-FR-1, EU-NL-1, EU-RO-1, EU-SE-1, EUR-IS-2, OC-AU-1, US-CA-2, US-GA-1, US-GA-2, US-IL-1, US-KS-2, US-NC-1, US-TX-3, US-TX-4, US-WA-1). | [networking](https://docs.runpod.io/pods/networking.md), [v1 create pod](https://docs.runpod.io/api-reference/pods/POST/pods.md) |

Consequences: between ordinary pods the choices are 100 Mbps private, or public-IP TCP with
NAT'd ports whose real bandwidth is host-dependent (v1 exposes `minDownloadMbps` /
`minUploadMbps` placement filters; v2 does not). NCCL needs many peer-to-peer ports
(`NCCL_SOCKET` opens listening ports dynamically), which NAT'd per-port mapping cannot
provide, so **synchronous NCCL training across plain pods is only viable via global
networking, at 100 Mbps** - about 4-5 orders of magnitude below an IB rail. No UDP also rules
out WireGuard/QUIC-based overlays in kernel mode; a userspace TCP mesh is the only DIY option.

### 1.5 Container model restrictions

- "Docker Compose is not supported, and you cannot run your own Docker instance on a Pod";
  no Windows ([pods overview](https://docs.runpod.io/pods/overview)).
- No privileged mode and no FUSE (`/dev/fuse` permission errors reported in a Jan 2026
  SkyPilot issue; community statements that RunPod "doesn't support privileged mode")
  ([SkyPilot issue #8592 mirror](https://www.goodfirstissue.org/skypilot-org/skypilot/issues/8592),
  [community thread](https://www.answeroverflow.com/m/1297834108768096257)) (unverified
  against official docs, which are silent).
- Kernel, NVIDIA driver and OFED are the host's. The only knob is placement filtering by
  `allowedCudaVersions` / `minCudaVersion`; the pod reports `cudaVersion`.
- Implications: no nested container runtime (so no K8s-in-pod with normal kubelets, no
  `enroot`/`docker` under Slurm), no `gcsfuse`/`s3fs`/`mountpoint-s3`, no `sysctl` tuning,
  no `modprobe`, no MTU changes, no custom CNI. SSH is just an `sshd` in the container
  started by the template with `PUBLIC_KEY`.

### 1.6 Storage

| Type | Scope | Persistence | Price | Notes |
|---|---|---|---|---|
| Container disk | Host-local | Lost on stop | $0.10/GB-mo while running | `disk` field. |
| Volume disk (`mounts.persistent`) | **Pinned to one physical host** | Survives stop, deleted with pod | $0.10 running, **$0.20/GB-mo stopped** | Deprecated in v2 in favour of network volumes; min 10 GB. A stopped pod can only restart if that host's GPUs are free, else "zero GPUs" / migrate ([zero GPUs](https://docs.runpod.io/pods/troubleshooting/zero-gpus.md), [migration](https://docs.runpod.io/pods/troubleshooting/pod-migration.md)). |
| Network volume | **Pinned to one data center**; Secure Cloud only | Independent of pods | $0.07/GB-mo (<1 TB), $0.05 (>1 TB); billed hourly | 200-400 MB/s typical, "up to 10 GB/s peak"; grow-only; >4 TB via support; attach only at pod creation; up to 4 per pod; S3-compatible API for loading data without compute; concurrent-writer semantics unspecified ([network volumes](https://docs.runpod.io/storage/network-volumes.md)). |
| High-performance network volume | Select US/EU data centers | same | $0.14/GB-mo (pricing page) | "Up to 3x throughput and 4x IOPS", "parallel storage architecture"; no in-place migration from standard ([high-performance storage](https://docs.runpod.io/storage/high-performance-storage.md)). |
| Global volume (beta) | Region-independent, object-backed | Until deleted | Storage + $0.005/1k class-A, $0.0005/1k class-B requests | Not POSIX: no locking, atomic rename, hard links or permission bits; eventually consistent; read-heavy use (weights); not mountable on clusters ([global volumes](https://docs.runpod.io/storage/globalvolume/overview.md)). |

The pinning rules interact badly with capacity hunting: choosing a network volume fixes the
data center, which shrinks the GPU pool ("GPU options depend on the volume's location"), and
a Reddit report describes every GPU showing unavailable once a network volume was attached
(unverified anecdote). No ingress/egress fees apply
([pod pricing](https://docs.runpod.io/pods/pricing.md)).

### 1.7 Spot / interruptible

- REST v1: `interruptible: true` "creates a spot Pod that can be stopped to free resources";
  no bid field ([v1 create pod](https://docs.runpod.io/api-reference/pods/POST/pods.md)).
- Notice: **SIGTERM, then SIGKILL 5 seconds later** per a RunPod docs page mirrored from the
  docs repo and community answers ([docs PR mirror](https://github.kakarurec.workers.dev/runpod/docs/pull/626/files/8e5aebdb3740725f1f0d080d1f112ba8267e20ee),
  [community answer](https://answeroverflow.com/m/1262811070603464704)). The current docs
  index has no spot page and the v2 OpenAPI spec contains no `interruptible`/`spot` field
  (checked by text search of the spec), so RunPod appears to be de-emphasising spot
  (unverified intent). Treat RunPod spot as: legacy API only, 5 s notice, volume disk kept.
- Global networking is "only for On-Demand GPU Pods" and Instant Clusters are on-demand, so
  **there is no spot path with a usable east-west network**.

### 1.8 Pricing and billing

- Per-second billing for compute and pod storage; network volumes hourly; no data transfer
  fees; savings plans deprecated ([pod pricing](https://docs.runpod.io/pods/pricing.md)).
- Balance reaching $0 stops pods; pods without a network volume are **terminated and data
  lost** (same source).
- List prices fetched from [runpod.io/pricing](https://www.runpod.io/pricing) (Secure Cloud,
  per GPU-hour): B200 $7.99, H200 $5.29, H100 SXM $3.99, H100 PCIe $2.89, A100 $1.79,
  L40S $1.09, RTX 5090 $1.19, RTX 4090 $0.89. Community prices on the same page's structured
  data were lower (e.g. H100 SXM $2.69) (unverified, extracted from page metadata). The pod's
  own `cost` field is the authoritative billed rate.

### 1.9 Provisioning latency

No primary source gives pod start-time distributions. Marketing claims "250 ms cold start";
third-party guides put H100 pods at 20-60 s to ready with cached images and minutes with
large image pulls (all unverified). Billing starts during image pull. What is documented: a
pod can sit `RUNNING` before its service is ready ("check the Telemetry tab"), and status
incidents such as a 4h41m "delayed pod starts" event on 2026-05-08 are recorded by a
third-party tracker ([pingoru](https://pingoru.io/providers/runpod)) (unverified). `grid`
must measure this itself (see `09-fast-bootstrap-preflight.md`).

### 1.10 Reliability complaints (all anecdotal unless noted)

- Maintenance: email notice with **no stated lead time**; unplanned outages may be notified
  after the fact; no credits mentioned; RunPod recommends checkpointing "every hour to every
  few hours" ([maintenance and outages](https://docs.runpod.io/pods/maintenance-and-outages.md)).
- Pods stuck in "RUNNING (waiting for runtime)" without a started container, pod-creation
  errors, proxy timeouts ([community thread](https://www.answeroverflow.com/m/1396791981903974561)).
- Host pinning: stopped pods lose their GPU to other renters (documented behaviour above).
- A third-party tracker counts 81 incidents in 12 months, 29 in the last 90 days
  ([pingoru](https://pingoru.io/providers/runpod)); official history is at
  [uptime.runpod.io](https://uptime.runpod.io/incidents/2025-09/2025-11).
- Hardware opacity and thin health checking per ClusterMAX (section 1.1).

---

## 2. Comparison across providers

Legend for "east-west": **IB** = InfiniBand with GPUDirect RDMA; **RoCE**; **TCP** = private
L3 only; **NAT** = public mapped ports only. "Self-serve" = API/console without a sales step.

| Provider / product | Alloc unit | Multi-node RDMA on demand? | East-west | API to create cluster | Min term / billing | Spot / preemption | Storage | Notes and sources |
|---|---|---|---|---|---|---|---|---|
| **RunPod Pods** | Container | No | 100 Mbps overlay or NAT TCP | `POST /v2/pods` | Per second | v1 `interruptible`, 5 s SIGTERM | Host disk; DC-pinned network volume | Section 1 |
| **RunPod Instant Clusters** | Container group | **Yes**, 2-8 nodes self-serve, to 512 GPUs via sales | 1600-3200 Gbps; IB/RoCE "where supported", else TCP | `POST /v2/clusters` (`podCount` 2-250) | Hourly rate, on-demand | None | One network volume | Section 1.3 |
| **Lambda 1-Click Clusters** | VM/bare-metal cluster | RDMA yes, but **reserved in whole weeks**, launched on reservation start day | Quantum-2 IB 400 Gb/s per GPU, rail-optimised, 3200 Gb/s per node; 2x100 GbE | Console reservation; no cluster-create API found | Weeks, prepaid invoice | None | 24 TB local NVMe per node; Lambda filesystems | 16-512 H100/B200 GPUs, 3 management nodes ([docs](https://docs.lambda.com/public-cloud/1-click-clusters/)). On-demand instance API: launch limited to ~1 request per 10-12 s, general 1 req/s ([rate limiting](https://old.docs.lambdalabs.com/cloud/rate-limiting/)) (exact current value unverified). |
| **Vast.ai** | Container (Docker) or VM, per host | No on demand; RoCE/IB "requires a separate cluster request", A100/H100/H200 only | Overlay network within a "physical cluster" (`vastai create overlay CLUSTER_ID NAME`), all ports, `eth0`; TCP | `vastai search offers cluster_id=...` then `create instance --env "-n NET"` | Per second; storage billed while stopped; **per-byte bandwidth charges set by host** | Interruptible (bid), "may be paused"; notice undocumented | Host-local | Overlays work only with Docker templates ([multi-node NCCL](https://docs.vast.ai/multi-node-training-using-torch-nccl), [pricing](https://docs.vast.ai/documentation/instances/pricing)). Dedicated clusters are a sales product ([clusters](https://vast.ai/products/clusters)). |
| **CoreWeave** | K8s node (CKS), Slurm via SUNK | Yes technically; account onboarding/capacity approval in practice (unverified how self-serve) | IB on selected instance types, exposed as pod resource `rdma/ib: 1`; TCP fallback if omitted | K8s node pools | Per-instance hourly; reserved discounts | Spot offered per aggregators (unverified) | Distributed file/object storage, dedicated VAST | Image must carry OFED/NCCL matching node kernel modules ([GPUDirect RDMA guide](https://docs.coreweave.com/products/networking/hpc-interconnect/use-gpudirect-rdma.md)); ClusterMAX top tier ([review](https://clustermax.semianalysis.com/cloudreview/coreweave)). |
| **Crusoe** | VM | **Yes** | IB partitions; `h100-80gb-sxm-ib.8x`/`h200-141gb-sxm-ib.8x` 8 HCAs, 3200 Gbps; `a100-80gb-sxm-ib.8x` 1600 Gbps | `crusoe networking ib-partitions create --ib-network-id`, then `crusoe compute vms create --ib-partition-id`; Terraform `crusoe_ib_partition` | On-demand hourly | Spot claimed by third parties with conflicting terms (unverified) | Persistent disks, shared FS | Max 5 partitions per IB network; needs `NCCL_TOPO_FILE`, `NCCL_IB_MERGE_VFS=0`, `NCCL_IB_HCA=^mlx5_0:1` ([IB docs](https://docs.crusoe.ai/networking/infiniband/managing-infiniband-networks)); capacity via `api/v1alpha5/capacities` ([support FAQ](https://support.crusoecloud.com/hc/en-us/articles/45906998547227-FAQ-Resolving-Out-of-Stock-Errors)). |
| **Nebius** | VM | **Yes** | IB, 400 Gbps per GPU (3.2 Tbps per 8-GPU VM), per-cluster P-Key isolation | `nebius compute gpu-cluster create --infiniband-fabric <fabric>`, then VM with `--gpu-cluster-id` (gRPC API, SDKs) | On-demand hourly | Preemptible VMs exist; whether they can join a gpu-cluster is undocumented (unverified) | Disks, shared filesystems, object storage | Fabric is tied to one GPU platform and region (e.g. `fabric-7` = H200 in eu-north1, `us-central1-b` = B200); cluster membership only settable at VM creation ([docs](https://docs.nebius.com/compute/clusters/gpu.md)). |
| **Together GPU Clusters** | Managed K8s or Slurm (Slinky) cluster | **Yes**, on-demand hourly, "terminate anytime" | 3.2 Tb/s IB on H100 SXM / H200 / B200 | Console, CLI `tg clusters create`, REST, Terraform | Hourly on-demand; reserved 3-180 days prepaid | Preemptible compute in preview, sub-hourly billing, reclaim any time | Shared volumes (survive cluster deletion), local NVMe | 8 to 512+ GPUs advertised ([overview](https://docs.together.ai/docs/gpu-clusters-overview.md), [product page](https://together.ai/instant)). |
| **Voltage Park** | Bare metal (IB) or VM (Ethernet) | **Yes** on bare metal | Bare metal 3.2 Tbps IB; VMs 100 Gbps Ethernet only | REST API + SDK (per third-party review; docs moved to a help center that did not render) (unverified) | No minimum term; "up within 15 mins" | None found | Local NVMe | Advertises 64-4088 H100; on-demand IB about $2.49-3.00/GPU-h, sources conflict ([compare page](https://voltagepark.com/compare/nebius)) (unverified). Docs domain now redirects to Lightning AI's help center. |
| **Hyperstack** | VM | No for on-demand VMs; IB only in "Supercloud" contracts | Ethernet tiers; SR-IOV up to 350 Gbps in specific region/image, for contracted users on request | REST API, Terraform | Hourly; stop/hibernate options | Spot VMs (unverified) | Volumes | [network types](https://docs.hyperstack.cloud/docs/network/network-types). |
| **FluidStack** | Managed clusters | Not self-serve; private/reserved builds | IB | Sales | Contract | n/a | n/a | Business has moved to large private clusters; on-demand marketplace status unclear (unverified) ([Sacra](https://sacra.com/research/fluidstack)). |
| **TensorDock** | VM (KVM) | No | Public IP / NAT | REST | Per second/hour | Was offered | Host-local | Acquired by Voltage Park (Mar 2025); 2026 reports of empty inventory (unverified). Treat as dormant. |
| **Paperspace / DigitalOcean Gradient** | VM (GPU Droplet) or bare metal | Multi-node only after contacting support; multiples of 8 GPUs | Extra NICs `eth2..eth9/17` for GPU traffic, `eth1` control | DO API for droplets | Hourly | None | Volumes, Spaces | [multi-node how-to](https://docs.digitalocean.com/products/droplets/how-to/gpu/configure-multi-node/), [bare metal](https://docs.digitalocean.com/products/bare-metal-gpus). Transport (RoCE vs TCP) not confirmed (unverified). |
| **Prime Intellect** | Pods/VMs brokered from ~24 providers; own multi-node | **Yes** (16-64+ H100, one public IP per 8-GPU node) for its multi-node product; else inherits underlying provider | Exposed as data: `interconnect` (Gbps), `interconnectType` | `GET /api/v1/availability/multi-node`, then create pod with `cloudId` | Hourly; some offers have `prepaidTime` | `isSpot` flag per offer | Network disks, `sharedDisk` | Response also carries `stockStatus`, `provisioningTime` (minutes), `security`, `prices.isVariable` ([API ref](https://docs.primeintellect.ai/api-reference/availability/get-multinode-availability.md)). Provider enum includes runpod, lambdalabs, hyperstack, crusoecloud, nebius, datacrunch, tensordock, vultr, akash. |
| **SF Compute** | K8s cluster (default), VMs | **Yes** on `h100i` | 3.2 Tb/s IB on `h100i`; `h100v` VMs have **no IB** | `sf` CLI: buy orders (GPUs, start, duration, limit price) | Market: from 1 node for 1 hour upward; sold in 8-GPU nodes | An order is a fixed block; resale possible | Cluster-local | Order can cancel if price too low; containers need `libibverbs` for RDMA ([docs](https://docs.sfcompute.com/docs/instance-types), site). Docs versions conflict (unverified details). |
| **AWS** (EC2, Capacity Blocks, HyperPod) | VM | **Yes**, quota-gated | EFA (SRD, not IB verbs) inside a cluster placement group | EC2 API / ParallelCluster / PCS / HyperPod | Per second on demand; Capacity Blocks 1-182 days, up to 64 instances, prepaid, non-cancellable, P4d-P6 families | Spot: 2-minute notice (from memory; not re-verified this sweep) | EBS, FSx for Lustre, S3 | [Capacity Blocks guide](https://awslabs.github.io/accelerated-compute-tutorials/ai-infra/capacity-blocks-guide/), [EC2 docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html). |
| **GCP** | VM | **Yes**, quota/reservation-gated | RoCE-class RDMA on newest GPU families; Falcon RDMA VPC profile on H4D | Compute API, GKE, Cluster Director | Per second; DWS flex-start up to 7 days; calendar-mode future reservations up to 90 days / 80 GPU VMs (preview as of Dec 2025) | Spot: 30 s notice (from memory) | PD, Filestore, Lustre, GCS | [AI Hypercomputer release notes](https://docs.cloud.google.com/ai-hypercomputer/docs/release-notes), [HPC flex-start](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/run-hpc-workloads). |
| **Azure** | VM | **Yes**, quota-gated | IB on ND-series (e.g. ND H100 v5: 3.2 Tbps per VM) within one VMSS/availability set | ARM, CycleCloud, AKS | Per second | Spot: at least 30 s via Scheduled Events | Managed disks, Azure Managed Lustre | [NDH100v5](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ndh100v5-series). |
| **OCI** | Bare metal | **Yes**, quota-gated | RoCE v2 cluster networks, <10 us latency claimed; up to 4096 BM nodes / 32768 GPUs per cluster | Cluster network / compute cluster API | Per second (1-min minimum, from memory) | Preemptible not offered for BM GPU shapes (unverified) | Block, FSS, Lustre | [Oracle BM GPU cluster guide](https://docs.oracle.com/en/solutions/deploy-bare-metal-gpu-cluster-for-ai/index.html). |

### 2.1 Three classes of provider

1. **RDMA cluster as a first-class, API-creatable object, hourly, no sales step**: RunPod
   Instant Clusters, Together, Nebius, Crusoe, Voltage Park bare metal, Prime Intellect
   multi-node, SF Compute `h100i`. These are `grid`'s primary targets for synchronous
   training. Common pattern: **create a fabric-scoped object first** (cluster, gpu-cluster,
   IB partition, placement group, overlay), then create nodes *into* it; membership cannot be
   changed after node creation (explicit at Nebius; implicit at RunPod).
2. **RDMA clusters behind quota, reservation or sales**: Lambda 1CC (weekly reservations),
   CoreWeave, hyperscalers, OCI, Hyperstack Supercloud, FluidStack, Vast dedicated clusters,
   DigitalOcean multi-node.
3. **Isolated nodes**: RunPod pods, Vast marketplace, TensorDock, Hyperstack on-demand VMs,
   Lambda on-demand instances, GPU Droplets. Good for single-node jobs (up to 8-10 GPUs),
   embarrassingly parallel sweeps, data preprocessing, eval/inference fleets, and
   low-communication distributed training.

### 2.2 Recurring differences that bite

- **Who sets NCCL up.** RunPod presets `NCCL_IB_HCA` etc.; Crusoe requires a topology file
  and `NCCL_IB_MERGE_VFS=0`; CoreWeave wants `NCCL_IB_HCA=ibp` and matching OFED in the
  image; Vast wants `NCCL_SOCKET_IFNAME=eth0`; RunPod wants `ens1`. Interface names for the
  fast path differ on every provider (`ens1-8`, `eth0`, `eth2-9`, `ibp*`, `mlx5_*`).
- **Rank hints.** RunPod injects `NODE_RANK`/`WORLD_SIZE`/`PRIMARY_ADDR`; nobody else does.
  `grid` should own rank assignment and treat provider hints as optional input (see
  `03-rank-configuration.md`).
- **Homogeneity is enforced by the providers' cluster objects.** Every RDMA cluster product
  above is single-SKU, single-fabric, single-data-center. Heterogeneity only arises *across*
  cluster objects or among isolated nodes - where there is no RDMA anyway.
- **Preemption notice ranges from 5 s (RunPod) through 30 s (GCP, Azure) to 2 min (AWS)**,
  and several marketplaces document none.
- **Availability data quality** ranges from a four-level enum (RunPod), to a capacities
  endpoint (Crusoe), to explicit offers with price and `provisioningTime` (Prime Intellect,
  Vast, SF Compute order book), to nothing but try-and-fail (hyperscaler on-demand).

---

## Implications for grid

1. **Model two resource kinds, not one**: `Node` and `Fabric` (cluster/partition/placement
   group/overlay). A `Fabric` has `provider`, `datacenter`, `gpu_sku`, `transport`,
   `max_nodes`, `mutable_membership`, and must be created before nodes. Providers without a
   fabric object get an implicit "null fabric" whose transport is `nat_tcp` or `overlay_tcp`.
2. **Capability flags the abstraction needs** (per offer, not per provider, because RunPod,
   Vast and Prime Intellect all vary by host/data center):

   | Flag | Values / examples |
   |---|---|
   | `alloc_unit` | `container`, `vm`, `bare_metal`, `k8s_node` |
   | `privileged`, `nested_containers`, `fuse`, `kernel_modules`, `sysctl` | bool each (all false on RunPod) |
   | `ew_transport` | `none`, `nat_tcp`, `overlay_tcp`, `roce`, `infiniband`, `efa` |
   | `ew_bandwidth_gbps`, `ew_rails`, `ew_ifnames` | 0.1 (RunPod global net), 100, 1600, 3200; `ens1..ens8` |
   | `rdma_guaranteed` | bool - false for RunPod Instant Clusters ("where the hardware supports it") so a runtime probe decides |
   | `udp`, `inbound_ports` | `none`, `mapped_random`, `mapped_symmetric`, `all`; `proxy_http_timeout_s` (100 on RunPod) |
   | `stable_address_across_restart` | false for RunPod port mappings and Community IPs |
   | `fabric_scope`, `max_nodes_self_serve`, `max_nodes_api`, `homogeneous_only`, `elastic_add`, `elastic_remove` | RunPod: datacenter, 8, 250, true, console-only, terminate-pod |
   | `storage_classes[]` | each with `scope` (`host`, `datacenter`, `region`, `global`), `posix`, `shared_rw`, `attach_at_create_only`, `throughput_mbps`, `survives_stop`, `survives_terminate` |
   | `preemptible`, `preempt_notice_s`, `preempt_signal`, `preempt_action` | RunPod: 5, SIGTERM, stop; AWS: 120 |
   | `restart_requires_same_host` | true for RunPod volume-disk pods |
   | `billing_granularity_s`, `min_commit_s`, `prepaid`, `billed_while_provisioning`, `stopped_storage_multiplier`, `egress_fee` | RunPod 1, 0, false, true, 2.0, none; Lambda 1CC 604800; Vast egress per byte |
   | `availability_signal` | `none`, `enum_level`, `count`, `offer_list`, `order_book` |
   | `create_idempotent`, `batch_create`, `rate_limit_source` | RunPod: false, clusters only, response headers |
   | `spend_cap_usd_per_h` | RunPod default 80 |
   | `injected_env` | map of provider rank/address variables |
   | `driver_owner` | `host` (RunPod, Vast containers) vs `image` (VM providers); `cuda_versions_selectable` |

3. **RunPod driver design.** Use REST v2 exclusively except for spot (v1). Implement the
   candidate-ordering loop the docs prescribe: catalog read with `include=AVAILABILITY`,
   order `(datacenter, gpu)` candidates, treat `400` as "next candidate", `403` as "skip
   pool", `402`/`422` as fatal. Because creates are not idempotent, tag each pod with a
   deterministic `name` (run id + slot) and **list-before-retry** to avoid duplicate billing.
   Drive the limiter from `RateLimit`/`RateLimit-Policy` headers, and poll with list calls.
4. **Pre-flight the account**, before promising a cluster size: spend limit ($80/h default),
   balance (at least one hour of the configuration), and observed rate-limit policy. A
   64-GPU H100 Instant Cluster costs about $307/h, nearly four times the default cap.
5. **Never use host-pinned storage for anything that must survive.** On RunPod use network
   volumes (choose the data center by joint GPU-plus-volume availability, since the volume
   pins placement) or treat nodes as cattle with object storage and no stop/start. Because
   FUSE is unavailable, object storage access must be SDK/`s5cmd`-style copy, or RunPod's
   S3-compatible API onto a network volume.
6. **Everything must run unprivileged in one container per node.** `grid`'s node agent,
   launcher, health checks and any overlay must be plain user-space processes over TCP. Bake
   the agent into (or bootstrap it from) the image; do not depend on Docker, systemd,
   `modprobe`, or UDP.
7. **Probe, do not trust, the network.** After provisioning, measure: presence of
   `/dev/infiniband`, HCA count and link rate, a short NCCL all-reduce or `ib_write_bw`
   between node pairs, and TCP throughput on the declared interface. Record results as the
   *actual* `ew_transport` for the scheduler. This is mandatory on RunPod, where two Instant
   Clusters of the same SKU can sit on different operators' fabrics.
8. **Assume 5 seconds of preemption notice in the worst case.** Checkpoint-on-signal is not
   viable for multi-hundred-GB states at 5 s; spot use on RunPod needs periodic asynchronous
   checkpoints and designs where losing a node loses bounded work (see
   `04-failure-taxonomy-recovery.md`).
9. **Plan for "big" meaning multi-fabric.** Self-serve RDMA tops out around 64 GPUs on
   RunPod and a few hundred elsewhere; beyond that a marketplace cluster is several fabrics
   joined by slow links (100 Mbps on RunPod global networking, or public internet). That
   pushes `grid` towards hierarchical topologies: synchronous within a fabric, infrequent
   synchronisation across fabrics (`08-heterogeneous-distributed-training.md`).
10. **Consider Prime Intellect's availability schema as prior art** for a normalised offer
    record (`interconnect`, `interconnectType`, `isSpot`, `stockStatus`, `provisioningTime`,
    `security`, `prepaidTime`), and as a possible meta-provider (`02-multicloud-provisioners.md`).

## Open questions

- **What transport do RunPod Instant Clusters actually deliver, per data center?** Need a
  hands-on survey: IB vs RoCE vs TCP, whether RDMA traverses the VXLAN overlay, measured
  NCCL all-reduce bandwidth, MTU, and how often the TCP fallback occurs.
- **RunPod real quotas**: actual `RateLimit-Policy` values for a normal account, any cap on
  concurrent pods, how quickly spend limits are raised, whether `podCount` up to 250 is
  reachable without sales, and whether `POST /v2/clusters` is all-or-nothing on partial
  capacity.
- **RunPod spot status**: is `interruptible` in REST v1 still honoured, is the 5 s notice
  current, how are spot prices set now that bid fields are gone, and will v1 be retired?
- **Provisioning latency distributions** on RunPod (create to `RUNNING` to SSH-ready) by GPU
  type, data center, image size and cache state; none are published.
- **Network volume behaviour under load**: concurrent multi-pod read/write semantics,
  real throughput of standard vs high-performance tiers, and the claimed availability
  collapse when a volume pins the data center.
- **Privileged/FUSE/capabilities on RunPod**: official statement, and whether Instant
  Cluster pods differ (they receive `/dev/infiniband` and unlimited memlock, so some device
  passthrough exists).
- **Cross-pod connectivity beyond 100 Mbps** for plain pods: real throughput of public-IP
  TCP between Secure Cloud data centers, and whether symmetric ports make a userspace mesh
  practical.
- **Second-sweep provider gaps**: Lambda's current 1CC minimum term and any cluster API;
  CoreWeave's self-serve reality and spot terms; Crusoe and Nebius preemptible instances
  inside IB partitions/gpu-clusters; Voltage Park's API after its docs moved; SF Compute's
  current VM/IB state; Together's cluster API limits and preemptible notice; DigitalOcean
  multi-node transport; Hyperstack and TensorDock current status; hyperscaler details
  (EFA/HyperPod flexible training plans, GCP A4/A3 Ultra RDMA, OCI limits) which were only
  lightly touched here.
- **ClusterMAX 3.0** reportedly published in September 2026; its ratings for the providers
  above were not retrieved.
