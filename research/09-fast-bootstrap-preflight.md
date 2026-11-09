# 09 - Fast bootstrap and preflight

Research date: 2026-11-09. Scope: what dominates the time from "request N nodes" to "first
optimizer step" on marketplace GPU providers, how to shrink it, and what to check before (and
while) a node is allowed to hold a rank. Sibling files: provider APIs and quotas are in
`07-provider-landscape.md`; NCCL transport tuning in `05-networking-interconnects.md`; failure
classes and recovery in `04-failure-taxonomy-recovery.md`; rank assignment in
`03-rank-configuration.md`; prior-art provisioners in `02-multicloud-provisioners.md`.

Conventions: every non-obvious claim has a source URL. "(unverified)" means I could not confirm
it from a primary source in this sweep. "(inference)" means it is my reasoning from sourced
facts, not something a source states.

## Summary

- Time-to-first-step is a sum of five mostly independent phases: **provision** (API call to
  container running), **image** (pull + unpack), **environment + data + checkpoint staging**,
  **mesh** (nodes can reach and authenticate each other), and **preflight** (node and cluster
  admission tests). On a cold host with a 10 GB PyTorch image, the image phase alone is
  ~3 minutes with defaults, and about 80% of that is single-threaded gzip unpack rather than
  network ([Crusoe](https://www.crusoe.ai/resources/blog/your-container-image-pull-isnt-slow-because-of-the-network)).
- The cheapest wins are not exotic: zstd layers, small images, weights and datasets kept out
  of the image, a provider-local cache for anything large, and doing phases in parallel per
  node rather than serially across the cluster.
- Lazy-pull snapshotters (eStargz, SOCI, Nydus) are host-side containerd plugins. A tenant on a
  container-only provider (RunPod pods) cannot install them (inference), and at least one
  report shows FUSE itself is unavailable inside RunPod containers. For `grid` the practical
  equivalent is "tiny image + agent that fetches the rest in parallel".
- Every serious operator gates node admission on active tests, and they converge on the same
  ladder: software sanity (seconds) -> DCGM level 2 (minutes) -> short GPU burn (about 5 min)
  -> single-node NCCL -> multi-node NCCL all-reduce. Together AI exposes exactly this as an
  optional provisioning step; CoreWeave re-runs a 20-30 minute verification hourly on idle
  nodes; Imbue kept 10-20% spare machines and found ~3% of machines break per week on a new
  H100 cluster.
- Slow nodes are as important as dead ones. A production study found fail-slows in 59% of
  512-1024 GPU jobs, adding 34.59% to job completion time ([Falcon](https://ar5iv.labs.arxiv.org/html/2410.12588)).
  Per-rank step-time telemetry is therefore part of bootstrap, not an afterthought.

## 1. Where the time goes

The table is a budget model for one node, built from the measurements cited in later sections.
The "typical" column is an estimate assembled from those numbers (inference), not a single
measured run; `grid` should replace it with its own probe data per provider.

| Phase | Dominant cost | Typical cold | Achievable | Main lever |
|---|---|---|---|---|
| Provision | Provider scheduling, host selection, capacity misses | tens of seconds to minutes, heavy tail (no trustworthy public distribution found) | bounded by a deadline | over-provision and cull (section 2) |
| Image | Layer decompress + unpack, serial by default | 184 s for a 10 GB gzip image | 41.5 s (13 GB, zstd + parallel unpack); ~0 s if host-cached | image format and size (section 3) |
| Env install | Wheel download + unpack (torch + CUDA libs are multi-GB) | 1-3 min with pip (unverified) | seconds from a warm cache with hardlinks | uv cache on shared or local volume (section 3.4) |
| Data + checkpoint | Object-store or volume throughput | 1 TB at 400 MB/s = ~42 min | 1 TB at 2.5 GB/s = ~7 min; ~0 with streaming | stream data, parallel-fetch checkpoint (section 4) |
| Mesh | Per-node SSH/agent handshakes if done serially | O(N) x seconds | O(1) wall time | phone-home agent (section 5) |
| Preflight | DCGM level 2, burn, NCCL tests | 5-15 min | 2-4 min tiered, concurrent with staging | tiering and overlap (section 6) |

Two structural observations:

1. Phases 2-6 are per-node and embarrassingly parallel, except the cluster-level NCCL test.
   Wall-clock time for the cluster is therefore the **slowest admitted node**, which makes
   tail-cutting (section 2) worth more than mean-shaving.
2. GPU preflight does not need the dataset, and dataset staging does not need the GPU. Running
   them concurrently hides most of the preflight cost (inference).

## 2. Parallel provisioning: over-provision, deadline, cull

**Why.** Cold-start latency on GPU providers is bimodal. One practitioner measurement across
serverless GPU providers saw requests finish in either ~6-8 s (warm) or ~90-122 s (cold), and
hedging across providers cut p95 from ~116.6 s to 29.4 s
([GPUHedge write-up, small sample, not peer reviewed](https://prismix.dev/news/422eac45811c)).
The same shape applies to pods: a host with the image cached starts quickly; a host without it
pays the full pull.

**Pattern.** Request N + k nodes concurrently, set a hard deadline, admit the first N that pass
preflight, terminate the rest. This is the provisioning analogue of hedged requests. Imbue's
operational version of the same idea is to hold 10-20% more machines than the run needs so
failed hosts can be swapped without stopping
([Imbue](https://imbue.com/research/70b-infrastructure/)).

Design points `grid` has to decide:

- **Sizing k.** Needs a per-provider, per-GPU-type distribution of time-to-ready and of
  preflight failure rate. I found no independent published distribution for RunPod, Lambda or
  Vast in this sweep; a probe harness is the only reliable source (see Open questions).
- **Billing granularity sets the cost of culling.** Where billing is per-second or per-minute,
  an over-provisioned node that is culled after 3 minutes costs cents; where it is hourly it
  does not. See `07-provider-landscape.md`.
- **Cull on rank, not just on pass/fail.** Preflight produces numbers (matmul GFLOPS, NCCL bus
  bandwidth, disk MB/s). With spare nodes, drop the slowest k even if all pass, because one
  slow rank sets synchronous step time (see `08-heterogeneous-distributed-training.md`).
- **Serial provisioning is the anti-pattern.** SkyPilot's release notes show how much was lost
  to it: v0.7 reported up to 3x faster provisioning and parallelised SSH setup for multi-node
  clusters ([SkyPilot 0.7](https://blog.skypilot.co/announcing-skypilot-0.7/)); v0.8 reported
  a 3x faster setup phase and up to 4x faster multi-node setup
  ([changelog](https://data.safetycli.com/changelogs/skypilot)).
- **Managed cluster products change the trade.** RunPod Instant Clusters provision nodes in one
  data center with a dedicated cluster network and pre-set rank variables, but self-serve is
  2-8 nodes (16-64 GPUs); up to 512 GPUs requires sales
  ([RunPod](https://docs.runpod.io/instant-clusters)). Beyond that, `grid` is assembling
  individual pods.

## 3. Container images

### 3.1 What a pull actually costs

Crusoe's measurements on containerd with local NVMe
([source](https://www.crusoe.ai/resources/blog/your-container-image-pull-isnt-slow-because-of-the-network)):

| Image | Size | gzip cold pull | zstd cold pull |
|---|---|---|---|
| CUDA base | 3.6 GB | 51.5 s | 17.4 s |
| PyTorch training | 10 GB | 183.7 s | 77.7 s |
| NeMo framework | 24.3 GB | 394 s | 152 s |

- Unpack is ~80% of pull time on 7-13 GB images. Raw decompression throughput: gzip 95 MB/s,
  zstd 681 MB/s.
- Registry locality barely matters: same-region registry 51.5 s vs NVIDIA public registry
  56.2 s.
- containerd tuning, all under `[plugins.'io.containerd.transfer.v1.local']`:
  `max_concurrent_downloads` (8 in the test) and `max_concurrent_unpacks` (new in containerd
  2.2, default 1). Raising downloads alone did nothing because unpack is serial. gzip + tuning:
  183.8 s -> 75.8 s; zstd + tuning on a 13 GB image: 41.5 s (4.43x).
- Build flags: BuildKit `compression=zstd,compression-level=3,force-compression=true`; skopeo
  `--dest-compress-format zstd --dest-force-compress-format`. Native zstd unpack needs
  containerd >= 1.5.0.

`grid` controls the **image format** on any provider, but the containerd settings only on
providers that give a VM or bare metal. On RunPod the host runtime is the provider's.

Microsoft's breakdown for LLM containers confirms that weights in the image are the worst case:
model weights compress ~1.00x, so gzip costs CPU for nothing; removing gzip saved ~36 s on an
8.7 GB image (88.3 s -> 56.3 s pull)
([Microsoft](https://techcommunity.microsoft.com/blog/linuxandopensourceblog/dissecting-llm-container-cold-start-where-the-time-actually-goes/4508831)).

### 3.2 Host-side layer caching on providers

- RunPod community answers describe the image being downloaded on first use and kept on the
  worker/host for later starts
  ([forum](https://answeroverflow.com/m/1270729627328974859),
  [forum](https://answeroverflow.com/m/1199755426456150086)). I found no official statement of
  cache lifetime, eviction, or whether popular base images are pre-seeded on pod hosts
  (unverified). Practical consequence (inference): basing the `grid` image on a layer stack the
  provider's own templates use maximises the chance that most layers are already on the host.
- dstack keeps "idle instances" in a fleet for reuse, which also preserves the image cache,
  but notes that idle-duration reuse is not applied to container-based backends such as RunPod
  ([dstack fleets](https://dstack.ai/docs/concepts/fleets/),
  [troubleshooting](https://dstack.ai/docs/guides/troubleshooting/)).

### 3.3 Lazy-pull snapshotters

| Tech | Mechanism | Evidence | Usable on container-only providers? |
|---|---|---|---|
| eStargz (stargz-snapshotter) | Seekable gzip layers, FUSE mount, fetch on read | Oracle test: startup 4 min -> 33 s on a GPU shape ([Oracle](https://docs.oracle.com/en/learn/oke-lazy-container-image-loading/index.html)); Grab found app startup 25.0 s vs 5.0 s for overlayfs/SOCI on one workload ([Grab](https://engineering.grab.com/docker-lazy-loading)) | No: host containerd plugin |
| SOCI | Separate index over unmodified images, range requests; also a parallel-pull mode | In production on EKS/Fargate; EKS Auto Mode uses parallel pull for GPU instances ([AWS](https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time/accelerate-pull-process)) | No: host plugin |
| Nydus (+ Dragonfly) | Chunk-addressed RAFS image, FUSE/EROFS, prefetch | Metadata "pull" ~0.7 s; with `prefetch_all=true` 1.32x faster than uncompressed overlayfs, without prefetch 2.87x slower; a 14 GB Python/CUDA stack exceeded 900 s startup and was excluded ([Microsoft](https://techcommunity.microsoft.com/blog/linuxandopensourceblog/dissecting-llm-container-cold-start-where-the-time-actually-goes/4508831)) | No: host plugin |
| Modal's runtime | Image as a ~5 MiB index, FUSE lazy file server, content-addressed cache | Own runtime ([Modal](https://modal.com/blog/jono-containers-talk)) | Only on Modal |

Conclusions for training workloads:

- Lazy pull pays off when the container touches a small fraction of its files. Importing torch
  plus CUDA libraries touches gigabytes, so the gain is smaller than for small services and can
  go negative through FUSE overhead, as the Nydus and eStargz counter-examples show.
- On container-only providers the tenant does not control the snapshotter (inference from the
  mechanism: all three are containerd snapshotter plugins on the host). RunPod additionally
  states that you cannot run your own Docker inside a pod
  ([RunPod pods overview](https://docs.runpod.io/pods/overview)).
- The tenant-level substitute is to do the lazy part in user space: a small image whose
  entrypoint fetches the heavy content in parallel, with the hot set first.

### 3.4 Slim image plus runtime environment

- uv with a warm cache on the same filesystem reinstalls torch in 0.53 s by hardlinking; with
  the cache on a different filesystem (`UV_LINK_MODE=copy`) it took 26.2 s
  ([benchmark, third party](https://pub.towardsai.net/uv-is-200x-faster-than-pip-put-it-in-docker-and-its-4x-e3851d45b8eb)).
  So `UV_CACHE_DIR` must be on the same filesystem as the venv to get the fast path; a cache on
  a network volume with the venv on local disk falls back to copy.
- `uv pip install torch --torch-backend=auto` selects the CUDA build from the detected driver
  ([guide](https://pydevtools.com/handbook/how-to/how-to-install-pytorch-with-uv/)), which
  matters on heterogeneous fleets where driver versions differ.
- conda-pack / pixi-pack style relocatable tarballs turn the environment into one object that
  can be fetched with the same parallel downloader as the checkpoint (unverified in this
  sweep: no timing numbers found).

**Bake vs fetch**, as a rule of thumb (inference):

| Content | Where | Why |
|---|---|---|
| OS, CUDA userspace, sshd, `grid` agent, s5cmd, uv | Image | Small, stable, likely host-cached |
| Python env (torch etc.) | Image if the provider caches layers well; otherwise cache tarball on volume or object store | Multi-GB; changes with the project |
| Training code | Fetched at start (git or tarball) | Changes every run |
| Weights, checkpoints, datasets | Never in the image | Incompressible, change often, defeat layer caching |

## 4. Dataset and checkpoint staging

### 4.1 Throughput reference points

| Path | Number | Source |
|---|---|---|
| s5cmd, 10,000 x 1 MiB objects, c5n.18xlarge | 19.7-20.8 Gb/s | [CRT benchmark](https://tingdaok.github.io/CRT-Binding-Performance/3pclient-benchmark/index.html) |
| rclone, same test | 15.4-18.7 Gb/s | same |
| Python CLI, same test | 3.7-4.8 Gb/s | same |
| s5cmd vs default AWS CLI, uploads | 19-26x faster | [DoiT](https://www.doit.com/blog/save-time-and-money-on-s3-data-transfers-surpass-aws-cli-performance-by-up-to-80x) |
| Mountpoint for S3, read per node, c6i.16xlarge | 4.85 GB/s | [AWS HPC blog](https://aws.amazon.com/blogs/hpc/improve-the-speed-and-cost-of-hpc-deployment-with-mountpoint-for-amazon-s3) |
| JuiceFS, MLPerf Storage v2.0, 10 nodes / 40 H100 | 108 GiB/s (3D U-Net), vendor-submitted | [JuiceFS](https://juicefs.com/en/blog/engineering/mlperf-storage-v2-ai-training-storage-performance) |
| RunPod network volume | 200-400 MB/s typical, "up to 10 GB/s peak"; high-performance tier up to 3x throughput | [RunPod](https://docs.runpod.io/storage/network-volumes) |
| RunPod network volume, user measurement | 438 MB/s sequential, poor on many small files | [forum](https://www.answeroverflow.com/m/1415080595020709938) |

These are cloud-provider instance numbers. On a marketplace host the binding constraint is
usually the host's WAN uplink, which is unknown until measured. Imbue, on owned hardware, had
only ~10 Gbit/s shared Ethernet to S3 for 511 hosts and built a local mirror with three-fold
replication to avoid saturating it ([Imbue](https://imbue.com/research/70b-infrastructure/)).

Tool notes: s5cmd's per-object part concurrency defaults to 5 and needs raising for large
objects ([Pure Storage](https://blog.purestorage.com/?p=106907)); rclone with default flags was
measured at less than half the AWS CLI's download speed in one test and needs tuning
(search-summarised, source not re-read). aria2 is the usual choice for HTTP(S) sources with
multi-connection ranges (no benchmark gathered this sweep).

### 4.2 Options compared

| Option | Time before first step | Requirements | Fit for marketplace nodes |
|---|---|---|---|
| Full copy to local NVMe (s5cmd) | Size / uplink | Local disk >= dataset | Good for checkpoints; poor for multi-TB datasets |
| Streaming dataset (WebDataset tar shards, Mosaic MDS, LitData, Lance) | Seconds: first shard | HTTP/S3 reachability, local shard cache | Best default for token datasets |
| FUSE mount (mountpoint-s3, JuiceFS, rclone mount) | Seconds | `/dev/fuse` and `CAP_SYS_ADMIN` | Often unavailable in containers |
| Provider network volume | ~0 if pre-populated | Same data center; attach at create | Good cache tier, single-DC |
| Peer-to-peer among nodes | Size / LAN speed after first seed | Node-to-node reachability | Good for checkpoint fan-out when WAN is the bottleneck |

Details that matter:

- **Streaming formats.** MDS (Mosaic StreamingDataset) shards and indexes samples so that
  shuffling and exact mid-epoch resume are deterministic; a comparison notes it checkpoints the
  stream position after every batch
  ([comparison](https://haoliyin.substack.com/p/multimodal-dataloaders-go-brrrrrrr)). An
  independent benchmark repo put LitData at 3040 img/s, WebDataset 2614, MDS 2642, with MDS
  slowest to first batch, though its own tables conflict
  ([repo mirror](https://gh.r-mj.com/Nano1337/benchmark-dataloader)). No LLM-token benchmark or
  Lance comparison was found. For pre-tokenised text, throughput demand is low (tokens/s x 2-4
  bytes), so determinism and resume semantics matter more than raw MB/s (inference).
- **FUSE in containers.** A SkyPilot issue (#8592, January 2026) reports JuiceFS failing to
  mount inside a RunPod container for lack of FUSE device permissions
  ([issue mirror](https://www.goodfirstissue.org/skypilot-org/skypilot/issues/8592); I could
  not open the issue body, so details are from the search summary). Treat FUSE as a per-node
  capability to probe, not an assumption.
- **RunPod network volumes.** One data center per volume; must be attached at pod creation and
  cannot be attached or detached later; Secure Cloud only; concurrent writers can corrupt data;
  an S3-compatible API allows populating a volume with no compute running; mounts at
  `/workspace` on pods and Instant Clusters
  ([RunPod](https://docs.runpod.io/storage/network-volumes)). The docs do not say whether many
  pods can attach one volume read-write at once (unverified).
- **Peer-to-peer.** Uber's Kraken: a 3 GB two-layer image pulled by 2,600 hosts concurrently
  with a 300 MB/s per-agent limit finished at p50 10 s, p99 18 s, p99.9 22 s; it claims 15k
  hosts per cluster, and warns it does not help if the registry is not the bottleneck
  ([Kraken](https://uber-kraken.readthedocs.io/)). Imbue used Kraken for image distribution
  with "almost no issues" ([Imbue](https://imbue.com/research/70b-infrastructure/)). Alibaba
  reports Dragonfly cutting registry egress by 99.5% for a 200-node, 500 MB distribution
  (secondary write-up: [source](https://besthub.dev/articles/how-alibaba-s-dragonfly-p2p-system-powers-20b-transfers-and-slashes-docker-image-traffic-d0b07220c03a)).
  Both assume nodes can open connections to each other, which on marketplace pods requires
  exposed TCP ports or an overlay (section 5). No evidence was found of either being used for
  checkpoint distribution across marketplace hosts.

### 4.3 Checkpoint specifics

Checkpoint write time is also a recovery-cost term: Falcon cites nearly 100 minutes to dump a
GPT2-100B checkpoint ([Falcon](https://ar5iv.labs.arxiv.org/html/2410.12588)), and a JuiceFS
user reports ~1.7 GB/s per process with `torch.save`
([JuiceFS](https://juicefs.com/en/blog/engineering/mlperf-storage-v2-ai-training-storage-performance),
vendor material). Sharded, per-rank checkpoints fetched with ranged parallel GETs let each rank
start as soon as its own shard lands (inference). Restore policy is covered in
`04-failure-taxonomy-recovery.md`.

## 5. Node mesh

### 5.1 Phone-home agent vs push over SSH

| | Push over SSH | Agent phones home |
|---|---|---|
| Controller needs | Inbound reachability to every node, a key on every node | Only an outbound-reachable endpoint |
| Works behind NAT / mapped ports | Needs per-node host:port bookkeeping | Yes |
| Scaling | Controller opens N sessions; fan-out limits apply | Nodes connect in parallel; O(1) wall time |
| Readiness signal | Poll until sshd answers | Arrival of the first heartbeat |
| Prior art | SkyPilot, Ray cluster launcher, Ansible | dstack `dstack-shim` + runner ([dstack](https://dstack.ai/docs/guides/troubleshooting/)), kubelet, Slurm `slurmd` |

On RunPod, pods "only support TCP and HTTP connections" and the start command and environment
are set through the create call (`dockerStartCmd`, `env`)
([RunPod API](https://docs.runpod.io/api-reference/pods/POST/pods),
[pods overview](https://docs.runpod.io/pods/overview)). That is enough to inject a one-line
bootstrap that downloads and starts an agent, with a join token in `env`. SkyPilot's RunPod
path requires an image with a default entrypoint and root login (or
`SKYPILOT_RUNPOD_DOCKER_USERNAME`), which shows the friction of the SSH-push model on
container providers ([SkyPilot](https://docs.skypilot.co/en/latest/examples/docker-containers.html)).

### 5.2 Addresses, overlays and discovery

- **Managed cluster network (best case).** RunPod Instant Clusters set `PRIMARY_ADDR`
  (`MASTER_ADDR`), `PRIMARY_PORT`, `NODE_ADDR`, `NODE_RANK`, `NUM_NODES`, `NUM_TRAINERS`,
  `HOST_NODE_ADDR`, `WORLD_SIZE`; inter-node traffic uses `ens1`-`ens8`; `eth0` is for external
  traffic only and its 172.x addresses time out between nodes; `NCCL_SOCKET_IFNAME=ens1` is
  required ([RunPod](https://docs.runpod.io/instant-clusters/configuration)). Any mesh `grid`
  builds should export the same variable shapes so user scripts port.
- **Provider private networking.** RunPod `globalNetworking` gives a pod a private IP
  reachable across data centers, but only for on-demand pods in some Secure Cloud data centers
  ([RunPod API](https://docs.runpod.io/api-reference/pods/POST/pods)).
- **Overlay.** Kernel WireGuard needs `/dev/net/tun` and `NET_ADMIN`. Tailscale's
  `--tun=userspace-networking` (`TS_USERSPACE`) works without a TUN device but exposes the
  tailnet only through a SOCKS5/HTTP proxy
  ([Tailscale](https://tailscale.com/docs/concepts/userspace-networking)), which NCCL's socket
  transport cannot use directly (inference). Whether RunPod pods expose TUN was not confirmed.
- **Fallback.** Public IP plus mapped TCP ports per node, with the controller as the registry
  of `(node, host, port)` tuples. Rendezvous and NCCL port ranges then need pinning; see
  `03-rank-configuration.md` and `05-networking-interconnects.md`.

### 5.3 Keys, secrets, clocks

- **SSH keys.** Generate one ephemeral cluster keypair per run; deliver it through the agent
  channel or `env`, never bake it into the image. `mpirun`-style launchers need passwordless
  node-to-node SSH; an agent-RPC launcher does not need SSH at all.
- **Secrets.** RunPod stores secrets and injects them by reference:
  `API_KEY={{ RUNPOD_SECRET_my_api_key }}`
  ([RunPod](https://docs.runpod.io/pods/templates/environment-variables)). A provider-neutral
  design is a short-lived join token in `env`, exchanged by the agent for scoped, expiring
  object-store credentials (inference). Plain `env` values are visible in the provider console
  and API.
- **Clock sync.** A container shares its host's clock and normally cannot set it (inference).
  Imbue hit clock skew early that broke HTTPS and apt
  ([Imbue](https://imbue.com/research/70b-infrastructure/)). `grid` can only measure offset
  (agent timestamp vs controller) and should reject outliers and never compare raw timestamps
  across nodes when computing step times.

### 5.4 Fan-out command execution

| Tool | Model | Notes |
|---|---|---|
| pdsh | Parallel ssh/rsh, sliding window (`-f`, default 32, unverified this sweep) | Simple; needs inbound SSH |
| ClusterShell `clush` | Sliding-window fanout; tree mode through gateways when `topology.conf` exists (default since 1.7); fanout propagates to gateways | [docs](https://clustershell.readthedocs.io/en/latest/tools/clush.html) |
| parallel-ssh | Python library and CLI | Same constraints as pdsh |
| `mpirun` / PMIx | Launch tree over ssh or a resource manager | See `03-rank-configuration.md` |
| Agent RPC | Controller publishes a command; agents on a persistent stream execute and stream back | No inbound ports; what dstack, Ray and kubelet do |

For hundreds of nodes behind mapped ports, agent RPC is the only model that does not depend on
per-node inbound reachability. SSH remains valuable as the human debugging path.

## 6. Preflight and burn-in

### 6.1 What the tools offer

**DCGM diagnostics** ([NVIDIA user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html),
[dcgmi diag reference](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgmi/dcgmi-diag.html)):

| Level | Adds | Duration (NVIDIA, Hopper) |
|---|---|---|
| `-r 1` | Software deployment checks: NVML/CUDA access, nouveau, pending page retirement, failed row remaps, blocked device nodes | < 2.5 s |
| `-r 2` | `memory` (pattern test, 75% allocation default), `pcie` (bandwidth, latency, link gen/width, replay count) | < 2.5 min (4 GPUs), < 10.5 min (8 GPUs); CLI reference says "approximately two minutes" |
| `-r 3` | `diagnostic` (FP64/32/16 matmul with integrity check, 180 s default), `targeted_stress` (30 s), `targeted_power` (120 s, must reach 75% of target), `memory_bandwidth`, `nvbandwidth`, single-node `nccl_tests` | < 10 min (4 GPUs), < 35 min (8 GPUs) |
| `-r 4` | `memtest` (600 s default), `pulse_test` (60 s) | < 45 min (4 GPUs), < 2.25 h (8 GPUs) |

Useful flags: `-j` (JSON), `--fail-early`, `-t <seconds>`, `-p plugin.param=value`,
`-c config.yaml` with per-SKU thresholds (the H100 example sets `min_pci_generation` 3.0 and
`min_pci_width` 16.0). NVIDIA positions level 1 as a readiness metric and levels 3-4 as
administrator post-mortem tools. DCGM's NCCL plugin is single-node only. DCGM needs
`nv-hostengine` and readable/writable device nodes, and consumer GPUs support only level 1; so
on container-only providers and on RTX-class cards it may not run, and `grid` needs its own
fallback checks.

**nccl-tests.** `all_reduce_perf -b 8 -e 8G -f 2 -g 8` under `mpirun -np <N> -H n1:8,n2:8`;
the reported bus bandwidth is normalised to be comparable with link speed
([NVIDIA Cluster Readiness Engine](https://docs.nvidia.com/cluster-readiness-engine/how-to-guides/run-nccl-benchmarks),
[Nebius example](https://docs.nebius.com/slurm-soperator/jobs/examples/nccl-all-reduce)).
`NCCL_DEBUG=INFO` shows which transport was selected, which is the quickest way to catch a
silent fallback from RDMA to TCP.

**Open-source check suites.**

- Imbue `cluster-health` (MIT): `health_checks`, `gpu_stress_test`, `host_validation`
  (NVLink within a host, InfiniBand between hosts), `ufm_events`, `ib_burn`
  ([repo](https://github.com/imbue-ai/cluster-health)).
- Meta GCM health checks: 21 checks including `check-dcgmi`, `check-nvidia-smi` (includes
  clock policy drift), `check-pci` (PCIe link status), `check-nccl`, `check-ib`,
  `check-syslogs`, `check-blockdev`, `memtest`; exit codes 0 OK, 1 WARN, 2 CRITICAL, 3 UNKNOWN
  ([GCM](https://facebookresearch.github.io/gcm/docs/GCM_Health_Checks/health_checks)).
  Secondary sources say they run from Slurm prolog/epilog but disagree on which hook runs DCGM.
- GPUd (Lepton, now under NVIDIA; Apache 2.0): daemon covering ECC, Xid/SXid from the kernel
  log, NVLink, temperature, clock speed, remapped rows, peermem, plus host and container
  runtime ([GitHub](https://github.com/leptonai/gpud),
  [NVIDIA docs](https://docs.nvidia.com/dgx-cloud/lepton/gpud/index.html)).
- NVSentinel (NVIDIA, Kubernetes): DCGM-based detection of thermal, ECC and Xid events;
  observe-only by default, with opt-in quarantine, drain and remediation, including per-GPU
  reset instead of node reboot ([NVSentinel](https://docs.nvidia.com/nvsentinel/getting-started/overview/)).

### 6.2 How operators gate nodes

| Operator | Admission | Continuous | Source |
|---|---|---|---|
| Together AI | Optional acceptance tests at provisioning (off by default because they add time): DCGM Diag (default level 2), GPU Burn (default 5 min), single-node NCCL, multi-node NCCL. Cluster states: Running Tests, Tests Failed, Ready | Passive signals: GPU fell off bus, Xid, uncorrectable ECC, row-remap failure, high PCIe replay rate, thermal throttling, IB rails down (< 8 active), link flapping, read-only filesystem | [Together](https://docs.together.ai/docs/health-checks) |
| Together thresholds | Single-node NCCL >= 300 GB/s; multi-node NCCL >= 330 GB/s; IB write >= 320 Gb/s; NVBandwidth CPU<->GPU >= 30 GB/s; local storage >= 2 GiB/s read, 1 GiB/s write; TorchTitan median MFU >= 30% (~15 min: 9 init + 6 run) | | same |
| Crusoe | Up to 30 tests, escalating sanity -> validation -> performance -> stress: DCGM level 2 then level 4, ECC, PCIe width/speed vs rated, sustained TDP, 20-cycle pulsed power, sysbench CPU/memory, NVMe throughput/IOPS/latency, iperf3, GPUDirect RDMA bandwidth, multi-node NCCL all-reduce; then real multi-node PyTorch jobs before handoff | | [Crusoe](https://docs.crusoe.ai/compute/virtual-machines/burn-in-validation) |
| CoreWeave | Day-1 proprietary burn-in of compute, network and storage | HPC Verification: 20-30 min, hourly, only on idle nodes, pre-empted by customer work; covers silent data corruption, FP8/FP16/BF16 kernel performance regression, 100% SM thermal load, all GPUs and IB; not billed; failures drain the node | [CoreWeave](https://docs.coreweave.com/platform/fleet-management/hpc-verification) |
| Imbue (511 hosts, 4,088 H100) | ~10% of machines failed first boot; "golden set" of known-good hosts; `ib_write_bw --use_cuda` for ~15 min to catch flapping links; multi-node NCCL soak of 12-24 h to surface stalls | Health checks before every relaunch; auto-relaunch picks the healthiest hosts; ~3% of machines break per week; failures concentrate in repeat offenders | [Imbue](https://imbue.com/research/70b-infrastructure/) |
| Meta (GCM) | Health checks around job lifecycle | Same suite periodically | [GCM](https://facebookresearch.github.io/gcm/docs/GCM_Health_Checks/health_checks) |
| Lambda | Nothing found this sweep | | |

Imbue's per-host checklist is the most concrete public list: GPU count, ECC enabled and clean,
NVLink topology up and error-free, disk utilisation <= 95%, GPU containers start under the
NVIDIA runtime, no Xid/SXid in dmesg (and every dmesg line matches an expected list),
InfiniBand error rates, GPUDirect RDMA enabled, VBIOS and NIC firmware versions, PCIe link
speed/width. Their MFU signatures are a useful diagnostic table: ~30% MFU from GDR
misconfigured on one host; 60-80% from degraded links or CPU throttling; persistent dips to
70% from GPU clock throttling; 10x dips every 15 minutes from a co-scheduled CPU-heavy job.

### 6.3 A tiered preflight for rented nodes

On a rented marketplace node, `grid` cannot fix hardware; it can only reject. That argues for
short tests that rank nodes, rather than long burn-in that qualifies them. Durations marked
"est." are my estimates.

| Tier | Checks | Time | Runs |
|---|---|---|---|
| T0 inventory | `nvidia-smi --query-gpu` (count, model, memory, driver, ECC mode, temperature, retired pages / remapped rows), CUDA and NCCL versions, kernel, CPU cores, RAM, free disk, `/dev/fuse`, `/dev/net/tun`, clock offset, `dmesg` Xid scan if readable; `dcgmi diag -r 1` where available | < 10 s (DCGM r1 < 2.5 s) | Every node, before anything else |
| T1 single-node | Per-GPU matmul burn with result check and GFLOPS; GPU memory pattern test over >= 75% of memory; PCIe link width/generation (`nvidia-smi -q` or `lspci -vv`); host<->GPU copy bandwidth; `fio` sequential read/write on scratch; download probe against the object store; `dcgmi diag -r 2` where it runs | 1-3 min est. (DCGM r2 alone: 2-10 min) | Every node, concurrent with data staging |
| T2 intra-node comms | Single-node `all_reduce_perf` over all local GPUs | 30-60 s est. | Multi-GPU nodes |
| T3 cluster | Pairwise reachability and TCP throughput between sampled node pairs; multi-node `all_reduce_perf` on the candidate set, with bisection on failure; uniformity check on driver, CUDA, NCCL, torch versions and GPU model | 1-3 min est. | Once on the candidate set |
| T4 soak (optional) | A few hundred real training steps at reduced size, comparing per-rank step times | 2-5 min est. | Before long, expensive runs |

Notes:

- **Bisection** is how Imbue isolated bad hosts: a failing 48-machine job was split into groups
  of eight, then six, to find the culprits.
- **Version uniformity** is a hard gate for one synchronous job; mixed driver or NCCL versions
  belong to `06-gpu-hardware-nuances.md` and `05-networking-interconnects.md`.
- **Soak tests catch what burn-in cannot.** Imbue needed 12-24 h of multi-node NCCL to surface
  stalls. `grid` cannot afford that up front, so the first minutes of the real run are the
  soak: keep spares warm until the step-time distribution is stable (inference).
- **Record results.** Failures concentrate in repeat offenders (Imbue), so a persistent
  per-host scorecard keyed by provider host/machine ID, where the provider exposes one, lets
  `grid` avoid known-bad hosts in later runs.

## 7. Observability during the run

- **Per-rank step time is the primary signal.** Falcon measured fail-slows in a production
  cluster: 1.5% of single-node jobs, 40% of four-node jobs and 59% of 512-1024 GPU jobs were
  affected, with average durations of 10, 24 and 72 minutes and completion-time increases of
  11.79%, 15.45% and 34.59%. Causes were CPU contention and thermal GPU degradation for
  computation, and network congestion for communication (42 of 43 four-node cases)
  ([Falcon](https://ar5iv.labs.arxiv.org/html/2410.12588)). ByteDance's five-month trace study
  likewise found stragglers are often not hardware faults
  ([OSDI '25, arXiv 2505.05713](https://www.emergentmind.com/papers/2505.05713)).
- **Detection without framework changes.** Falcon hooks NCCL calls with `LD_PRELOAD`, infers
  iteration boundaries from the autocorrelation of the call sequence (first lag with ACF
  >= 0.95), flags change points with Bayesian online change-point detection (threshold 0.9),
  ignores changes under 10%, then marks communication groups whose transfer time exceeds 1.1x
  the median and validates with GEMM and peer-to-peer link tests. Mitigation escalates by
  ski-rental cost: do nothing, rebalance micro-batches (seconds), re-map topology (< 1 min),
  checkpoint and restart on healthy nodes. Minder reports detecting faulty machines in 3.6 s
  on average with precision 0.904
  ([search summary; paper not re-read](https://www.catalyzex.com/author/Zhanghan%20Wang)).
- **In a synchronous job every rank reports the same wall-clock step time**, so the useful
  per-rank metrics are the components: data-loader wait, forward/backward compute time, and
  time blocked in collectives (inference). A rank that computes slowly shows low wait and high
  compute; its peers show the opposite.
- **GPU telemetry.** Imbue installed Prometheus node exporter and DCGM exporter on every host
  and watched DCGM clock-throttle events, excluding `gpu_idle` and `power_cap`. Where DCGM
  cannot run in the container, the agent can poll NVML directly for utilisation, temperature,
  power, clock throttle reasons, ECC counters and Xid events (inference).
- **Hang detection.** Imbue set an NCCL watchdog timeout of 600,000 ms and used py-spy and GDB
  on stalled processes. The agent should capture stack dumps from every rank before killing a
  stalled job. Details in `04-failure-taxonomy-recovery.md`.
- **Synchronised noise.** Imbue traced a gradual MFU sag to Python garbage collection and fixed
  it by running GC at fixed intervals on all hosts at once.
- **Logs.** With an agent on every node, per-rank stdout/stderr can stream over the existing
  agent connection, tagged with rank and host, and spooled to local disk so a controller
  disconnect loses nothing (inference). This avoids deploying a separate log stack on
  short-lived nodes.

## Implications for grid

1. **Ship a phone-home agent as the core primitive.** A single static binary started from the
   pod's start command with a join token. It gives readiness signalling, command fan-out, log
   and metric streaming, and preflight execution without inbound reachability, and it works on
   container-only providers. Keep sshd as a debugging path, not the control path.
2. **Run a per-node pipeline, not cluster-wide stages.** Each node independently does
   T0 -> (T1 preflight || env install || checkpoint fetch) -> reports. The controller admits
   the first N that pass and culls the rest at a deadline.
3. **Over-provision by default, with k derived from measured data.** Start with a
   conservative default informed by Imbue's 10-20% spares, and replace it with per-provider
   time-to-ready and failure-rate distributions collected by `grid` itself. Cull by rank on
   preflight scores, not just pass/fail.
4. **Own the image.** Publish a small zstd-compressed base (CUDA userspace, agent, uv, s5cmd,
   nccl-tests, a matmul burn tool), layered on whatever base the provider is most likely to
   have cached. Never put weights or datasets in the image. Put the Python environment in a
   cacheable artifact, with the uv cache on the same filesystem as the venv.
5. **Do not depend on lazy-pull snapshotters, FUSE, TUN or DCGM.** Probe each as a node
   capability in T0 and use it when present. Every one of them has a user-space fallback:
   parallel fetch, streaming datasets, mapped TCP ports, and NVML-based checks.
6. **Stream datasets, parallel-fetch checkpoints.** Default to a streaming shard format with
   deterministic resume for training data. Fetch the checkpoint with ranged parallel GETs, and
   add a peer-fetch path (first node seeds the rest over the cluster network) for cases where
   the WAN uplink is the bottleneck. Use provider network volumes as a single-data-center
   cache, populated through the S3-compatible API before the cluster exists.
7. **Adopt the operators' admission ladder, tiered and time-boxed.** T0 inventory (seconds),
   T1 single-node burn and I/O (1-3 min, overlapped with staging), T2/T3 NCCL all-reduce with
   bisection, optional T4 soak. Thresholds should be relative to the fleet median for that GPU
   model, since absolute numbers like Together's 300 GB/s assume a specific HGX + InfiniBand
   build.
8. **Export RunPod-compatible rank variables** (`NODE_RANK`, `NUM_NODES`, `NUM_TRAINERS`,
   `WORLD_SIZE`, `PRIMARY_ADDR`, `PRIMARY_PORT`, `NODE_ADDR`) on every mesh `grid` builds, so
   scripts written for Instant Clusters run unchanged.
9. **Make straggler telemetry part of the launcher.** Emit per-rank data-wait, compute and
   collective time every step; run change-point detection in the controller; keep warm spares
   through the first minutes of the run, since that is the real soak test.
10. **Keep a host scorecard across runs.** Persist preflight results and in-run incidents per
    provider host identifier, and feed it into placement and culling.

## Open questions

- **Provisioning latency distributions.** No trustworthy public p50/p95/p99 time-to-ready for
  RunPod, Lambda, Vast or others, by GPU type, region, and cached vs uncached image. Needs a
  probe harness; coordinate with `07-provider-landscape.md`.
- **RunPod host image cache.** Lifetime, eviction, whether template base images are pre-seeded
  on pod hosts, and whether zstd layers are supported by the host runtime version. Official
  documentation was not found.
- **Container capabilities per provider.** Is `/dev/fuse` or `/dev/net/tun` available in
  RunPod pods (Secure vs Community Cloud), Vast, and others? Can `nv-hostengine` and
  `dcgmi diag -r 2` run in an unprivileged pod? Is `dmesg` readable for Xid scanning? I only
  have one indirect report (SkyPilot #8592) for FUSE.
- **Network volume concurrency.** Can hundreds of pods attach one RunPod network volume at
  once, and what is the aggregate read throughput under that load?
- **Peer-to-peer checkpoint distribution on marketplace networks.** No evidence found for
  Dragonfly, Kraken or BitTorrent used across marketplace pods; needs an experiment on
  node-to-node bandwidth over mapped ports.
- **Environment install timings.** No measured numbers for cold uv install of a current
  torch + CUDA stack, wheel sizes for current builds, or conda-pack / pixi-pack restore times.
- **Streaming dataset formats for token data.** No head-to-head benchmark of MDS, LitData,
  WebDataset and Lance on pre-tokenised LLM corpora, and no comparison of their resume
  semantics under elastic world size.
- **Preflight thresholds for non-reference hardware.** What all-reduce bus bandwidth and
  matmul throughput to expect per GPU model over TCP at 10/25/100 GbE; needs data from
  `05-networking-interconnects.md` and `06-gpu-hardware-nuances.md`.
- **Lambda and Meta internals.** Nothing found on Lambda's node acceptance process; GCM's
  prolog/epilog wiring is only described by conflicting secondary sources.
- **Fan-out tool defaults** (pdsh `-f`, clush fanout) were not confirmed from primary docs.
- **Falcon-style detection cost.** Overhead of `LD_PRELOAD` NCCL hooking in practice, and
  whether a simpler in-framework timer gives equivalent signal for `grid`.
