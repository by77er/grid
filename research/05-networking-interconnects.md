# 05 — Networking and interconnects for multi-node training

Research sweep date: 2026-11-09. NCCL documentation current at the time of writing is 2.32.3.
Sibling files referenced: `03-rank-configuration.md` (rank/parallelism mapping), `04-failure-taxonomy-recovery.md`
(failure handling), `06-gpu-hardware-nuances.md` (NVLink/PCIe detail), `07-provider-landscape.md` (provider
products), `08-heterogeneous-distributed-training.md` (algorithms that tolerate slow links),
`09-fast-bootstrap-preflight.md` (preflight checks).

## Summary

- The inter-node network decides whether synchronous multi-node training is viable at all. Two independent
  public benchmarks of FSDP show a **10x to 57x** throughput collapse when the same GPUs talk over TCP Ethernet
  instead of 400 Gb/s InfiniBand ([Azure H100 study](https://jingchaozhang.github.io/IB-vs-Ethernet-Fine-Tuning-at-Scale),
  [Nebius H100/H200 study](https://maknee.github.io/blog/2025/Network-And-Storage-Training-Skypilot)).
- NCCL's TCP socket transport does not get near line rate on fast links. Measured `all_reduce_perf` bus bandwidth
  over TCP is about **3–6 GB/s** even on 80–100 Gb/s NICs, against **390–480 GB/s** on 8x400 Gb/s InfiniBand
  (sources in the quantification section). Treat "100GbE without RDMA" as roughly a 25–50 Gb/s network for NCCL.
- A simple model predicts the damage: per-step network time is about `2S / busbw` for DDP and `3S / busbw` for
  FSDP, where `S` is the gradient/parameter byte size. It reproduces the published Azure number to within 3%.
- NCCL is synchronous and bandwidth-homogeneous by design: a ring runs at the speed of its slowest hop, so one
  slow node or one mis-selected interface drags down the whole job. NCCL has gained real resilience features
  since 2.24 (RAS, `ncclCommShrink`, IB port failover and recovery), but all the link-level ones are IB/RoCE only.
- On marketplace providers the network a pod actually gets ranges from 100 Mb/s (RunPod global networking)
  to 8x400 Gb/s RDMA (RunPod Instant Clusters, Lambda 1-Click Clusters). RunPod pods do not support UDP, which
  rules out direct WireGuard/Tailscale/Nebula tunnels there. NCCL needs every rank to reach every other rank
  directly on arbitrary TCP ports, so NAT with mapped ports is not enough.
- grid should measure, not assume: a 2–3 minute preflight (interface inventory, `iperf3`, `ib_write_bw`,
  2-node `all_reduce_perf`) is enough to classify each node pair into a network tier and derive placement.

## 1. Interconnect technologies

| Technology | What it is | Typical per-port speed | Loss handling | How NCCL uses it |
|---|---|---|---|---|
| NVLink / NVSwitch | GPU-to-GPU link inside a node (or rack, for NVL72) | H100: about 900 GB/s aggregate per GPU ([cited in Azure study](https://jingchaozhang.github.io/IB-vs-Ethernet-Fine-Tuning-at-Scale)) | Lossless, credit based | P2P transport; NVLS (NVLink SHARP) offload on NVSwitch gen 3 with Hopper and later ([env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)) |
| InfiniBand | Lossless, credit-based fabric with a subnet manager; native RDMA | HDR 200 Gb/s ([Megatron-LM, Selene](https://ar5iv.labs.arxiv.org/html/2104.04473)), NDR 400 Gb/s ([Azure ND H100 v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series)), 800 Gb/s with CX8 ([NCCL 2.27 blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27)) | Link-level flow control, adaptive routing | Built-in `IB` net (verbs, RC queue pairs) |
| RoCEv2 | RDMA verbs carried in UDP/IP over Ethernet | 100–400 Gb/s | Needs PFC and/or ECN-based congestion control configured on the fabric | Same `IB` net; needs a RoCE v2 GID and usually a traffic class |
| AWS EFA | AWS's own NIC with the SRD transport: reliable, out-of-order, sprayed over ECMP paths | 3.2 Tb/s aggregate on p5/p5e/p5en/p6-b200 ([NVIDIA Dynamo EFA guide](https://docs.nvidia.com/dynamo/latest/kubernetes/installation/rdma-setup/efa-on-aws)) | Reliability in SRD | Not the `IB` net. Needs the [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) plugin (NCCL -> libfabric -> EFA) |
| Google Cloud | RoCE on ConnectX-7 with a dedicated "RoCE network profile" VPC, one subnet per NIC; gVNIC for host traffic | 3.2 Tb/s inter-node on A3 Ultra / A4 ([GCP docs](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-vm)) | Managed by Google | `IB` net over RoCE, delivered via a Google-provided NCCL/RDMA installer |
| Azure | Real InfiniBand passed to the VM: one 400 Gb/s Quantum-2 CX7 port per GPU | 3.2 Tb/s per VM ([Azure docs](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series)) | IB | `IB` net |
| Plain TCP Ethernet | Kernel TCP sockets | 1–100 Gb/s nominal | TCP retransmit | Built-in `Socket` net; data is staged through pinned host memory |

Notes that matter for grid:

- **RoCE vs IB at the NCCL level are the same code path** (`net_ib.cc`), but RoCE has more ways to be
  misconfigured. Since NCCL 2.21 the GID is selected dynamically and NVIDIA says not to set
  `NCCL_IB_GID_INDEX`; before 2.21 a wrong index produces `ibv_modify_qp failed with error Invalid argument`
  ([NCCL networking troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_sources/troubleshooting/networking_troubleshooting.rst.txt)).
  `NCCL_IB_TC` should be set to the fabric vendor's value on RoCE (same source).
- **EFA is not InfiniBand.** EFA exposes verbs devices but supports only SRD and UD queue pairs, not the RC
  type NCCL's `IB` net uses, so creating an RC QP fails (community sources:
  [UCCL EFA guide](https://uccl-project.github.io/posts/efa-programming),
  [rdma-core man page](https://man7.org/linux/man-pages/man3/efadv_create_driver_qp.3.html) for the SRD
  description). Without the plugin, NCCL on AWS silently runs over TCP sockets. The plugin needs libfabric
  endpoints of type `FI_EP_RDM` with `FI_TAGGED`/`FI_MSG`, plus `FI_HMEM` for GPUDirect
  ([aws-ofi-nccl README](https://github.com/aws/aws-ofi-nccl)).
- **GPUDirect RDMA (GDR)** lets the NIC read and write GPU memory directly, skipping the copy through host
  memory. The GPU buffer is registered either through the `nv_peer_mem` module or the kernel's dma-buf path
  (`NCCL_DMABUF_ENABLE`, default 1 since 2.13, auto-disabled if unsupported;
  [env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)). NCCL only uses GDR when the
  NIC is close enough to the GPU in the PCI tree, controlled by `NCCL_NET_GDR_LEVEL` (see section 3).

### Rail-optimised fat tree

In a rail-optimised design, NIC *k* of every node connects to leaf switch *k*, so each "rail" is its own
leaf-level network and same-rail traffic never needs the spine
([NVIDIA NCCL 2.12 blog](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/),
[NVIDIA IB cluster bring-up](https://docs.nvidia.com/networking/display/ibclusterbringupprocedure/setting+the+infiniband+cluster+topology)).
Lambda documents its clusters as "non-blocking fabric in a rail-optimized topology"
([Lambda docs](https://docs.lambdalabs.com/1-click-clusters/getting-started)). Selene used a three-level
(leaf, spine, core) fat tree with 850 switches ([Megatron-LM](https://ar5iv.labs.arxiv.org/html/2104.04473)).

Consequences for NCCL:

- GPU *i* is normally paired with NIC *i* (same PCI switch), so a ring that enters a node on rail 3 should leave
  on rail 3. `NCCL_CROSS_NIC=0` enforces this; the default `2` prefers it but allows crossing
  ([env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).
- **PXN** (NCCL 2.12+) moves data over NVLink to the local GPU that sits on the destination's rail, then out of
  that GPU's NIC, so point-to-point traffic avoids the spine and messages get aggregated (same NVIDIA blog).
  Controlled by `NCCL_PXN_DISABLE` and `NCCL_P2P_PXN_LEVEL` (default 2, always use).
- Some fabrics are strictly rail-only (no cross-rail route at all). NCCL 2.29.7 added "GIN support for platforms
  without cross-rail connectivity" and 2.30.5 added `NCCL_NET_MERGE_POLICY=RAIL`
  ([2.29.7 notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/release-notes/rel_2-29-7.html),
  [env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

## 2. How NCCL picks a transport

NCCL separates *intra-node* transports from *inter-node* "net" transports
([Demystifying NCCL, arXiv 2507.04786](https://arxiv.org/html/2507.04786v1), analysing NCCL 2.19.1):

- **Intra-node:** `P2P` (NVLink if present, else PCIe; `P2P_DIRECT` when ranks share a process) and `SHM`
  (host shared memory, used when P2P is unavailable or poor, for example across CPU sockets).
- **Inter-node, `Socket`:** the GPU copies into a CUDA-pinned host buffer, which is sent with ordinary socket
  calls; the receiver reverses it. Two extra PCIe copies and the kernel TCP stack are in the path.
- **Inter-node, `IB`:** RDMA writes over RC queue pairs, driven by a CPU proxy thread. Data is staged in host
  memory unless GDR applies. Each rank pair has a forward QP (data `RDMA_WRITE`s followed by a zero-byte
  `RDMA_WRITE_WITH_IMM` carrying the size) and a reverse QP for clear-to-send; a loopback `RDMA_READ` flushes
  GDR writes.

Selection order: an external plugin if one is found (`NCCL_NET_PLUGIN`: tries `libnccl-net-<name>.so`, then
`libnccl-net.so`, then internal; `none` forces internal), then internal `IB`, then internal `Socket`.
`NCCL_NET` forces one by name (`IB`, `Socket`, or a plugin name) since 2.10
([env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

Two traps:

1. **`NCCL_IB_DISABLE=1` does not guarantee TCP.** In the Azure study the container shipped an external RDMA
   plugin (`libnccl-net.so`), and `NCCL_IB_DISABLE=1` alone still ran over RDMA; `NCCL_NET=Socket` was what
   actually forced sockets ([source](https://jingchaozhang.github.io/IB-vs-Ethernet-Fine-Tuning-at-Scale)).
   The reverse trap is more dangerous for grid: a missing device or plugin makes NCCL fall back to sockets
   without failing. Since 2.32 such fallbacks are logged at the new `ATTN` level
   ([2.32.3 notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-32-3.html)).
2. **The bootstrap network is always TCP.** Even with IB data paths, ranks rendezvous and exchange connection
   information over TCP sockets on the interface chosen by `NCCL_SOCKET_IFNAME`. Interfaces that are UP but
   cannot reach other nodes cause init failures or hangs
   ([troubleshooting doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_sources/troubleshooting/networking_troubleshooting.rst.txt)).

To verify what was chosen, run with `NCCL_DEBUG=INFO` (default subsystems `INIT,BOOTSTRAP,ENV`; add
`NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING`) and read the `NET/IB` or `NET/Socket` lines.

## 3. NCCL environment variables that matter

All from the [NCCL environment variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
(2.32.3) unless noted. NCCL also reads these from a config file (`NCCL_CONF_FILE`, default `~/.nccl.conf`,
since 2.23), which is a convenient injection point for grid.

### Interface and transport selection

| Variable | Meaning | Default / notes |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | Comma-separated interface prefixes; `^` excludes, `=` exact match (`=eth0,eth1`, `^docker0`) | Auto: skips `lo` and `docker*` unless nothing else exists, favours `ib*`. Wrong value returns `ncclInvalidUsage` since 2.24 ([2.24 blog](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24)) |
| `NCCL_SOCKET_FAMILY` | `AF_INET` or `AF_INET6` | |
| `NCCL_NET` | Force net by name: `IB`, `Socket`, plugin name | unset |
| `NCCL_NET_PLUGIN` | Plugin suffix/library, or `none` | internal |
| `NCCL_IB_DISABLE` | `1` disables the internal IB/RoCE net | |
| `NCCL_IB_HCA` | `<hca>[:<port>[:<rail>[:<plane>]]]` list; `^` excludes, `=` exact; max 32 HCAs | all devices |
| `NCCL_IB_GID_INDEX` | RoCE GID index | `AUTO` (dynamic since 2.21) |
| `NCCL_IB_ADDR_RANGE`, `NCCL_IB_ADDR_FAMILY`, `NCCL_IB_ROCE_VERSION_NUM` | Constrain auto GID selection (CIDR, family, RoCE version) | ignored, `AF_INET`, `2` (2.21) |
| `NCCL_IB_SL`, `NCCL_IB_TC`, `NCCL_IB_PKEY`, `NCCL_IB_PKEY_VALUE` | Service level, traffic class, partition key | `AUTO`, `AUTO`, `0`, unset (PKEY_VALUE since 2.31) |
| `NCCL_CROSS_NIC` | `0` same NIC per ring/tree, `1` allow different, `2` prefer same | `2` |
| `NCCL_NETDEVS_POLICY` | `MAX:N` or `ALL`: how many net devices each GPU may use | auto (2.28) |
| `NCCL_IB_MERGE_NICS`, `NCCL_NET_MERGE_LEVEL`, `NCCL_NET_FORCE_MERGE`, `NCCL_NET_MERGE_POLICY` | NIC fusion: merge several physical NICs into one virtual device | merge dual-port NICs (`PORT`); see [2.24 blog](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24) |
| `NCCL_OOB_NET_ENABLE`, `NCCL_OOB_NET_IFNAME` | Use a separate out-of-band network for bootstrap | 2.23 |

### GPUDirect and intra-node P2P

| Variable | Meaning | Default / notes |
|---|---|---|
| `NCCL_NET_GDR_LEVEL` | Max GPU–NIC distance for GDR: `LOC` (never), `PIX` (same PCI switch), `PXB` (via PCI switches), `PHB` (same NUMA node, through CPU), `SYS` (always) | auto by platform. Use strings, not integers |
| `NCCL_NET_GDR_READ` | Use GDR on the send side too | `1` on NVLink platforms, else `0` (PCIe reads are "slightly slower on some platforms") |
| `NCCL_NET_GDR_C2C` | GDR via CPU-attached NIC where CPU–GPU is C2C | `1` since 2.27 |
| `NCCL_P2P_DISABLE` | `1` disables GPU P2P (falls back to SHM) | |
| `NCCL_P2P_LEVEL` | Max distance for P2P: `LOC`, `NVL`, `PIX`, `PXB`, `PHB`, `SYS` | auto |
| `NCCL_P2P_DIRECT_DISABLE` | `1` disables direct user-buffer access between GPUs in one process | |
| `NCCL_P2P_PXN_LEVEL`, `NCCL_PXN_DISABLE` | PXN control | `2`, `0` |
| `NCCL_P2P_NET_CHUNKSIZE` | Send/recv network message size | 131072 bytes |
| `NCCL_SHM_DISABLE` | `1` disables shared-memory transport | |
| `NCCL_NVLS_ENABLE` | NVLink SHARP: `0` off, `1`/`2` on if supported | `2`; init fails if resources cannot be allocated (2.29 reverted the 2.27/2.28 silent fallback) |

### Socket transport tuning

| Variable | Meaning | Default / notes |
|---|---|---|
| `NCCL_SOCKET_NTHREADS` | CPU helper threads per connection, 1–16 | 1; 2 on AWS; 4 on GCP gVNIC |
| `NCCL_NSOCKS_PERTHREAD` | Sockets per helper thread | 1; 8 on AWS. Product with NTHREADS must be <= 64 |
| `NCCL_SOCKET_RETRY_CNT`, `NCCL_SOCKET_RETRY_SLEEP_MSEC` | Connect retries after `ETIMEDOUT`/`ECONNREFUSED`/`EHOSTUNREACH` | 34, 100 ms (2.24) |
| `NCCL_SOCKET_POLL_TIMEOUT_MSEC` | Poll timeout | 0 (2.28) |
| `NCCL_BUFFSIZE` | Per-channel buffer | 4194304 (4 MiB) |
| `NCCL_SOCKET_INLINE`, `NCCL_SOCKET_MIN_TASKSIZE` | TCP inlining and task size, named in the [cross-DC blog](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness) | not in the 2.32.3 env reference (unverified defaults) |

The defaults of 1 thread x 1 socket mean a generic (non-AWS, non-GCP) host sends each connection through a
single TCP stream. AMD's RCCL docs, which mirror NCCL, suggest `NSOCKS_PERTHREAD=4` for generic 100G networks
([RCCL env docs](https://rocm.docs.amd.com/projects/rccl/en/latest/userguide/source/env.html)).

### IB reliability and tuning

| Variable | Meaning | Default / notes |
|---|---|---|
| `NCCL_IB_TIMEOUT` | QP timeout = 4.096 us x 2^value | 20 since 2.23 (18 in 2.14, 14 before) |
| `NCCL_IB_RETRY_CNT` | 0–7 | 7 |
| `NCCL_IB_QPS_PER_CONNECTION` | 1–128 QPs per connection, adds ECMP entropy on RoCE | 1 |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | `1` split each message over QPs, `0` round-robin | 0 since 2.20 |
| `NCCL_IB_AR_THRESHOLD`, `NCCL_IB_ADAPTIVE_ROUTING` | Adaptive-routing message split | 8192 bytes; AR on for IB, off for RoCE |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 0 off, 1 force, 2 auto | 2 |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | Fatal async IB events stop communication and surface as errors | 1 (2.23) |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | `1` enables automatic failover between a GPU's local IB/RoCE ports | off; 2.29.7 ([notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/release-notes/rel_2-29-7.html)) |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY` | `1` enables recovery from transient port faults without re-init | off; 2.30.3 ([notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html)) |
| `NCCL_IB_EVENT_BASED_LB` | Split traffic across a fused device in proportion to current link speed | 0 (2.31) |
| `NCCL_IB_OOO_RQ` | Out-of-order receive (Dynamic Direct Path multipath) | 0 (2.30) |

### Algorithm, protocol, debugging

| Variable | Meaning | Default / notes |
|---|---|---|
| `NCCL_ALGO` | Allowed algorithms: `Ring`, `Tree`, `CollnetDirect`, `CollnetChain`, `NVLS`, `NVLSTree`, `PAT` (2.23+). `^` excludes. Since 2.24 per-function: `NCCL_ALGO="allreduce:tree;broadcast:ring"` | auto. Invalid tokens now fail; no silent fallback to ring |
| `NCCL_PROTO` | `LL`, `LL128`, `Simple` | auto. Docs warn that forcing `LL128` on unsupported platforms can corrupt data |
| `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` | Channel count bounds (replace `NCCL_MIN/MAX_NCHANNELS`, deprecated 2.17) | <= 64 |
| `NCCL_DEBUG` | `VERSION`, `WARN`, `ATTN` (2.32), `INFO`, `TRACE` | |
| `NCCL_DEBUG_SUBSYS` | `INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC,PROXY,NVLS,BOOTSTRAP,REG,RAS,...` | `INIT,BOOTSTRAP,ENV` |
| `NCCL_DEBUG_FILE` | Log path, `%h` hostname, `%p` PID | stdout |
| `NCCL_TOPO_DUMP_FILE` | Write detected topology XML (physical devices only; fused virtual NICs are not shown) | unset |
| `NCCL_TOPO_FILE` | Load a topology XML | `/var/run/nvidia-topologyd/virtualTopology.xml` if present |
| `NCCL_IGNORE_NET_MISMATCH` | `0` fails init when net device counts differ across ranks | 1 (ignore), 2.30 |
| `NCCL_RAS_ENABLE`, `NCCL_RAS_ADDR`, `NCCL_RAS_TIMEOUT_FACTOR` | RAS subsystem | 1, `localhost:28028`, 1 (2.24) |

## 4. Ring, tree and friends

From the [Demystifying NCCL paper](https://arxiv.org/html/2507.04786v1):

- **Ring AllReduce** over *k* ranks takes `2k-1` steps per chunk: one `send`, `k-2` `recvReduceSend`, one
  `recvReduceCopySend`, `k-2` `recvCopySend`, one `recv`. Bandwidth-optimal (each byte crosses each link about
  twice) but latency grows linearly with rank count.
- **Tree AllReduce** uses a double binary tree across nodes (GPUs inside a node are chained): reduce up, then
  broadcast down. No node is interior in both trees. Latency grows with `log(nodes)`.
- Measured on 16 GH200 nodes: ring wins for large messages, tree for small ones; `LL`/`LL128` protocols win
  below about 64 KiB, `Simple` wins at large sizes. `LL` reaches only 25–50% of peak bandwidth and forces data
  through host memory, so it cannot use GDR.
- **PAT** (2.23+) and **NVLS/NVLSTree** (NVSwitch offload) are newer options; 2.24 tuned the PAT-to-ring
  transition at scale ([2.24 blog](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24)).
- Work is split over **channels**. Each channel is one CUDA block on its own SM and can exit the node through a
  different NIC, which is how NCCL uses all rails. On IB there are by default 2 logical channels per remote GPU
  per NIC.

Gradient sync for LLMs is dominated by multi-gigabyte messages, so ring + `Simple` is what normally runs, and
the relevant metric is steady-state bandwidth, not latency. Tensor-parallel activations are smaller and far more
frequent, so latency and the tree/LL path matter there.

## 5. Mixed NIC speeds and one slow node

- **Collectives are synchronous.** In a ring every byte passes every hop, so throughput is set by the slowest
  link. This is the standard criticism of NCCL in the Blink paper
  ([arXiv 1910.04940](https://www.arxiv-vanity.com/papers/1910.04940/)) and is restated for degraded links in
  "Don't Let a Few Network Failures Slow the Entire AllReduce" ([arXiv 2606.01680](https://arxiv.org/abs/2606.01680)):
  when a NIC fails NCCL routes through the node's surviving NICs, training continues, but the degraded node stays
  on the ring's critical path and slows everyone. That paper reports existing fault-tolerant schemes cost up to
  57% overhead with up to 50% bandwidth loss on one node, and proposes an algorithm (OptCC) within 2–6% of the
  fault-free ring.
- **Topology is negotiated down.** Each rank builds its own topology graph and ranks then agree on a common
  channel count and bandwidth estimate, taking the minimum across ranks (from reading NCCL's `init.cc`; unverified
  in this sweep). The practical effect: one node with a single NIC caps the number of channels for the whole
  communicator.
- **Mismatched device counts** across ranks are tolerated by default (`NCCL_IGNORE_NET_MISMATCH=1`); set it to
  `0` to make a heterogeneous cluster fail at init instead of running in a degraded shape (2.30).
- **Socket link speed is read from the OS.** For virtual interfaces (veth, overlay, WireGuard) the reported
  speed is often missing or wrong, so NCCL's cost model works from a default guess (unverified which value in
  current releases). The tuner cannot be trusted on overlays; measure instead.
- **`LL128` in heterogeneous GPU jobs** was enabled on Hopper and later in 2.30.3
  ([notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html)); earlier versions disabled it
  when GPU generations were mixed (see `06-gpu-hardware-nuances.md`).
- **Mixed transports in one communicator are not generally supported.** A communicator uses the first net that
  loads successfully. The experimental cross-datacenter work adds `NCCL_ALLNET_ENABLE=1` (load all nets, disables
  CollNet), a per-device `fabricId` (set via `NCCL_IB_HCA=device:port:fabricID`, decoded with
  `NCCL_IB_FABRICID_MAXRAIL`), and `NCCL_SCATTER_XDC=1` so that rings cross the inter-DC link through different
  nodes; an AllReduce is composed as intra-DC ReduceScatter, inter-DC AllReduce, intra-DC AllGather, with
  `2*(nDc-1)` cross-DC connections ([NVIDIA cross-DC deep dive](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness)).
  It lives on a `cross-dc` branch / PR #1659; whether it is in a tagged release is unverified.

The honest conclusion: stock NCCL does not do heterogeneity-aware collectives. If nodes differ in network tier,
either keep each synchronous communicator inside one tier, or move the heterogeneity up into the training
algorithm (`08-heterogeneous-distributed-training.md`).

## 6. Recent NCCL releases: resilience and observability

| Version | Feature | Source |
|---|---|---|
| 2.21 | Dynamic RoCE GID selection; dual-port fusion | [troubleshooting doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_sources/troubleshooting/networking_troubleshooting.rst.txt) |
| 2.22 | `NCCL_RUNTIME_CONNECT=1`: peers connect lazily at first use rather than at init | [env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) |
| 2.23 | PAT algorithm; `NCCL_IB_TIMEOUT` default 20; async IB events returned as errors; OOB bootstrap net; `.nccl.conf` | env docs |
| 2.24 | **RAS subsystem**: one thread per process, TCP keep-alives between processes, queried with `ncclras` or `nc localhost 28028`; reports per-communicator status `OK`/`MISMATCH`/`INCOMPLETE`/`DEAD`. NIC fusion. Strict `NCCL_ALGO`/`NCCL_PROTO`. Socket connect retries | [2.24 blog](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24) |
| 2.27 | **`ncclCommShrink`**: build a new communicator without failed ranks; `NCCL_SHRINK_DEFAULT` (planned, finishes in-flight ops) or `NCCL_SHRINK_ABORT` (after a failure, aborts in-flight ops). Symmetric memory kernels; SHARP for AllGather/ReduceScatter; Direct NIC | [2.27 blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27) |
| 2.28 | RAS JSON output for machine parsing; `NCCL_NETDEVS_POLICY` | [2.28.7 notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-7.html) |
| 2.29 | **IB/RoCE port failover** (`NCCL_IB_RESILIENCY_PORT_FAILOVER=1`) for GPUs with multiple local ports; `ncclCommSuspend`/`ncclCommResume`; Apache 2.0 licence; RAS realtime monitoring | [2.29.7 notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/release-notes/rel_2-29-7.html) |
| 2.30 | **IB port recovery** (`NCCL_IB_RESILIENCY_PORT_RECOVERY=1`); out-of-order receive; net-count mismatch checks; parameter query API (`ncclParamGet`) | [2.30.3 notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html) |
| 2.31 | `NCCL_IB_PKEY_VALUE`, `NCCL_IB_EVENT_BASED_LB` | env docs |
| 2.32 | Optional **TLS for NCCL-owned socket traffic** via `ncclSetEncryption` (needs an OpenSSL 3 build); RAS diagnostics for NVLink/NIC state and speed, PCI/GDR config, Xid events, GPU-resident progress counters for stalled kernels; `ATTN` log level; Git-revision mismatch across ranks reported at init | [2.32.3 notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-32-3.html) |

Limits to keep in mind:

- Port failover and recovery are **IB/RoCE only** and need more than one local port or device per GPU. Nothing
  equivalent exists for the socket transport: a dropped TCP connection mid-collective is still a fatal
  communicator error.
- `ncclCommShrink` is a library primitive. The NVIDIA blog does not say how failed ranks are detected or what
  the framework must do afterwards; PyTorch-level use is a question for `04-failure-taxonomy-recovery.md`.
- A RAS *API* (rather than the text/JSON socket) was on the Q2 2026 roadmap
  ([roadmap issue #2090](https://github.com/NVIDIA/nccl/issues/2090)); whether it shipped is unverified.
  A whole-process CUDA checkpoint helper exists in `contrib/nccl_checkpoint` (same issue).
- The NCCL version is whatever the framework wheel bundles, not what the host has. grid must read it at runtime
  (`NCCL_DEBUG=VERSION`) before relying on any feature above.

## 7. How much bandwidth does training need?

### Model

Bus bandwidth as reported by nccl-tests is defined so that it can be compared with hardware link speed:
`busbw = algbw x factor`, with factor `2(n-1)/n` for AllReduce and `(n-1)/n` for AllGather and ReduceScatter,
and `algbw = size / time` ([nccl-tests PERFORMANCE.md](https://raw.githubusercontent.com/NVIDIA/nccl-tests/master/doc/PERFORMANCE.md)).
Inverting it gives per-step network time (my derivation, for large `n`):

- **DDP**: one AllReduce over all gradients, `S = 4 bytes x params` (fp32). Time ~ `2S / busbw`.
- **FSDP (full shard)**: AllGather parameters for forward, AllGather again for backward, ReduceScatter
  gradients. Time ~ `3S / busbw`, with `S` in the parameter dtype.
- **Tensor parallelism**: `8*b*s*h*(t-1)/t` elements per layer per device per microbatch, from four blocking
  all-reduces per layer; **pipeline parallelism**: `b*s*h` per stage boundary per microbatch
  ([Megatron-LM, section 3.2](https://ar5iv.labs.arxiv.org/html/2104.04473)).

Check against published data: Qwen2.5-7B in bf16 is about 14 GB; measured socket busbw was 3.19 GB/s;
the model predicts `3 x 14 / 3.19 = 13.2 s`; the measured Ethernet overhead was 13.5 s per step
([Azure study](https://jingchaozhang.github.io/IB-vs-Ethernet-Fine-Tuning-at-Scale)).

### What NCCL actually achieves per network type

| Network | Measured NCCL busbw | Source |
|---|---|---|
| 8x400 Gb/s IB, 2 nodes x 8 H200 | 462 GB/s average (512 MB–8 GB messages) | [SkyPilot CoreWeave example](https://docs.skypilot.ai/en/latest/examples/performance/coreweave_infiniband.html) |
| 8x400 Gb/s IB, Azure ND H100 v5 | about 392 GB/s | Azure study |
| Same CoreWeave nodes without IB | 3.75 GB/s | SkyPilot example |
| Azure 80 Gb/s front-end NIC, `NCCL_NET=Socket` | 3.19 GB/s (about 25.5 Gb/s) | Azure study |
| AWS p3dn 100 Gb/s, TCP, NCCL 2.4.x patched | 6.39 GB/s; iperf3 showed 92 Gb/s | [NCCL issue #209](https://github.com/NVIDIA/nccl/issues/209) |
| GCP 32 Gb/s, TCP, 2 nodes x 1 GPU, `NTHREADS=8` | about 3.9 GB/s (30.6–31.4 Gb/s), near line rate | NCCL issue #209 |
| GCP 32 Gb/s, TCP, 2 nodes x 8 GPUs | about 1.8 GB/s (14.5–14.9 Gb/s) | NCCL issue #209 |

So the socket path can approach line rate up to roughly 25–30 Gb/s with tuning, and tops out around 25–50 Gb/s
on faster NICs. An older study of Horovod over TCP found the same shape: at most 32 Gb/s used on a 100 Gb/s
link, with CPU only 14–25% busy, and blamed the transport implementation
([Zhang et al. 2020](https://arxiv.org/pdf/2006.10103)).

### Per-step sync time (derived from the model above)

Assumed effective busbw: 10GbE 1.1 GB/s, 25GbE 2.8 GB/s, "100GbE TCP" 4 GB/s (measured range 3–6),
1x400 Gb/s IB 45 GB/s, 8x400 Gb/s IB 400 GB/s.

| Workload | Bytes `S` | 10GbE TCP | 25GbE TCP | 100GbE TCP | 1x400G IB | 8x400G IB |
|---|---|---|---|---|---|---|
| DDP, 1B params fp32 | 4 GB | 7.3 s | 2.9 s | 2.0 s | 0.18 s | 0.02 s |
| DDP, 7B params fp32 | 28 GB | 51 s | 20 s | 14 s | 1.2 s | 0.14 s |
| FSDP, 7B bf16 | 14 GB | 38 s | 15 s | 10.5 s | 0.93 s | 0.11 s |
| FSDP, 70B bf16 | 140 GB | 382 s | 150 s | 105 s | 9.3 s | 1.05 s |

Compute time per step for these models on an 8-GPU node is of the order of 0.5 s (7B, Azure study: 502 ms) to
a few seconds, so:

- 8x400 Gb/s RDMA keeps sync at a few percent to tens of percent of step time (Azure: 96% efficiency for 7B at
  8 nodes, 83% for 72B).
- A single 400 Gb/s rail per node is workable for models up to roughly 10B parameters.
- Any TCP Ethernet tier makes per-step synchronous sync of a multi-billion-parameter model the dominant cost.
  Published results: 26–28x slower for 7B, 38–45x for 72B, 56–57x for Mixtral 8x7B (MoE communicates all 46.7B
  parameters for about 12.9B active) on Azure; 39.8 s vs 4.4 s per step for Gemma 3 12B on 10GbE vs IB on Nebius
  ([Nebius study](https://maknee.github.io/blog/2025/Network-And-Storage-Training-Skypilot)).
- Ethernet is not hopeless when communication overlaps compute and the model is modest: IBM and PyTorch
  reported 95% scaling efficiency for T5-3B and 80–90% for T5-11B on up to 512 A100s over Ethernet without
  InfiniBand, after fixing FSDP's communication/computation overlap
  ([PyTorch blog](https://pytorch.org/blog/scaling-pytorch-fsdp-for-training-foundation-models-on-ibm-cloud/));
  peak utilisation was reported as 32 GB/s by HPCwire, so that was still a very fast Ethernet.

### Tensor parallelism

Example (my arithmetic): hidden 8192, sequence 4096, microbatch 1, bf16, `t = 8`, 80 layers.
`b*s*h` = 33.5M elements = 67 MB. Per layer per device: `8 x 67 MB x 7/8` = 470 MB. Per microbatch per device:
37.6 GB, spread over 320 blocking all-reduces. At 1.1 GB/s that is 34 s per microbatch; on NVLink it is tens of
milliseconds. Pipeline parallelism at the same shape sends 67 MB per boundary per direction: 60 ms on 10GbE.

Rule: **TP only inside an NVLink domain; DP/FSDP only inside an RDMA tier; PP (and local-SGD style outer loops)
are the only axes that tolerate TCP-class links.** This matches Megatron-LM's guidance that tensor parallelism
should be used up to the number of GPUs in a server and pipeline parallelism across servers. Rank-to-group
mapping is in `03-rank-configuration.md`.

## 8. Container and cloud realities

### Reachability

- NCCL needs **direct L3 reachability between all ranks on arbitrary TCP ports**. Each rank listens on kernel-
  assigned ephemeral ports and advertises its own interface address. Vast.ai's guide states it plainly: "Overlay
  communication is direct on all ports, which NCCL requires"
  ([Vast docs](https://docs.vast.ai/multi-node-training-using-torch-nccl)).
- The only supported way to restrict ports is the kernel sysctl `net.ipv4.ip_local_port_range` (NVIDIA's example:
  50000–51000) plus matching firewall rules
  ([troubleshooting doc](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_sources/troubleshooting/networking_troubleshooting.rst.txt)).
  There is no NCCL env var for a port range. In an unprivileged container that sysctl may not be writable
  (unverified per provider).
- Pods behind NAT with a handful of mapped TCP ports cannot run NCCL directly: the advertised address is the
  private one and the listening port is not the mapped one. They need an overlay that gives each pod a routable
  address.

### What specific providers give you (network view only; products in `07-provider-landscape.md`)

| Provider / product | Network between nodes | Notes |
|---|---|---|
| RunPod Instant Clusters | Up to 8 high-bandwidth interfaces `ens1`–`ens8` on an overlay with static `NODE_ADDR`; `eth0` is management/internet only. RDMA where hardware supports it: RunPod adds `/dev/infiniband/uverbs*`, removes the memlock limit, sets `NCCL_IB_HCA` and `NCCL_IB_DISABLE=0`, and `NCCL_IB_GID_INDEX` on RoCE. Otherwise NCCL falls back to TCP over `ens*` | `NCCL_SOCKET_IFNAME=ens1` is required; using the `172.x` `eth0` addresses causes timeouts. "All ports are available." ([RunPod docs](https://docs.runpod.io/instant-clusters/configuration.md)). Marketing quotes 800–3200 Gb/s east-west depending on configuration ([product page](https://www.runpod.io/product/instant-clusters)) |
| RunPod ordinary pods, global networking | Private IP and `POD_ID.runpod.internal` DNS, **100 Mb/s** between pods, 17 datacentres, NVIDIA pods only | [RunPod docs](https://docs.runpod.io/pods/networking.md). Useless for gradient sync; adequate for control plane |
| RunPod ordinary pods, public | HTTP proxy via Cloudflare (100 s limit, max 10 ports) or TCP mapped to a public IP with a random external port that changes on reset; symmetric mapping by requesting ports above 70000 (`RUNPOD_TCP_PORT_70000`). **"Pods do not support UDP."** | [RunPod docs](https://docs.runpod.io/pods/configuration/expose-ports.md) |
| Vast.ai | Default: instances on different machines are on separate bridges behind NAT. `vastai create overlay CLUSTER_ID NAME` creates a private LAN for instances in the same physical cluster (`eth0` overlay, `eth1` internet). Docker templates only. RoCE/IB only by separate cluster request (A100/H100/H200) | [Vast docs](https://docs.vast.ai/multi-node-training-using-torch-nccl). No bandwidth figures published |
| Lambda 1-Click Clusters | Quantum-2 400 Gb/s IB, non-blocking, rail-optimised, up to 3200 Gb/s per node; up to 200 Gb/s in-band Ethernet | [Lambda docs](https://docs.lambdalabs.com/1-click-clusters/getting-started) |

### RDMA inside containers

What a container needs to use verbs (from the RunPod behaviour above, the NCCL troubleshooting doc and the
sources below):

- Device nodes `/dev/infiniband/uverbsN` (and `rdma_cm`) visible in the container.
- Unlimited locked memory. Without it NCCL prints `Call to ibv_create_qp failed` or `Call to ibv_reg_mr failed`
  (NCCL troubleshooting doc). In rootless setups the container cannot raise the limit itself; the runtime must
  ([Alibaba eRDMA docs](https://help.aliyun.com/en/ecs/user-guide/configure-erdma-in-a-docker-container) use
  `--ulimit memlock=-1`). Kubernetes recipes add the `IPC_LOCK` capability and request a device-plugin resource
  such as `rdma/hca` or `rdma/ib`.
- For RoCE, the RDMA subsystem's network-namespace mode matters. In the default `shared` mode
  (`rdma system set netns shared`) the device is visible to all namespaces ([gVisor RDMA guide](https://gvisor.dev/docs/user_guide/rdma/)),
  but RoCE GIDs derive from the IP addresses of the underlying netdev, so a pod on a veth/bridge network that
  does not own that netdev has no usable RoCE v2 GID; the usual answers are host networking, macvlan/ipvlan on
  the RDMA NIC, or an SR-IOV VF per pod (partly unverified; see
  [Mellanox RoCE containers talk](https://openfabrics.org/images/2018workshop/presentations/115_PPandit_RoCEContainers.pdf)
  and the [network operator docs](https://mellanox.github.io/network-operator-docs/quick-start/macvlan-rdma-shared.html)).
  Native InfiniBand addresses by LID and does not have this problem.
- GDR additionally needs the host kernel to support dma-buf registration or to have `nv_peer_mem` loaded; on
  kernels 5.12+ the dma-buf path removes the need for the peer-memory module
  ([NVIDIA Dynamo EFA guide](https://docs.nvidia.com/dynamo/latest/kubernetes/installation/rdma-setup/efa-on-aws)).
  None of this is fixable from inside an unprivileged pod: grid can only detect it.

### MTU

- NVIDIA's NCCL networking doc does not discuss MTU. Datacentre fabrics typically run jumbo frames; Google
  specifies MTU 8896 for the gVNIC host networks on A3 Ultra / A4
  ([GCP docs](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-vm)).
- Every encapsulation layer subtracts from the inner MTU (VXLAN about 50 bytes, WireGuard 60–80 bytes, hence its
  common 1420 default on a 1500 underlay; standard figures, not re-verified in this sweep). A mismatch shows up
  as connections that establish and then hang on the first large message when path-MTU discovery is blocked.
  Preflight should send a do-not-fragment ping at the expected size: `ping -M do -s <mtu-28> <peer>`.

### Encrypted meshes: WireGuard, Tailscale, Nebula

| Setup | Single-stream TCP throughput | Source |
|---|---|---|
| No tunnel, 10 Gb/s baseline | about 9.5 Gb/s | [Tailscale 2022](https://tailscale.com/blog/throughput-improvements) |
| Kernel WireGuard, i5-12400, 25 Gb/s NICs | 11.8 Gb/s | [Tailscale "more throughput"](https://tailscale.com/blog/more-throughput) |
| Tailscale wireguard-go with UDP GSO/GRO, same hosts | 13.0 Gb/s | same |
| Kernel WireGuard, AWS c6i.8xlarge | 2.67 Gb/s | same |
| Tailscale wireguard-go with GSO/GRO, AWS c6i.8xlarge | 7.32 Gb/s | same |

- UDP GSO needs Linux 4.18+, UDP GRO 5.0+; Tailscale's docs say version 1.54+ on kernel 6.2+ enables the UDP
  offload path ([Tailscale performance docs](https://tailscale.com/docs/reference/best-practices/performance.md)).
- WireGuard encrypts per flow on a limited set of cores; research on multi-tunnel designs reports near-linear
  scaling with cores and up to 18x a single tunnel ([Netdev 0x18 talk](https://netdevconf.info/0x18/sessions/talk/achieving-linear-cpu-scaling-in-wireguard-with-an-efficient-multi-tunnel-architecture.html)).
  I found no published kernel-WireGuard result above about 13 Gb/s per tunnel and no Nebula benchmark at 25 Gb/s+.
- So an encrypted mesh caps the usable link at roughly **5–13 Gb/s per tunnel**, which is the same order as
  NCCL's own socket ceiling on a 10–25GbE underlay. On such links the tunnel is not the first bottleneck; on a
  100GbE underlay it is.
- Tunnels are UDP. On RunPod ordinary pods (no UDP) a mesh can only work through a TCP relay (for Tailscale,
  DERP over HTTPS), which is far slower, and without `/dev/net/tun` only in userspace-networking mode
  (inference from the RunPod UDP restriction; unverified in practice).
- A tunnel never carries RDMA. Meshes are for the TCP tier and the control plane only.
- NCCL 2.32's `ncclSetEncryption` offers TLS for NCCL's own socket traffic without a tunnel, but needs an
  OpenSSL 3 build of NCCL and framework support to call it.

### Cross-region and cross-provider links

Per-step synchronous collectives across regions are not realistic at LLM sizes: the table in section 7 already
shows tens of seconds per step at 10 Gb/s. NVIDIA's cross-DC design assumes dedicated inter-DC capacity in the
hundreds of Gb/s (their worked example needs 400 Gb/s with scattering, 1.6 Tb/s without) and recommends
measuring the pairwise inter-DC connection before tuning `NCCL_SCATTER_XDC`, channel counts,
`NCCL_IB_QPS_PER_CONNECTION` (IB/RoCE) or `NCCL_NSOCKS_PERTHREAD`/`NCCL_SOCKET_NTHREADS` (TCP)
([cross-DC deep dive](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness)).
For marketplace capacity spread over regions or providers, the workable pattern is one synchronous island per
RDMA fabric, joined by an infrequent-sync algorithm (`08-heterogeneous-distributed-training.md`).

## 9. Measuring a network quickly

| Step | Tool and command | What it tells you | Time |
|---|---|---|---|
| 1. Inventory | `ip -br addr`, `cat /sys/class/net/*/speed`, `ls /dev/infiniband`, `ibstat` / `ibv_devinfo`, `rdma link`, `ulimit -l` | Interfaces, nominal speed, RDMA devices, link layer (InfiniBand vs Ethernet), port state must be `Active`/`LinkUp`, rate not downshifted, memlock | seconds |
| 2. Reachability and MTU | TCP connect on an arbitrary high port in both directions; `ping -M do -s <size>` | Whether NCCL's "all ports, direct" requirement holds; path MTU | seconds |
| 3. Intra-node | [`nvbandwidth`](https://github.com/NVIDIA/nvbandwidth) `-t device_to_device_memcpy_read_ce` (add `-j` for JSON) | GPU-to-GPU matrix in GB/s: NVLink vs PCIe, broken links (see `06-gpu-hardware-nuances.md`) | about 1 min |
| 4. TCP | `iperf3 -c <peer> -P 8 --bidir -t 10` (iperf3 >= 3.16 is multi-threaded; older versions run all streams on one core, per [ESnet](https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/multi-stream-iperf3/)) | Achievable TCP rate. Run bidirectionally: NVIDIA devs recommend this to match NCCL's traffic pattern ([issue #209](https://github.com/NVIDIA/nccl/issues/209)) | 10–20 s per pair |
| 5. RDMA | `ib_write_bw -d <dev> -a` server/client, with `--use_cuda=<gpu>` (optionally `--use_cuda_dmabuf`) to test GDR; `ib_write_lat`; `rping` for RoCE connectivity | Raw verbs bandwidth per rail, whether GPU memory registration works | 10–20 s per rail |
| 6. NCCL end to end | [`nccl-tests`](https://github.com/NVIDIA/nccl-tests): `mpirun -np <N*g> -N <g> ./build/all_reduce_perf -b 512M -e 8G -f 2 -g 1` with `NCCL_DEBUG=INFO` | The only number that matters: busbw at large sizes, plus the log line that says which net was selected | 1–2 min |
| 7. Counters during 5/6 | `rdma statistic` (`rnr_nak_retry_err`, `packet_seq_err`, `local_ack_timeout_err`), `perfquery -x <lid>` (`SymbolErrorCounter`, `LinkDownedCounter`), `ethtool -S` for PFC/ECN/drops | Flaky cables, congestion, lossy RoCE | free |

The IB and RoCE checks in rows 1, 5 and 7 are the ones NVIDIA lists in its
[networking troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_sources/troubleshooting/networking_troubleshooting.rst.txt).
nccl-tests needs MPI for multi-node runs (`make MPI=1`), so the image must carry an MPI and passwordless SSH, or
grid needs its own small harness that calls NCCL through PyTorch (`torch.distributed` all_reduce on a large
tensor gives the same number with no MPI).

A third-party rule of thumb says `all_reduce_perf` below about 50 GB/s on H100 nodes means traffic is on
Ethernet rather than the RDMA fabric ([markaicode](https://www.markaicode.com/vs/provision-gpu-clusters-on-runpod-vs-lambda-labs-in-15-minutes/),
unverified). The measured values in section 7 (3–6 GB/s vs 390–480 GB/s) make the two cases easy to tell apart.

### From measurements to a placement decision

1. **Build a pairwise matrix** of measured busbw (or iperf3/ib_write_bw as proxies). Full N^2 is unnecessary:
   test each node against 2–3 others plus a rotating ring; nodes in the same provider cluster normally share a
   tier.
2. **Classify links into tiers.** Suggested thresholds, derived from section 7:

   | Tier | Measured NCCL busbw | Typical cause | Allowed synchronous traffic |
   |---|---|---|---|
   | A | >= 100 GB/s | Multi-rail RDMA with GDR | Per-step DDP/FSDP for any model size; TP still intra-node |
   | B | 20–100 GB/s | Single-rail RDMA, or RDMA without GDR | Per-step DDP/FSDP up to roughly 10B parameters; PP |
   | C | 1–6 GB/s | TCP on 10–100GbE | PP boundaries; per-step sync only for small models (< 1B); otherwise infrequent sync |
   | D | < 1 GB/s | WAN, relayed mesh, 100 Mb/s private nets | Control plane, checkpoints, infrequent-sync algorithms only |

3. **Cluster nodes into islands**: connected components where every internal link is tier A or B. A node whose
   links to an island are a tier below the rest is a straggler; exclude it or demote it, because a ring runs at
   the slowest hop.
4. **Map parallelism onto the hierarchy** (see `03-rank-configuration.md`): TP within the NVLink domain reported
   by nvbandwidth; DP/FSDP within an island; PP or local-SGD/DiLoCo-style outer loops across islands.
5. **Estimate before launching**: with model byte size `S` and the island's minimum busbw `B`, predicted sync
   time is `2S/B` (DDP) or `3S/B` (FSDP). If that exceeds a set fraction of measured compute time per step, grid
   should warn and propose a different layout rather than start a job that will run at 3% utilisation.
6. **Pin the configuration** so NCCL cannot choose differently from what was measured: `NCCL_SOCKET_IFNAME`
   (exact match with `=`), `NCCL_IB_HCA`, `NCCL_NET`, and `NCCL_IGNORE_NET_MISMATCH=0`.

## Implications for grid

1. **Network tier is a first-class property of a node pair**, discovered by measurement in preflight, stored,
   and used by the planner. Provider labels ("InfiniBand cluster") are hints only; the RDMA path can be silently
   absent.
2. **Fail loudly on silent TCP fallback.** After launch, parse `NCCL_DEBUG=INFO` output (or `ATTN` on 2.32+) and
   confirm the selected net and device list match the plan. Set `NCCL_NET=IB` where RDMA is expected so a missing
   device is an init error rather than a 30x slowdown. On AWS, check that the OFI plugin loaded.
3. **Generate NCCL configuration per provider** rather than exposing raw env vars: for example RunPod Instant
   Clusters get `NCCL_SOCKET_IFNAME=ens1` and keep the provider-set `NCCL_IB_HCA`; Vast overlays get
   `NCCL_SOCKET_IFNAME=eth0`. Write it to `NCCL_CONF_FILE` so every launcher path inherits it.
4. **TCP-tier defaults.** On generic hosts set `NCCL_SOCKET_NTHREADS`/`NCCL_NSOCKS_PERTHREAD` above 1 (start at
   4 x 4, product <= 64) and sweep quickly in preflight; the defaults of 1 x 1 are tuned only for AWS and GCP.
5. **Refuse or restructure, do not just run.** Use the `2S/B` and `3S/B` estimates to reject per-step
   synchronous training on tier C/D links for large models and suggest infrequent-sync methods instead.
6. **Do not rely on overlays for data-plane traffic.** WireGuard-class meshes cap at roughly 5–13 Gb/s per
   tunnel and cannot exist on providers without UDP (RunPod pods). Use the provider's native cluster network for
   NCCL; reserve a mesh (or plain TLS over TCP) for the control plane, where 100 Mb/s is enough.
7. **Check container prerequisites for RDMA** in preflight: `/dev/infiniband/uverbs*` present, `ulimit -l`
   unlimited, `ibstat` Active, GDR registration works (`ib_write_bw --use_cuda`). These cannot be repaired from
   inside a pod, so the outcome is a tier downgrade or node rejection.
8. **Detect the bundled NCCL version** and enable resilience features conditionally:
   `NCCL_IB_RESILIENCY_PORT_FAILOVER=1` (>= 2.29.7), `NCCL_IB_RESILIENCY_PORT_RECOVERY=1` (>= 2.30.3),
   `NCCL_IB_EVENT_BASED_LB=1` (>= 2.31). Poll RAS (`NCCL_RAS_ADDR`, JSON since 2.28) from the node agent as a
   hang detector; give each job on a shared node its own RAS port.
9. **Homogeneous islands.** Keep each NCCL communicator inside one tier with equal NIC counts, and set
   `NCCL_IGNORE_NET_MISMATCH=0` so accidental heterogeneity is caught at init.
10. **Keep a continuous cheap signal** after preflight: per-step collective time from the framework profiler
    plus RDMA/ethtool error counters, so that a link that degrades mid-run is identified as the straggler
    (hand-off to `04-failure-taxonomy-recovery.md`).

## Open questions

- **RunPod Instant Cluster reality**: actual `all_reduce_perf` busbw per GPU type, how often RDMA is present
  versus TCP-only `ens*`, MTU on the overlay, and whether the 100 Mb/s global-networking figure still holds. The
  one community thread on low NCCL bandwidth across RunPod pods could not be fetched (HTTP 429). Needs hands-on
  measurement.
- **Which NCCL version current PyTorch wheels bundle**, and whether PyTorch exposes `ncclCommShrink`,
  `ncclSetEncryption` or RAS in `torch.distributed`. Decides which features in section 6 are usable.
- **NCCL socket transport on modern hardware**: all public TCP numbers found are either old (2019–2020) or
  incidental. A controlled sweep of `NCCL_SOCKET_NTHREADS` x `NCCL_NSOCKS_PERTHREAD` on 25/100GbE with current
  NCCL would firm up the tier C numbers. Also unverified: the link-speed value NCCL assumes for virtual
  interfaces, and the exact min-across-ranks topology negotiation.
- **RoCE in pods without host networking**: exact GID/netns behaviour per provider, and whether
  `rdma system set netns exclusive` setups appear on marketplaces.
- **Cross-DC NCCL**: whether `NCCL_ALLNET_ENABLE` / `NCCL_SCATTER_XDC` are in a tagged release, and whether a
  mixed IB + TCP communicator works in practice.
- **aws-ofi-nccl details**: current release, `OFI_NCCL_*` variables and protocol modes were not retrieved.
- **Encrypted transport at speed**: no data found for kernel WireGuard above about 13 Gb/s, for Nebula at all,
  or for the cost of NCCL's own TLS mode.
- **Other providers' intra-cluster networks** (CoreWeave, Crusoe, Nebius, Vast overlay bandwidth) at the level
  of interface names and required env vars; overlaps with `07-provider-landscape.md`.
- **Heterogeneity-aware collectives** (OptCC, R2CCL and similar): whether any is usable as an NCCL plugin or
  tuner today.
