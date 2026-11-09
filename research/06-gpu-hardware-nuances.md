# 06 - GPU hardware nuances for a training orchestrator

Research date: 2026-11-09. Scope: what differs between the accelerators and server forms that can
actually be rented in late 2026, which of those differences a training orchestrator must model, what
mixing them in one job costs, and how to fingerprint a node into a capability record. Networking
between nodes is in `05-networking-interconnects.md`; failure handling (Xid, ECC, fall-off-the-bus)
is in `04-failure-taxonomy-recovery.md`; heterogeneity-tolerant *algorithms* are in
`08-heterogeneous-distributed-training.md`; preflight/burn-in is in `09-fast-bootstrap-preflight.md`;
provider APIs and SKUs are in `07-provider-landscape.md`.

Claims marked **(unverified)** were not confirmed against a primary source during this sweep (most
are well-known datasheet values recalled from memory, and should be re-checked before being
hard-coded).

## Summary

- "GPU model" is not a sufficient scheduling key. The same silicon ships in forms that behave
  differently for training: H100 **SXM** (HGX baseboard, 8 GPUs behind NVSwitch, 900 GB/s NVLink per
  GPU) versus H100 **PCIe** (no NVSwitch; GPU-to-GPU traffic over PCIe unless a 2-way bridge is
  fitted) versus H100 **NVL** (94 GB, bridged pairs). RunPod lists all three as separate SKUs at
  $3.99, $2.89 and $3.19 per GPU-hour ([RunPod pricing](https://www.runpod.io/pricing), read
  2026-11-09).
- The 2026 rentable ladder is roughly: consumer RTX 3090/4090/5090 (24-32 GB, no NVLink, no
  supported P2P) -> workstation/"inference" datacenter cards A6000/A40/L40/L40S/RTX 6000 Ada/RTX PRO
  6000 Blackwell (48-96 GB, PCIe only) -> A100 (40/80 GB, NVLink 3) -> H100/H200 (80/141 GB, NVLink 4)
  -> B200/B300 (180/288 GB, NVLink 5) -> GB200/GB300 NVL72 racks (72 GPUs in one NVLink domain, Arm
  hosts). Rubin (288 GB HBM4, NVLink 6) is in production and shipping to named customers but had
  no published hourly price found in this sweep.
- Prices **rose** through 2026 after years of decline: H100 SXM went from about $2.99 to $3.99 at
  RunPod and Lambda, Nebius raised all SKUs on 2026-10-01
  ([Neurohive, 2026-11-05](https://neurohive.io/en/guides/cloud-gpu-pricing/)). Heterogeneity
  tolerance is therefore an economic feature, not just a robustness one.
- Software compatibility is a hard constraint, not a tuning matter: Blackwell needs CUDA >= 12.8
  and a 570+ driver; CUDA 13 needs driver >= 580 and drops everything below Turing; PyTorch 2.15
  stops shipping CUDA 12 wheels altogether
  ([PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
  The host driver is the provider's, not ours, so the image must be chosen to fit the node.
- In a synchronous data-parallel job the slowest rank sets step time and the smallest-memory rank
  sets the per-rank batch, unless the launcher assigns per-rank batch sizes and weights gradients
  by sample count. Cross-architecture runs are not bit-reproducible; PyTorch says so explicitly.
- A node should be fingerprinted in a few seconds (nvidia-smi queries, `nvidia-smi topo -m`,
  `nvidia-smi nvlink -s`, `lscpu`, `lspci`, cgroup limits) into a capability record, then grouped
  into *classes*; the default policy should be one class per synchronous group.

## 1. Accelerator catalogue

### 1.1 NVIDIA datacenter parts

| Part | Arch / CC | VRAM | Intra-node GPU link | Typical node | GPU power | Notes |
|---|---|---|---|---|---|---|
| A100 40/80 (SXM4) | Ampere / 8.0 | 40 GB HBM2, 80 GB HBM2e | NVLink 3, 600 GB/s per GPU via NVSwitch (unverified exact; "600 GB/s (Gen3)" per [Cyfuture KB](https://cyfuture.cloud/kb/gpu/how-does-nvlink-differ-between-h100-a100-and-h200)) | 8x in HGX A100 / DGX A100, 2x EPYC, PCIe Gen4 | 400 W (unverified) | MIG up to 7 instances. No FP8. |
| A100 PCIe | Ampere / 8.0 | 40/80 GB | PCIe Gen4; optional bridge pairs (unverified) | 1-8x in generic 4U | 250-300 W (unverified) | RunPod lists "A100 PCIe" and "A100 SXM" separately. |
| H100 SXM5 | Hopper / 9.0 | 80 GB HBM3, 3.35 TB/s | NVLink 4 through 3rd-gen NVSwitch: 900 GB/s GPU-to-GPU, 7.2 TB/s aggregate per baseboard | 8x HGX H100 | up to 700 W | Figures from [NVIDIA HGX enterprise RA](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html). |
| H100 PCIe | Hopper / 9.0 | 80 GB | PCIe Gen5 x16 only (about 128 GB/s nominal, vendor figure) | 1-8x in generic servers | 350 W default (third-party figure, [io.net](https://io.net/p/h100-sxm-vs-pcie-which-gpu-configuration-for-ai-training)) | Lower power cap means lower per-GPU throughput than SXM as well as slower links. |
| H100 NVL | Hopper / 9.0 | 94 GB | 2-way NVLink bridge, 600 GB/s per pair (third-party, see [Spheron](https://www.spheron.network/blog/h100-nvl-vs-sxm5-vs-pcie-form-factor-guide/)) | 2/4/8x in pairs | about 400 W (unverified) | NVLink islands of 2: "has NVLink" is not "all-to-all NVLink". |
| H200 SXM5 | Hopper / 9.0 | 141 GB HBM3e, 4.8 TB/s | same as H100 SXM (900 GB/s, NVSwitch) | 8x HGX H200 | 700 W ([power-capping study](https://arxiv.org/html/2603.16164v2) sweeps to 700 W) | Same compute as H100 (32 PFLOPS per baseboard); memory is the difference. |
| H200 NVL | Hopper / 9.0 | 141 GB | 2- or 4-way bridges, 900 GB/s per NVIDIA product brief (as reported in search; retailer lists a "1.8 TB/s" 4-way bridge, conflict unresolved) | 2-8x PCIe | (unverified) | |
| B200 | Blackwell / 10.0 | 180 GB HBM3e, up to 8 TB/s | NVLink 5 through 4th-gen NVSwitch: 1800 GB/s GPU-to-GPU, 14.4 TB/s aggregate | 8x HGX B200 | configurable up to 1 kW | NVIDIA RA above. Some vendors quote 192 GB; 180 GB is NVIDIA's number. |
| B300 (Blackwell Ultra) | Blackwell / 10.0 | 288 GB HBM3e | NVLink 5, 1800 GB/s (third-party) | 8x HGX B300, 1:1 ConnectX-8 SuperNIC up to 800 Gb/s per GPU ([FiberMall overview](https://www.fibermall.com/blog/nvidia-hgx-b300-overview.htm)) | 1.2-1.4 kW, sources disagree (unverified) | MIG table lists 288 GB; one vendor cites "270 GB usable" (unverified). |
| GB200 NVL72 | Blackwell / 10.0, **aarch64 host** | 13.4 TB HBM3e per rack (about 186 GB/GPU) | 130 TB/s NVLink across all 72 GPUs (3.6 TB/s per 2-GPU superchip) | 18 compute trays x (2 Grace + 4 GPU); 36 Grace CPUs, 2,592 Neoverse V2 cores, 17 TB LPDDR5X | about 120 kW/rack (unverified), liquid | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/). FP8 720 PFLOPS sparse, BF16 360 PFLOPS sparse per rack. |
| GB300 NVL72 | Blackwell Ultra, aarch64 host | 72 x 288 GB, about 20 TB pooled | 130 TB/s NVLink 5 | as GB200 | 132-142 kW nominal, about 155 kW peak, vendor-dependent ([Spheron summary of HPE/Lenovo/Vertiv figures](https://www.spheron.network/blog/gb300-nvl72-power-consumption-and-liquid-cooling-requirement/)) | Rented as whole racks or large slices by neoclouds; not a marketplace SKU. |
| Rubin (VR200) | Rubin | 288 GB HBM4, 22 TB/s | NVLink 6, about 3.5-3.6 TB/s per GPU | Vera Rubin NVL72 (also branded NVL144), 88-core Vera CPUs, ConnectX-9 | not disclosed | [Glenn Lockwood's notes](https://glennklockwood.com/garden/processors/r200). See 1.4. |

"CC" is CUDA compute capability, which determines which compiled kernels can run (section 3.1).

### 1.2 NVIDIA workstation, "universal" and consumer parts

| Part | Arch / CC | VRAM | NVLink | GPU power | ECC | Notes |
|---|---|---|---|---|---|---|
| RTX A6000 / A40 | Ampere / 8.6 | 48 GB GDDR6 | 2-way bridge on A6000 (unverified) | 300 W (unverified) | yes (unverified) | Cheapest 48 GB: $0.33-0.59/h on RunPod. |
| L40 / L40S | Ada / 8.9 | 48 GB GDDR6 | none | 350 W (L40S, unverified) | yes (unverified) | Passive, PCIe Gen4. No MIG (absent from [MIG supported list](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)). FP8 tensor cores (Ada is listed for FP8 in [Transformer Engine docs](https://nvidia.github.io/TransformerEngine/index.html)). |
| RTX 6000 Ada | Ada / 8.9 | 48 GB | none | 300 W (unverified) | yes (unverified) | Blower workstation card; supported P2P over PCIe (passed `simpleP2P` in [Puget's test](https://www.pugetsystems.com/labs/hpc/problems-with-rtx4090-multigpu-and-amd-vs-intel-vs-rtx6000ada-or-rtx3090/)). |
| RTX PRO 6000 Blackwell | Blackwell / 12.0 | 96 GB | none | 600 W (unverified) | yes (unverified) | MIG up to 4 instances; RunPod sells "Pro 6000 MIG 48GB/24GB" slices. $2.49/h whole. |
| RTX 3090 | Ampere / 8.6 | 24 GB GDDR6X | 2-way bridge (last GeForce with NVLink) (unverified) | 350 W (unverified) | no | Puget measured 16.5 GB/s GPU-GPU without P2P. |
| RTX 4090 | Ada / 8.9 | 24 GB GDDR6X | none | 450 W (unverified) | driver toggle existed ([Notebookcheck](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-departs-from-RTX-3090-Ti-and-RTX-4090-flagship-tradition-drops-VRAM-ECC-for-pro-workloads.958141.0.html)) | P2P disabled by driver since 525.105.17 (section 2.2). |
| RTX 5090 | Blackwell / 12.0 | 32 GB GDDR7 | none | 575 W (unverified) | no toggle; GDDR7 on-die ECC only | Needs driver 570+ and CUDA 12.8+ (sm_120). |

### 1.3 AMD Instinct

| Part | Arch | VRAM | Power | Typical node | Software |
|---|---|---|---|---|---|
| MI300X | CDNA 3 | 192 GB HBM3, 5.3 TB/s | 750 W | 8x OAM on a universal baseboard, GPU-GPU over Infinity Fabric / PCIe (topology details unverified) | ROCm; RCCL replaces NCCL |
| MI325X | CDNA 3 (same compute die) | 256 GB HBM3e, about 6 TB/s | 1000 W (vendor listing) | drop-in on the MI300X 8-way baseboard | as above |
| MI355X | CDNA 4 | 288 GB HBM3e, 8 TB/s | 1400 W | 8-GPU node pooling 2,304 GB ([SemiAnalysis InferenceX](https://inferencex.semianalysis.com/chips/mi355x)) | ROCm 7.x, FP8 path |

Software state: ROCm's latest line is 7.14 (PyTorch 2.12, JAX 0.10). 7.14 "introduces significant
enhancements" to RCCL including FP8 communication on MI355X; 7.14.1 fixed a collective-performance
regression for distributed training on MI300X caused by InfiniBand fault-injection instrumentation
left active in 7.14.0 ([ROCm 7.14 blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html),
[release notes](https://rocm.docs.amd.com/en/docs-7.14.1/_sources/about/release-notes.md)). The ROCm
7.0 RC notes said multi-node RCCL would come "in a future release", i.e. multi-node training on AMD
matured only during 2025-26. In a power-capping study the MI300X reached 16,843 tokens/s per GPU on
LLM continued pre-training at 750 W versus 30,271 for H100 and 34,107 for H200 at 700 W, and caps
below 400 W were not reliably enforced ([arXiv 2603.16164](https://arxiv.org/html/2603.16164v2)).
MI355X rental prices found ranged from $2.59/h (Vultr, via an aggregator) to $7.20/h (CoreWeave,
March 2026 data); sources are inconsistent.

### 1.4 Is Rubin rentable?

NVIDIA said on 2026-07-21 that Vera Rubin had reached full production and that systems were
shipping to customers including OpenAI, CoreWeave, Google Cloud, Azure and Meta, with AWS, Google
Cloud, Microsoft, OCI, CoreWeave, Lambda, Nebius and Nscale named as first cloud deployers
([GCN, Aug 2026](https://gcn.com/?p=20421)); another outlet stressed that volume shipments were
still ahead ([The Next Web](https://thenextweb.com/news/nvidia-vera-rubin-full-production-customers)).
This sweep found **no published hourly price and no self-serve SKU** for Rubin. Treat it as "exists
in hyperscaler/neocloud reserved contracts, not on marketplaces". Design consequence: the capability
schema must not hard-code an enum of architectures; a node reporting an unknown model with 288 GB
and compute capability above 12.x should still be schedulable if the image supports it.

### 1.5 Rental prices (on-demand, per GPU-hour, early November 2026)

| GPU | RunPod Secure / Community | Elsewhere |
|---|---|---|
| RTX A5000 24 GB | $0.27 / $0.16 | |
| RTX 3090 | $0.50 / $0.22 | |
| RTX 4090 | $0.89 / $0.34 | Vast from about $0.30 |
| RTX 5090 | $1.19 / $0.69 | Vast from about $0.50 |
| RTX A6000 / A40 | $0.59 / $0.33-0.35 | |
| RTX 6000 Ada | $0.99 / $0.74 | |
| L40S | $1.09 / $0.79 | Nebius from $1.55, AWS g6e $1.86 |
| A100 80 GB PCIe / SXM | $1.79 / $1.19-1.39 | Lambda $2.79, AWS p4de $3.43 |
| RTX PRO 6000 96 GB | $2.49 / $1.69 | |
| H100 PCIe / NVL / SXM | $2.89 / $3.19 / $3.99 (community $1.99 / $2.59 / $2.69) | Vast $2.14, Lambda $3.99, Nebius $4.50, CoreWeave $6.16, AWS p5 $6.88, GCP $11.06 |
| H200 | $5.29 / $3.59 | Vast median $3.85, Crusoe $4.29, Nebius $5.40, CoreWeave $6.31, AWS $7.91 |
| B200 | $7.99 / $5.98 | Lambda $6.69, Nebius $8.50, AWS p6-b200 $14.24 |
| B300 | $8.99 / $6.94 | Nebius $9.50, AWS p6-b300 $17.80 |

Sources: [RunPod pricing page](https://www.runpod.io/pricing) (community figures come from the
page's structured data), [Neurohive](https://neurohive.io/en/guides/cloud-gpu-pricing/),
[Thunder Compute H200 tracker, 2026-11-03](https://www.thundercompute.com/blog/nvidia-h200-pricing).
RunPod raised H200 from $4.59 to $5.29 on 2026-10-09; Nebius raised H200 by 20% on 2026-09-29.
The same RunPod page shows the host shape bundled per GPU, which matters for section 2.4: H100 SXM
gets 20 vCPU / 125 GB RAM, B200 28 vCPU / 283 GB, but an RTX 4090 gets 6 vCPU / 41 GB and an RTX
5090 9 vCPU / 35 GB.

## 2. Nuances that change how a job must be laid out

### 2.1 SXM/HGX baseboards versus PCIe cards

An HGX baseboard is a fixed appliance: 8 SXM modules, NVSwitch chips giving every GPU a full-rate
NVLink path to every other, and NVIDIA's reference host around it: 2 sockets, at least 48 (56
recommended) physical cores per socket, at least 1.5 TB RAM, 8x PCIe Gen5 x16, one BlueField-3
SuperNIC per GPU at up to 400 Gb/s (minimum 8x200 or 4x400), a separate north-south DPU, and at
least 2 TB NVMe per socket for training nodes
([NVIDIA RA](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)).
Knowing "HGX H100" therefore predicts a great deal about the node.

A "PCIe" SKU predicts almost nothing: the cards may be 1 to 10 per chassis, on one root complex or
behind PLX switches, split across sockets, with any NIC. Vendor benchmarks claim large gaps for
communication-heavy training (io.net reports 80 h versus 118 h for a 70B fine-tune and 3% versus
18% of GPU time waiting on gradient sync; this is a provider's own unverified benchmark,
[io.net](https://io.net/p/h100-sxm-vs-pcie-which-gpu-configuration-for-ai-training)). The direction
is reliable even if the magnitude is not: tensor-parallel and pipeline-parallel groups (see
`03-rank-configuration.md`) should be placed inside NVLink domains; PCIe-only nodes are best used
as pure data-parallel replicas or single-GPU ranks.

NVL72 racks invert the usual hierarchy: the NVLink domain is 72 GPUs across 18 trays, so "node"
(tray, 4 GPUs) is smaller than "NVLink domain" (rack). NCCL's topology dump on such systems
includes every host in the NVLink domain
([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)). The hosts are
Arm (Grace), so every wheel and container must be `linux/arm64`.

### 2.2 P2P over NVLink versus PCIe, and the consumer-card trap

NCCL chooses the intra-node transport by GPU "distance". `NCCL_P2P_LEVEL` sets the maximum
distance at which direct P2P is used: `LOC` (never), `NVL` (NVLink only), `PIX` (same PCI switch),
`PXB` (through PCI switches), `PHB` (same NUMA node, through the CPU host bridge), `SYS` (across
NUMA nodes). `NCCL_P2P_DISABLE=1` turns P2P off; without P2P NCCL falls back to shared-memory
(`NCCL_SHM_DISABLE` controls that), i.e. GPU -> host RAM -> GPU copies
([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

GeForce cards are the special case. Puget Systems found that dual RTX 4090s on driver 525.85.05
*reported* peer access but returned wrong data: `simpleP2P` failed verification, DDP hung, and
minGPT died with `RuntimeError: probability tensor contains either inf, nan or element < 0`.
NVIDIA's fix in 525.105.17 was to disable P2P on GeForce; NVIDIA's stated intent is that P2P is
unavailable there. Measured copy bandwidth was about 54 GB/s with P2P versus 31 GB/s without on
the 4090 pair, and 16.5 GB/s on RTX 3090s without P2P; RTX 6000 Ada passed the P2P tests
([Puget Systems](https://www.pugetsystems.com/labs/hpc/problems-with-rtx4090-multigpu-and-amd-vs-intel-vs-rtx6000ada-or-rtx3090/)).
Community forks of the open kernel modules (tinygrad's, and later forks covering 30/40/50-series)
re-enable P2P, but that needs control of the host kernel module, which a marketplace tenant does
not have ([write-up](https://allanchan339.github.io/2025/05/21/4090-P2P)).

Consequences: on consumer multi-GPU pods, intra-node all-reduce goes through host memory, so it
competes with the dataloader for memory bandwidth and CPU; and a multi-GPU consumer pod deserves a
correctness check (a known-answer all-reduce), not only a bandwidth check.

### 2.3 NUMA, PCIe layout and CPU pinning

`nvidia-smi topo -m` prints the GPU/NIC connection matrix plus each GPU's CPU and NUMA affinity;
`-mp` shows PCI-only connectivity ignoring NVLink; `-p2p r|w|n|a|p` prints the pairwise P2P
capability matrix for read, write, NVLink, atomics or PCIe
([nvidia-smi man page](https://man.archlinux.org/man/nvidia-smi.1.en)). What to act on:

- **Rank-to-core binding.** NCCL by default uses the intersection of the inherited CPU affinity
  and the GPU's affinity, and keeps the inherited mask if the intersection is empty;
  `NCCL_IGNORE_CPU_AFFINITY=1` uses only the GPU's affinity. Either way it cannot escape a cpuset
  or cgroup limit (NCCL env docs). In a container that was handed cores on the wrong socket, no
  env var fixes the placement; grid can only detect it.
- **GPU-to-NIC pairing.** HGX nodes pair each GPU with a NIC under the same PCIe switch; a rank
  whose GPU sends through a NIC on the other socket pays a cross-NUMA hop. The topo matrix is how
  to discover the pairs (details in `05-networking-interconnects.md`).
- **Link training.** A GPU that negotiated PCIe Gen3 x8 instead of Gen5 x16 works but stages data
  slowly; `pcie.link.gen.current` / `pcie.link.width.current` versus their `.max` counterparts
  catch this cheaply.
- **Virtualised topology.** In VMs the guest sees a flattened PCI tree. NCCL loads
  `/var/run/nvidia-topologyd/virtualTopology.xml` if present and accepts `NCCL_TOPO_FILE`;
  `NCCL_TOPO_DUMP_FILE` writes what NCCL detected, which is a ready-made fingerprint artifact.

### 2.4 Dataloader CPU starvation

The CPU-per-GPU ratio varies by more than 3x across SKUs. NVIDIA's HGX reference implies at least
12 physical cores and about 190 GB RAM per GPU; RunPod's bundled shapes are 20-28 vCPU per GPU on
H100 SXM/B200 but 6 vCPU and 41 GB on an RTX 4090, and 8 vCPU on an A100 PCIe
([RunPod pricing](https://www.runpod.io/pricing)). A vCPU is usually a hyperthread, so 6 vCPU is
about 3 cores for the training process, NCCL threads, the dataloader workers and the checkpoint
writer. For LLM training on pre-tokenised data this is fine; with on-the-fly tokenisation,
augmentation or decompression it is not. Two further traps: `os.cpu_count()` inside a container
reports host cores rather than the cgroup quota (so default `num_workers` heuristics oversubscribe),
and host RAM below VRAM (5090: 35 GB RAM for 32 GB VRAM; H100 NVL: 94 GB for 94 GB) makes
CPU-side checkpoint assembly or pinned-memory staging liable to OOM. grid should read
`cpu.max` / `cpuset.cpus.effective` and `memory.max` from cgroup v2, not `/proc/cpuinfo`.

### 2.5 MIG and vGPU

MIG partitions a GPU into isolated instances: up to 7 on A100, H100, H200, B200, B300 and GB200;
up to 4 on A30 and RTX PRO 6000 Blackwell; not available on L40S, RTX 6000 Ada or GeForce
([MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)).
Providers now sell MIG slices as SKUs (RunPod "Pro 6000 MIG 48GB" at $1.29/h). For training this
matters in three ways: a slice looks like a GPU with an odd memory size and a `MIG-<uuid>` device
identifier; MIG devices do not support GPU-to-GPU P2P and a process is generally limited to one MIG
device (unverified in this sweep; from the MIG user guide's limitations), so multi-slice jobs are
effectively multi-node jobs; and a "whole" GPU left in MIG mode exposes less memory than expected.
The fingerprint should record `mig.mode.current` and refuse MIG slices for multi-GPU ranks by
default.

### 2.6 Precision and feature support

| Feature | Ampere (A100, A6000, 3090) | Ada (L40S, 6000 Ada, 4090) | Hopper (H100/H200) | Blackwell (B200/B300, 5090, PRO 6000) |
|---|---|---|---|---|
| TF32, FP16, BF16 tensor cores | yes | yes | yes | yes |
| FP8 tensor cores | no | yes | yes | yes |
| FP4 (NVFP4) | no | no | no | yes |
| MIG | A100/A30 only | no | yes | datacenter + PRO only |

BF16 autocast needs Ampere or newer; `torch.cuda.is_bf16_supported()` checks both the build and the
device, and on Volta the check can pass while math ops fail
([PyTorch forums](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629/8)). FP8 training
via NVIDIA Transformer Engine is available on "Hopper, Ada, and Blackwell" with FP16/BF16
optimisations on Ampere and later ([Transformer Engine docs](https://nvidia.github.io/TransformerEngine/index.html)).
So a job configured for FP8 cannot include A100, A6000 or 3090 ranks without those ranks running
a different numeric path, which is a correctness question, not only a speed one. On AMD, Transformer
Engine v2.12 is validated on MI300X/MI350X/MI355X per the ROCm 7.14 material.

### 2.7 ECC, memory errors and consumer cards

Datacenter and professional cards have ECC on GPU memory plus row remapping / page retirement,
exposed through `nvidia-smi --query-remapped-rows` and `--query-retired-pages`
([man page](https://man.archlinux.org/man/nvidia-smi.1.en)). GeForce does not offer equivalent
protection: the 3090 Ti and 4090 had a driver-level ECC toggle (off by default), the 5090 dropped
the toggle and relies on GDDR7's on-die ECC (single-bit correct, double-bit detect per JEDEC) and
link-level CRC with retry
([Notebookcheck](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-departs-from-RTX-3090-Ti-and-RTX-4090-flagship-tradition-drops-VRAM-ECC-for-pro-workloads.958141.0.html)).
There is no error counter the tenant can read on those cards. Combined with section 2.2, a
consumer rank is more likely to contribute a silently wrong gradient; mitigations (gradient-norm
outlier checks per rank, periodic cross-rank loss agreement on a fixed batch) belong in
`04-failure-taxonomy-recovery.md`.

### 2.8 Licence terms for consumer cards

The NVIDIA GeForce software licence (version dated 2025-02-25) states in section 2.8 that the
software "is not licensed for datacenter deployment"
([NVIDIA GeForce licence](https://www.nvidia.com/en-us/drivers/geforce-license/)). The clause dates
from late 2017, originally with a blockchain-processing exception, and "datacenter" is not defined
([DCD](https://www.datacenterdynamics.com/en/news/nvidia-updates-geforce-eula-to-prohibit-data-center-use/)).
The obligation sits with whoever installs the driver, i.e. the host, not the tenant; marketplace
"community" tiers are where such cards live. For grid this is a supply-risk and enterprise-policy
flag, not a technical block: expose a policy switch such as `allow_consumer_gpus` (default on for
individuals, off-able for organisations). This is not legal advice.

### 2.9 Power and thermals

- **Limits.** `nvidia-smi -pl <watts>` sets a per-GPU cap within the min/max reported by
  `nvidia-smi -q -d POWER`, requires root, and does not survive a reboot or driver reload
  ([man page](https://man.archlinux.org/man/nvidia-smi.1.en),
  [Netdata guide](https://www.netdata.cloud/guides/nvidia-gpu/nvidia-gpu-sw-power-cap-throttling/)).
  Tenants in containers usually cannot set it, but hosts do, so two "identical" 4090s may be capped
  differently. Record `power.limit` against `power.default_limit`.
- **Cost of a cap.** For LLM continued pre-training, H100 throughput fell about 8% at 500 W, 18% at
  400 W and 41% at 300 W relative to 700 W; H200 fell 17% at 500 W and 35% at 400 W; energy
  efficiency peaked at 400 W (H100) and 500 W (H200), and the authors conclude there is no
  universal optimum (percentages computed from Table 4 of
  [arXiv 2603.16164](https://arxiv.org/html/2603.16164v2)). A capped host is thus a 10-40% slower
  rank that reports the same model name.
- **Throttling.** Thermal or power throttling shows up as clocks stuck low; Puget saw GPU clocks
  pinned near 626 MHz with low power draw on a faulty platform. Consumer cards in dense chassis and
  5090s on 12V-2x6 connectors are the usual suspects (unverified generalisation). The fingerprint
  should include a short load test that reads clock, temperature and throttle reasons under load.
- **Density.** 8x B200 at up to 1 kW each, and NVL72 racks at 132-155 kW, are liquid-cooled and
  exist only in purpose-built sites, which correlates with good networking. Air-cooled 8x 4090
  rigs correlate with the opposite.

## 3. Software compatibility: driver, CUDA, wheels, containers

### 3.1 Compute capability versus wheel contents

A binary runs on a GPU only if it contains kernels for that compute capability (or an older one of
the same major). Current state:

- Blackwell needs CUDA 12.8+; sm_120 (RTX 50, PRO Blackwell) kernels first shipped in stable
  PyTorch 2.7.0 with cu128 wheels (April 2025). Older wheels fail with
  `CUDA error: no kernel image is available for execution on the device`
  ([NVIDIA migration guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)).
- PyTorch removed Maxwell and Pascal from CUDA 12.8/12.9 builds starting with 2.8
  ([dev-discuss](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)).
- From PyTorch 2.11 the default PyPI wheel is CUDA 13.0 (as reported in search results; unverified
  against release notes). PyTorch 2.14 is the last release with any CUDA 12.x wheel; 2.15 ships
  CUDA 13.x only, retaining Turing (sm_75), Ampere (sm_80, sm_86), Hopper (sm_90) and Blackwell
  (sm_100, sm_120); pre-Turing needs a source build with `TORCH_CUDA_ARCH_LIST`
  ([dev-discuss notice, 2026-09-04](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
- `torch.cuda.get_arch_list()` reports what the installed build contains;
  `torch.cuda.get_device_capability()` reports what the device is. Comparing them is a one-line
  preflight.

Every GPU in section 1 is Ampere or newer, so a single CUDA 13 image covers all NVIDIA parts
**if the host driver is new enough**, which is the real constraint.

### 3.2 Host driver is the provider's choice

The container brings CUDA user-space libraries; the kernel driver and `libcuda`/`libnvidia-ml` are
injected from the host by the NVIDIA Container Toolkit. Rules:

- A driver supports its CUDA major and older ones. CUDA 13.0 GA needs Linux driver >= 580.65.06
  (13.0 U1: 580.82.07, U2: 580.95.05)
  ([CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html.md)).
  Current datacenter branches are R580 (LTSB, EOL June 2028), R595 (production, EOL March 2027)
  and R620 (LTSB, EOL September 2029), all topping out at CUDA 13 without the forward-compat
  package ([NVIDIA driver branches](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html)).
- Community reports say many rented RTX 5090 hosts still run 570-branch drivers
  ([NVIDIA forum thread](https://forums.developer.nvidia.com/t/effective-pytorch-and-cuda/348230?page=2);
  anecdotal). A 570 host cannot run a cu130 wheel without forward compatibility.
- Forward compatibility (`cuda-compat-<major>-<minor>` package, libraries under
  `/usr/local/cuda-*/compat/`) lets a newer toolkit run on an older driver "subject to platform
  and GPU support" ([CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/)).
  The failure signature when it is not supported is
  `forward compatibility was attempted on non supported HW` (CUDA error 804), which is the classic
  result on GeForce (hardware scope unverified here). The inverse failure, the image's compat
  libraries shadowing the host's via `LD_LIBRARY_PATH`, is described by NVIDIA as the most common
  cause of CUDA init error 803
  ([NIM troubleshooting](https://docs.nvidia.com/nim/large-language-models/3.0.0/troubleshooting/cuda-driver.html)).
- `Failed to initialize NVML: Driver/library version mismatch` means user-space and kernel module
  disagree: either the host upgraded the driver without reloading the module, or image libraries
  replaced the host's ([libnvidia-container issue](https://gitlab.com/nvidia/container-toolkit/libnvidia-container/-/issues/3)).
  Not fixable by the tenant; replace the node.
- Images can gate themselves: `NVIDIA_REQUIRE_CUDA="cuda>=13.0 driver>=580"` (space = OR, comma =
  AND), `NVIDIA_REQUIRE_ARCH`, `NVIDIA_REQUIRE_BRAND`; if unmet "the container does not start".
  `NVIDIA_DISABLE_REQUIRE=true` disables the checks. `NVIDIA_DRIVER_CAPABILITIES` defaults to
  `utility,compute` ([Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)).
  A provider template that fails to start with no logs is often this.

Practical matrix for grid's image selector:

| Host driver | Usable wheel | Covers |
|---|---|---|
| < 570 | cu126 or older (PyTorch <= 2.14) | Ampere, Ada, Hopper; **not** Blackwell |
| 570-579 | cu128/cu129 (PyTorch 2.7-2.14) | everything through Blackwell |
| >= 580 | cu130+ (PyTorch 2.11+ default, only option from 2.15) | Turing through Blackwell (and Rubin when supported) |

Mixing a pre-570 A100 host and a Blackwell host in one job therefore forces PyTorch <= 2.14 with
two different wheel variants, or excluding one side. AMD nodes need a separate ROCm image entirely
(PyTorch 2.12 on ROCm 7.14), and GB200/GB300 need `arm64` images.

## 4. Heterogeneity inside one job

### 4.1 What goes wrong

1. **Slowest rank sets the pace.** Synchronous SGD waits for every rank's gradient each step. With
   equal per-rank batches, step time is the maximum over ranks; an H100 next to an A100 idles for
   the difference. The power-cap data above shows that even same-model ranks can differ by 10-40%.
2. **Smallest memory sets the batch.** With one global per-rank batch size, a 24 GB card caps a
   180 GB card. For LLMs it is worse than batch size: the model, optimizer state and activations at
   the chosen sequence length must fit on the smallest rank at all. Mixed memory is tolerable for
   data parallelism over a model that fits everywhere; it is not tolerable inside a tensor-parallel
   or FSDP-sharded group, where shards are equal-sized.
3. **Collectives degrade to the worst link.** A ring all-reduce runs at the bandwidth of its
   slowest hop; a single PCIe-only or TCP-only member drags an NVLink/InfiniBand group (see
   `05-networking-interconnects.md`).
4. **Numerics differ.** PyTorch: "Completely reproducible results are not guaranteed across PyTorch
   releases, individual commits, or different platforms", and determinism controls apply only "for
   a specific platform, device, and PyTorch release"
   ([PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)).
   Different architectures select different cuDNN/cuBLAS kernels and SDPA backends with different
   reduction orders; FP8-capable and FP8-incapable ranks would run different arithmetic. For SGD
   this is noise of the same character as data-order noise (unverified as a general claim, but
   standard practice treats it so); it does mean a heterogeneous run cannot be replayed bit-for-bit
   and that resuming a checkpoint on different hardware changes the trajectory.
5. **Feature intersection.** Compile flags, FP8, `torch.compile` kernels, flash-attention builds
   and MIG constraints must all be satisfied by every rank, or selected per rank.
6. **Software intersection.** Section 3: the driver/wheel matrix may have an empty intersection.

### 4.2 Coping techniques

| Technique | Mechanism | Evidence |
|---|---|---|
| Group by class (default) | Only identical classes share a synchronous group; different classes run separate jobs or separate replicas joined by a looser algorithm (local SGD / DiLoCo, see `08`) | Operational norm; avoids all six problems |
| Per-rank batch sizing | Give each rank a batch proportional to its measured throughput so all ranks finish a step together | LB-BSP reports over 54% improvement on a 16-node heterogeneous GPU cluster ([arXiv 1806.02508](https://arxiv.org/abs/1806.02508)) |
| Batch-weighted gradient averaging | Weight each rank's gradient by its sample (or token) count; plain averaging over-represents samples from small batches | LB-BSP; [Cannikin](https://arxiv.org/html/2402.05302v1); [HetSeq](https://arxiv.org/pdf/2009.14783) weights by batch size or token count |
| Predictive configuration | Model per-node speed versus batch to find the optimal split when total batch changes | Cannikin ("OptPerf") |
| Decouple compute from memory | Split the batch unevenly by compute, shard training state by memory, use per-rank gradient accumulation so small-memory GPUs do several micro-batches | Cephalo: 1.2x-10.8x throughput over prior methods on 9 transformer models, clusters mixing A10G/V100/T4 ([arXiv 2411.01075](https://arxiv.org/abs/2411.01075), ICS 2025) |
| Adaptive batch + staleness compensation | Adjust batch per worker at runtime and compensate delayed gradients | [ABS-SGD, arXiv 2308.15164](https://arxiv.org/abs/2308.15164) |

Implementation notes for the weighted approach in a PyTorch DDP-style loop: DDP's all-reduce
averages gradients with equal weight, so each rank must scale its loss by `local_tokens *
world_size / global_tokens` (or sum-reduce the loss and divide by the all-reduced global token
count) before `backward()`. LLM batches should be measured in tokens, not sequences. Throughput
must be measured, not looked up by model name, because of power caps, PCIe link width and CPU
starvation. BatchNorm-style statistics are not a concern for transformers; dropout and data
sharding need per-rank RNG streams that do not depend on batch size.

Hard limits: per-rank batch sizing does not help tensor/pipeline parallel groups or FSDP shards,
which assume symmetric members; there the only safe rule is identical class.

## 5. Fingerprinting a node into a capability record

Goal: under about 10 s of static probing plus an optional 30-60 s micro-benchmark, with no root.
nvidia-smi's output format is not guaranteed stable across drivers and NVIDIA recommends NVML for
tooling ([nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html)); use `pynvml`
where available and CSV queries as the fallback. Field names below should be validated against
`nvidia-smi --help-query-gpu` on each driver branch (the list varies by driver and GPU).

| Probe | Command | Yields |
|---|---|---|
| GPU identity | `nvidia-smi --query-gpu=index,name,uuid,serial,pci.bus_id,memory.total,driver_version,vbios_version,compute_cap --format=csv,noheader,nounits` | model, VRAM, driver, compute capability |
| GPU state | `--query-gpu=ecc.mode.current,mig.mode.current,compute_mode,persistence_mode,power.limit,power.default_limit,power.max_limit,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max` | ECC present, MIG, power cap, degraded PCIe link |
| Memory health | `nvidia-smi --query-remapped-rows=gpu_uuid,remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure --format=csv` (field names unverified), `--query-retired-pages=...` | prior memory faults |
| Topology | `nvidia-smi topo -m`, `nvidia-smi topo -mp`, `nvidia-smi topo -p2p n` / `-p2p p` / `-p2p r` | NVLink versus PCIe adjacency, NUMA/CPU affinity per GPU, NIC adjacency, P2P support matrix |
| NVLink | `nvidia-smi nvlink -s` (link state and per-link bandwidth), `-c` (capabilities), `-e` (replay/recovery/CRC error counters), `-R` (remote endpoint) | NVLink generation, link count, whether peers are GPUs or switches, link health |
| NCCL view | run a 1-step NCCL init with `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` and `NCCL_DEBUG=INFO` | what NCCL will actually use, including transport chosen |
| CPU | `lscpu -J`, `numactl -H`, `cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/memory.max` | arch (x86_64/aarch64), sockets, NUMA nodes, *usable* cores and RAM |
| PCI | `lspci -nn -d 10de:` and `lspci -tv`; `lspci -nn -d 15b3:` for Mellanox NICs; `-d 1002:` for AMD | PCI device IDs (distinguishes SXM/PCIe/NVL variants), switch hierarchy, NIC models |
| Storage | `lsblk -J -o NAME,MODEL,SIZE,ROTA,TRAN`, `df -h`, short `fio` | local NVMe versus network volume |
| Software | `torch.version.cuda`, `torch.cuda.get_arch_list()`, `torch.cuda.get_device_capability()`, `torch.cuda.is_bf16_supported()`, `nvidia-container-cli --version` if visible | wheel/driver fit |
| Health (optional, needs DCGM on host or privileged container) | `dcgmi diag -r 1` (fast deployment check), `-r 2` (about 2 min), `-r 3` (about 15 min; adds memory_bandwidth, targeted_stress, targeted_power), `-r 4` (adds memtest, pulse_test) | hardware burn-in ([dcgmi diag reference](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgmi/dcgmi-diag.html)) |
| AMD | `rocm-smi --showhw --showmeminfo vram --showtopo` / `amd-smi static` (unverified flags) | equivalent record |

Suggested record (illustrative):

```json
{
  "node_id": "…", "provider": "runpod", "arch": "x86_64",
  "cpu": {"model": "EPYC 9554", "sockets": 2, "numa_nodes": 2, "usable_cores": 20, "usable_ram_gb": 125},
  "gpus": [{"index": 0, "uuid": "GPU-…", "name": "NVIDIA H100 80GB HBM3", "pci_device_id": "10de:2330",
            "form": "sxm", "vram_mib": 81559, "compute_cap": "9.0", "numa_node": 0,
            "ecc": true, "mig": false, "power_limit_w": 700, "power_default_w": 700,
            "pcie": {"gen": 5, "width": 16, "gen_max": 5, "width_max": 16},
            "remapped_rows": {"uncorrectable": 0, "pending": false}}],
  "gpu_links": {"kind": "nvlink-switch", "nvlink_gen": 4, "per_gpu_gbps_bytes": 900, "p2p_supported": true,
                "islands": [[0,1,2,3,4,5,6,7]]},
  "nics": [{"model": "BlueField-3", "speed_gbps": 400, "link_layer": "ethernet", "numa_node": 0}],
  "storage": {"local_nvme_gb": 3500, "seq_write_mbps": 2800},
  "software": {"driver": "580.95.05", "max_cuda": "13.0", "container_toolkit": "…", "kernel": "6.8"},
  "features": {"bf16": true, "fp8": true, "tf32": true},
  "measured": {"tokens_per_s_ref": 30100, "allreduce_gbps": 410, "h2d_gbps": 22, "dataloader_cores_free": 12},
  "class_key": "h100-sxm|80g|nvsw4|x86_64|drv580|cap700"
}
```

Key design points: `islands` (sets of GPUs with full-rate P2P) rather than a boolean `has_nvlink`,
so H100 NVL pairs and NVL72 racks are expressible; `form` derived from PCI device ID plus NVLink
topology, not from the marketing name; `measured` fields from a reference micro-benchmark, because
power caps and CPU starvation are invisible to static probes; and a `class_key` built from the
fields that must match for a symmetric group (model, VRAM, link kind, CPU arch, driver major,
power cap bucket).

## Implications for grid

1. **Schedule on capability records, not SKU names.** Provider SKU strings ("H100", "A100") hide
   form factor, power cap, CPU share and driver. Fingerprint every node at join and derive a
   `class_key`.
2. **Default policy: one class per synchronous group.** Tensor/pipeline/FSDP groups require
   identical class and must sit inside one P2P island. Cross-class mixing is opt-in and limited to
   the data-parallel axis, with token-weighted gradient averaging and throughput-proportional
   per-rank batches, or handed to the looser algorithms in `08`.
3. **Pick the image per node from the driver.** Maintain a small image matrix (cu126 / cu128 /
   cu130 x amd64/arm64, plus ROCm) and select by `driver_version` and `compute_cap`; verify with
   `torch.cuda.get_arch_list()` before joining the job. Set `NVIDIA_REQUIRE_*` deliberately so an
   unsuitable host fails fast, and never let image compat libraries shadow host libraries.
4. **Reject or quarantine by rule.** Pre-570 driver with Blackwell; PCIe link below max
   width/gen; uncorrectable remapped rows or pending retirements; `power.limit` well below default
   (or re-class it); MIG enabled when whole GPUs were requested; usable cores per GPU below a
   job-declared minimum; host RAM below a multiple of VRAM when CPU-side checkpointing is used.
5. **Treat consumer multi-GPU pods as multiple single-GPU ranks with a slow local link.** No
   supported P2P, no ECC, no error counters; add a known-answer all-reduce test and per-rank
   gradient sanity checks. Provide an `allow_consumer_gpus` policy flag because of the licence
   clause.
6. **Measure, then trust.** A 30-60 s reference step (fixed model, fixed tokens) per node gives the
   only number that captures power caps, thermals, CPU starvation and memory bandwidth together.
   Use it for batch apportioning and to detect outliers within a class (for example more than 10%
   below class median).
7. **Size dataloader workers from cgroup limits** and reserve cores for NCCL and checkpoint I/O;
   expose a `min_cpu_per_gpu` job parameter and prefer pre-tokenised shards on low-CPU classes.
8. **Keep the architecture list open-ended.** Rubin, B300 and MI355X arrived within about a year;
   the schema should carry raw capability numbers and treat unknown models as schedulable when
   software checks pass.
9. **Expose NVLink domain as a first-class placement unit** above "node" so NVL72-style racks and
   below "node" so bridged pairs both map onto the same abstraction.

## Open questions

- **Verify the (unverified) datasheet values** for A100, L40S, A6000, RTX 6000 Ada, 3090/4090/5090
  and RTX PRO 6000 Blackwell (power, NVLink bridge support, ECC) against NVIDIA datasheets; this
  sweep confirmed HGX H100/H200/B200 and GB200 NVL72 from NVIDIA pages but not those.
- **B300 power and usable memory** (1.2 kW versus 1.4 kW; 288 GB versus "270 GB usable") and the
  H200 NVL 4-way bridge bandwidth (900 GB/s versus 1.8 TB/s) are contradictory across sources.
- **What do marketplace hosts actually run?** A survey of driver versions, power caps, PCIe
  topology and vCPU-to-core mapping on real RunPod/Vast pods (especially multi-GPU 4090/5090
  pods) would replace anecdotes; this needs hands-on probing, not web research.
- **What can a tenant container see?** Whether `nvidia-smi topo`, `nvlink`, `lspci -tv`, NUMA
  information and DCGM are usable inside unprivileged provider containers and inside VM-based
  offerings (flattened PCI tree, virtual topology file) is untested.
- **Forward-compat scope.** Exactly which GPUs support the `cuda-compat` forward-compatibility
  path, and the precise minimum driver for CUDA 12.8/12.9, were not confirmed from the CUDA
  compatibility subpages.
- **MIG limits for training** (P2P between MIG devices, multiple MIG devices per process) should
  be confirmed against the current MIG user guide.
- **AMD node internals.** Infinity Fabric topology and bandwidth on the 8-way MI300X/MI355X
  baseboards, RCCL transport selection, `amd-smi` fingerprint fields, and real multi-node training
  scaling under ROCm 7.14 need a dedicated sweep.
- **Quantified cost of mixing.** No public measurement was found of LLM pre-training convergence
  with mixed Hopper/Blackwell ranks or mixed FP8/BF16 ranks; Cephalo/Cannikin evaluate older GPUs
  and smaller models. An experiment (for example H100 + A100 + 4090 on a small GPT) would settle
  whether token-weighted DP is good enough in practice.
- **Rubin on rentable clouds**: SKU names, prices, driver/CUDA minimums and whether the host is
  Vera (Arm) everywhere.
- **Silent corruption rates on consumer cards** under sustained training load: no quantitative
  source found.
