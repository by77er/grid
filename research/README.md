# Research: GPU cluster orchestration for LLM training

Goal of the project (`grid`): a tool that spins up large GPU clusters very quickly on
marketplace-style providers (RunPod and similar) for **training** LLMs, tolerating
heterogeneous hardware and networks.

This directory holds the background research. Each file is one research direction, written
independently, with sources cited inline. `SYNTHESIS.md` pulls the threads together.

## Research directions

| # | File | Question |
|---|------|----------|
| 01 | `01-orchestrators-schedulers.md` | How do Slurm, Kubernetes (Kubeflow Trainer, Volcano, Kueue, JobSet), Ray etc. model a distributed training job, gang-schedule it, and handle heterogeneous nodes? |
| 02 | `02-multicloud-provisioners.md` | Prior art closest to `grid`: SkyPilot, dstack, Prime Intellect, Modal, Ray cluster launcher, Slurm-on-K8s. What do they do, where do they fall short? |
| 03 | `03-rank-configuration.md` | How are ranks / world size / rendezvous configured across launchers (torchrun, MPI/PMIx, srun, Accelerate, DeepSpeed, JAX), and how do ranks map onto DP/TP/PP/EP parallelism groups? |
| 04 | `04-failure-taxonomy-recovery.md` | Classes of failure in large GPU training (hardware, network, software, preemption, silent corruption), how each is detected, and how each is recovered from. |
| 05 | `05-networking-interconnects.md` | InfiniBand / RoCE / EFA / plain TCP, NCCL transport selection and tuning, topology, and what happens on mixed or slow networks. |
| 06 | `06-gpu-hardware-nuances.md` | What is different about each GPU generation / server configuration (memory, NVLink, PCIe/NUMA, drivers, precision support), and what mixing them in one job costs. |
| 07 | `07-provider-landscape.md` | RunPod and peers (Lambda, Vast, CoreWeave, Crusoe, Nebius, ...): APIs, provisioning latency, cluster products, networking, storage, container restrictions, spot semantics. |
| 08 | `08-heterogeneous-distributed-training.md` | Training algorithms that tolerate heterogeneous / slow / unreliable links and nodes: local SGD, DiLoCo, async methods, straggler mitigation, heterogeneity-aware data parallelism. |
| 09 | `09-fast-bootstrap-preflight.md` | What dominates time-to-first-step: image pulls, data/checkpoint staging, node mesh setup, and preflight health checks / burn-in; how to make each fast. |

### Serving (added scope)

`grid` should also cover **serving** LLMs, both for direct consumption and as the inference
side of RL training loops.

| # | File | Question |
|---|------|----------|
| 10 | `10-inference-engines.md` | vLLM, SGLang, TGI, TensorRT-LLM/Triton, Dynamo, llama.cpp: how each is launched and scaled across GPUs/nodes, and how hardware and heterogeneity affect them. |
| 11 | `11-serving-fleet-orchestration.md` | Fleet-level serving: routing (cache-aware), autoscaling, cold starts, health checking, failure recovery, spot handling, multi-region/provider replicas. |
| 12 | `12-inference-for-rl.md` | Inference inside RL loops: actor/learner architectures, throughput-oriented generation, weight sync from trainer to servers, co-scheduling training and inference on one cluster. |

### Second sweep (gap closing from source code and specs)

| # | File | Question |
|---|------|----------|
| 13 | `13-in-place-repair-elasticity.md` | Can survivors keep running when a node dies? NCCL shrink/grow, PyTorch support, TorchElastic internals, NVIDIA resiliency tooling. |
| 14 | `14-cross-island-sync-implementations.md` | Code grid could adopt for the slow-link tier (PCCL, hivemind, DiLoCo implementations, object-store relay), for training and RL weight broadcast. |
| 15 | `15-checkpoint-resharding-data-resume.md` | What lets a job resume on a different cluster shape: checkpoint formats, resharding, world-size-independent data resume. |
| 16 | `16-provider-apis-ground-truth.md` | Exact RunPod (and peer) API shapes, how SkyPilot/dstack drive them, and user-reported problems. |
| 17 | `17-serving-source-checks.md` | vLLM/SGLang/router behaviour verified at current versions: death semantics, weight updates, non-Kubernetes routers, mid-stream failover. |

### Design-space studies

| # | File | Question |
|---|------|----------|
| 18 | `18-state-transfer-design-space.md` | Weight transfer and checkpoint methods as a selectable, tunable axis: the catalogue of methods, the trade-off axes, the knobs for sitting between methods, and how grid exposes the choice. |

## Conventions

- Every non-obvious claim carries a source URL. Claims that could not be verified are marked `(unverified)`.
- Each file ends with **Implications for grid** and **Open questions**.
- Files 01–12 were not edited after the second sweep. Files 13–17 each have a
  **Corrections to first sweep** section; where they disagree with 01–12, they take precedence.
- Nothing was run on real hardware or against a live provider account. All findings come from
  docs, specs, source code, papers and issue threads read on 2026-11-09.
