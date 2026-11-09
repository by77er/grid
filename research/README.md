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

## Conventions

- Every non-obvious claim carries a source URL. Claims that could not be verified are marked `(unverified)`.
- Each file ends with **Implications for grid** and **Open questions**.
