# 12 — Inference inside RL loops for LLMs

Researched 2026-11-09. Sibling files: `10-inference-engines.md` (engine launch/scaling), `11-serving-fleet-orchestration.md` (routing, autoscaling, spot), `04-failure-taxonomy-recovery.md`, `05-networking-interconnects.md`, `08-heterogeneous-distributed-training.md`.

## Summary

- **The dominant 2026 architecture is three (or four) decoupled services**: an inference fleet (vLLM or SGLang) holding the current policy, a CPU-side orchestrator/buffer that drives environments and packs batches, a trainer process group (FSDP2 / Megatron), and, for agentic work, a separate sandbox fleet. prime-rl, verl's fully-async recipe, AReaL, SkyRL, slime/miles, NeMo RL, Trinity-RFT, Atropos and Cursor's Composer stack all have this shape. Synchronous "colocated" designs (TRL colocate, verl hybrid engine, ART shared mode) that time-slice the same GPUs between generation and training survive for small scale and are measurably slower at scale (1.6-2.8x in published comparisons).
- **Weight sync is now an engine-native feature, not framework glue.** vLLM ships a `start → update → finish` weight-transfer lifecycle with NCCL, CUDA-IPC, sparse, M2N-reshard and RDMA/NIXL backends plus `/pause` / `/resume` ([vLLM docs](https://docs.vllm.ai/en/latest/training/weight_transfer/), [vLLM blog 2026-05-28](https://vllm.ai/blog/2026-05-28-native-rl-apis)); SGLang ships `/update_weights_from_{disk,tensor,distributed}` and `/pause_generation` ([SGLang docs](https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md)). Sync cost ranges from ~20 ms (7B, NCCL bucketed) through ~20 s (1T params, 256 GPUs) to ~14 min (62 GB over public internet).
- **Staleness is tolerated but bounded.** Typical bounds are 1-8 optimizer steps of policy lag, enforced by version tagging and rejection, optionally corrected by truncated importance sampling.
- **For agentic RL the bottleneck moves off the GPUs**: sandboxes (CPU fleet, start-up latency, stragglers) gate throughput; published runs show inference engines at 23% KV-cache use with empty queues while waiting on sandboxes.
- **What a cluster tool must add for RL** (beyond training or serving alone): role-typed pools with different reliability/network classes, a weight-distribution plane with policy-version bookkeeping, fleet-level update gating, a CPU sandbox fleet, a ratio-tuning feedback loop, and the ability to flip a GPU's role.

---

## 1. Architectures in use

### 1.1 The canonical decoupled loop

prime-rl is the cleanest statement of the pattern ([overview](https://docs.primeintellect.ai/prime-rl/overview.md)):

| Process | Hardware | Role |
|---|---|---|
| Inference | GPU fleet, vLLM | Holds current policy. Orchestrator calls the token-in route `/inference/v1/generate`; OpenAI chat/completions routes also exposed |
| Orchestrator | CPU only | Runs `verifiers` environments (each in an isolated subprocess with an env-worker pool), drives multi-turn rollouts, computes advantages, packs batches, relays weights trainer → inference |
| Trainer | GPU, FSDP2 process group | Consumes packed rollouts, steps optimizer; EP (DeepEP), CP, FP8, LoRA supported |

Transports: rollouts trainer↔orchestrator over **ZMQ by default** (each trainer rank *requests* a step, so the trainer only holds the batch it is training on — pull-based backpressure), or `rollout_transport.type = "filesystem"`; weight broadcast over **NCCL**, falling back to filesystem when LoRA is on or no inference server is configured ([overview](https://docs.primeintellect.ai/prime-rl/overview.md)). Multi-node is SLURM-only (`[deployment] type = "multi_node"`, `num_train_nodes`, `num_infer_nodes`, `gpus_per_node` default 8); each inference node runs an independent vLLM replica with TP/DP contained in the node, fronted by one router on node 0 ([scaling](https://docs.primeintellect.ai/prime-rl/scaling)).

verl's fully-async recipe has the same four boxes under different names: **Rollouter → MessageQueue → Trainer**, plus a **ParameterSynchronizer** (NCCL, based on Moonshot's checkpoint-engine) ([verl docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html)). Cursor's Composer 2 stack is "four decoupled services: training, environments, inference, evaluations", on Ray + PyTorch, with GPU compute in 3 regions and CPU compute in 4 ([arXiv 2603.24477 §6.2](https://arxiv.org/html/2603.24477)).

### 1.2 Framework map

Largely from the Hugging Face survey of 16 async-RL libraries (March 2026) ([HF blog](https://huggingface.co/blog/async-rl-training-landscape)), cross-checked against primary docs where I fetched them.

| Framework | Orchestration | Weight sync | Staleness control | In-flight rollouts at sync | Trainer backend |
|---|---|---|---|---|---|
| prime-rl (Prime Intellect) | asyncio + ZMQ/filesystem; SLURM | NCCL; filesystem safetensors + HTTP; NIXL/ModelExpress | version + depth (`max_async_level`, default 2) + IS | group cancellation | FSDP2 |
| verl (ByteDance) | Ray | NCCL + checkpoint-engine buckets | IS correction; `staleness_threshold` | explicit save/resume (`partial_rollout`) | FSDP1/2, Megatron |
| AReaL (Ant/inclusionAI) | asyncio + HTTP RPC; Ray/Slurm pluggable | NCCL chunked or filesystem | depth bound + optional IS (cap 5.0) | soft pause; in-flight finish | FSDP2, Megatron, Archon |
| SkyRL (NovaSky) | Ray | NCCL group; vLLM native APIs | depth bound | abort + retry with prefix | FSDP/FSDP2, Megatron-Bridge |
| slime (THUDM) / miles (radixark) | Ray | NCCL (disaggregated) or CUDA IPC (colocated), bucketed | IS (TIS + OPSM in miles) | abort and recycle to buffer | Megatron-LM (+FSDP2 in miles) |
| NeMo RL (NVIDIA) | Ray | NCCL, CUDA IPC-ZMQ, HTTP | version rejection | in-flight continuation | DTensor, Megatron-Bridge |
| PipelineRL (ServiceNow) | asyncio + Redis/JSONL | NCCL + HTTP notify, **mid-sequence** | version rejection (`max_lag`) | implicit continuation | DeepSpeed ZeRO-3 |
| open-instruct (AI2) | Ray | NCCL broadcast | depth (`async_steps` default 1, production 8) | drain or in-flight | DeepSpeed |
| ART (OpenPipe/CoreWeave) | asyncio + subprocesses | LoRA adapter swap in vLLM | none (sync) in shared mode | full sleep | Unsloth, Megatron |
| Atropos (Nous) | FastAPI microservices; trajectory API buffer | filesystem checkpoint + vLLM restart | depth bound | not supported | PyTorch / TRL / Axolotl / Tinker |
| Trinity-RFT (ModelScope) | explorer / buffer / trainer + Synchronizer | `sync_method: nccl | checkpoint | memory`, `sync_interval`, `sync_timeout: 1200` | `sync_interval`, `sync_offset` | — | verl |
| TorchForge (Meta) | Monarch | torchstore + shared memory | version rejection (`max_policy_age`) | not supported | FSDP2 (TorchTitan) |
| TRL GRPO (HF) | in-process or `trl vllm-serve` | in-proc load (colocate) or HTTP (server) | none (sync) | n/a | Accelerate |

Sources for rows not in the survey: Trinity-RFT [synchronizer docs](https://modelscope.github.io/Trinity-RFT/tutorial/synchronizer.html) and [config guide](https://modelscope.github.io/Trinity-RFT/v0.1.0/tutorial/trinity_configs.html); TRL [vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration).

Layered-on agent frameworks do not own the GPUs but shape the request pattern:

- **rLLM**: a Model Gateway captures token IDs and logprobs from arbitrary agent code; training backends are `verl` (Ray, multi-node), `tinker` (LoRA, remote) or `fireworks` ([rLLM docs](https://rllm-project.readthedocs.io/)).
- **Agent Lightning** (Microsoft): a `LightningStore` holds the task queue, rollouts, attempts, spans and versioned resources; runners dequeue and stream spans back; an LLM proxy sits between agent and model. Algorithm and runners only talk to the store, so they can be on different machines ([docs](https://microsoft.github.io/agent-lightning/0.3.0/deep-dive/store)).
- **Atropos**: environments are independent microservices that push scored trajectory groups to a central trajectory API; trainers pull batches ([Nous intro](https://nousresearch.com/introducing-atropos)). One aggregator claims the repo is archived (unverified).
- **Tinker** (Thinking Machines): hosted version of the same split — `forward_backward` / `optim_step` on a `TrainingClient`, then `save_weights_and_get_sampling_client()`; LoRA only ([Tinker docs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)). How Tinker multiplexes base models across tenants is not documented (unverified).
- **NVIDIA Dynamo** now has an RL listener: `DYN_ENABLE_RL=true DYN_RL_PORT=8001 python -m dynamo.frontend`, worker `python -m dynamo.vllm --model ... --enable-rl`, discovery at `GET /v1/rl/workers` (protocol_version 1) ([Dynamo RL reference](https://docs.nvidia.com/dynamo/dev/reinforcement-learning/rl-integration-reference)).

### 1.3 Colocated (time-sliced) versus disaggregated

**Colocated**: the same GPUs alternate between generation and optimizer step.

- TRL: `GRPOConfig(use_vllm=True, vllm_mode="colocate")`; vLLM is launched inside the trainer's `torch.distributed` group, capped by `vllm_gpu_memory_utilization` (example 0.3), and can call vLLM `sleep()` level 2 between phases. HF measured colocate 1.26-1.73x faster than server mode on 8 GPUs, mostly because server mode burns 1 of 8 GPUs on a mostly-idle server ([HF blog](https://huggingface.co/blog/vllm-colocate)). TRL PR #5255 (March 2026) made colocate the default ([vLLM TRL docs](https://docs.vllm.ai/en/latest/training/trl/)). Axolotl's docs make the opposite point: colocate is slower when generation and training could overlap ([Axolotl](https://docs.axolotl.ai/docs/vllm_serving.html)).
- verl hybrid engine: training weights are offloaded, vLLM generates a full batch, then the system swaps back ([AMD/verl walkthrough](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html)).
- vLLM sleep mode is the primitive: `--enable-sleep-mode` with `VLLM_SERVER_DEV_MODE=1`; level 1 moves weights to CPU RAM and drops KV cache; level 2 drops both ("e.g. RLHF weight update"); wake in two stages — weights first, `reload_weights`, then `POST /wake_up?tags=kv_cache` — so KV memory is not allocated during the update ([vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)). SGLang equivalent: `--enable-memory-saver`, `POST /release_memory_occupation` / `/resume_memory_occupation` with tags `kv_cache`, `weights`; release asserts no ongoing requests ([SGLang](https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md)).
- ART shared `LocalBackend` pauses inference during training; ART rejects that mode for its `PipelineTrainer` and requires dedicated mode (`trainer_gpu_ids` and `inference_gpu_ids` in `_internal_config`), where vLLM keeps serving and the LoRA is hot-reloaded after each step ([ART backend docs](https://art.openpipe.ai/fundamentals/art-backend/); key names from a PR mirror, treat as (unverified) against current source).

**Disaggregated async** wins once rollouts are long or variable. verl's own numbers, same total GPU count, Qwen2.5-Math-7B DAPO on H20 ([verl docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html)):

| GPUs | Colocate, 400 steps | Fully async, 400 steps | Speedup |
|---|---|---|---|
| 32 (16:16) | 3d 17h 5m | 1d 9h 26m | 2.66x |
| 64 (32:32) | 1d 17h 32m | 21h 40m | 1.92x |
| 128 (64:64) | 1d 16h 48m | 17h 22m | 2.35x |

Decomposed at 128 GPUs: streaming alone (no staleness) ≈1.6x; adding `staleness_threshold=0.5` with `partial_rollout=True` reaches 2.35x. AReaL reports up to 2.77x versus synchronous at equal GPU count ([arXiv 2505.24298](https://arxiv.org/abs/2505.24298v4)); PipelineRL ≈2x on 128 H100 ([arXiv 2509.19128](https://arxiv.org/abs/2509.19128v2)). The cause is the synchronous barrier: a batch waits for its longest sequence; miles measured 32% of rollout time occurring after 90% of trajectories had finished ([LMSYS/miles blog](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl)).

---

## 2. Getting weights from trainer to inference servers

### 2.1 Mechanisms and measured cost

| Mechanism | Requires | Measured cost | Source |
|---|---|---|---|
| NCCL broadcast, trainer ranks → inference ranks | Same NCCL-reachable network, shared process group (master addr/port, `rank_offset`, `world_size`) | ~100-500 ms typical; verl bucketed ~20 ms for 7B | [HF survey](https://huggingface.co/blog/async-rl-training-landscape) |
| NCCL + checkpoint-engine buckets (verl, Megatron, H20) | as above | 7B (4/4 ranks) 0.12 s → 0.02 s; 30B MoE (16/16) 15.76 s → 4.38 s; 235B MoE (64/64) 58.57 s → 23.70 s | [verl docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html) |
| checkpoint-engine broadcast / P2P (Kimi-K2 1T, FP8, 256x H20) | ParameterServer colocated with engines | ~20 s headline; ~21.5 s broadcast, ~34.5 s P2P (secondary source) | [PyPI](https://pypi.org/project/checkpoint-engine/), [secondary](https://uxproduct.blogspot.com/2025/09/moonshotais-checkpoint-engine.html) |
| CUDA IPC handles (colocated, zero-copy; ~64-byte handle per tensor) | Same GPU/host | Qwen3-30B-A3B on 8x H100: 60 s → 7 s after parallel gather, 512 MB buckets (2000 → 120 calls), flattening, load caching | [slime blog](https://hebiao064.github.io/rl-weight-sync) |
| NCCL cross-node, 30B MoE, 1 train node → 3 inference nodes (MI355X) | Pinned NCCL interface | 23 s per sync against a 208 s step | [miles blog](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl) |
| Filesystem safetensors + HTTP notify | Shared FS or object store | "medium"; slime's disk mode writes a full HF checkpoint `weight_v{N:06d}/` per sync | [HF survey](https://huggingface.co/blog/async-rl-training-landscape) |
| Filesystem + engine restart (Atropos) | none | "very high" (full cold start) | same |
| LoRA adapter only | Adapter-capable engine | rank-32 adapter on 7B+ ≈ 50 MB; transfer negligible | same |
| Object store + per-rank delta compression (Composer 2, 1T MoE) | S3 only; no trainer↔inference connectivity | every step; diffs "a handful of gigabytes"; ≥98% size cut; minutes-level cross-cluster staleness | [arXiv 2603.24477](https://arxiv.org/html/2603.24477), [Fireworks case study](https://fireworks.ai/customers/cursor-composer-2-reinforcement-learning) |
| HTTP relay tree over public internet (Shardcast, INTELLECT-2) | nginx relays; SHA-256 check | 62 GB in ~14 min (≈590 Mb/s); verified batch ready 22-29 min after broadcast | [arXiv 2505.07291](https://arxiv.org/html/2505.07291) |

Two families emerge. **In-cluster push** (NCCL / IPC / RDMA) is sub-second to tens of seconds but needs trainer and inference ranks in one routable, ideally RDMA, fabric and a rendezvous. **Store-and-pull** (object store, HTTP relays, ModelExpress) tolerates WAN, NAT and unrelated providers, at seconds-to-minutes lag; delta compression and LoRA make it viable for large models.

### 2.2 vLLM API surface (as of docs "latest", ~v0.32)

All require `VLLM_SERVER_DEV_MODE=1`; server flag `--weight-transfer-config '{"backend": "nccl"}'` ([docs](https://docs.vllm.ai/en/latest/training/weight_transfer/)):

- `POST /init_weight_transfer_engine` (once), `POST /start_weight_update`, `POST /update_weights` (repeatable, chunked), `POST /finish_weight_update` (post-processing such as FP8 re-quantisation; can commit a `weight_version`), `POST /shutdown_weight_transfer_engine`, `POST /update_weight_version`, `GET /weight_info`, `GET /get_world_size`, `POST /pause`, `POST /resume`.
- Backends: `nccl`, `ipc`, `sparse_nccl` (sparse patches), `sharded_rdt` (NIXL or Ray Direct Transport, pull-based, for large EP MoE), `nccl_m2n` (trainer and inference sharded differently), and ModelExpress (version-based, including object-store checkpoints and deltas).
- Trainer side calls one `send_weights()`; every trainer rank builds an engine, rank 0 sends. Default packed buffer ≈1 GB per bucket, double-buffered ([HF survey](https://huggingface.co/blog/async-rl-training-landscape)).
- **Failure hazard**: a failed `/update_weights` leaves the model mid-reload while `/health` still returns 200; `/finish_weight_update` is rejected; the full start→update→finish sequence must be resent. Health checks alone cannot tell a serving tool that a replica is unusable.
- The API has churned (v0.18 used a single `/update_weights`), so pin engine versions.
- Validated at scale: prime-rl, fully async, GLM-5.1-FP8 inference on 16 nodes x 8 H200 (two 4P+4D replicas, DPEP32, 1 TB CPU KV offload per node, cache-aware `vllm-router`) and BF16 trainer on another 16 x 8 H200, stable 100+ steps ([vLLM blog](https://vllm.ai/blog/2026-05-28-native-rl-apis)).

LoRA path: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`, `POST /v1/load_lora_adapter` with `lora_name`, `lora_path`, and `load_inplace=true` to replace an adapter under the same name; `/v1/unload_lora_adapter`; `--max-loras`, `--max-lora-rank` set at startup ([vLLM LoRA docs](https://docs.vllm.ai/en/v0.20.2/features/lora/)).

### 2.3 SGLang API surface

([SGLang for RL](https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md))

- `POST /update_weights_from_disk` — `model_path`, optional `abort_all_requests`, `weight_version`, `flush_cache` (default true), `keep_pause`, `recapture_cuda_graph`, `torch_empty_cache`.
- `POST /update_weights_from_tensor` — `serialized_named_tensors` (per-TP payloads via `MultiprocessingSerializer.serialize`), `load_format` `direct | flattened_bucket`; colocated only.
- `POST /init_weights_update_group` (`master_address`, `master_port`, `rank_offset`, `world_size`, `group_name` default `weight_update_group`, `backend` default `nccl`) → `POST /update_weights_from_distributed` (`names`, `dtypes`, `shapes`) → `POST /destroy_weights_update_group`.
- `POST /begin_weight_update` / `/end_weight_update` session; out-of-session calls return HTTP 400 and leave weights unchanged.
- **Per-token weight versions**: `meta_info.weight_versions` is a list of `{version, start, end}` half-open spans over output tokens; `--enable-prefill-weight-versions` adds prompt spans. This is the only engine-level provenance for mixed-policy trajectories I found; Dynamo's shared response does not attach a version per token and its router does not select workers by weight version ([Dynamo](https://docs.nvidia.com/dynamo/dev/reinforcement-learning/rollout-weight-updates)).
- `--enable-deterministic-inference` reduces batch-shape variance for true on-policy training.

### 2.4 Fleet-level semantics

Dynamo's guidance is the most explicit statement that an engine API is not a fleet protocol ([Dynamo rollout weight updates](https://docs.nvidia.com/dynamo/dev/reinforcement-learning/rollout-weight-updates)): the framework must gate new rollouts, resolve worker membership, pause, transfer, invalidate KV state on every tier (GPU, host, disk, shared), verify the version on every worker, and run a post-update generation before reopening. "Per-worker success is not fleet-wide atomicity." `weight_version` is caller-supplied metadata, not a tensor digest. A group-initialisation timeout must be treated as worker failure because the backend may stay blocked. A replacement worker can pull a version from a peer (inference-to-inference via ModelExpress), but the framework decides which version and when it joins.

---

## 3. In-flight rollouts across a weight update

| Strategy | Engine primitive | Used by | Cost |
|---|---|---|---|
| Drain: wait for all requests | vLLM `/pause` mode `wait` | sync designs, Dynamo example | Idle time bounded by longest sequence; no async |
| Abort and retry / recycle | vLLM mode `abort`; SGLang `pause_generation` mode `abort` | slime, miles, SkyRL (retries with generated prefix) | Client retry; wasted or re-prefilled tokens |
| Retract to queue | SGLang mode `retract` (running requests go back to waiting; KV may be flushed and recomputed) | SGLang-based stacks | Recompute prefix under new weights |
| Freeze and continue | vLLM mode `keep` (scheduler stops, state preserved); SGLang `in_place` (a later `flush_cache` fails) | AReaL-style, PipelineRL, NeMo RL | Mixed-policy trajectory; choose to keep or discard KV |
| Save and resume | verl `partial_rollout=True` with `staleness_threshold>0` | verl | Framework-side bookkeeping |
| Cancel the whole group | orchestrator | prime-rl | Lost work, clean on-policy groups |

Sources: [vLLM blog](https://vllm.ai/blog/2026-05-28-native-rl-apis), [SGLang docs](https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md), [HF survey](https://huggingface.co/blog/async-rl-training-landscape), [verl docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html).

KV cache: entries computed under old weights are invalid under new ones even for identical tokens ([Dynamo](https://docs.nvidia.com/dynamo/dev/reinforcement-learning/rollout-weight-updates)). PipelineRL found that reusing the stale cache for mixed-policy sequences only slightly increased divergence, and that recomputing is fast but lowers throughput ([arXiv 2509.19128](https://arxiv.org/abs/2509.19128v2)); the Motif 3 report retains KV across refits, bounds trajectory age to one optimizer step and truncates importance weights to [0.2, 5.0] ([arXiv 2608.09119](https://arxiv.org/pdf/2608.09119), via search summary). vLLM had a deadlock when pausing data-parallel + expert-parallel deployments; the fix moved pause into `EngineCore` with a local-then-global pause agreed by an all-reduce every 32 steps ([vLLM blog](https://vllm.ai/blog/2026-05-28-native-rl-apis)).

Staleness bounds in practice:

| System | Knob | Value |
|---|---|---|
| prime-rl | `max_async_level` (inference may run k steps ahead of trainer) | default 2 ([docs via search](https://docs.primeintellect.ai/prime-rl/overview)) |
| INTELLECT-2 | async steps | 2 in production; up to 4 matched sync baseline on a 1.5B ablation ([paper](https://arxiv.org/html/2505.07291)) |
| open-instruct | `async_steps` | default 1, production 8 ([HF survey](https://huggingface.co/blog/async-rl-training-landscape)) |
| verl | `staleness_threshold` (fraction of extra stale samples), `trigger_parameter_sync_step` | 0.1-0.5; sync every 4 local updates ([verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html), [AMD](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html)) |
| AReaL | depth bound + decoupled loss | IS cap 5.0 ([HF survey](https://huggingface.co/blog/async-rl-training-landscape)) |
| Composer 2 | cross-region delta chain | "minutes-level" ([Fireworks](https://fireworks.ai/customers/cursor-composer-2-reinforcement-learning)) |

verl's ablation shows zero staleness is not the best point: at 128 GPUs, `staleness_threshold` 0 gave 0.2844 max accuracy in 1d 1h 53m; 0.1 gave 0.3542 in 19h 59m.

---

## 4. Inference:training GPU ratio

| Run | Inference : training | Notes |
|---|---|---|
| TRL server mode | 1 : 7 | Short single-turn completions ([HF](https://huggingface.co/blog/vllm-colocate)) |
| prime-rl 8-GPU example | 6 : 2 | `--deployment.num-infer-gpus 6 --deployment.num-train-gpus 2 --inference.vllm.data-parallel-size 6` ([scaling](https://docs.primeintellect.ai/prime-rl/scaling)) |
| prime-rl Nemotron-3-Super 120B SWE | 1 node : 4 nodes | Trainer-heavy for a large MoE |
| prime-rl GLM-5.1 | 16 : 16 nodes (8x H200 each) | ([vLLM blog](https://vllm.ai/blog/2026-05-28-native-rl-apis)) |
| verl 7B | 1 : 1 (16:16 … 64:64) | ([verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html)) |
| verl Qwen3-30B-A3B | "96:32", described as skewed and far from optimal | Which side is 96 is not stated in what I read (unverified) |
| miles 30B SWE agent | 3 nodes : 1 node (24 SGLang engines) | Sandbox-bound, not GPU-bound ([miles](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl)) |
| INTELLECT-2 32B | ≈4 : 1 compute | ([paper §5](https://arxiv.org/html/2505.07291)) |

There is no universal ratio: it runs from 1:7 to 4:1 and is driven by tokens generated per trained token (rollout length, group size G, turns), trainer memory tricks, and whether sandboxes gate generation. Reference throughput for sizing: ~6,300 output tok/s for a 7B and ~1,200 tok/s for a 32B per H100 (bf16, offline); a 512-rollout batch at 32K tokens takes ~3.7 h on one GPU for 32B, ~28 min on eight ([HF survey](https://huggingface.co/blog/async-rl-training-landscape)).

Signals used to tune it:

- **Trainer idle ratio** (verl async on MI355X: 6-9% idle, 91-94% busy at 4:4) ([AMD](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html)).
- **Step-time decomposition** (miles: 63% actor update, 29% waiting for rollouts, 8% logprob recompute, sync 23 s).
- **Engine saturation**: KV-cache utilisation and queue depth (miles: 23% KV, empty queue ⇒ add sandboxes, not GPUs).
- **Buffer depth versus staleness bound**: if rollouts are discarded as stale, there is too much inference; if the trainer waits, too little. prime-rl notes k=1 with equal step timings gives zero idle on both sides.

---

## 5. Throughput-oriented serving for rollouts

RL inference differs from API serving in what is requested and what is optimised:

- **Token-in, token-out.** Trainers need exact prompt token IDs, sampled token IDs and per-token logprobs; re-tokenising chat text drifts. prime-rl uses vLLM's `/inference/v1/generate` via its `renderers` package ([overview](https://docs.primeintellect.ai/prime-rl/overview.md)); vLLM ≥0.10.2 supports `return_token_ids=True` ([Strands docs](https://strandsagents.com/docs/community/model-providers/vllm/)); NeMo Gym toggles `return_token_id_information` ([NeMo Gym](https://docs.nvidia.com/nemo/gym/latest/model-server/vllm.md)); Dynamo accepts a token-ID array as `prompt` with `nvext.extra_fields: ["completion_token_ids"]` and `nvext.prompt_logprobs` ([Dynamo](https://docs.nvidia.com/dynamo/dev/reinforcement-learning/rl-integration-reference)). Before admitting a sample, check token/logprob count alignment, terminal state, tokenizer/model match.
- **Logprob fidelity matters.** miles tracks trainer-versus-inference logprob gap (~0.011) and still recomputes logprobs on the trainer (8% of step). FP8 inference with BF16 training is used in production (prime-rl GLM-5.1), with re-quantisation in `finish_weight_update`. For MoE, Cursor replays the inference router's expert choices in training and filters implausible samples ([Composer 2](https://arxiv.org/html/2603.24477)). Heterogeneous actor GPUs add numeric variance; INTELLECT-2's TOPLOC hashes exist partly to detect precision changes (~1% throughput overhead).
- **Throughput over latency.** Goal is tokens/GPU/s at high concurrency; TTFT is irrelevant. Published per-GPU rollout figures: ~1,050 (GRPO, VL-7B) to ~2,994 (DAPO, Math-7B) tok/GPU/s async on MI355X; sync GRPO Qwen2-7B 3,830 (MI355X) vs 3,595 (B300) ([AMD](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html)). Engine flags (max concurrent sequences, GPU memory fraction, prefix caching) are covered in `10-inference-engines.md`; I did not find an RL-specific tuning guide (unverified which values labs use).
- **Sampling.** Temperature ≈1 sampling with G samples per prompt (4-16 in the configs above); `n`-way sampling shares the prompt prefix, so prefix caching and cache-aware/sticky routing matter (prime-rl uses `vllm-router` with sticky routing; SGLang Model Gateway dispatches per turn).
- **Long, heavy-tailed generations.** Cap context: uncapped trajectories reached 113K-186K tokens, filled the KV cache and timed out queued requests; a 64K cap and a turn cap gave ~3x faster steps ([miles](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl)). In multi-agent chains a 5x p90 per agent compounds to ~25x ([HF survey](https://huggingface.co/blog/async-rl-training-landscape)).
- **Client-side caps bite first.** miles hit, in order: an HTTP pool capped at 100, a single proxy process serialising LLM calls, ~31 Docker networks per host, a 100 pulls/hour image limit, and one CPU core handling all agent requests.
- **Topology.** Per-node independent replicas (TP/DP inside a node) behind one router is the norm; miles ran 8 single-GPU engines per node at 16 trajectories each.

Synthetic data generation and large evals are the degenerate case of this section: same throughput-oriented fleet, no weight sync, no staleness, fully preemptible, embarrassingly parallel. Online evals inside RL are not free, though: Composer 2 acquires an "evaluation deployment lease", moves GPUs to it and does a cross-region weight sync per evaluated step.

---

## 6. Environment and sandbox fleets

| System | Scale | Mechanics |
|---|---|---|
| Cursor Anyrun | hundreds of thousands of pods per cluster; >500 pods/s scheduled per cluster | Firecracker VMs on x86 + ARM; scheduling on live CPU/memory/disk pressure; fork on same node else live-migrate; egress via TCP-level proxy with injected root CA; rollout-level memory snapshots ([Composer 2](https://arxiv.org/html/2603.24477)) |
| DeepSeek DSec | ~160 nodes, ~30K cores; ~3M sandboxes/day, ~380K concurrent, >5,000 creations/s; up to 3,200 containers or 800 microVMs per node | Bursts up to 32K sandboxes per job; power-of-k placement; sandboxes sit **outside the preemptible GPU pool** so a preempted trainer reconnects without replay; on-demand image loading; overflow to cloud VMs above 80% utilisation ([arXiv 2609.22978](https://www.alphaxiv.org/abs/2609.22978.md)) |
| GKE Agent Sandbox | >30,000 sandboxes per cluster; worst-case wait 7.5 min → <10 s | ([Google Cloud blog](https://cloud.google.com/blog/products/containers-kubernetes/accelerate-agentic-rl-with-gke-agent-sandbox), vendor claim) |
| miles + Harbor | 384 concurrent | E2B microVMs (2 vCPU, 4 GB): ready 1.9 s median / 3.7 s p99 versus 11.7 s local Docker under load; tunnel adds ~0.26 s per LLM call; self-hosted E2B start-up ≈20% of trajectory time ([miles](https://www.lmsys.org/blog/2026-10-22-miles-mi355x-agentic-rl)) |
| CoreWeave Sandboxes, Modal, E2B, Prime Sandboxes | "thousands per step" | Hosted; vendor claims ([CoreWeave](https://coreweave.com/blog/run-agentic-workloads-safely-at-scale-with-coreweave-sandboxes), [Modal](https://modal.com/resources/best-sandboxes-rl-environments)) |

Consequences: the sandbox fleet is a CPU/RAM fleet with its own image cache, network policy and burst scheduler, on the critical path of GPU utilisation. Infrastructure errors (3.1% of trajectories in miles) score reward 0 unless filtered, so sandbox failure is a *training-signal* bug, not just an availability bug. Warm pools and snapshots matter more than cores.

---

## 7. Backpressure, fault tolerance, elasticity

**Buffering/backpressure.** Pull-based trainers (prime-rl ZMQ request-per-step; verl `require_batches * ppo_mini_batch_size` from a MessageQueue; Atropos trajectory API; Agent Lightning store) plus a staleness bound form the control loop: the bound caps how far producers may run ahead, and over-production is discarded, not queued forever. Cursor uses a central reconciler with slot-based sample lifecycle and the Ray object store spilling to local NVMe.

**Fault asymmetry.** Actors are stateless given a policy version; losing one costs its in-flight rollouts. The learner holds optimizer state and is a gang-scheduled process group. Published practice:

- Cursor: three checkpoint tiers — rollout-level (environment memory snapshot), group-level (advantage-tagged sequences with policy versions on NFS), step-level (weights); warm standby nodes; passive + active health checks; drain-and-replace for live code updates.
- INTELLECT-2: heartbeats in Redis with expiry; workers join/leave freely and poll a step-counter endpoint; relays keep the last 5 checkpoints; checksum mismatch → skip to next checkpoint, not retry.
- miles: watchdog auto-resume; 3 checkpoint resumes in a 145-step run; a cross-node NCCL sync hang caused by a duplicate Docker IP, fixed by pinning the NCCL interface.
- DSec: sandboxes pause when their job is preempted and resume transparently on the next request.
- A failed weight update is a silent failure mode (health 200, model unusable) — see §2.2.

**Elasticity and heterogeneity.** INTELLECT-2 is the existence proof that actors can be permissionless, heterogeneous, consumer-grade, WAN-connected machines at ~4:1 inference:training compute, with 2-step async and ~14 min broadcasts; the price is verification (TOPLOC) and policy lag of tens of minutes. Composer 2 is the industrial version: inference in US and EU clusters with *no direct connection to the training cluster*, joined only by an S3 delta chain, and inference capacity shared with production traffic. DSec keeps CPU sandboxes off the preemptible pool. I found no public numbers on spot-preemption rates for RL actors specifically (see Open questions).

---

## Implications for grid

1. **Model an RL job as typed pools, not one gang.** At minimum: `learner` (gang-scheduled, RDMA/NCCL-capable, reliable, checkpointed — the `01`-`05` training path), `actor` (N independent engine replicas, any GPU, preemptible, individually replaceable — the `10`/`11` serving path), `env` (CPU/RAM nodes or an external sandbox provider), and a small `control` node (orchestrator, buffer, router, version registry). Each pool needs its own reliability class, network class and autoscaling policy.
2. **Provide a weight-distribution plane with two tiers.** Tier A: in-cluster push — grid must place actors that will use NCCL/NIXL on the same fabric as the learner, open the rendezvous ports, pin `NCCL_SOCKET_IFNAME`-style interface selection, and hand the framework `master_address`/`master_port`/`rank_offset`/`world_size`. Tier B: store-and-pull — an object-store bucket or relay tree with content-addressed, immutable version IDs, delta support, checksum verification and retention (keep last K). Tier B is the only option for cross-provider, NAT'd or consumer actors, and is what the largest published system uses.
3. **Own fleet-level update orchestration or expose the hooks for it.** Per replica: gate → pause(mode) → transfer → finish → invalidate KV → verify version → probe generation → reopen. Track `policy_version` per replica as first-class state; never infer readiness from `/health` during RL. Support mixed-version fleets deliberately (async) and expose version in routing so an orchestrator can pin a rollout group to one version.
4. **Late joiners and replacements need a "current policy" bootstrap** that is not the base checkpoint: pull latest version from store or a peer before admitting traffic. This is where fast bootstrap (`09`) meets RL: time-to-first-token for a new actor includes the delta chain replay.
5. **Engine control endpoints are privileged.** `VLLM_SERVER_DEV_MODE=1`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, SGLang update routes must be reachable from the learner/orchestrator and not from anything else; grid's network model needs a private control plane distinct from the generation data plane.
6. **Support role flips.** For small clusters and for the tail of a job, the same GPUs should switch between actor and learner (sleep level 2 / memory-saver, or process restart). Expose it as a pool-resize operation with drain semantics; default to dedicated pools above roughly one node, given the 1.6-2.8x async advantage.
7. **Ship a ratio controller, or at least the metrics for one.** Export trainer idle fraction, wait-for-rollouts time, sync time, engine KV utilisation, queue depth, stale-discard rate, sandbox wait time. Start from a ratio derived from tokens-generated-per-trained-token and correct online; let the user set the staleness bound and let the controller scale actors (cheap, heterogeneous) around a fixed learner.
8. **Treat sandboxes as a third fleet.** Either provision CPU nodes with image caching, shared Docker network, raised pull limits and warm pools, or integrate external providers (E2B, Modal, Prime Sandboxes, CoreWeave) and account for tunnel latency (~0.26 s/call). Keep sandbox state off preemptible GPU nodes.
9. **Heterogeneity is acceptable on actors with guardrails.** Mixed GPU types change throughput per replica (router must weight by capacity) and numerics (FP8 vs BF16, kernel differences); grid should record engine build, dtype and GPU model per replica so the trainer can filter or importance-correct, and should homogenise at least within a rollout group.
10. **Checkpoint tiers.** Learner step checkpoints (weights + optimizer) on durable storage; the rollout buffer persisted (filesystem transport or NVMe spill) so a learner restart does not discard hours of generation; rollouts tagged with policy version so they can be reused or rejected after resume.
11. **Synthetic data and evals reuse the actor pool** with weight sync disabled; the same tool should launch "actors only" as a batch-generation job on the cheapest capacity.

## Open questions

- **Spot/preemptible actors**: no public preemption-rate or cost data for RL actor fleets on marketplaces; how fast must an actor rejoin for it to be worth it, given delta-chain replay time? Needs a measurement sweep on RunPod/Vast (cross-reference `07`, `11`).
- **NCCL weight broadcast across heterogeneous or TCP-only networks**: all sub-second numbers are from homogeneous RDMA/NVLink clusters. What does `nccl` or `sharded_rdt` cost over 10-25 GbE TCP between marketplace nodes, and at what model size does object-store delta win? (See `05`.)
- **Delta compression details**: Cursor reports ≥98% reduction and "a handful of gigabytes" for 1T params but not the codec or reconstruction time; vLLM's `sparse_nccl` and ModelExpress XOR-delta formats need a source read.
- **prime-rl async internals**: the dedicated async page 404'd; `max_async_level` default and group-cancellation behaviour are from search summaries and the HF survey, and docs disagree on the default rollout transport (ZMQ versus filesystem). Read the repo.
- **verl 30B "96:32" split direction**, and whether verl has separated vLLM from the trainer process in ≥0.7.
- **Throughput-tuning values** (max concurrent sequences, KV fraction, prefix cache, speculative features) that RL frameworks set by default — not found in docs; read framework launch code.
- **Tinker / Fireworks / W&B Serverless RL internals**: multi-tenant LoRA multiplexing and hot-load latency are undocumented.
- **LoRA at scale**: adapter-only sync is nearly free for dense models, but a 64-expert MoE needs 200+ adapters; no measurements of multi-adapter hot-swap latency in vLLM under load.
- **Lab infrastructure beyond Cursor, Moonshot, DeepSeek, Prime Intellect**: nothing concrete found from OpenAI, Anthropic, Google or Meta on actor:learner ratios or staleness; Motif 3 and the Kimi K2 paper deserve a full read.
- **Atropos maintenance status**, ART `LocalBackend` config key names, and NeMo RL's current sync defaults were only seen via secondary sources.
- **Sandbox cost model**: CPU-core-hours per GPU-hour for agentic RL is not published anywhere I found; DSec's ~30K cores per scale unit and miles' 2 vCPU/4 GB per trajectory are the only anchors.
