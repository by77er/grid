# 08 - Training algorithms for heterogeneous, slow or unreliable clusters

Research sweep date: 2026-11-09. Sibling files referenced by name: `03-rank-configuration.md`
(process groups), `04-failure-taxonomy-recovery.md` (detection/recovery), `05-networking-interconnects.md`
(transports), `06-gpu-hardware-nuances.md` (mixed GPUs), `07-provider-landscape.md` (what links
marketplace providers actually give you).

## Summary

- Fully synchronous data parallelism (DDP/FSDP) is the quality reference but is a lock-step system:
  every step waits for the slowest rank and any rank loss stops the job. It needs on the order of
  100-400 Gbit/s between replicas to keep GPUs busy at 1B-100B parameters (simulated; see section 4).
  It is only appropriate inside an "island" with a fast, uniform fabric.
- The one family with LLM-scale evidence for crossing slow links is **local-update training with an
  outer optimizer (DiLoCo and descendants)**. Workers run H inner AdamW steps independently, then
  exchange a parameter delta ("pseudo-gradient" / outer gradient) that is fed to an outer Nesterov
  optimizer. Published results now go to 10B parameters with eval loss within about +/-0.3% of data
  parallel for M <= 4 replicas, and bandwidth needs of roughly 1-10 Gbit/s (plain) down to
  0.1-2 Gbit/s (streaming + overlap + 4-bit quantisation).
- 2026 moved the state of the art from "low communication" to "decoupled": Google's Decoupled DiLoCo
  (April 2026) makes the outer loop asynchronous with a quorum, a grace window and token-weighted
  merging, and demonstrates mixed accelerator generations and failure injection. This is the closest
  published design to what `grid` needs across islands.
- Quality cost is real and grows with the number of replicas M, shrinks with model size, and is
  under-characterised on downstream evals (one independent replication found worse MMLU/GSM8K/HumanEval
  at equal pretraining loss). Treat M as a budget: few large islands beat many small ones.
- Classical asynchronous SGD, gossip SGD, elastic averaging and heavy per-step gradient compression
  have little or no evidence at LLM pretraining scale; they matter mostly as ingredients (error
  feedback, weighting by work done, delayed momentum, pairwise averaging).
- Open implementations exist but are young: Prime Intellect `prime`/PCCL (MIT), OpenDiLoCo on hivemind,
  Nous Psyche (Apache-2.0, Rust), Gensyn NoLoCo, exo gym (simulator). None is a drop-in "torchrun" flag.
  Google's Streaming/Decoupled DiLoCo have no released production code (unverified: none found).

Recommended default for `grid`: **hierarchical** - synchronous FSDP inside each homogeneous island,
DiLoCo-style outer loop across islands, with the orchestrator owning island grouping, outer-step
coordination and membership. Section 7 gives the decision rules.

---

## 1. Baseline: synchronous data parallelism and the straggler problem

Mechanism: each rank computes gradients on its shard of the global batch; an all-reduce
(NCCL/Gloo/MPI) averages them every step; all ranks apply the identical update. DDP and FSDP both
require gang scheduling and lock-step execution, so step time is `max_i(t_i) + t_allreduce` and one
slow or dead rank blocks everyone. PyTorch's own write-up states that with stragglers "all the
processes have to wait" before synchronising, and proposes hierarchical SGD (sync small groups every
step, global sync less often) so a random straggler slows only its group
(https://pytorch.org/blog/straggler-mitigation/). GPUs blocked in all-reduce still report ~100%
utilisation, so `nvidia-smi` does not reveal the straggler
(https://discuss.pytorch.org/t/distributed-training-is-even-slower-sometimes/77927); per-rank step
timing is required (cross-ref `04-failure-taxonomy-recovery.md`).

Communication volume: one full gradient per step per replica group. Rule of thumb: bytes per step is
about `N_params x bytes_per_value` (2 bytes at bf16, 4 at fp32) and must complete in a small fraction
of the step time. Google's simulator puts the bandwidth needed for 80% compute utilisation at
152.6 Gbit/s (1B params, 0.1 s step), 222.3 Gbit/s (10B, 0.8 s) and 323.8 Gbit/s (100B, 4.9 s)
(Streaming DiLoCo Table 4, https://arxiv.org/html/2501.18512v1). This is why sync DP stops at the
edge of an RDMA/400GbE fabric.

It does work across sites when the operator owns the WAN. Google states Gemini Ultra 1.0 was trained
across multiple datacenters with "model parallelism inside superpods and data parallelism across
them" using synchronous training (https://arxiv.org/pdf/2312.11805), and Gemini 2.5 used synchronous
data-parallel training over multiple 8960-chip TPUv5p pods across datacenters
(https://arxiv.org/pdf/2507.06261). Microsoft links Fairwater Wisconsin and Atlanta (~700 miles) over
a dedicated "AI WAN" as one "AI superfactory"
(https://news.microsoft.com/source/features/ai/from-wisconsin-to-atlanta-microsoft-connects-datacenters-to-build-its-first-ai-superfactory/);
whether single pretraining runs span both sites synchronously is not stated in primary material
(unverified). None of this transfers to marketplace providers, which offer no private high-bandwidth
WAN between regions (cross-ref `07-provider-landscape.md`).

### 1.1 Straggler mitigation that keeps synchronous semantics

| Technique | Mechanism | Evidence / status |
|---|---|---|
| Backup workers | Launch N+b workers, take the first N gradients each step, discard the rest. Avoids async noise while cutting tail latency. | Chen et al. 2016, "Revisiting Distributed Synchronous SGD" (https://arxiv.org/abs/1604.00981). Parameter-server era (TensorFlow `SyncReplicasOptimizer`); not available in NCCL all-reduce DDP/FSDP because collectives need every rank. Wastes b/N of compute. |
| Heterogeneity-aware batch sizing (LB-BSP) | Tune each worker's batch size at sync barriers so all finish together; aggregate gradients weighted by batch size. On GPUs the time-vs-batch relation is non-linear, so tuning is iterative. | Chen et al., SoCC'20 (https://arxiv.org/abs/1806.02508): up to 54% faster in a 16-node heterogeneous GPU cluster. |
| Dynamic batching (PID) | Proportional controller adjusts per-worker mini-batch to equalise iteration time. | "Taming Resource Heterogeneity in Distributed ML Training with Dynamic Batching" (https://arxiv.org/abs/2305.12213). |
| Cannikin | Learns a cluster performance model online and predicts the optimal per-node batch for a new total batch without iterative search. | https://arxiv.org/html/2402.05302v1 |
| Poplar | Per-GPU batch-size search per ZeRO stage; argues FLOPs-proportional allocation mispredicts real speed. Six GPU types tested. | AAAI'25, https://arxiv.org/abs/2408.12596 |
| Cephalo | Splits global batch by capability and adds gradient accumulation to decouple memory from compute; reports 1.2x-10.8x throughput vs prior methods. | ICS'25, https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-58.pdf |
| Hierarchical / grouped sync | Sync every step in small groups, globally every k steps. | PyTorch blog above; precursor of the island design. |

Key correctness detail for uneven batches: the averaged gradient must be weighted by samples (or
tokens) per rank, otherwise small-batch ranks are over-weighted. With FSDP (sharded parameters) the
gradient reduction is a reduce-scatter inside the framework, so weighting must be applied to the loss
(scale each rank's loss by `local_tokens / global_tokens x world_size`) rather than at the collective.
All of these fix *speed* heterogeneity only; none tolerates node loss, and none helps with slow links.
Mixed-memory ranks additionally constrain the maximum per-rank micro-batch (cross-ref
`06-gpu-hardware-nuances.md`).

---

## 2. Classical alternatives (pre-LLM lineage)

The characterisations in this section are summarised from the cited papers; the arXiv pages were not
re-fetched in this sweep except where a number is quoted with its own source elsewhere.

| Family | Mechanism | Comm. volume / frequency | Node loss | Speed heterogeneity | LLM-scale evidence |
|---|---|---|---|---|---|
| Async SGD (Hogwild, parameter server) | Workers push gradients computed on stale parameters whenever ready. Hogwild: https://arxiv.org/abs/1106.5730 | Full gradient per worker step | Excellent (workers are independent) | Excellent in throughput, but slow workers produce staler gradients | None for transformer pretraining; staleness plus adaptive optimisers/momentum is unstable |
| Staleness-aware async | Scale LR by 1/staleness (https://arxiv.org/abs/1511.05950); delay compensation DC-ASGD (https://arxiv.org/abs/1609.08326) | Same | Same | Better | None at scale |
| Local SGD / FedAvg | H local steps, then average parameters. FedAvg: https://arxiv.org/abs/1602.05629; analysis: https://arxiv.org/abs/1805.09767 | Full model every H steps | Easy to drop a worker at a round boundary | A slow worker delays the round unless H_i varies | Yes, via DiLoCo (section 3) |
| Post-local SGD | Sync large-batch first, switch to local SGD after the first LR decay; improves generalisation. Lin et al. https://arxiv.org/abs/1808.07217 | As local SGD in phase 2 | As local SGD | As local SGD | Vision only; DiLoCo's "pretrain 24k steps then DiLoCo" is the same idea, and from-scratch cost only ~0.1 PPL (https://arxiv.org/html/2311.08105v3) |
| Elastic averaging (EASGD) | Each worker is pulled toward a centre variable by an elastic term; centre moves toward workers. https://arxiv.org/abs/1412.6651 | Full model every tau steps to a centre | Good (centre survives) | Good (async variant) | None |
| Slow-momentum / Lookahead | Outer momentum on averaged local iterates (SlowMo https://arxiv.org/abs/1910.00643; Lookahead https://arxiv.org/abs/1907.08610) | As local SGD | - | - | Direct ancestors of DiLoCo's outer Nesterov; DiLoCo with M=1 is Lookahead-like and beats plain data parallel (section 3) |
| Decentralised gossip: D-PSGD, SGP, AD-PSGD | No all-reduce; each node averages with neighbours. SGP uses push-sum on directed graphs (https://arxiv.org/abs/1811.10792); AD-PSGD is asynchronous pairwise averaging (https://arxiv.org/abs/1710.06952) | Full model to 1-2 peers per step | Good; no global barrier | AD-PSGD removes the barrier entirely | Small; replicas never agree exactly (consensus error). Revived for LLMs as NoLoCo and GASLoC (section 5) |
| Moshpit SGD | Averaging in small dynamic groups arranged so information mixes in few rounds; DHT-based matchmaking. https://arxiv.org/abs/2103.03239 | Group all-reduce | Designed for churn | Yes | ALBERT-scale; implemented in hivemind |

### 2.1 Gradient compression and quantisation

| Method | What is sent | Typical ratio | Notes |
|---|---|---|---|
| 1-bit SGD (Seide et al. 2014) / signSGD (https://arxiv.org/abs/1802.04434) | Sign per coordinate; majority vote aggregation | 32x vs fp32 | Needs error feedback to converge in general; majority vote is not a sum, so it does not compose with NCCL all-reduce |
| QSGD (https://arxiv.org/abs/1610.02132), TernGrad (https://arxiv.org/abs/1705.09878) | Stochastic low-bit quantisation | 4-16x | Unbiased |
| Top-k with memory (https://arxiv.org/abs/1809.07599), Deep Gradient Compression (https://arxiv.org/abs/1712.01887) | k largest coordinates + indices, residual kept locally (error feedback) | 100-1000x claimed on CNN/RNN | Sparse payloads are not all-reducible; need gather + scatter-add |
| PowerSGD (https://arxiv.org/abs/1905.13727) | Rank-r factors via one power-iteration step, with error feedback | ~100x+ | Linear, so it *is* all-reduce compatible |
| DeMo (https://arxiv.org/abs/2411.19870) | Per-worker decoupled momentum, DCT then top-k of the fast components, momentum buffer acts as error feedback | Up to 85x less data per GPU than AdamW-DDP at 300M and 1B params | Per-step method; basis of Nous DisTrO/Psyche |

What matters at LLM scale: compressing the **per-step gradient** has little adoption because inside a
datacenter the all-reduce is not the bottleneck and compression costs GPU time. Compressing the
**infrequent outer gradient** is where it pays. Evidence there is consistent on one point and mixed on
another:

- Low-precision quantisation of outer gradients is nearly free: fp16 with no degradation in OpenDiLoCo
  (https://arxiv.org/abs/2407.07852); int8 with a mu +/- 6 sigma range and 256 buckets (codebook = bucket
  mean, reduction in fp32) in INTELLECT-1 (https://arxiv.org/html/2412.01152v1); 4-bit float E3M0 with
  fp32 accumulation in Streaming DiLoCo (https://arxiv.org/html/2501.18512v1).
- Dropping values is contested. Streaming DiLoCo reports top-k/FedDropout/DARE on outer gradients
  hurt while 4-bit precision did not; original DiLoCo found sign-based pruning of 50% cost +0.39% PPL
  and 75% cost +1.66% (https://arxiv.org/html/2311.08105v3). MuLoCo (Muon as inner optimiser) and
  SparseLoCo report that top-k plus quantisation works *if* error feedback is used, with MuLoCo
  citing 8x less data at 2-bit quantisation and experiments to 3.1B parameters
  (https://arxiv.org/abs/2505.23725; https://alphaxiv.org/abs/2508.15706 - details from secondary
  summaries, unverified).

---

## 3. DiLoCo family: the LLM-scale evidence

### 3.1 Algorithm

```
theta_global, outer_opt = Nesterov(lr=0.7, momentum=0.9)
repeat:
    for each replica m in parallel:            # replica = one island running sync DP/FSDP
        theta_m = theta_global
        run H AdamW steps on shard m           # inner optimiser state stays local, never synced
        delta_m = theta_global - theta_m       # outer gradient / pseudo-gradient
    delta = weighted_mean(delta_m)             # the only cross-replica communication
    theta_global = outer_opt.step(theta_global, delta)
```

Outer SGD with lr 1 is FedAvg; Nesterov was empirically best, and outer Adam was unstable. Inner
optimiser states are not exchanged (it would add ~3x payload)
(https://arxiv.org/html/2311.08105v3). Extra memory: a copy of the last global parameters plus outer
momentum per replica; Streaming DiLoCo quotes ~5/3x parameter memory overall, held on CPU in
practice (https://arxiv.org/html/2501.18512v1; INTELLECT-1 does the outer step on CPU, 5-10 s).

### 3.2 Published results

| Work | Scale | H | M (replicas) | Quality vs sync data-parallel | Communication | Source |
|---|---|---|---|---|---|---|
| DiLoCo (DeepMind, 2023) | 60M-400M, C4 | 500 default (50-2000 tested) | 8 default (4-64) | 150M: 15.02 PPL vs 15.30 for 8x-batch DP; H=1000 costs +2.9% PPL vs H=50 | 500x fewer sync points | https://arxiv.org/html/2311.08105v3 |
| Scaling laws for DiLoCo (NeurIPS 2025) | 35M-2.4B fitted, 4B/10B validated | 30 (1-300 ablated) | 1, 2, 4, 8 | 2.4B: M=2 -0.1%, M=4 +0.3%, M=8 +1.1% eval loss; 10B: M=1/2 -0.19%, M=4 +0.29% | >100x less total; 10B needs ~184 Gbit/s (DP) vs 6.2 (H=100) or 2.0 (H=300) for 80% utilisation | https://arxiv.org/html/2503.09799v1 |
| Streaming DiLoCo (Jan 2025) | 35M-4B; 1B on Dolma to 250B tokens | 30-100 | 2-4 | 1B/C4: 2.48 vs 2.49 loss; Dolma 250B tokens: 2.45 vs 2.45 | ~400x fewer bits; see table below | https://arxiv.org/html/2501.18512v1 |
| Eager updates (Feb 2025) | 35M-1B | 30-500 | 2+ | 500M: 2.69 (eager) vs 2.68 (blocking) vs 3.01 (naive delayed) | Sync may take a full inner phase | https://arxiv.org/html/2502.12996v1 |
| OpenDiLoCo (Prime Intellect, 2024) | to ~1B | 500 | 4-8 | Replicates DiLoCo | 90-95% compute utilisation across 3 countries; fp16 all-reduce | https://arxiv.org/abs/2407.07852 |
| INTELLECT-1 (Prime Intellect, 2024) | 10B, 1T tokens | 100 (~38 min on 8xH100) | up to 14 nodes (112 H100), 30 providers | No same-recipe centralised loss baseline reported | int8, "400x"; all-reduce median 103 s (US), 382 s (US+EU), 469 s (global) | https://arxiv.org/html/2412.01152v1 |
| Decoupled DiLoCo (Google, Apr 2026) | 2B-9B dense, 2.8B-active MoE, up to 1T tokens | 24 per fragment, P=24 fragments, tau=2 | 8 | "Matches" DP at 2B/5B/9B on text+vision evals | "two orders of magnitude less bandwidth" | https://arxiv.org/html/2604.21428v1 |

Robust findings across these:

1. **The penalty scales with M and shrinks with model size.** At 35M parameters M=8 costs +3.9% eval
   loss; at 2.4B it costs +1.1%. M=1 (no communication saving, just the outer momentum) is 0.1-0.4%
   *better* than data parallel (scaling-laws paper). Diminishing returns beyond 8 replicas were
   already visible in the original paper.
2. **DiLoCo tolerates larger global batches** (batch-size exponent ~0.44-0.51 vs ~0.28 for DP), which
   is what lets it use more hardware in parallel. The optimal outer LR depends on M (and grows with
   H), not on model size, for N >= 335M.
3. **Dropped rounds are survivable.** Dropping each replica's outer gradient with probability 50%
   cost only +2.1% PPL with non-i.i.d. shards; changing the number of replicas over time (1->8, 8->1,
   4->8) gave quality tracking total compute, not the schedule (original paper). This is the
   empirical basis for elastic membership.
4. **Non-i.i.d. shards are fine**; outer gradients were weighted by shard size.
5. **Caveats.** The 4B/10B numbers use extrapolated hyperparameters with ~0.2% margins and no variance
   reported; wall-clock claims are from an idealised simulator. An independent replication on
   nanochat found competitive pretraining loss but worse MMLU, GSM8K and HumanEval after mid-training
   and SFT, not recovered by switching back to DDP (https://arxiv.org/abs/2511.13761; small models,
   unverified at larger scale). Gensyn's NoLoCo paper measured DiLoCo behind FSDP on perplexity at
   6.8B (C4: 17.3 vs 15.7), larger gap with small models and wide DP (https://arxiv.org/html/2506.10911v1).
   Results from DeepMind and from third parties therefore disagree on the size of the gap; tuning
   (outer LR per M, H, batch) appears to explain much of it but this is unresolved.

### 3.3 Hiding the synchronisation: streaming, overlap, eager

Plain DiLoCo blocks every replica while the full model delta is all-reduced: a peak-bandwidth spike
and idle GPUs. Three modifications (Streaming DiLoCo):

- **Fragments**: sync one group of layers at a time on a staggered schedule (strided pattern, 3-layer
  fragments), so peak bandwidth falls by `fragment_layers / total_layers`. Freezing un-synced layers
  instead was much worse (+20% loss).
- **Overlap**: keep training for tau inner steps while the fragment is in flight, then merge with
  mixing factor alpha. Loss degradation < 0.2% up to tau = 10; recommended tau ~ 5; workers with
  different delays are tolerated.
- **Quantise**: 4-bit outer gradients.

Simulated bandwidth (Gbit/s) to reach a compute-utilisation target, H = 100
(Table 4, https://arxiv.org/html/2501.18512v1):

| Model (step time) | Method | 50% | 80% | 95% |
|---|---|---|---|---|
| 1B (0.1 s) | Data-parallel | 86.8 | 152.6 | 222.3 |
| | DiLoCo | 1.4 | 6.2 | 23.3 |
| | Streaming + overlap + FP4 | 0.4 | 0.9 | 2.0 |
| 10B (0.8 s) | Data-parallel | 104.8 | 222.3 | 268.3 |
| | DiLoCo | 1.7 | 7.5 | 33.9 |
| | Streaming + overlap + FP4 | 0.4 | 0.9 | 1.4 |
| 100B (4.9 s) | Data-parallel | 184.2 | 323.8 | 390.7 |
| | DiLoCo | 3.0 | 11.0 | 49.4 |
| | Streaming + overlap + FP4 | 0.5 | 0.9 | 1.1 |

**Eager updates** go further: start the next inner phase immediately, all-reduce in the background for
a whole phase, and when the stale average arrives replace the worker's own stale term with its fresh
one: `delta~ = (1/M)(delta_m(t) - delta_m(t-H)) + delta_avg(t-H)`. Tolerated sync latency becomes
H x step time; simulated requirement for 95% utilisation at 10B drops to 1.4 Gbit/s (H=30) or
0.3 Gbit/s (H=100) (https://arxiv.org/html/2502.12996v1). Naively applying the delayed average
without the eager correction is much worse (3.01 vs 2.68 loss). PCCL's "async DiLoCo" mode
(one-step-delayed outer update via concurrent collectives) and Psyche's "overlapped" mode (results of
step k applied at step k+2) are the same idea in shipped code (https://arxiv.org/html/2505.14065v1;
https://nousresearch.com/nous-psyche/).

### 3.4 Asynchrony and heterogeneous replica speed

- **Async Local-SGD** (DeepMind, Jan 2024): each worker updates the global model as soon as it
  finishes its H steps. Naive version converges *slower* than sync because outer momentum is applied
  to stale pseudo-gradients. Fixes: Delayed Nesterov (apply momentum only every N arrivals) and
  Dynamic Local Updates (give each worker a number of local steps proportional to its speed so all
  finish together). Matches sync in perplexity per update and beats it in wall-clock, but only shown
  to 150M parameters (https://arxiv.org/abs/2401.09135).
- **Decoupled DiLoCo** (Google, Apr 2026) is the production-shaped answer. M learners never talk to
  each other; a CPU-only **syncer** holds global parameters and outer-optimiser state. Learners post
  per-fragment step/token counters every step. For each fragment the syncer waits for a **minimum
  quorum K** (default 1), optionally extends by an **adaptive grace window** bounded by the slack in
  the overlap (`grace <= gamma x slack`, slack = tau x step_time minus quorum+sync time, tracked by
  EMA), merges with **token-weighted** averaging (weight = tokens x tokens/steps), applies Nesterov,
  and broadcasts; learners overwrite the fragment without pausing. A missing learner just gets weight
  0. Heterogeneity test: 4 learners on one TPU generation and 4 on another, slowest 18% behind plus
  10% injected variance; K=8 (synchronous) gave 84% goodput, K=1 without grace gave 100% goodput but
  clearly worse quality (vision avg 25.7 vs 37.6), K=1 **with** grace gave 100% goodput and matched
  quality. Failure simulation (MTBI 1 year/chip): at 1.2M simulated chips goodput is 88% for M=8 vs
  58% for elastic data-parallel. Rule of thumb from the paper: M learners behave like a single
  learner with M times fewer chips for availability purposes
  (https://arxiv.org/html/2604.21428v1). Goodput figures are modelled, not measured.
- **HeLoCo** (May 2026): momentum-guided correction of stale pseudo-gradients for async DiLoCo with
  mixed-speed workers and non-IID data (https://arxiv.org/abs/2606.00271; not read in full).

---

## 4. Real decentralised runs and their systems

| System / run | Algorithm | Scale achieved | Membership and failure handling | Open code | Maturity note |
|---|---|---|---|---|---|
| Prime Intellect `prime` + INTELLECT-1 | Hybrid DiLoCo (across nodes) + FSDP2 (inside node), int8 ring all-reduce | 10B, 1T tokens, 42 days (Oct 10 - Nov 22 2024), up to 14 nodes / 112 H100, 3 continents | ElasticDeviceMesh: heartbeat every 2 s, eviction after 6 s, "deathrattle" on graceful exit, failed all-reduce retried without the dead node; joiners fetch checkpoint P2P (30-60 min) and enter at the next outer step; ring order chosen to maximise the minimum-bandwidth edge | Yes | Loss of 4/12 nodes at once destabilised training and required checkpoint restore; Tailscale VPN was a bottleneck; 500 Mbit/s-4 Gbit/s links (https://arxiv.org/html/2412.01152v1) |
| PCCL (Prime Collective Communications Library) | Fault-tolerant collectives over public-internet TCP; shared-state hash sync | Benchmarks: 6 EU nodes 3.67 Gbit/s single-connection, ~45.7 Gbit/s with 128 connections; 18 NA+EU nodes 90.5 s for a 1.07 GB reduce single-op | Master-client; peers are *registered* then *accepted* by vote; aborted collectives restore the buffer and are retried with the smaller group; out-of-date peers resync P2P; ATSP ring optimisation | MIT, C99 + Python bindings | One major operation at a time per group; scheduler-latency sensitive (https://arxiv.org/html/2505.14065v1; https://primeintellect.ai/blog/pccl) |
| hivemind / OpenDiLoCo | DHT (no master), decentralised averaging, Moshpit; OpenDiLoCo implements DiLoCo on `hivemind` | ~1B across 3 countries, 90-95% utilisation | Peers discovered via DHT; averaging rounds tolerate missing peers | Yes (https://github.com/learning-at-home/hivemind) | Last tagged release found is 1.1.10, roughly 2023 (unverified date); Prime Intellect moved off it to PCCL |
| Petals | Inference/fine-tuning over volunteer-hosted layers (not pretraining) | - | Rerouting around failed servers | Yes (https://arxiv.org/abs/2209.01188) | Out of scope for training; cross-ref `10-inference-engines.md` |
| SWARM parallelism | Stochastically wired pipelines between stage replicas, rebalanced on failure; compressed activations | 1B shared params (13B before sharing) on preemptible T4s, < 200 Mbit/s | Designed for preemption | Yes (https://arxiv.org/abs/2301.11913) | Research code |
| Pluralis Agora / Pluralis-8B (Jul 2026) | Pipeline-parallel sharding (7 stages) with low-rank activation compression (rank ~51 at d=5120); each stage's replicas average a rotating 5% of parameters every 20 steps, sample-weighted, in a background thread | 8.6B, 500B tokens, 40 days, 330 contributor nodes (55% RTX 4090/5090, 22% L40S), ~170k tokens/s, 63% of centralised H100 efficiency | Admission gate: >= 24 GB VRAM, >= 200 Mbit/s down, <= 80 ms latency; joiners more than 3 steps stale go through weight-sync then optimiser-sync phases at zero weight; workers self-terminate after failing 2 of 3 all-reduce rounds | Paper (https://arxiv.org/abs/2607.13332) | Head/tail stages and reducers ran on operator nodes, i.e. not fully decentralised |
| Nous Psyche (DisTrO/DeMo) | Per-step DeMo-style update: DCT + top-k of decoupled momentum, optional sign-only (`quantize_1bit`, > 3x extra), overlapped application | Hermes 4.3 (36B, post-training) on 24 nodes at 144k tokens/s, reported to beat the centralised FSDP+AdamW twin on downstream evals; Consilience 40B pretraining run on testnet | Solana coordinator contract; "epochs" (e.g. 500 steps) with phases Waiting -> Warmup -> Training -> Witness -> Cooldown; join/leave at epoch boundaries; unhealthy clients voted out and their data reassigned; iroh (QUIC, hole punching) for P2P | Apache-2.0, Rust (https://github.com/PsycheFoundation/psyche) | Verification unsolved; no independent replication of quality claims (https://nousresearch.com/nous-psyche/; https://nousresearch.com/introducing-hermes-4-3/) |
| Gensyn NoLoCo | No all-reduce: each outer step averages with one random peer, with a modified Nesterov term; random pipeline routing | 125M-6.8B, up to 64 accelerators; outer step every 50 inner steps | Pairwise only, so no global barrier | https://github.com/gensyn-ai/noloco | Slightly better than DiLoCo, still behind FSDP in their runs (https://arxiv.org/html/2506.10911v1) |
| exo gym, SPARTA | Simulator (one subprocess per node) implementing DDP, DiLoCo, SPARTA, DeMo; SPARTA averages ~0.05-0.1% of parameters very frequently with no outer optimiser | MNIST / nanoGPT scale | - | https://github.com/exo-explore/gym; https://blog.exolabs.net/day-12 | Research tool, useful for prototyping outer-loop policies cheaply |

Other 2026 directions seen but not read in depth: GASLoC, gossip-style outer optimisation claiming to
"significantly outperform DiLoCo" under heterogeneous bandwidth (https://arxiv.org/abs/2606.11081);
Factored Gossip DiLoCo (https://arxiv.org/pdf/2606.22768); LOSCAR-SGD, local training plus sparse
averaging with overlap (https://arxiv.org/html/2605.20866).

Observations for an orchestrator:

- Every working system has an **explicit membership service separate from the collective library**
  (PRIME's key-value store + heartbeats, PCCL's master, Psyche's coordinator, Decoupled DiLoCo's
  syncer, Agora's Authorizer). NCCL/MPI world membership is static, so the cross-island layer cannot
  be an NCCL communicator (cross-ref `03-rank-configuration.md`).
- **Joining is the expensive event**, not leaving: a joiner must fetch full weights (30-60 min P2P for
  10B in INTELLECT-1) and, if admitted mid-phase, contributes a zero pseudo-gradient that caused loss
  spikes. Agora rate-limits admissions for the same reason.
- **Correlated loss is the dangerous failure**: losing 1/3 of nodes at once broke INTELLECT-1; a
  single provider/region outage would do the same, so islands should not all share one failure domain.
- **WAN throughput is a tuning problem**: multiple parallel TCP connections raised PCCL throughput by
  roughly 10x; ring ordering by measured bandwidth matters; VPN overlays cap throughput
  (cross-ref `05-networking-interconnects.md`).
- Pipeline-parallel-over-internet (SWARM, Agora) targets nodes too small to hold the model. `grid`
  rents datacenter GPUs, so the data-parallel family is the relevant one; Agora's 63% efficiency
  figure is a useful upper bound on what that route costs.

---

## 5. Comparison at a glance

| Topology | Cross-replica bytes | Sync frequency | Bandwidth needed (10B model, ~80% util.) | Tolerates node loss | Tolerates speed variance | Quality vs sync at LLM scale | Open implementation maturity |
|---|---|---|---|---|---|---|---|
| Sync DDP/FSDP | Full gradient | Every step | ~220 Gbit/s (simulated) | No (restart from checkpoint, or elastic re-form) | No (slowest rank) | Reference | Production |
| Sync + per-rank batch sizing | Same | Every step | Same | No | Yes (static/slow drift) | Same if token-weighted | Papers, small codebases; easy to implement |
| Sync + per-step compression (DeMo/DisTrO) | 1-2 orders less | Every step | ~Gbit/s (unverified) | Via protocol (Psyche epochs) | Limited | Claimed parity, vendor-reported | Psyche: usable, young |
| DiLoCo (blocking) | Full delta, fp16/int8 | Every H = 30-500 steps | ~7.5 Gbit/s at H=100 | At outer-step boundaries | Only with per-replica H or weighting | -0.2% to +1.1% loss depending on M | `prime`, OpenDiLoCo |
| Streaming/eager DiLoCo | Fragments, 4-bit, overlapped | Continuous trickle | 0.3-1.4 Gbit/s | Same | tau differences tolerated | ~parity at 1B-4B | PCCL async mode; no reference Google code found |
| Decoupled (async, quorum) | Fragments to syncer | Continuous | ~100x less than DP | Yes, no global stall | Yes (quorum + grace + token weights) | Parity reported at 2B-9B, M=8 | None public (unverified) |
| Gossip (NoLoCo, GASLoC) | Full model to one peer | Every ~50 steps | Low, no all-reduce | Yes | Yes | Behind FSDP in NoLoCo's runs | Research code |
| Classic async SGD | Full gradient | Every worker step | High | Yes | Yes | No evidence; expected poor | Parameter-server legacy |

---

## 6. What "heterogeneous" does to each layer

- **Different speeds inside an island** (e.g. mixed H100/A100): sync step runs at the slowest rank.
  Fix by grouping (do not mix) or by per-rank batch sizing with token-weighted loss. Persistent ~20%
  spread is tolerable with batch sizing; a spread of 2x usually means a different island.
- **Different speeds between islands**: in blocking DiLoCo the outer step waits for the slowest
  island. Options, in increasing complexity: (a) give each island a step budget H_i proportional to
  its measured steps/s so all reach the barrier together (Dynamic Local Updates); (b) fix a wall-clock
  phase length and weight each delta by tokens processed (Decoupled DiLoCo's weighting); (c) quorum +
  grace window so the outer step proceeds without laggards.
- **Different island sizes**: weight outer gradients by tokens (shard size in the original paper). A
  very small island adds an M-penalty for little compute; prefer merging small groups or using them
  as spares.
- **Different link quality**: all-reduce over a ring is bounded by the weakest edge. Measure pairwise
  bandwidth, order the ring (PRIME/PCCL), or use a star through a well-connected syncer (Decoupled
  DiLoCo), or pairwise gossip (NoLoCo).
- **Different memory**: constrains micro-batch and whether a node can hold the DiLoCo extras (global
  copy + outer momentum, normally in host RAM).

---

## 7. Decision framework

Inputs the orchestrator should measure in preflight (cross-ref `09-fast-bootstrap-preflight.md`):
`B_ij` pairwise TCP/RDMA throughput, `RTT_ij`, per-node tokens/s `s_i` on the actual model, and
expected interruption rate per node. Model inputs: parameter count N, single-island step time `t_step`.

Useful estimates:

- Sync DP needs roughly `2 x N x 16 bits / t_step` of all-reduce traffic per replica group at bf16;
  require this to be well under the fabric bandwidth and RTT << t_step.
- Blocking outer sync time: `T_sync ~ c x N x b / B_min`, with b = bits per parameter (16, 8 or 4) and
  c ~ 2 for a ring. Compute utilisation = `H x t_step / (H x t_step + T_sync)`. Example check:
  10B x 8 bit = 10 GB; INTELLECT-1 measured 103-469 s per sync against ~38 min of compute, i.e.
  83-96% utilisation.
- Overlapped/eager sync is free if `T_sync <= H x t_step` (eager) or fragment sync fits in
  `tau x t_step` (streaming).

| Measured condition | Topology | Why |
|---|---|---|
| All nodes on one RDMA/high-speed fabric (>= ~100 Gbit/s effective, sub-ms RTT), speed spread <= ~10%, low interruption rate | **Pure sync FSDP/DDP**, one island | Reference quality, simplest; recover by checkpoint-restart |
| Same fabric, speed spread 10-50% | Sync + per-rank batch sizing (token-weighted), or split by GPU type into islands | Avoid idling fast ranks |
| Several fast fabrics joined by 1-25 Gbit/s links, RTT 1-150 ms (different pods, datacenters, providers) | **Hierarchical**: sync inside each island, DiLoCo outer loop across, M kept small (2-8), H = 30-100 to start, int8/fp16 deltas | Evidence to 10B; +/-0.3% loss at M <= 4 |
| As above but links 0.1-1 Gbit/s, or sync would cost > 10% of wall-clock | Hierarchical with overlapped/eager outer step and 4-8 bit fragments | Hides sync behind compute |
| Islands with different speeds (> ~15%) or frequent preemption (spot) | Hierarchical with **asynchronous** outer loop: syncer, quorum K, grace window, token-weighted merge | Decoupled DiLoCo result: no global stall, parity only with grace window |
| Many single nodes, each too weakly connected to form any island | Each node is its own replica (INTELLECT-1 style) with M capped; or do not use them for pretraining | Quality penalty grows with M; consider them for eval/data jobs instead |
| Nodes too small to hold the model | Pipeline-over-internet (Agora/SWARM) | Out of `grid`'s primary scope; ~63% efficiency |

Fully asynchronous per-step SGD is not recommended in any cell: no LLM-scale evidence.

What the orchestrator must provide per topology:

| Capability | Pure sync | Hierarchical (blocking outer) | Hierarchical (async outer) |
|---|---|---|---|
| Island grouping | Trivial (one) | Cluster nodes by measured bandwidth/RTT graph and GPU class; choose M; balance tokens/s per island | Same, plus per-island speed estimate kept current |
| Rank/process groups | One static world | Static world *per island* (NCCL); separate fault-tolerant cross-island group (TCP/QUIC, not NCCL) with rank 0 of each island as its delegate | Same; cross-island group replaced or fronted by a syncer service on CPU |
| Outer-step coordination | - | Global barrier at step H (or per-island H_i); ring/tree order from measured bandwidth; quantisation; retry on abort with remaining members | Per-fragment schedule, quorum K, grace bound from EMA of step and sync time, token counters, weight 0 for absent islands |
| Membership changes | Stop, re-form world, restore checkpoint | Islands join/leave only at outer boundaries; joiner state transfer from a peer or object store; admission rate limit; zero-weight warm-in | Continuous; stale islands resync then rejoin; syncer shard persists for an absent island |
| Failure inside an island | Job restart | Island restarts alone from the last global parameters (it loses at most H steps); others continue or wait up to a timeout | Island is simply excluded until it returns |
| Data | Global sampler | Disjoint shards per island (non-i.i.d. acceptable); reassign a dead island's shard | Same, with token accounting per island |
| Checkpointing | Global | Global parameters + outer-optimiser state at outer boundaries (cheap, consistent cut); inner optimiser state per island | Syncer state is the checkpoint of record |
| Hyperparameters | - | Outer LR as a function of M (re-tune when M changes), H, inner LR schedule driven by global token count | Plus K, gamma, tau |

---

## Implications for grid

1. **Make "island" a first-class object.** An island is a set of nodes with a measured fast, uniform
   fabric and one GPU class; it runs unmodified `torchrun`-style FSDP. The cluster is a set of
   islands. A single-island cluster degenerates to ordinary sync training, so one code path covers
   both.
2. **Measure, then group.** Preflight must produce the pairwise bandwidth/RTT matrix and per-node
   tokens/s on the real model, then cluster. Minimise M subject to using the hardware: published
   quality cost rises with M (+0.3% at M=4, +1.1% at M=8 for 2.4B) and falls with model size.
3. **Ship a cross-island runtime that is not NCCL**: a small coordinator/syncer (CPU node or sidecar)
   plus a fault-tolerant delta exchange over multi-connection TCP/QUIC with int8/fp16 quantisation.
   Evaluate PCCL (MIT) as the collective layer before writing one; its registered/accepted peer model
   and abort-and-retry semantics match the need. Avoid VPN overlays on the data path.
4. **Start with blocking DiLoCo, design for decoupled.** Blocking outer steps at H ~ 100 with int8 are
   proven at 10B and simple. But the coordinator's interface should already carry per-island step and
   token counters, weights, and a quorum/deadline, so that overlapped and asynchronous modes
   (Decoupled DiLoCo's K, grace window, token weighting) are configuration, not a rewrite.
5. **Membership policy lives in the orchestrator.** Heartbeats (PRIME used 2 s / 6 s), graceful-exit
   signal on preemption notice, admission only at outer boundaries, rate-limited joins, state
   transfer from object storage rather than slow P2P where possible, and automatic outer-LR adjustment
   when M changes. Spread islands across failure domains; losing a third of replicas at once was
   fatal in INTELLECT-1.
6. **Within-island heterogeneity: refuse or rebalance.** Default to homogeneous islands. Offer
   per-rank batch sizing with token-weighted loss as an opt-in for moderate spreads.
7. **Report goodput, not GPU utilisation.** Track compute utilisation
   (`inner time / (inner + sync)`), per-island tokens/s, sync duration and bytes, and M over time;
   these are the numbers every paper above uses and they expose stragglers that `nvidia-smi` hides.
8. **Be honest in the UX about quality risk.** Offer a single-island sync mode as the reference and
   make it easy to run a short A/B (same token budget) before committing a long multi-island run;
   downstream-eval parity is not settled in the literature.
9. **DiLoCo with M=1 is a free option**: outer Nesterov on a single island improved loss slightly in
   Google's study. It also means the multi-island code path is exercised even on one island.

## Open questions

- **Downstream quality at >= 10B with M >= 4.** DeepMind reports parity; nanochat-DiLoCo and NoLoCo
  report gaps. No independent, well-tuned replication at multi-billion scale was found. Needs a
  second sweep and probably our own experiments.
- **Is there a public implementation of Streaming or Decoupled DiLoCo?** None found; PCCL's async
  mode and Psyche's overlap are the nearest. Check Google/JAX releases and `prime` successors.
- **Interaction with FSDP sharding, tensor/pipeline parallelism and MoE** inside islands: Decoupled
  DiLoCo notes MoE load-balancing losses are only local per learner. How do outer steps interact with
  LR warmup, weight decay, EMA and mixed-precision optimiser state?
- **Per-island H vs wall-clock phases vs quorum**: which heterogeneity policy loses least when islands
  differ by 2x in speed? Async Local-SGD evidence stops at 150M parameters; HeLoCo and GASLoC were not
  read in full.
- **Outer-gradient sparsification**: Streaming DiLoCo says top-k hurts; MuLoCo/SparseLoCo say it works
  with error feedback. Which holds at 10B+, and does Muon as inner optimiser change the answer?
- **Hyperparameter transfer**: outer LR depends on M; what is the right automatic schedule when M
  changes mid-run because islands join or leave?
- **Psyche/DisTrO**: no peer-reviewed or third-party verification of per-step compression at 36-40B;
  absolute bandwidth numbers and current status of the Consilience run were not found.
- **Fine-tuning and RL workloads** (cross-ref `12-inference-for-rl.md`): almost all evidence is
  pretraining from scratch; behaviour of local-update methods for short fine-tunes is unstudied here.
- **Hyperscaler practice**: whether Microsoft/OpenAI/Google cross-site runs are synchronous over
  private WAN or use DiLoCo-like schemes in production is not disclosed beyond Google's "synchronous"
  statements for Gemini 1.0/2.5 and its 2026 multi-site claims.
- **Real marketplace link measurements**: the framework above needs actual pod-to-pod and
  region-to-region throughput for RunPod-class providers (from `05`/`07`) to pick defaults for H and
  quantisation.
