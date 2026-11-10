# 18 - State transfer and checkpointing as a selectable design space

Written 2026-11-10. Requirement from the user: "I want to be able to pick between and compromise
between weight transfer / checkpoint methods". This file maps the methods by which model state
moves and persists, names the axes they trade on, and proposes how grid exposes them.

Facts already in sibling files are cited by filename (`04`-`17`; where they disagree, 13-17 win).
New web sources carry a URL. Three kinds of statement are kept apart:

- **[F]** fact from a cited source.
- **[D]** derived here; the arithmetic is shown.
- **[P]** design proposal for grid.

Nothing was run. Sizes use decimal GB.

## Summary

- State movement serves seven purposes with different requirements, and the cheapest method for
  one is unsafe for another (a 2 GB lossy sparse delta suits cross-island sync and must never be the
  recovery copy).
- Seventeen catalogued methods reduce to six orthogonal choices: **payload, codec, transport,
  trigger, commit, retention**. "Picking" is choosing a preset per purpose; "compromising" is
  overriding one field, most of which are dials (interval, bit-width, density, keyframe period, H,
  staleness bound, fragments, generations).
- Sizes per billion parameters [D]: full training state 13.5 GB, bf16 weights 2 GB, int8 delta
  1 GB, Templar sparse codec 27 MB. At 70B the largest and smallest payloads differ by about 500x,
  so payload and codec matter more than transport.
- Two derived results change earlier advice. With async saves the loss-of-work window grows by the
  background persist time, so Young/Daly needs an extra term. On pods where node loss dominates and
  local disk dies with the pod, a local tier without peer replication buys little.
- grid can provide transport, commit, retention, integrity and tiering without job cooperation.
  Payload selection, lossy codecs, live peer pull and in-memory push need hooks.

## 1. Purposes

| | Purpose | Sender, receiver | Must have | Can give up |
|---|---|---|---|---|
| a | Durable checkpoint for fault recovery | trainer to storage outside every failure domain | lossless, complete (optimizer, cursor), atomic, verifiable, several generations | speed of read-back |
| b | Fast local restart | trainer to itself | read-back in seconds, low blocking time | durability against node loss, portability |
| c | Bootstrap a joining or replacement node | peers or storage to one node | current state, no stall of survivors | history; same-shape only is acceptable |
| d | Cross-island training sync | island to islands | bounded round time on tier C/D links (`05`), tolerance of absent peers | exactness: lossy deltas are accepted (`08`, `14`) |
| e | Trainer to inference servers (RL) | learner to N actors | bounded policy lag, version identity per replica | optimizer state; exact dtype (FP8 actors) |
| f | Model rollout / cold start | registry to N replicas | immutable version, fan-out, fast load to VRAM | write speed |
| g | Migration to another shape or provider | cluster A to cluster B | logical addressing, global-unit progress (`15` contract) | speed; rank-local extras |

## 2. Reference numbers

**Bytes per event** [D], with N = parameters in billions:

| Payload and codec | bytes | 1B | 10B | 70B | Basis |
|---|---|---|---|---|---|
| Full training state (bf16 weights, fp32 master + two Adam moments) | 13.5 GB x N | 13.5 GB | 135 GB | 945 GB | `15` s8: 7B = 13 + 82 GB |
| Weights only, bf16 | 2 GB x N | 2 GB | 20 GB | 140 GB | 16 bit/param |
| Weights, lossless byte-grouped entropy coding | ~1.3 GB x N | 1.3 GB | 13 GB | 93 GB | ZipNN: BF16 models to 66-67% |
| Lossless XOR delta of weights between snapshots | 0.2-1.2 GB x N | 0.2-1.2 GB | 2-12 GB | 14-84 GB | LMC: ratio < 0.6, falling to 0.1-0.2 late in training |
| Dense int8 delta | 1 GB x N | 1 GB | 10 GB | 70 GB | `14` s9 |
| Dense 4-bit delta | 0.5 GB x N | 0.5 GB | 5 GB | 35 GB | `14` s9 |
| Chunked top-k (64 of 4096), 12-bit index, 2-bit value | 27 MB x N | 27 MB | 273 MB | 1.9 GB | `14` s6: 0.219 bit/param |
| LoRA rank 32 on q,k,v,o, bf16 | 16 x L x r x d bytes | 23 MB | 67 MB (7B shape) | 336 MB | below |

LoRA arithmetic [D]: each adapted d x d matrix adds 2rd parameters; four per layer gives 8Lrd
parameters, two bytes each. (L, d) = (22, 2048), (32, 4096), (80, 8192) give 11.5M, 33.5M and
167.8M parameters. `12` quotes about 50 MB for rank 32 on a 7B model, the same order.

Lossless compression sources [F]: ZipNN compresses BF16 models to 66.3-67.4% (zstd alone 77.7% on
Llama-3.1) at 1.15 GB/s single-threaded, and XORs for deltas
(https://arxiv.org/html/2411.05239v2). IBM's LMC reports delta ratios under 0.6, improving to about
0.1 (weights) and 0.2 (biases) as BLOOM training converges, at 2.78 GiB/s on 16 cores
(https://arxiv.org/html/2505.09810v1). Neither measures optimizer moments; the late-run ratios are
for one model.

**Rates** [F] unless marked:

| Path | Rate | Source |
|---|---|---|
| GPU to pinned host memory | 25 GB/s per GPU peak | `15` s8 |
| `torch.save` flush | ~1 GB/s per writer | `15` s8 |
| RunPod network volume | 200-400 MB/s typical | `09` s4.1 |
| Object store over a 1 Gbit/s uplink | 0.125 GB/s per node | [D] |
| NCCL sockets, tier C | 1-6 GB/s | `05` s9 |
| tensorizer HTTP streaming / local | ~1 GiB/s / 2.25-4.6 GiB/s | `15` s8 |

## 3. Catalogue

Each method is one row across three tables. Tiers A-D are the link tiers of `05` s9; "HTTPS-out"
means outbound TCP 443 only.

### 3.1 What moves, over what, how much, how long

| # | Method | What moves | Transport and tier | Bytes per event | Blocking time | Staleness / loss-of-work window |
|---|---|---|---|---|---|---|
| M1 | Full blocking checkpoint to shared or network volume | Full state | Filesystem mount; data-center local | 13.5 GB x N | Whole write. 70B, 8 nodes: 118 s at 1 GB/s, 394 s at 0.3 GB/s [D] | Interval tau |
| M2 | Async sharded checkpoint (DCP `async_save`, Megatron + NVRx) | Full state, one shard per rank | Host RAM, then any fsspec target; Gloo between ranks (`15`) | same | GPU-to-host copy only: 6-14 s measured at 7B-13B (`15`) | tau plus background persist time P [D] |
| M3 | Local NVMe tier with background upload | Full state locally, a subset uploaded | Local disk; then HTTPS-out | same, twice | Stage or local write | tau_local if the node survives; tau_durable + P if not |
| M4 | Object-store direct per-rank upload | Full state or weights | HTTPS-out multipart, every node in parallel | same | None if staged; else the upload | tau + P; P = S / (n x uplink) |
| M5 | Lossless incremental (XOR delta + entropy coding) | Delta of weights against a named base | Any | 0.1-0.6 x base | CPU: ~0.2-0.3 GiB/s per core (LMC) | As carrier; restore adds chain replay |
| M6 | Quantised dense delta (fp8, int8, 4-bit) | Pseudo-gradient or weight delta | Any; used on tier C/D | 0.5-1 GB x N | Snapshot + quantise; hidden if overlapped | H steps + round time |
| M7 | Sparsified delta, top-k with error feedback (Templar codec) | 1.56% of delta entries + residual kept locally | Object store, HTTPS-out | 27 MB x N | Chunked top-k on GPU or CPU | H steps + round; dropped mass waits in the residual |
| M8 | Low-rank / LoRA only | Adapter tensors | Any; HTTP load route | 20-340 MB total | None on the engine: hot-swap (`12`, `17`) | Per publish |
| M9 | In-memory peer replication (buddy) | A rank's staged state to k partner ranks | Process group or TCP inside the island; tier B/C | 13.5 GB x N x k | Stage; replication in background | tau_replica |
| M10 | Live pull from a healthy peer (torchft) | Live `state_dict` of one replica group | HTTP or process group; flat network (`13`, `14`) | State of one replica | Survivors: none. Joiner: S / B_peer | Zero: joiner gets the current step |
| M11 | In-cluster collective push (NCCL broadcast; vLLM `nccl`, `sparse_nccl`, `nccl_m2n`; SGLang distributed update; checkpoint-engine) | Weights, GPU to GPU | NCCL: tier A/B, all ports reachable | 2 GB x N | Lockstep: 0.02 s (7B) to 23.7 s (235B MoE); 23 s for 30B across nodes (`12` s2.1) | About one trainer step |
| M12 | CUDA IPC handles | Nothing is copied: handles of ~64 bytes per tensor | Same host, same GPU view | ~0 | 7 s for 30B after tuning (`12`) | None |
| M13 | RDMA / NIXL pull (vLLM `sharded_rdt`, ModelExpress GPU peer) | Weight shards pulled by receivers | RDMA; NIXL back ends include UCX, GDS, POSIX, object stores (https://docs.nvidia.com/nixl/getting-started/architecture) | 2 GB x N | Receiver-driven | About one step |
| M14 | Shared-filesystem reload (`/update_weights_from_disk`, prime-rl `filesystem`) | Weights as safetensors, or adapter | Shared mount | 2 GB x N | Trainer write + engine reload | Write + read |
| M15 | Object-store store-and-pull, content addressed (ModelExpress S3 deltas, Composer 2 chain) | Immutable full versions and deltas | HTTPS-out | 2 GB x N full; deltas "a handful of gigabytes" at 1T (`12`) | Snapshot only | Two legs, 2S / B; minutes at Composer |
| M16 | Peer-to-peer swarm among nodes (Dragonfly, Kraken, Shardcast relays) | Pieces of an immutable artefact | TCP between nodes; one inbound port each | S per node; origin serves about S once | None for the sender | S / B_seed + swarm time |
| M17 | Streaming and cache loaders (tensorizer, Run:ai streamer, fastsafetensors, `ipc_cache`, SGLang weight cache) | Weights, storage or host cache to VRAM | HTTP/S3, NVMe, Unix socket | 2 GB x N read | The load itself | n/a |

### 3.2 Fidelity, durability, failure behaviour, integrity

| # | Fidelity | Durability | Half-finished transfer; atomicity; rollback | Integrity |
|---|---|---|---|---|
| M1 | Lossless | Survives pods; pinned to one data center (`04`) | Partial directory. DCP commits by renaming `.metadata` but defaults to `overwrite=True` (`15`) | None in DCP or safetensors; Megatron SHA-256 opt-in (`15`) |
| M2 | Lossless | As target | Staged copy lost with the process; previous generation intact if directories are immutable | As M1 |
| M3 | Lossless | Local: process failure only. NVRx local tier is shape-locked (`15`) | Local and durable generations can diverge; restore picks the newest complete one that is reachable | grid hashes in the uploader [P] |
| M4 | Lossless | Outside all failure domains | Multipart upload is invisible until completed (`14` s9); the manifest written last is the commit | Per-part transport checksum + grid SHA-256 manifest |
| M5 | Lossless | As carrier, but a delta is useless without its base | A broken link breaks every later version until the next keyframe | Hash of base and of the reconstructed result; ModelExpress uses adler32 and strict lineage (`14`) |
| M6 | Lossy. fp16, int8 and 4-bit outer gradients reported free or nearly free (`08` s2.1) | None: an exchange format | A missing delta is a dropped round; 50% drop was survivable in DiLoCo (`08`) | Checksum per object |
| M7 | Lossy and contested: SparseLoCo reports lower loss than dense DiLoCo up to 2B (https://arxiv.org/abs/2508.15706); Streaming DiLoCo found top-k hurt (`08`); Covenant-72B has no same-recipe baseline (`14`) | None; the residual is extra state | Absent upload is skipped; a lost residual silently loses mass | Checksum per object |
| M8 | Exact for adapter training; an approximation for anything else | As carrier | Adapter swap succeeds or leaves the old one | File hash |
| M9 | Lossless | Survives loss of fewer than k + 1 buddies; not a provider outage | Partner holds the previous generation until the new one is complete | Hash on receipt [P] |
| M10 | Lossless, but only valid if the source's step committed (`13`) | None | Joiner retries; `should_commit` can pass after a failed reduce (`14`) | None; torchft loads pickle from trusted peers only |
| M11 | Lossless; FP8 re-quantised at finish (`12`) | None | Engine left mid-reload while `/health` is 200; no rollback; restart the sequence (`12`, `14`) | vLLM `/weight_checker`, SGLang `/weights_checker` (`17`) |
| M12 | Lossless | None | Same as M11 | Same |
| M13 | Lossless | None | "A failed update leaves inference paused ... there is no rollback" (`14`) | Checker routes |
| M14 | Lossless | As the mount | Sentinel files `.started`/`.finished` (`14`); a reader can race a writer without them | None by default |
| M15 | Lossless (XOR + zstd) or adapter | Durable and versioned | Immutable versions; a manifest or conditional put is the commit; receivers behind replay the chain, up to 64 links (`14`) | Content hash is the address |
| M16 | Lossless | As origin | Piece-level resume | Piece hashes; Shardcast SHA-256, mismatch skips to the next checkpoint (`12`) |
| M17 | Lossless | n/a | Failed load is a failed start | tensorizer CRC32 + SHA-256 per tensor (`15`) |

Atomic commit on object stores [F]: S3 accepts `If-None-Match: *` on `PutObject` and
`CompleteMultipartUpload`
(https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html); R2
supports conditional `PutObject`, where among concurrent writers "at most one request succeeds"
(https://developers.cloudflare.com/r2/api/s3/extensions/). That is a compare-and-swap for `latest`
pointers without a lock service.

### 3.3 Privileges, cost, maturity

| # | Needs on unprivileged pods | Dollar drivers | Maturity and what to reuse |
|---|---|---|---|
| M1 | Volume attached at creation, single data center; RunPod warns about concurrent writers (`09`) | $0.05-0.14 per GB-month (`07`); GPU time while blocked | Every framework |
| M2 | Host RAM for one extra copy of per-rank state; Gloo TCP between all ranks (`15`) | Host RAM | DCP `async_save`; torchtitan `async_with_pinned_mem`; Megatron `--async-save` with NVRx |
| M3 | Local disk >= one generation; container disk is wiped on restart (`16`) | Disk $0.10 per GB-month (`07`) | Megatron `non_persistent_ckpt_type=local`; else grid agent |
| M4 | HTTPS-out only; scoped credentials | Storage; write operations; egress on restore | DCP `FsspecWriter`; s5cmd; Templar's multipart code (`14`) |
| M5 | CPU cores | CPU; saves storage and transfer | `zipnn` (PyPI); ModelExpress `XOR_DELTA`; no framework ships it (`15`) |
| M6 | Nothing extra | Bytes moved; egress if priced | torchft fp8, PCCL min-max, INTELLECT-1 int8 (`14`) |
| M7 | Nothing extra | Smallest bytes | Templar `compress.py`, MIT (`14`) |
| M8 | Engine flag (`VLLM_ALLOW_RUNTIME_LORA_UPDATING`, which forces `--api-server-count 1`) (`17`) | Negligible | vLLM and SGLang LoRA routes (`17`); PEFT |
| M9 | TCP between buddies; RAM or disk on the partner | Partner memory | NVRx `CliqueReplicationStrategy`; DCP `_ReplicationStager` (experimental) (`15`) |
| M10 | Flat routable network, ephemeral ports, advertises `gethostname()`: fails on NAT-mapped pods (`14`) | None | torchft `HTTPTransport`, `PGTransport`; PCCL shared-state sync |
| M11 | NCCL reachability on arbitrary ports: not across NAT (`05`); dev-mode routes are unauthenticated (`17`) | Trainer stall | vLLM weight-transfer back ends; SGLang; checkpoint-engine |
| M12 | Same container or shared GPU IPC view; unverified on RunPod (`17`) | None | slime/miles colocated mode; vLLM `ipc` |
| M13 | `/dev/infiniband`, memlock: detect, cannot fix (`05`); ModelExpress needs gRPC + Redis (`14`) | None | vLLM `sharded_rdt`, `modelexpress` (unreleased dependency, `14`) |
| M14 | Shared mount | Volume | prime-rl `filesystem`; SGLang `/update_weights_from_disk` |
| M15 | HTTPS-out only | Storage; egress x replicas unless zero-egress (`14` s9) | ModelExpress S3 refit; HF Xet as a chunk-dedup reference (64 KB content-defined chunks, https://huggingface.co/docs/xet/deduplication) |
| M16 | One inbound TCP port per peer; pointless over RunPod's 100 Mb/s private network (`05`) | Origin egress falls to about one copy | Dragonfly `dfget hf://` (https://www.cncf.io/blog/2026/04/06/peer-to-peer-acceleration-for-ai-model-distribution-with-dragonfly/); Kraken (`09`); untested on pods |
| M17 | `ipc_cache`: same user, same node. CRIU snapshots need root: unusable. GDS needs host support (`17`) | Daemon holds VRAM | vLLM `tensorizer`, `runai_streamer`, `instanttensor`, `ipc_cache`; SGLang weight-cache daemon (`17`) |

## 4. Trade-off axes

| Axis | One end | Other end |
|---|---|---|
| X1 Completeness of payload | M8 adapter; M6/M7 delta; M11-M15 weights | M1-M4 full state |
| X2 Fidelity | M7 (0.219 bit), M6 (4-8 bit) | Everything lossless |
| X3 Durability domain | M10-M13 none; M2 staging: process | M9 buddy; M1 data center; M4/M15 outside the provider |
| X4 Blocking time on the step | M1 minutes; M11 seconds to tens of seconds | M2/M4 staged; M15, M16 none on the sender |
| X5 Staleness / loss-of-work | M10, M12 zero | M15 minutes; M1 at long intervals, hours |
| X6 Network tier required | M12 same host; M11, M13 tier A/B | M4, M7, M15 HTTPS-out on tier D |
| X7 Portability across shapes | M3 local NVRx, M9, rank-keyed formats: shape-locked | Logical formats (`15`) through M4 |
| X8 Failure atomicity | M11-M13: no rollback | M4, M15: invisible until committed |
| X9 Coupling | M11, prime-rl handshakes: lockstep ("a broadcast nobody receives blocks and then fails", `14`) | M7, M15: store-and-forward, absentees skipped |
| X10 Dollar cost | Dense sync through an egress-priced store ($18k-33k per month in `14` s9, list prices unverified) | M7, M8; zero-egress stores |

## 5. The dials

### 5.1 Interval against overhead (X4, X5)

[F] Young/Daly: tau* = sqrt(2 x delta x M), overhead about sqrt(2 x delta / M) + R/M (`04` s4.1).

[D] With an async save the newest *complete* checkpoint lags by the background persist time P, and
a new save cannot usefully start before the previous one lands (torchtitan waits for the previous
staging future, `15`). So:

```
overhead(tau) = delta/tau + (tau/2 + P)/M + R/M,    tau* = max( sqrt(2*delta*M), P )
```

Worked case [D]: 70B, 945 GB of state, 8 nodes. Blocking to a WAN store at 0.2 GB/s per node:
delta = 945/(8 x 0.2) = 591 s, call it 600 s. Blocking to a 1 GB/s volume: 118 s, call it 120 s.
Async: delta = 10 s (within the 6-14 s measured in `15`), P = 600 s to the WAN store.

| Job MTBF M | Blocking, delta = 600 s | Blocking, delta = 120 s | Async, delta = 10 s, P = 600 s |
|---|---|---|---|
| 30 h | tau 190 min, 10.5% | tau 85 min, 4.7% | tau 24.5 min, 1.4% + P/M 0.6% = 1.9% |
| 8 h | tau 98 min, 20% | tau 44 min, 9.1% | tau 12.6 min, 2.6% + 2.1% = 4.7% |
| 2 h | tau 49 min, 41% (first-order formula is rough here) | tau 22 min, 18% | tau clamped to 10 min: 1.7% + 12.5% = 14.2% |

Arithmetic for the 8 h row: sqrt(2 x 600 x 28800) = 5879 s; sqrt(1200/28800) = 0.204.
sqrt(2 x 10 x 28800) = 759 s; 10/759 + 380/28800 = 2.6%; P/M = 600/28800 = 2.1%.

The dials are tau and P; more uploading nodes, a faster store or a smaller payload (5.4) lower P.

### 5.2 Tiered generations (X3, X5)

[F] Multi-level checkpointing is established HPC practice: SCR reported up to 35% better machine
efficiency, with cheap levels sufficient for 85% of failures (https://www.osti.gov/biblio/984082;
figures from a search summary of the abstract). LLM implementations are in `15` (shape-locked).

[D] Two-level model. Let f be the fraction of failures after which the local copy is unreadable
(node gone). Local level: delta1, tau1. Durable level: uploaded in the background every tau2 with
persist time P2.

```
overhead = delta1/tau1 + tau1/(2M) + f * (tau2/2 + P2)/M + R/M
```

At 70B, M = 8 h, delta1 = 10 s, tau1 = 12.6 min, P2 = 10 min, tau2 = 60 min: the base is 2.6%; the
durable term is f x (1800 + 600)/28800 = f x 8.3%. With f = 0.7 the total is 8.4%, worse than the
4.7% of single-level async-to-durable every 12.6 min. With f = 0.1 it is 3.4%.

Consequence: on unprivileged pods every reset-class Xid becomes node replacement (`04`) and
container disk does not survive (`16`), so f is high. A local tier then pays off only with M9 (a
buddy copy turns node loss back into a level-1 recovery) or when uploading every generation is
unaffordable. Steps on this dial: {local, local + buddy k = 1 or 2, durable} and their intervals,
plus an occasional portable full keyframe.

### 5.3 Full against delta, with keyframes (X1, X10)

A delta version costs 0.1-0.6 of a full one (section 2) but restore must replay base + chain.

- Dial: `keyframe_every = K`. Longest chain is K - 1 (ZipNN's own framing, same URL as above).
  ModelExpress caps replay at `max_replay_chain_length` 64 (`14`).
- [D] Storage per K versions: S x (1 + (K - 1) x r) against S x K. At r = 0.4, K = 8: 3.8 S
  against 8 S. Worst-case restore reads the same 3.8 S: at 70B weights and 0.125 GB/s, 4,256 s
  against 1,120 s for a full version. Storage falls as restore time rises.
- Intermediate setting: deltas against the keyframe rather than the previous version give chain
  length 1 at a worse ratio; ZipNN reports a base at distance 5 or 10 is "still far better than
  standalone compression".
- For RL actors the chain is also a late joiner's bootstrap cost (`12`).

### 5.4 Which state, how often (X1)

Weights are 2 of the 13.5 GB per billion parameters, about 15% [D]. Split cadences:

- weights every tau_w, optimizer every tau_o = m x tau_w.
- [D] Mixing the two is not an exact resume. Step count, LR position and data cursor must come
  from the weights' step. This sits between `15`'s Tier D (weights-only warm start) and exact
  resume, and must be labelled in run metadata. No source measures the quality effect.
- Stepped options: full state; weights + moments at reduced precision (lossy, ExCP-style: about 70x
  on Pythia-410M with "nearly lossless performance", https://arxiv.org/abs/2406.11257); weights
  only; adapter only.

### 5.5 Bit-width and density with error feedback (X2)

[F] Evidence from `08` and `14`: fp16, int8 and 4-bit outer gradients were free or nearly free;
dropping values without error feedback hurt in Streaming DiLoCo; with error feedback SparseLoCo and
Covenant-72B run at 1-3% density and 2 bits.

Dials: `bits` in {16, 8, 4, 2}; `density` from 1.0 down to 0.0156; `error_feedback`; chunk size.
[D] Sparse payload = N x density x (index_bits + value_bits) / 8. Dense int8 (1 GB x N) and the
Templar point (27 MB x N) are 37x apart; an intermediate setting such as density 10% with 12-bit
indices and 4-bit values is 0.1 x 16 = 1.6 bit/param = 200 MB x N.

Constraint: error feedback makes the residual part of training state. It must be checkpointed with
the island, or a restart silently drops the accumulated mass.

### 5.6 Sync frequency and staleness bounds (X5, X9)

- Cross-island: H inner steps per round. `14` s9 gives utilisation by H and model size; overlap
  hides the round when round time <= H x t_step. Dials: H, overlap delay (`fragment_sync_delay`),
  mixing alpha, quorum K of M, grace window.
- RL: publish every k trainer steps; `max_staleness` bounds how old a sampling policy may be
  (prime-rl default 8, runs at 64; `14` s10). Engines tag tokens with weight versions (`17`).
- [D] Feasibility: `ceil(T_transfer / t_step) + k <= max_staleness`. This inequality links codec,
  transport, cadence and bound; grid checks it whenever one of them is overridden.

### 5.7 Push against pull crossover (X4, X6, X9)

[D] Push (M11) costs one transfer leg and holds trainer and receivers in lockstep. Store-and-pull
(M15) costs two legs, 2S/B, but the trainer only pays the snapshot. Pull is forced when NCCL
reachability is absent. Where both exist, pull wins once the lag it adds fits the bound; the
bandwidth needed is

```
B* = 2 * S * 8 / (s * t_step)      (Gbit/s, S in GB, s = steps of lag allowed for transfer)
```

- 70B full weights, s = 4, t_step = 200 s: 2 x 140 x 8 / 800 = 2.8 Gbit/s.
- 10B: 2 x 20 x 8 / 800 = 0.4 Gbit/s.
- 70B with a lossless delta at ratio 0.2 (28 GB): 0.56 Gbit/s. With LoRA (0.34 GB): 7 Mbit/s.

Reference points [F]: NCCL across nodes took 23 s for 30B against a 208 s step; Shardcast moved
62 GB in 14 min (`12` s2.1). grid measures B in preflight, so it knows which side of B* each actor
pool is on.

### 5.8 Fragments (X4, X5)

Streaming DiLoCo splits parameters into F fragments synced at staggered steps; torchft implements
it (`num_fragments`; `14` s5). [D] Peak bytes per sync fall to S/F while the whole model is still
refreshed every H steps, so bandwidth demand is smoothed, not reduced. The same idea applies to RL
(vLLM `/update_weights` is repeatable and chunked, `12`) and to checkpoint uploads. A durable
checkpoint taken mid-cycle must include per-fragment round counters and outer-optimizer state.

### 5.9 Fallback ladders (X8)

A ladder is an ordered list of transports for one purpose; grid tries the first, and on timeout or
integrity failure degrades. Examples:

- (c) joiner: live peer pull (M10) -> buddy replica (M9) -> local generation -> durable store (M4).
- (e) actor update: NCCL push (M11) -> object-store delta (M15) -> full version (M15) -> restart
  with a streaming loader (M17). A failed in-place update marks the replica unusable (no rollback
  exists, `14`), so the last rung is always a clean reload.
- (f) cold start: `ipc_cache` (M17) -> island swarm (M16) -> object store (M15).

Each rung has a deadline from its predicted time; the worst case is their sum.

### 5.10 Unsafe or pointless combinations

| Combination | Why |
|---|---|
| Lossy codec (M6, M7, low-rank approximation) as the only durable recovery copy | Reconstruction is not the trained state; error-feedback residual is lost |
| Delta chain with no keyframe, or retention that deletes a base still referenced | One missing link invalidates every later version; restore time unbounded |
| Deltas computed against the sender's exact weights while receivers hold a lossy or re-quantised base (FP8 actors) | Receiver drifts; XOR deltas only apply to a bit-identical base [D] |
| Rank-keyed format or shape-locked local tier used for (g) | Does not reshard (`15`) |
| Local-only tier on container disk as the sole copy | Wiped with the pod (`16`) |
| Buddy replica in the same host or cluster product as its source, as the only non-local copy | Correlated loss; an Instant Cluster is one failure domain (`16`) |
| Single generation with overwrite in place | A crash mid-write leaves no checkpoint; no rollback for corruption (`04`) |
| Async interval shorter than persist time P | Saves queue or block; the newest complete generation never advances |
| Push (M11, M13) to preemptible or NAT-separated receivers | Unreachable (`05`), or the handshake stalls the trainer until timeout (3600 s in prime-rl, `14`) |
| Live peer pull (M10) across providers | Pickle over an unauthenticated port; advertises an unreachable hostname (`14`) |
| Sparse delta to inference servers without error feedback or periodic full refresh | Actor weights diverge from the learner; logprob mismatch grows unobserved |
| LoRA transport when the trainer updates all weights | Cannot represent the update |
| Dense sync through an egress-priced store | Egress exceeds everything else (`14` s9) |
| Weights and optimizer restored from different steps and reported as exact resume | It is a discontinuity (5.4) |

## 6. Proposed abstraction [P]

### 6.1 Vocabulary

Six fields. Every catalogue method is a combination; a policy is one value per field, plus an
optional `fallback` list of further policies.

| Field | Values | Dials |
|---|---|---|
| `payload` | `content`: weights, optimizer, ema, cursor, rng, residual. `form`: full, delta(base), adapter. `layout`: logical or rank_local. `fragments`: F | optimizer cadence multiple; F |
| `codec` | `dtype`: native, bf16, fp8, int8, int4, int2. `sparsify`: none or topk(k, chunk). `error_feedback`. `lossless`: none, zstd, bytegroup, xor+zstd | bits, density |
| `transport` | `path`: local, shared_fs, object_store, peer, collective, ipc, rdma, swarm. `mode`: push or pull. `fanin`: per_rank or funnel | parallel parts, part size |
| `trigger` | every(steps or seconds), young_daly, outer_boundary(H), on_publish, on_signal, staleness(max) | tau, H, k, max_staleness |
| `commit` | `blocking`: block, staged, background. `marker`: manifest_last, conditional_put, sentinel, quorum(K of M), per_replica. `verify`: none, crc, sha256, tensor_checksum. `on_failure`: keep_previous, discard_round, mark_unusable | quorum, deadline, grace |
| `retention` | keep_last, keyframe_every, keep_every (sparse history), max_chain, tiers | generations, K |

Every catalogue method is a point in this space. The presets of 6.2 name the methods they
combine; the rest are one-field variants: M1 is `durable-portable` with `transport.path: shared_fs`
and `commit.blocking: block`; M12, M13 and M14 are `engine-push` with `transport.path` set to `ipc`,
`rdma` or `shared_fs` (the last with `commit.marker: sentinel`); M16 is `registry-pull` with
`transport.swarm` set.

### 6.2 Configuration sketch

```yaml
state:
  defaults: {store: r2://grid-run-42, verify: sha256}
  policies:
    recover:                      # (a)
      preset: durable-portable
      trigger: {young_daly: {min: 5m, max: 60m}}     # override: or every: 15m
      retention: {keep_last: 3, keep_every: 6h}
    restart:                      # (b)
      preset: local-buddy
      payload: {optimizer_every: 1}
      transport: {replicas: 1}
    join:                         # (c)
      ladder: [peer-heal, local-buddy, durable-portable]
    island_sync:                  # (d)
      preset: relay-int8
      trigger: {outer_boundary: {H: 100, overlap: true}}
      codec: {dtype: int4}        # one-field compromise toward relay-sparse
      commit: {quorum: "3 of 4", grace: 120s}
    rl_weights:                   # (e)
      ladder: [engine-push, store-delta]
      trigger: {on_publish: {every_steps: 1}, staleness: {max: 8}}
      retention: {keyframe_every: 16, max_chain: 16}
    rollout:                      # (f)
      preset: registry-pull
      transport: {swarm: island, loader: runai_streamer}
    migrate:                      # (g)
      preset: durable-portable
      commit: {verify: sha256, self_test: true}
```

Presets:

| Preset | payload | codec | transport | trigger | commit | retention | Methods |
|---|---|---|---|---|---|---|---|
| `durable-portable` | full, logical | native | object_store, per_rank | young_daly | staged + background, manifest_last, sha256 | 3 + sparse | M2 + M4 |
| `durable-compact` | weights full; optimizer every m-th | bytegroup; xor against keyframe | object_store | young_daly | same | keyframe_every 8 | M4 + M5 |
| `local-buddy` | full, rank_local | native | local + peer | every 5-15 min | staged | 2 | M3 + M9 |
| `peer-heal` | full, live | native | peer, pull | on join | step commit | none | M10 |
| `relay-int8` | delta(global) | int8 | object_store | outer_boundary | quorum | 2 rounds | M6 |
| `relay-sparse` | delta + residual | topk 64/4096, 2-bit, error feedback | object_store | outer_boundary | deadline | 2 rounds | M7 |
| `engine-push` | weights | native | collective or rdma | on_publish | block, tensor_checksum | none | M11, M13 |
| `store-delta` | delta(base) | xor + zstd | object_store, pull | on_publish | conditional_put | keyframe_every 16 | M15 |
| `adapter` | adapter | native | object_store or shared_fs | on_publish | per_replica | versions | M8 |
| `registry-pull` | weights, immutable | as stored | object_store, optional swarm; streaming loader | on start | sha256 | versions | M15-M17 |

### 6.3 Validation before launch

| Check | Source of truth | Outcome |
|---|---|---|
| Link tier between sender and receivers | Preflight pairwise matrix (`05` s9) | `collective`/`rdma` refused below tier B; ladder falls to the next rung |
| Port reachability | Preflight TCP connect both ways; mapped-port table (`16`) | `peer`, `swarm` and Gloo-dependent async saves refused where not symmetric |
| Checkpoint capability of the job | Declared capabilities + `grid ckpt verify` (`15`) | `layout: logical` required for (g); rank-keyed jobs limited to same-shape |
| Host RAM for staging | cgroup limit (`09`) against per-rank state | `staged` downgraded to `block` or to fragments |
| Local disk, and whether it survives restart | Provider driver (`16`) | `local` allowed only with a buddy or durable tier behind it |
| Store pricing and location | Driver table: storage, operation and egress prices | Warn or refuse dense sync on egress-priced stores |
| Store semantics | Probe conditional put at start | Selects `conditional_put` or `manifest_last` |
| Engine support | Engine version and flags (`17`) | `engine-push`, LoRA and checker routes enabled or not |
| Rules of 5.10; staleness inequality (5.6) | Static; cost model | Errors, each with an explicit override |

### 6.4 Cost model and auto-selection

Inputs: parameter count and bytes per parameter per payload; G GPUs on n nodes; measured GPU-to-host
rate, local write rate, per-node uplink and downlink to the store, pairwise link rates; MTBF
estimate M and node-loss fraction f (prior from `04`, updated from grid's own telemetry); step
time; restart time R; prices.

Outputs, using the formulas of section 5:

- per purpose: blocking time, persist time, loss-of-work window, overhead, bytes and dollars per
  month, for the chosen policy **and each alternative preset**;
- a recommended default: lowest predicted overhead among policies that pass validation;
- sensitivity to M and uplink halving.

Example output for the 70B, 8-node, M = 8 h case of 5.1 [D]:

```
recover   durable-portable  tau 12.6m  block 10s  persist 10m  overhead 4.7%   <- recommended
          blocking volume   tau 44m    block 120s              overhead 9.1%
          blocking WAN      tau 98m    block 600s              overhead 20%
          local-buddy + hourly durable (f=0.1)                 overhead 3.4%   (needs peer ports)
```

Dollars [D] for `durable-portable` on R2 at this cadence: 3 generations = 2.8 TB x $0.015 = $43
per month; multipart parts of 64 MB give 945,000/64 = 14,766 writes per checkpoint, x 4.76 per hour
x 720 h = 50.6M operations x $4.50 per million = $228 per month; 512 MB parts cut that to $28.
Part size is therefore a cost dial. R2 prices are from `14` s9.

M and f are the weakest inputs: start from the pessimistic column of `04` s1 and re-derive as
failures are observed.

### 6.5 Hooks

What grid can do with **no job cooperation**, given only that the job writes complete checkpoints
into a directory grid names:

- transport, tiering, retention, hashing, manifest commit, restore staging, swarm distribution;
- lossless compression at rest, and XOR deltas between generations when file layouts match;
- the interval, where the framework takes it as a launch flag (torchtitan `interval`);
- for serving: loader choice, preload daemons, calls to the engine's update, LoRA and checker
  routes, version bookkeeping per replica.

What needs a hook:

| Choice | Interface the job or engine must expose |
|---|---|
| `payload.content` split, weights-only export | A save entry point taking a list of parts; or separate files per part |
| `layout: logical` | A reshardable format (`15` contract) |
| `commit.blocking: staged` | Async staging in the framework (DCP stager, NVRx) |
| Trigger changes at run time, `on_signal` | Poll a grid control file or socket each step, or a signal handler |
| Lossy codecs, error feedback, fragments, outer loop | An optimizer post-step hook in the shape of torchft's (`14`): snapshot at H - delay, merge at H |
| `peer` live pull | `state_dict()` / `load_state_dict()` callbacks and a step-commit protocol (torchft `Manager`) |
| `collective` / `rdma` push to engines | Trainer calls the engine's sender (vLLM `send_weights`), with rendezvous details supplied by grid |
| RL staleness | Engine returns weight-version spans; orchestrator applies the bound (`14`, `17`) |

Minimum contract: (1) a directory protocol (the job writes `step-N.tmp/`, renames to `step-N/`);
(2) a capability declaration in the job spec (format, layout, async, parts, trigger mechanism);
(3) an optional Python shim with `should_save()`, `save(parts)`, `restore(parts, path)` and
`publish(version)` for run-time switching. A preset needing a missing capability is rejected at
submit time with the nearest feasible alternative.

## 7. Evidence gaps

Reading cannot settle these.

1. Lossless delta ratios for LLM pretraining checkpoints at 5-30 minute spacing, for weights and
   Adam moments separately.
2. Quality effect of restoring weights and optimizer from different steps (5.4).
3. Upload throughput from marketplace pods to R2 and provider S3 endpoints with 8-64 parts, which
   sets P and therefore every number in 5.1.
4. Host RAM headroom for staged saves under container memory limits; f and MTBF on marketplace
   hardware.
5. Buddy replication over tier C sockets: time per 100 GB and effect on step time.
6. Whether `ipc_cache` and CUDA IPC work in unprivileged pods.
7. NCCL push to engines over tier C TCP against store-and-pull: the measured crossover.
8. Dense int8 against intermediate densities against the Templar point at equal token budget.
9. XOR + zstd delta size per RL step, and apply time on FP8 actors holding a bf16 base.
10. Swarm distribution inside an Instant Cluster against parallel pulls from a zero-egress store.
11. SHA-256 throughput in the uploader at multi-GB/s.

## Corrections and refinements to earlier files

- **`15` s8, "lossless deltas buy little"** and its unverified citations. ExCP and Check-N-Run are
  now confirmed from their abstracts (about 70x on Pythia-410M, lossy at the data level; 6-17x
  write bandwidth on recommendation models, where only part of the model changes per step:
  https://arxiv.org/abs/2010.08679). The "buy little" judgment is too strong for weights: published
  lossless results are 66% standalone for bf16 and delta ratios of 0.6 falling toward 0.1-0.2. It
  remains untested for optimizer moments, which are 85% of a full checkpoint.
- **`14` s10, XOR + zstd "expected ratio is modest and unmeasured"**: partly measured elsewhere.
  ZipNN reports 56% for XOR deltas between fp32 fine-tunes against 83.7% standalone, and that zstd
  beats Huffman coding only when more than 90% of delta bytes are zero.
- **`04` s4.1 table**: correct for blocking saves. For async saves add (P/M) to the overhead and
  clamp tau to at least P (5.1).
- **`04` s4.2 and `15` on two tiers**: an HPC primary source exists (SCR). The tier pays off in
  proportion to (1 - f); on pods it needs buddy replication (5.2).
- **`14` Implication 3** (manifest written last as commit point): S3 and R2 conditional puts allow
  a stronger compare-and-swap commit.

## Implications for grid

1. Make state policy a per-purpose table with the six fields of 6.1, not a global checkpoint
   setting. Ship the ten presets of 6.2; every override is one field.
2. Default (a) to `durable-portable`, with the interval from the extended Young/Daly formula.
3. Offer `local-buddy` only where preflight shows symmetric ports inside the island, and never as
   the sole copy.
4. Default (d) to `relay-int8` with overlap; expose bits and density as dials toward `relay-sparse`
   and persist the error-feedback residual in island checkpoints.
5. Default (e) to a ladder: engine push inside a fabric, object-store delta across providers, clean
   reload last. Enforce the staleness inequality at submit time and treat a failed in-place update
   as an unusable replica.
6. Treat (f) as content-addressed immutable versions pulled by each replica; swarm only inside an
   island; streaming loaders or cache daemons for the last hop.
7. Build the agent-side pieces first, since they need no job cooperation: per-node uploader with
   hashing, manifest commit, keyframe-aware retention, restore staging.
8. Build the cost model as part of `grid plan`: show predicted overhead, bytes and dollars for the
   chosen policy and for the alternatives, from preflight measurements.
9. Encode section 5.10 as static validation with explicit overrides.
10. Record in each manifest the policy that produced it (parts, codec, base hash, saving shape).

## Open questions

- Is a mixed-step restore (5.4) acceptable as a named mode, or should grid refuse it until its
  quality effect is measured?
- Who owns the error-feedback residual and outer-optimizer state on restart: the island's
  checkpoint, or the coordinator of `14`?
- For FP8 actors, where is the bf16 base for XOR deltas held (host RAM, local disk), and what does
  that cost per replica?
- Can lossless delta coding run inside the uploader fast enough to lower P, or does CPU become the
  new bottleneck on nodes with few cores per GPU (`06`)?
- What prior for M and f should grid ship before it has telemetry, per provider and tier?
- Does any provider's S3-compatible endpoint (RunPod network-volume API) support conditional puts
  and multipart semantics well enough to serve as the commit store, or must the manifest live on
  R2-class storage even when data sits on a provider volume?
