# 14 - Cross-island sync: implementations grid could adopt for the slow-link tier

Second sweep, 2026-11-09. Builds on `08-heterogeneous-distributed-training.md` (algorithms),
`12-inference-for-rl.md` (RL weight sync) and `05-networking-interconnects.md` (what RunPod-class
networks allow). Method: repositories were shallow-cloned and read at the commits named below;
GitHub's API was not reachable from this session, so issue lists come from the rendered issue pages.
Anything not read in source is marked (unverified).

Commits read:

| Repo | Commit / version | Date | Licence |
|---|---|---|---|
| https://github.com/PrimeIntellect-ai/pccl | `2af926a`; PyPI `pypccl` 1.0.0 (only release, 2025-05-21) | 2025-09-12 | MIT |
| https://github.com/PrimeIntellect-ai/prime-diloco (`prime`, INTELLECT-1 code) | `de5b931` | 2025-04-10 | (not checked) |
| https://github.com/PrimeIntellect-ai/OpenDiloco | `2d750e5` | 2025-01-13 | (not checked) |
| https://github.com/PrimeIntellect-ai/prime-rl | `83ca6fd` | 2026-11-09 | Apache-2.0 |
| https://github.com/learning-at-home/hivemind | `4bd43b7`; PyPI 1.1.12 (2026-01-03) | 2026-01-04 | MIT |
| https://github.com/PsycheFoundation/psyche | `0bdb13d` | 2026-03-20 | Apache-2.0 |
| https://github.com/meta-pytorch/torchft | `43e3b6d`; PyPI 0.3.1 (2026-11-03), nightly 2026.11.9 | 2026-11-06 | BSD-3-Clause |
| https://github.com/pytorch/torchtitan | `8a33dbf` | 2026-11-09 | (not checked) |
| https://github.com/facebookresearch/MuLoCo | `475bbf4` | 2026-02-26 | MIT per README badge; no LICENSE file at root |
| https://github.com/one-covenant/SparseLoCo | `276fa59` | 2025-08-22 | MIT |
| https://github.com/one-covenant/templar | `0fefc2a` | 2026-10-04 | MIT |
| https://github.com/exalsius/nanochat-diloco | `8b00f95` | 2025-11-06 | (not checked) |
| https://github.com/gensyn-ai/noloco | `a1b4a42` | 2026-03-26 | (not checked) |
| https://github.com/vllm-project/vllm | `692cf2d`; PyPI 0.33.0 | 2026-11-09 | Apache-2.0 |

## Summary

- **No existing library is a drop-in outer-loop transport for RunPod-style pods** (outbound TCP, a
  few mapped inbound TCP ports with external port != internal port, no UDP). PCCL, torchft and
  `prime` all advertise the port a process bound locally and assume it is reachable as-is; Psyche's
  transport is QUIC/UDP. Only hivemind (separate listen and announce addresses) and object-store
  exchange (outbound HTTPS only) fit without patches.
- **Object-store relay is proven at 72B, not just plausible.** Covenant-72B (Templar, Bittensor
  subnet 3) trained a 72B model for ~1.1T tokens with every peer uploading compressed
  pseudo-gradients to its own Cloudflare R2 bucket and downloading the others'. Reported: H = 30,
  20-minute compute window, ~70 s communication per round, ~94.5% compute utilisation, assumed
  links 110 Mb/s up / 500 Mb/s down (https://arxiv.org/html/2603.08163v2). The code is MIT.
- **A public Streaming DiLoCo implementation exists**: `torchft.local_sgd.DiLoCo` (Meta, BSD-3),
  wired into torchtitan under `torchtitan/experiments/torchft`. It has fragments, overlap
  (`fragment_sync_delay`) and mixing (`fragment_update_alpha`), fp8 quantised all-reduce, a
  lighthouse quorum service and live peer-to-peer healing. It is labelled experimental and its
  transports need a flat network. No public Decoupled DiLoCo (syncer, quorum K, grace window) code
  was found.
- **PCCL is the most complete WAN collective library but is dormant**: one PyPI release (May 2025),
  last commit September 2025, 8 issues total. Its semantics (unanimous voting, one operation at a
  time, abort-and-retry on any peer loss, master crash ends the run) are synchronous by construction.
- **prime-rl no longer has `max_async_level`.** The bound is now `orchestrator.max_staleness`
  (default 8), applied per token as a loss mask; rollouts are never cancelled for staleness. Weight
  broadcast is a four-sentinel handshake in a shared directory with pluggable data planes
  (`filesystem`, `nccl`, `nixl`, `modelexpress`). None crosses providers by itself.
- **vLLM 0.33 has an object-storage path for weight updates**: the `modelexpress` backend installs
  immutable versions and can replay zstd-compressed XOR deltas from S3-compatible storage. This is
  the natural cross-provider RL weight tier.
- **Recommendation: build grid's outer-loop coordinator; adopt pieces.** Build a small syncer with an
  object-store data plane as the universal baseline and an optional direct-TCP fast path. Reuse
  torchft's DiLoCo fragment logic, Templar's chunked top-k codec, PCCL (patched) where symmetric
  ports exist, and ModelExpress for RL actors.

---

## 1. PCCL (Prime Collective Communications Library)

**API** (`include/pccl.h`): `pcclCreateCommunicator` (params `master_address`, `peer_group`,
`p2p_connection_pool_size`), `pcclConnect`, `pcclUpdateTopology`, `pcclArePeersPending`,
`pcclOptimizeTopology`, `pcclAllReduce`, `pcclAllReduceAsync` / `pcclAwaitAsyncReduce`,
`pcclAllReduceMultipleWithRetry(descriptors, count, comm, info, max_in_flight)`,
`pcclSynchronizeSharedState(comm, shared_state, strategy, info)`, and master lifecycle
`pcclCreateMaster` / `pcclRunMaster` / `pcclInterruptMaster`. Python: `pip install pypccl`, Python
3.12+, CUDA 12+ optional.

**Discovery and ports** (`ccoip/public_include/ccoip.h`, `ccoip/src/cpp/ccoip_client_handler.cpp`,
`ccoip_master_handler.cpp`):

- `CCOIP_PROTOCOL_PORT_MASTER 48148` (persistent control connection from every peer),
  `CCOIP_PROTOCOL_PORT_SHARED_STATE 48149`, `CCOIP_PROTOCOL_PORT_P2P 48150`,
  `CCOIP_PROTOCOL_PORT_BANDWIDTH_BENCHMARK 48151`.
- Each peer opens three listening sockets at "the first free port above the specified port
  number ... which will bump on failure to bind", then reports `p2p_listen_port`,
  `shared_state_listen_port`, `bandwidth_benchmark_listen_port` to the master. The source comment
  states "The only asserted static port is the master listening port."
- The master hands other peers `{.inet = client_info.socket_address.inet, .port =
  client_info.variable_ports.p2p_listen_port}`: the IP is the **source address the master observed**,
  the port is the peer's **local bind port**. Both directions are dialled (each peer opens its TX
  connection to the other's P2P port), so every peer needs inbound reachability.
- The base ports are compile-time constants; the only environment variable in the library is
  `PCCL_MULTIPLEX_CHUNK_SIZE` (default `8388608 * 8` bytes).

**RunPod fit**: fails unpatched. A pod's mapped TCP port has a random external number, so the
advertised `48150` is wrong; the symmetric-port mechanism (ports above 70000, see `05`) hands out a
port number PCCL cannot be told to bind. It would also need the pod's outbound source IP to equal its
inbound public IP (unverified for RunPod). The fix is small (make the three base ports and the
advertised address configurable) and the licence allows it, but it means carrying a fork of a C++20
library. All `p2p_connection_pool_size` connections share the one P2P port, so the requirement is
three inbound TCP ports per PCCL peer.

**Joiner state**: `pcclSynchronizeSharedState` with a `revision` counter and per-tensor hashes
(`simplehash`, CPU or CUDA). The master elects the most popular `(key, hash)` set; outdated or
mismatched peers open an ephemeral TCP connection to a designated up-to-date peer's shared-state
port, then re-hash. Strategies: `PCCL_SHARED_STATE_SYNC_STRATEGY_ENFORCE_POPULAR`, `_RECEIVE_ONLY`,
`_SEND_ONLY`. A joiner blocks in `pcclConnect` until all accepted peers vote to accept it via
`pcclUpdateTopology`. State is never stored by the master: "CCoIP itself is not designed to
'retain' shared state" (`docs/md/05-ImplementationNotes/03_MasterOrchestration.md`).

**Failure semantics**: if a peer vanishes mid-collective, "The All-Reduce call returns an error to
other peers ... The remaining peers can simply retry"; worst case one fewer peer per retry. "Master
Crash = End of Run ... no built-in master re-election or replication." "One Operation at a Time
(Per Group)": every phase needs unanimous agreement of accepted peers.

**Compression**: `pcclQuantizationOptions_t` with `pcclQuantMinMax` or `pcclQuantZeroPointScale` to
any of `pcclUint2 .. pcclDouble`; the async DiLoCo example uses `DataType.UINT8`. No sparsification.

**Async mode**: `docs/md/07-DiLoCo.../03-AsyncDiloco.md` is a one-phase-delayed outer update: a
background thread runs `all_reduce_multiple_with_retry` while the next inner phase runs. It is
overlap, not asynchrony: every peer must still participate in every reduce, and newcomers need a
second shared-state sync to "eavesdrop" on the in-flight result.

**With FSDP** (`docs/md/8_CommonFootguns.md`): either one PCCL peer per FSDP rank with
`peer_group` = FSDP rank (requires identical sharding in every island, and 3 ports per rank), or a
single process per island that memory-maps all shards. The docs warn: "When using FSDP, the fault
tolerance of PCCL is significantly reduced ... the remaining peers will block forever during the
next all-gather" until the NCCL timeout.

**Maturity**: issues page shows 8 issues, 1 open (#28 "PCCL Vs Cross over support in NCCL", Jan
2026); the last closed (#27, Jan 2026) is a build break from a dependency update
(https://github.com/PrimeIntellect-ai/pccl/issues). `pccl` does not appear in prime-rl's
`pyproject.toml`, although prime-rl now has a `uv run pretrain` entrypoint. Throughput figures remain
those of the paper quoted in `08`; nothing newer was found.

## 2. `prime` (INTELLECT-1 code) and OpenDiLoCo

`prime-diloco/src/zeroband/comms.py`: membership lives in `torch.distributed.TCPStore` instances on
`global_addr:global_port` (+ rank); defaults `ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS=2`,
`ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS=10`, `ZERO_BAND_GLOBAL_PG_TIMEOUT_SECONDS=600`,
`ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS=300`, `MAX_JOINERS = 100`. Joiners download the latest
checkpoint from a peer's sidecar HTTP server serving `/dev/shm`, then "join the outer step with zero
pseudo-gradients". Compression is `uniform_8bit_quantize` / `quantile_8bit_quantize`
(`compression.py`). The README states public-IP forwarding gave "poor or unstable p2p bandwidth on
some compute providers" and that a VPN was used; a VPN is not available on pods without UDP. This
code is superseded by PCCL and should not be adopted.

OpenDiLoCo (`open_diloco/hivemind_diloco.py`, 738 lines) pins `hivemind @ git+...@213bff9`.
`DiLoCoOptimizer` takes `num_inner_steps`, `grad_compression`, `matchmaking_time=15.0`,
`all_reduce_strategy` (`WAIT_FOR_ALL` or `NO_WAIT`), `timeout_waiting_for_peers`. It also ships a
plain `train_diloco_torch.py` on NCCL. Useful as a reference for how DiLoCo maps onto hivemind's
averagers, not as a dependency.

## 3. hivemind

- **Discovery**: Kademlia DHT over a libp2p daemon; peers bootstrap from `initial_peers` multiaddrs.
- **NAT** (`hivemind/p2p/p2p_daemon.py`): `host_maddrs` (listen; default `/ip4/127.0.0.1/tcp/0`)
  is separate from `announce_maddrs` ("Visible multiaddrs that the peer will announce"; an explicit
  port is required). Also `use_relay=True`, `use_auto_relay=False`, `auto_nat=True`,
  `nat_port_map=True`, `force_reachability`, `use_ipfs`. `quic` is "Deprecated, has no effect since
  libp2p 0.17.0", so the transport is TCP. `client_mode=True` makes a peer outbound-only.
- **RunPod fit**: works with one mapped TCP port per peer (listen on the internal port, announce
  `/ip4/<public>/tcp/<external>`), or none for `client_mode` peers as long as some peers are public.
  This is the only P2P stack surveyed that fits without patching.
- **Averaging** (`hivemind/averaging/averager.py`): `DecentralizedAverager` defaults
  `min_group_size=2`, `min_matchmaking_time=5.0`, `request_timeout=3.0`, `declare_state_period=30`,
  optional `allreduce_timeout`, `next_chunk_timeout`, `sender_timeout`, `reducer_timeout`,
  `compression`, `state_compression`, `bandwidth`. A group that loses a peer continues with the parts
  it has; rounds are best-effort.
- **Joiner state**: `load_state_from_peers()` downloads "the latest optimizer state [from] one of
  the existing peer[s]".
- **Compression** (`hivemind/compression/`): `floating.py`, `quantization.py`, `adaptive.py`.
- **Throughput**: no primary measurement found for multi-GB payloads. Prime Intellect's README says
  its custom stack improved bandwidth utilisation "by up to 40x compared to our OpenDiLoCo release",
  which is indirect evidence that hivemind's Python/protobuf data path is slow for 10 GB tensors.
- **Maturity**: maintained at low volume (1.1.11 April 2025, 1.1.12 January 2026).

Verdict: suitable as a rendezvous layer, not as the bulk data plane.

## 4. Nous Psyche

Rust workspace version 0.2.0; networking on `iroh = 0.97.0`, `iroh-relay`, `iroh-blobs = 0.99`,
`iroh-gossip` (`Cargo.toml`). `shared/network/src/lib.rs` defines `DiscoveryMode {Local, N0}` and
`RelayKind {Disabled, Psyche, N0}` with Psyche relays `use1-1.relay.nousresearch.psyche.iroh.link`
and `usw1-1...`. iroh's direct paths are QUIC over UDP; on pods without UDP, traffic would have to
stay on the relays (inference from iroh's design; unverified in practice), which are a shared,
rate-limited service. Joiner state comes from `enum Checkpoint { Ephemeral, Dummy, Hub(HubRepo),
P2P(HubRepo), Gcs(GcsRepo), P2PGcs(GcsRepo) }` (`shared/coordinator/src/model.rs`): Hugging Face
Hub, GCS, or peers. The optimiser is `OptimizerDefinition::Distro { compression_decay,
compression_topk, compression_chunk, quantize_1bit, .. }`; the book says "only the DisTrO optimizer
is supported when training models on Psyche". There is an `architectures/centralized` server/client
pair that removes the Solana dependency. The repository's last commit is March 2026.

Verdict: per-step DisTrO plus a UDP transport makes it a poor fit for grid's island model; the
`Checkpoint` enum (object store or P2P for joiners) is the idea worth copying.

## 5. torchft and torchtitan: a public Streaming DiLoCo

`torchft/local_sgd.py` contains `LocalSGD`, `_StreamingDiLoCoFragment` and `DiLoCo`. The `DiLoCo`
docstring: "The class implements a more general version of DiLoco, Streaming DiLoCo, which
synchronizes fragments of pseudogradients at different steps." Constructor arguments:
`model_fragments`, `inner_optimizer`, `outer_optimizer` (one or one per fragment), `sync_every`,
`backup_device`, `pin_memory`, `should_quantize`, `fragment_sync_delay` ("the 'tao' parameter in the
Streaming DiLoCo paper"), `fragment_update_alpha`. Constraints in code: `sync_every` must be a
multiple of the fragment count and `fragment_sync_delay < sync_every // len(model_fragments)`.

Mechanics, from `_step_post_hook`: at local step `sync_every - fragment_sync_delay` it calls
`manager.start_quorum()` and `prepare_sync()` (compute pseudo-gradient, start all-reduce without
waiting); at `sync_every` it calls `perform_sync()`, which waits, asks `manager.should_commit()`,
and only then applies the outer optimiser and `lerp_`-merges with `fragment_update_alpha`. On
failure "the step will be discarded and the model parameters will reset back to the last time DiLoCo
synchronized". A code comment admits `should_commit` "can return success even if the allreduce
failed" when the group was reconfigured in flight.

Supporting pieces:

- **Lighthouse** (`src/lighthouse.rs`): flags `--bind` (default `[::]:29510`), `--join_timeout_ms`
  (60000), `--min_replicas`, `--quorum_tick_ms` (100), `--heartbeat_timeout_ms` (5000). Managers
  find it via `TORCHFT_LIGHTHOUSE`.
- **Manager** (`torchft/manager.py`): defaults `timeout`, `quorum_timeout`, `connect_timeout` all
  60 s; overrides `TORCHFT_TIMEOUT_SEC`, `TORCHFT_QUORUM_TIMEOUT_SEC`,
  `TORCHFT_CONNECT_TIMEOUT_SEC`, `TORCHFT_QUORUM_RETRIES`, `TORCHFT_MANAGER_PORT`. It binds
  `[::]:{port}` and advertises `hostname: str = socket.gethostname()` plus the replica group's
  `store_addr:store_port` (`MASTER_ADDR`/`MASTER_PORT`).
- **Process groups** (`torchft/process_group.py`): `ProcessGroupGloo`, `ProcessGroupNCCL`,
  `ProcessGroupXCCL`, `ProcessGroupBabyGloo`/`BabyNCCL` (subprocess-isolated so a hung collective
  can be killed), `ProcessGroupTorchComms`. torchtitan exposes `process_group: Literal["gloo",
  "nccl", "mccl"] = "gloo"` and `process_group_timeout_ms: int = 10000`.
- **Healing** (`torchft/checkpointing/`): `HTTPTransport` serves the live state dict from a healthy
  peer at `http://{socket.gethostname()}:{port}/checkpoint/` and loads with `weights_only=False`
  (pickle; trusted peers only); `PGTransport` sends it over the process group.
- **Quantisation**: `allreduce_quantized` with Triton fp8 kernels and per-row scales
  (`torchft/quantization.py`); torchtitan's `should_quantize` defaults to `False` because it "does
  utilize the GPU".
- **torchtitan** (`torchtitan/experiments/torchft/`): `semi_sync_method` (`"diloco"` or
  `"local_sgd"`), `sync_steps: int = 5`, `fragment_sync_delay: int = 0`,
  `fragment_update_alpha: float = 0.0`, `num_fragments: int = 1`, `replica_id`, `group_size`. Each
  replica group is a separately launched torchtitan instance running FSDP internally: exactly the
  island model. The README says the smallest alive replica ID does checkpoint saving.
- **MuLoCo** bundles forks of torchtitan and torchft and adds `QuantizedOuterSGD` and Muon as inner
  optimiser; its README notes the published experiments used an internal dataset.

**RunPod fit**: poor. The cross-replica group is Gloo or NCCL bootstrapped from a TCPStore, i.e.
full-mesh connections to self-reported addresses on ephemeral ports, and the HTTP healing transport
advertises `gethostname()` on an ephemeral port. It needs a flat routable network between all
islands. It is also still a quorum-per-sync design: a slow replica delays the fragment, and there is
no token weighting or grace window. README: "LocalSGD and DiLoCo are currently experimental"; open
issues include #39 "LocalSGD / DiLoCo support" and #171 "Towards Native Fault Tolerance for
Semi-Synchronous Training" (https://github.com/meta-pytorch/torchft/issues?q=is%3Aissue+diloco).

Verdict: the best available reference for the trainer-side hook (where to snapshot, when to start
the reduce, how to merge), and directly usable when islands share a private network.

## 6. SparseLoCo, Templar and Covenant-72B: the object-store relay in production

SparseLoCo's reference repo is single-cluster research code: `src/tplr/sparseloco.py` exchanges with
`dist.all_gather` over an NCCL process group, with baselines for DiLoCo, DeMo and AdamW. The
production system is Templar (`templar/src/tplr/comms.py`, 2822 lines; `compress.py`, 872 lines):

- **Transport**: S3 API against `https://{account_id}.r2.cloudflarestorage.com`, one bucket per
  peer. `s3_put_object` switches to multipart above `100 * 1024 * 1024` bytes; `upload_large_file`
  and `download_large_file` run parallel parts with resume (`_get_download_resume_info`), part size
  overridable by `TPLR_S3_PART_SIZE`; download timeout is `max(300, file_size / (25 * 1024 *
  1024))`, i.e. the code budgets for 25 MB/s.
- **Round structure**: time windows driven by chain blocks (`blocks_per_window = 115`); peers
  `put` their compressed pseudo-gradient, then `gather` from the selected peers
  (`gather_peer_count = 20`, `gather_share = 0.75`, gather timeout 600 s,
  `time_window_delta_seconds = 70`). A late or missing upload is simply absent from the gather.
- **Codec** (`compress.py`, `hparams/hparams.json` with `model_size = 70B`): `ChunkingTransformer`
  splits tensors into chunks; `TopKCompressor` keeps `topk_compression = 64` values per chunk;
  indices use `pack_12bit_indices` ("Every 2 indices (24 bits) are packed into 3 uint8 values");
  values are quantised with `quantization_bins = 4`, `quantization_range = 6`. `inner_steps = 30`,
  `outer_learning_rate = 0.4`.
- **Paper figures** (https://arxiv.org/html/2603.08163v2): k = 64 per chunk of 4096 (~1.56%
  density), 2-bit values, "a compression ratio of more than 146x"; target 20 peers per round,
  observed 24.4 active and 16.9 contributing on average; peers expected to have at least 8 B200
  GPUs, each sharding with FSDP; fixed 20-minute compute window; ~70 s average communication;
  ~94.5% utilisation; assumed 500 Mb/s down and 110 Mb/s up. Quality is reported against other
  models of similar size (e.g. MMLU 67.1 vs 65.6 for LLaMA-2-70B), not a same-recipe centralised run.
- The README says the Covenant-72B run is complete and the protocol "is not currently active".

Check on the ratio: (64 x (12 + 2) bits) / (4096 x 32 bits) = 896 / 131072 = 1/146.3. So the payload
is 0.219 bits per parameter: about 1.97 GB per peer per round at 72B (derived, not stated in the
paper). At the paper's minimum 110 Mb/s uplink that upload alone is ~143 s, so the reported 70 s
average implies typical peers were faster than the stated minimum (my inference).

How joiners got state was not found in the paper; the repo has `dcp_checkpoint.py` and
`checkpoint_init_window` (not read in detail).

## 7. Other DiLoCo code

- **nanochat-diloco** (`nanochat/diloco.py`): plain blocking DiLoCo via `dist.all_reduce(...,
  op=AVG)` on NCCL inside one cluster. No WAN transport.
- **NoLoCo**: research code on `torch.distributed` process groups (`create_process_groups.py`).
- **exo gym**: simulator (see `08`).
- **Flower**: not evaluated; nothing found suggesting it handles LLM-scale pretraining state better
  than the above (unverified).
- **Decoupled DiLoCo**: no code found. A search hit claiming a DeepMind repository could not be
  corroborated and is disregarded.

## 8. Comparison

| | Discovery | Inbound needs | RunPod pods (mapped TCP, no UDP) | Joiner state | Peer dies mid-sync | Built-in compression | Maturity |
|---|---|---|---|---|---|---|---|
| PCCL | Master `48148` | 3 TCP ports per peer, advertised = local bind port, observed source IP | No, without a patch | Hash-elected P2P transfer | Op aborts for all; retry with fewer | min-max / zero-point quantisation to 2-16 bit | 1 release, dormant since 2025-09 |
| `prime` | TCPStore | Flat network or VPN | No | HTTP from peer's `/dev/shm` | Heartbeat 2 s / timeout 10 s, retry | int8 | Superseded |
| hivemind | DHT, `initial_peers` | 1 TCP port, announce address separate; or none in client mode | Yes | `load_state_from_peers` | Round continues best-effort | fp16, quantile/uniform 8-bit | Maintained, slow data path |
| Psyche | Coordinator + iroh | UDP for direct paths | Relay only (unverified) | Hub / GCS / P2P | Epoch/witness protocol | DisTrO top-k, 1-bit | Last commit 2026-03 |
| torchft `DiLoCo` | Lighthouse `29510` | Flat network: TCPStore, Gloo/NCCL mesh, HTTP on ephemeral ports | No | Live HTTP or PG transfer from a healthy replica | Fragment discarded, params reset, new quorum | fp8 per-row scale | Active; DiLoCo "experimental" |
| Templar | Chain + bucket listing | None (outbound HTTPS) | Yes | Checkpoints in buckets | Upload absent from gather; others proceed | chunked top-k, 12-bit indices, 2-bit values | Ran 72B / 1.1T tokens; now inactive |

## 9. Object-store relay: design and arithmetic

**Design.** Each island's delegate uploads its (quantised) delta for round r to
`deltas/r/<island>/<shard>`; a CPU coordinator downloads the deltas present at the deadline, merges
(token-weighted), applies the outer optimiser, and writes `global/r/` plus a small manifest object
last; islands poll the manifest and download. Variant B (Templar): no coordinator data path; every
island downloads the other islands' sparse deltas and applies the same deterministic outer step.

Properties that the P2P designs lack: outbound HTTPS only; a multipart upload is invisible until
`CompleteMultipartUpload`, so a peer dying mid-upload leaves no partial state and stalls nobody;
joiners read `global/latest` from the store (no peer has to serve it); the store is the checkpoint
of record; slow islands delay only themselves.

**Payload size** S = N x b / 8:

| N | bf16 (16 b) | int8 (8 b) | 4-bit | SparseLoCo codec (0.219 b) |
|---|---|---|---|---|
| 1B | 2 GB | 1 GB | 0.5 GB | 27 MB |
| 10B | 20 GB | 10 GB | 5 GB | 273 MB |
| 70B | 140 GB | 70 GB | 35 GB | 1.9 GB |

**Round latency.** A store-and-forward round has four transfer legs (island up, coordinator down,
coordinator up, island down); each takes 8S/B seconds at island link rate B, assuming the
coordinator has at least M times that bandwidth. Dense int8:

| N | per leg at 1 Gbit/s | round (4 legs) at 1 Gbit/s | round at 5 Gbit/s |
|---|---|---|---|
| 1B | 8 s | 32 s | 6.4 s |
| 10B | 80 s | 320 s | 64 s |
| 70B | 560 s | 2240 s | 448 s |

Variant B with the sparse codec at 70B and M = 8: upload 1.9 GB (15 s at 1 Gbit/s) plus download
7 x 1.9 GB = 13.4 GB (107 s): about 2 minutes, consistent in order of magnitude with Covenant's 70 s.

**Utilisation.** Inner-phase length is H x t_step. Measured marketplace-scale step times: ~23 s at
10B on 8 H100 (INTELLECT-1: H = 100 in ~38 min, see `08`) and ~40 s at 72B on 8 B200 (Covenant: H = 30
in 20 min). For 1B I assume 1 s (assumption). Blocking utilisation = phase / (phase + round), dense
int8 at 1 Gbit/s:

| N | H = 50 | H = 100 | H = 500 |
|---|---|---|---|
| 1B (phase 50 / 100 / 500 s) | 61% | 76% | 94% |
| 10B (1150 / 2300 / 11500 s) | 78% | 88% | 97% |
| 70B (2000 / 4000 / 20000 s) | 47% | 64% | 90% |

With a one-phase-delayed (eager/overlapped) outer step the transfer is free whenever round <= phase.
That holds everywhere above except 70B at H = 50 (2240 s > 2000 s), which needs 4-bit, 2 Gbit/s, or
the sparse codec. So the relay's extra legs matter only for blocking mode; with overlap, dense int8
through a bucket is viable from 1B to 70B at H >= 100 on 1 Gbit/s links.

**Cost.** Cloudflare R2: storage $0.015/GB-month, Class A (writes, including `UploadPart`)
$4.50/million, Class B (reads) $0.36/million, egress free
(https://developers.cloudflare.com/r2/pricing/). Example: 70B, int8, H = 100, M = 4, overlapped,
phase 4000 s: 648 rounds/month. Volume: islands upload 4 x 70 GB x 648 = 181 TB and download 181 TB;
the coordinator downloads 181 TB and uploads 45 TB. With 64 MB parts, 5 objects of 1094 parts per
round is ~3.5 M Class A operations (~$16/month) and ~7 M ranged reads (~$2.50/month); keeping two
rounds is ~700 GB (~$10.50/month). Total about $30/month on R2. The same traffic on a store that
charges egress would be dominated by it: 363 TB/month read at roughly $0.05-0.09/GB is
$18k-33k/month (AWS S3 list prices from memory, unverified), comparable to the GPU bill of a small
island. Zero-egress storage (or a coordinator and bucket co-located with the islands) is therefore a
requirement, not an optimisation. At 1B with H = 100 (phase 100 s) there are ~26k rounds/month and
similar volume (4 x 1 GB x 25,920 = 104 TB each way), so small models do not make this cheaper.

**Unknowns**: sustained multipart throughput from a RunPod pod to R2 (or to RunPod's own
S3-compatible network volumes) is unmeasured; whether providers meter or throttle pod egress is not
established here (see `07`/`16`).

## 10. RL weight broadcast across providers

**prime-rl at `83ca6fd`.**

- Staleness: the string `max_async_level` does not occur in the repository. The setting is
  `max_staleness: Annotated[int, Live()] = Field(8, ge=0)`
  (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:568`). `docs/algorithms.md`:
  token staleness "is the policy version a batch trains on (n-1 at step n) minus the policy version
  that sampled the token. Engines label each sampled token with that version"; the orchestrator
  "masks tokens past the bound out of the loss, and drops a sample only when no token is left to
  train. Rollouts are never cancelled for staleness." `min_fresh_version(step, max_staleness)`
  returns `(step - 1) - max_staleness`. The setting is live-editable through `POST /v1/actions` with
  `update_config`. `docs/ipo.md` shows a run with `max_staleness = 64`.
- Policy versioning: step-numbered; `/update_weights` posts `{"weight_dir", "weight_version":
  step}` and the server calls `client.update_weight_version(...)` so tokens carry
  `weight_versions` spans (`src/prime_rl/inference/vllm/server.py`).
- Transport (`src/prime_rl/transports/weights/base.py`): every backend runs the same handshake in a
  broadcast directory with markers `.sender_ready`, `.receiver_ready`, `.started`, `.finished`.
  "The trainer therefore runs in lockstep with its consumer on every transport - a broadcast nobody
  receives blocks and then fails." `BaseWeightBroadcastConfig.timeout: int = 3600`. Only `step` and
  `step - 1` are kept.
- Backends (`configs/shared.py`): `filesystem` (safetensors HF checkpoint, or PEFT adapter
  hot-swapped via `/load_lora_adapter`), `nccl` (rendezvous port `29501`), `nixl` and
  `modelexpress` (gRPC port `8001`). Default: NCCL when an inference server is configured,
  filesystem with LoRA. Rollout transport default is `ZMQTransportConfig()` (port `5555`).
- ModelExpress in prime-rl (`docs/model-express.md`): "Inference GPUs pull their shards over RDMA
  (NIXL)"; "Trainer rank 0 and the orchestrator must share the broadcast directory"; "A failed
  update leaves inference paused; restart the trainer and inference together, since there is no
  rollback." No S3 or object-storage code path exists in prime-rl's source.

So prime-rl assumes one cluster with a shared filesystem. Across providers the only path is
`filesystem` over a directory that grid replicates itself.

**vLLM 0.33** (`vllm/distributed/weight_transfer/factory.py`): registered engines are `nccl`,
`ipc`, `sparse_nccl`, `sharded_rdt`, `nccl_m2n`, `modelexpress`. NCCL-family backends rendezvous via
`master_address`/`master_port` or `nccl_unique_id_b64` and need NCCL reachability (not available
across NAT, see `05`). `sparse_nccl` sends `O(nnz)` flat-index patches but still over NCCL. The
`modelexpress` engine (`docs/training/weight_transfer/modelexpress.md`) installs immutable published
versions "using GPU peer transfer or object storage" and can "load full checkpoints or replay
checkpoint deltas from S3". Init fields: `object_storage_type` ("S3"), `initial_base_version_id`,
`seed_checkpoint_path`, `refit_checkpoint_dir`, `refit_checkpoint_max_size_gb`,
`object_storage_endpoint_url` ("endpoint override for an S3-compatible service"),
`max_transfer_attempts` (3), `max_replay_chain_length` (64), `rpc_timeout_seconds` (30.0). "An error
does not roll back weights that were already applied." The doc warns the backend depends on
unreleased ModelExpress `main` (prime-rl pins `modelexpress==0.8.0.dev20261022`).

ModelExpress's S3 guide
(https://github.com/ai-dynamo/modelexpress/blob/main/docs/rl/S3_DELTA_WEIGHT_REFIT.md, read via a
summarising fetch): payload format `XOR_DELTA` with `compression_format` zstd and `adler32`
checksums; strict lineage (each delta names its `base_version_id`); a receiver several versions
behind replays the chain and installs only the target; `MX_GENERATOR_SOURCE_ORDER=OBJECT_STORAGE`
for S3-only operation; `MX_S3_UPLOAD_WORKERS` (8), `MX_REFIT_DOWNLOAD_WORKERS` (16),
`MX_REFIT_DELTA_BUCKET_BYTES` (536870912); each host without a shared filesystem needs its own
cache. No throughput numbers are published. It still needs a ModelExpress server (gRPC plus Redis)
reachable by all engines.

XOR+zstd of bf16 weights after a small optimiser step compresses only where high bytes are
unchanged; expected ratio is modest and unmeasured here. LoRA adapters remain the cheap case: tens
to hundreds of MB via any store, hot-swapped without pausing the engine.

## Corrections to first sweep

1. `12-inference-for-rl.md` sections 1.2, 3 and Open questions: "prime-rl `max_async_level`, default
   2" and "Cancel the whole group | prime-rl". At `83ca6fd` the setting is
   `orchestrator.max_staleness`, default 8, enforced as a per-token loss mask; staleness never
   cancels rollouts. `GroupCancellation` exists but is for groups the orchestrator abandons.
2. `12` Open questions, "docs disagree on the default rollout transport": code default is ZMQ
   (`rollout_transport: TransportConfig = ZMQTransportConfig()`); weight broadcast defaults to NCCL
   when inference is configured, filesystem under LoRA.
3. `12` section 2.2 "~v0.32": current is 0.33.0. `modelexpress` is a registered native backend, and
   its object-storage mode is XOR deltas with zstd, replayed by lineage.
4. `08` Summary and Open questions: "Google's Streaming/Decoupled DiLoCo have no released production
   code". Still true of Google's own code, but a public Streaming DiLoCo implementation exists in
   torchft (`DiLoCo` with `fragment_sync_delay`, `fragment_update_alpha`) and is integrated in
   torchtitan. Decoupled DiLoCo: still none.
5. `08` section 4, hivemind: "Last tagged release found is 1.1.10, roughly 2023". PyPI shows 1.1.11
   (2025-04-20) and 1.1.12 (2026-01-03).
6. `08` section 4, PRIME: "heartbeat every 2 s, eviction after 6 s". The code default timeout is 10 s
   (`ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS`); the 6 s figure may be the paper's run setting.
7. `08` section 2.1 and section 4: SparseLoCo details were "from secondary summaries, unverified" and
   the Templar/Covenant run was absent. Now verified from code and paper: k = 64 per 4096-chunk,
   12-bit indices, 2-bit values, 146x, used at 72B over R2. This is LLM-scale evidence that
   error-feedback top-k on outer gradients works, against Streaming DiLoCo's negative result (without
   a same-recipe centralised baseline).
8. `08` Implication 3, "Evaluate PCCL (MIT) as the collective layer before writing one": evaluated.
   It cannot run on NAT-mapped pods unpatched, is synchronous-only, and has had no release since May
   2025.
9. `08` section 3.3 describes PCCL's "async DiLoCo" as the same idea as eager updates. It is a
   one-phase overlap, but without the eager correction term and with every peer still required in
   every reduce.

## Implications for grid

1. **Build the outer-loop coordinator.** Nothing surveyed gives asynchronous, quorum-based merging
   over NAT-restricted links. The component is small if the data plane is an object store.
2. **Make object storage the baseline cross-island data plane**, on zero-egress storage. It needs no
   inbound ports, tolerates death mid-transfer by construction, doubles as joiner bootstrap and
   checkpoint of record, and with an overlapped outer step costs no utilisation at H >= 100 on
   1 Gbit/s links up to 70B. Direct TCP (PCCL-style multi-connection streams) is an optional fast
   path for blocking mode or very small H where both ends have symmetric public ports.
3. **Minimum coordinator interface** (HTTP/JSON control, object-store data):
   - `join(island, gpu_class, tokens_per_s) -> {run, global_version, state_manifest_uri, codec, H,
     fragment_schedule}`
   - `heartbeat(island, inner_step, tokens_since_sync, eta_next_sync)`; `leave(island, reason)`
   - `commit_delta(island, round, fragment, base_version, steps, tokens, object_uris, checksums)`
     (called after uploads complete; idempotent)
   - `poll_update(island, have_version) -> {version, object_uris, applied_weights}` or long-poll
   - Policy held by the coordinator: quorum K, deadline, grace bound, token weighting, outer
     optimiser and its state, admission rate limit, outer LR as a function of live M.
   Versions are monotonic integers plus content hashes; a manifest object written last is the commit
   point (the same idea as prime-rl's `.finished` marker and Templar's window keys).
4. **Trainer-side hook**: follow torchft's shape: an optimizer post-step hook that snapshots at
   `H - delay`, uploads in the background, and at `H` merges with mixing factor alpha, with a CPU
   backup copy of the last global parameters. Under FSDP each rank should upload its own shard in
   canonical parameter coordinates (name, offset, length) so islands with different rank counts
   interoperate and no gather to rank 0 is needed.
5. **Codecs**: ship dense int8 (PCCL-style min-max, or INTELLECT-1's bucket-mean) first; add
   Templar's chunked top-k + 12-bit indices + 2-bit values with error feedback as an opt-in, since it
   cuts 70B payloads from 70 GB to ~2 GB and removes the coordinator from the data path.
6. **Where islands share a routable private network**, torchtitan + torchft semi-sync is usable as
   is and gives a reference result to compare grid's relay against.
7. **RL**: treat weight distribution as a versioned artefact in the same store. Same-island actors
   use vLLM `nccl`/`sharded_rdt`; cross-provider actors use LoRA adapters through `filesystem`
   semantics, or full weights through vLLM's `modelexpress` backend with S3-compatible deltas. grid
   must own the version registry and the gating (`max_staleness`-style bounds), and must treat a
   failed update as "replica unusable" because neither prime-rl nor ModelExpress rolls back.
8. **If PCCL is used at all**, run one PCCL peer per island (not per FSDP rank), patch port and
   address advertisement, and supervise the master; treat it as a throughput optimisation.

## Still open (needs hands-on experiment)

- Sustained multipart upload/download throughput from RunPod pods (and peers) to R2 and to
  provider-native S3-compatible storage, with 8-64 parallel parts; whether pod egress is throttled.
- Whether a RunPod pod's outbound source IP equals its mapped inbound public IP, and how many TCP
  ports a pod can map (decides if patched PCCL or hivemind direct paths are viable).
- torchft `DiLoCo` under a custom process group whose all-reduce is backed by the relay: is the
  `ProcessGroup` abstraction enough, or does quorum/healing assume low-latency collectives?
- Merge quality for asynchronous, token-weighted object-store rounds with islands of different speed
  at >= 1B parameters; no public code or result exists beyond the Decoupled DiLoCo paper.
- Dense int8 versus chunked top-k + 2-bit with error feedback at equal token budget on a 1B-10B
  model inside grid's own stack; Covenant-72B has no same-recipe centralised baseline.
- Time to snapshot and quantise a 70B delta from FSDP shards without stalling the inner loop (CPU
  cores, host RAM for backup copy plus upload buffers).
- ModelExpress S3 delta size and apply time for a full-weight RL step (XOR+zstd ratio on bf16), and
  its stability once 0.8.0 is released.
- Covenant-72B joiner bootstrap procedure and the exact per-round payload size (the 1.97 GB figure
  here is derived).
