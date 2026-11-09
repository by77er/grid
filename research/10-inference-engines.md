# 10 — LLM inference engines and how hardware shapes them

Research date: 2026-11-09. Engine versions current at that date: vLLM v0.33.0 (2 Nov), SGLang v0.5.23 (2 Nov), TensorRT-LLM v1.4.0rc3 (6 Nov), Dynamo v1.6.2 (5 Nov), TGI 3.3.5 (maintenance mode). Sibling files: `11-serving-fleet-orchestration.md` (routing, autoscaling, cold starts at fleet level), `12-inference-for-rl.md` (weight sync, rollout throughput), `05-networking-interconnects.md`, `06-gpu-hardware-nuances.md`, `09-fast-bootstrap-preflight.md`.

## Summary

- **Two engines matter for grid's GPU fleet: vLLM and SGLang.** Both are PyTorch-based, OpenAI-compatible, start in about a minute for a 70B FP8 model, release every two weeks, and now carry first-class RL hooks (in-place LoRA reload, weight-transfer backends, weight versioning). TGI is in maintenance mode and its README points users to vLLM and SGLang ([TGI README](https://cdn.jsdelivr.net/gh/huggingface/text-generation-inference@main/README.md)). NIM 2.x is a packaging of vLLM only ([NIM release notes via search](https://docs.nvidia.com/nim/large-language-models/latest/release-notes.html)). TensorRT-LLM has become a PyTorch-backed engine too: its 1.4 notes call PyTorch "the sole execution backend" and remove the TensorRT-engine compatibility surfaces ([TRT-LLM releases](https://github.com/NVIDIA/TensorRT-LLM/releases)).
- **The engine boundary is one process group per replica**, exposing an HTTP port, `/health`, `/metrics`. Everything above that (routing, disaggregation orchestration, autoscaling) is Dynamo / llm-d / SGLang Model Gateway territory and belongs to file 11.
- **Heterogeneity is fine across replicas and bad inside one.** No engine documents support for mixing GPU models inside one tensor-parallel group; the practical effect is lowest-common-denominator memory and speed. Mixed fleets should be expressed as homogeneous replicas behind a router.
- **Interconnect matters only when a model does not fit one node, or for prefill/decode disaggregation.** Cross-node TP over TCP works but is "inefficient"; vLLM docs tell you to check NCCL logs for `NET/IB/GDRDMA` versus `NET/Socket` ([vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)). KV transfer for disaggregation wants RDMA (NIXL/UCX or Mooncake). On marketplace hardware without RDMA, the right answer is almost always "pick a quantisation that fits one node".
- **Cold start is dominated by CPU-bound initialisation and CUDA graph capture, not by weight I/O**, once weights are local. Both engines shipped restart accelerators in H2 2026 (`vllm preload` + `--load-format ipc_cache`, CRIU snapshots, SGLang InstantTensor, Dynamo ModelExpress and Snapshot).
- **Engine benchmarks differ by single-digit to low-double-digit percent on one GPU**; the big multipliers (2x to 20x) come from quantisation (FP8/NVFP4), speculative decoding/MTP, prefix-cache hit rate, and rack-scale disaggregation, per SemiAnalysis InferenceX.

---

## 1. Engine landscape at a glance

| Engine | Status (Nov 2026) | Runtime | Best at | Multi-node single replica | Disaggregated P/D |
|---|---|---|---|---|---|
| vLLM | v0.33.0, biweekly; Model Runner V2 default since v0.29 | PyTorch 2.14, CUDA 13.0 default | Broadest model/hardware coverage, default choice | Ray or native `--nnodes` multiprocessing | Experimental, via KV connectors (NIXL, Mooncake, LMCache) |
| SGLang | v0.5.23, biweekly | PyTorch, CUDA 13 (CUDA 12 lane retired in v0.5.20) | Prefix-heavy workloads (radix cache), DeepSeek-class MoE, RL rollouts | Native `--nnodes/--node-rank/--dist-init-addr` | First-class, Mooncake or NIXL, own router |
| TensorRT-LLM | v1.4.0rc3 | PyTorch backend only now | Peak per-GPU throughput on NVIDIA, esp. Blackwell | MPI/Slurm (`trtllm-llmapi-launch`) | First-class, NIXL |
| Dynamo | v1.6.2 | Rust/Python orchestration over vLLM / SGLang / TRT-LLM | Fleet layer: KV-aware router, planner, migration | Via backend; K8s operator injects `DYNAMO_RANK` | Core feature |
| NIM | 2.x/3.x doc trees | Container around vLLM | Supported, pre-tuned packaging | As vLLM | As vLLM |
| TGI | 3.3.5, maintenance mode | Rust router + Python shards | Legacy only | No | No |
| llama.cpp / Ollama | rolling builds | C++ ggml, GGUF | CPU, Apple, consumer GPUs, tiny footprint | `--rpc` (experimental) | No |
| LMDeploy | 0.12.x; PyPI wheels blocked since April 2026 | TurboMind C++ | INT4 throughput, InternLM/Qwen families | No (proxy over replicas) | No |
| MLX (mlx-lm, vllm-mlx) | n/a | Apple MLX | Apple Silicon | No | No |

Sources: [vLLM releases](https://github.com/vllm-project/vllm/releases), [SGLang releases](https://github.com/sgl-project/sglang/releases), [TRT-LLM releases](https://github.com/NVIDIA/TensorRT-LLM/releases), [Dynamo releases](https://github.com/ai-dynamo/dynamo/releases), [LMDeploy on PyPI](https://pypi.org/project/lmdeploy/0.12.3/).

---

## 2. vLLM

### 2.1 Launch and key flags

`vllm serve <model> [flags]`. The older `python -m vllm.entrypoints.openai.api_server` entrypoint is deprecated since v0.29 ([releases](https://github.com/vllm-project/vllm/releases)). Flags from the [engine args reference](https://docs.vllm.ai/en/latest/configuration/engine_args.html):

| Flag | Default | Notes |
|---|---|---|
| `--tensor-parallel-size`, `-tp` | 1 | Megatron-style TP; set to GPUs per node |
| `--pipeline-parallel-size`, `-pp` | 1 | Layer split; supports uneven splits; docs recommend PP over TP for GPUs without NVLink such as L40S |
| `--data-parallel-size`, `-dp` | 1 | Replicated engines behind one API server; each rank is a separate core-engine process over ZMQ |
| `--gpu-memory-utilization` (alias `--device-memory-utilization`) | docs now say 1.0 with extensible KV cache, otherwise 0.92 | Fraction of each GPU the process may claim; vLLM pre-allocates the KV cache from what is left after weights |
| `--kv-cache-memory-bytes` | None | Explicit KV budget, alternative to the fraction |
| `--max-model-len` | from model config | Lower it to make a model fit; KV pool must hold at least one full-length sequence |
| `--max-num-seqs`, `--max-num-batched-tokens` | (see `--help`) | Batch caps per step; `--max-num-seqs` applies per DP rank |
| `--max-num-active-seqs` | (v0.31) | Caps RUNNING admission separately from `max_num_seqs` |
| `--max-num-queued-reqs`, `--max-num-queued-tokens` | (v0.29) | Queue admission control; rejections counted in `vllm:admission_rejections` |
| `--enable-chunked-prefill` / `--disable-chunked-prefill`, `--long-prefill-token-threshold` (default 0) | | Threshold now adapts to number of waiting prefills (v0.31) |
| `--enable-prefix-caching` / `--disable-prefix-caching`, `--prefix-caching-hash-algo` | sha256 | Alternatives: `sha256_cbor`, `xxhash`, `xxhash_cbor` |
| `--scheduling-policy` | `fcfs` | or `priority` |
| `--quantization`, `-q` | auto-detect | `quantization="fp8"` was replaced by `fp8_per_tensor` in v0.31 |
| `--enable-lora`, `--max-loras` (1), `--max-lora-rank` (16), `--max-cpu-loras`, `--lora-modules` | | See 2.5 |
| `--load-format` | `auto` | See 2.6 |
| `--distributed-executor-backend` | `mp` single node, `ray` multi-node | choices `mp`, `ray`, `uni`, `external_launcher` |
| `--device-ids` | | Since v0.24 vLLM no longer sets `CUDA_VISIBLE_DEVICES` internally (reported via secondary source, [af.net](https://af.net/cn/realtime/vllm-v0-24-0-release-minimax-m3-support-and-enhanced-model-optimizations/)) |
| `--enforce-eager` | False | Disables CUDA graphs; since v0.31 also disables JIT kernel warmup. Faster startup, slower steady state |
| `--cpu-offload-gb` | 0 | Spill weights to host RAM; a last resort |
| `--api-server-count` | 1 | Multiple API processes; since v0.32 connections go to the least-loaded worker |

### 2.2 Scheduling and batching

Continuous batching with an iteration-level scheduler; prefill is chunked so that decodes are not starved by long prompts, and prefix caching is block-hash based (16-token blocks in the KV-offloading benchmark setup, [vLLM blog](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)). Recent scheduler changes worth knowing: the waiting queue schedules requests that already hold KV blocks first (v0.31); `max_tokens` is no longer reserved from the context window (v0.32); requests sent while generation is paused get HTTP 503 (v0.33) ([releases](https://github.com/vllm-project/vllm/releases)).

### 2.3 Multi-GPU and multi-node

From the [parallelism doc](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html):

- Fits one GPU: no distribution. Fits one node: `--tensor-parallel-size <gpus>`. Does not fit one node: TP = GPUs per node, PP = number of nodes (for example TP=8, PP=2).
- **Ray path:** `examples/ray_serving/run_cluster.sh vllm/vllm-openai <HEAD_IP> --head|--worker <HF_HOME> -e VLLM_HOST_IP=<this node IP>`, then `vllm serve ... --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray` from one node. The legacy Ray compiled-graph executor was removed in v0.33.
- **Native multiprocessing path (no Ray):** head `vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr <HEAD_IP>`; worker identical with `--node-rank 1 --headless`. This is torchrun-shaped rendezvous and maps directly onto what grid already does for training (file 03).
- **Interconnect:** "InfiniBand is preferred" for cross-node TP. Add `--privileged -e NCCL_IB_HCA=mlx5`; GPUDirect RDMA needs `--ipc=host --shm-size=16G -v /dev/shm:/dev/shm` (Docker) or `IPC_LOCK` plus a memory-backed `/dev/shm` (Kubernetes). Verify with `NCCL_DEBUG=TRACE`: `[send] via NET/IB/GDRDMA` is good, `[send] via NET/Socket` means TCP. A `vllm/vllm-openai:v0.33.0-efa` image bundles AWS EFA libraries.
- **Security:** inter-node traffic is unencrypted and the data format permits arbitrary code execution by anyone on that network; `VLLM_HOST_IP` must be a private address.
- All nodes need the identical environment and the model at the same path.
- **Data parallel** ([DP doc](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)): `--data-parallel-size 4 --tensor-parallel-size 2` on 8 GPUs; multi-node via `--data-parallel-size-local`, `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-start-rank`, `--headless`; `--data-parallel-hybrid-lb` (one API server per node, upstream LB) and `--data-parallel-external-lb` (each rank its own endpoint). For MLA/MoE models the ranks must stay in lock-step (a DP coordinator runs dummy passes on idle ranks), so DP for those models is one failure domain, not N.
- No-NVLink boxes: FlashInfer all-reduce is default for TP since v0.29 (`VLLM_ALLREDUCE_USE_FLASHINFER=0` to opt out); an opt-in PCIe IPC all-reduce exists for NVLink-less boxes (v0.30). The known error `NCCL WARN Cuda failure 217 'peer access is not supported between these two devices'` is worked around with `NCCL_CUMEM_HOST_ENABLE=0` ([troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting.html)).

### 2.4 KV caching, offloading, disaggregation

- **CPU offload:** `--kv_offloading_backend native --kv_offloading_size <GiB>` (older form: `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",...}'`). On H100 with Llama-3.1-8B, loading KV from host RAM cut TTFT 2x to 22x depending on prompt size and raised throughput up to about 9x at high CPU-cache hit rate; transfers run at about 50 GB/s one-way via `cudaMemcpyAsync` ([vLLM blog](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)). Host RAM therefore directly buys cache capacity.
- **Disaggregated prefill** is still labelled experimental. Connectors: `NixlConnector` (requires NIXL >= 1.5 since v0.33), `LMCacheConnectorV1`, `LMCacheMPConnector`, `MooncakeConnector`, `MoRIIOConnector` (ROCm), `MultiConnector`, `OffloadingConnector`, `FlexKVConnectorV1`. Config is `--kv-transfer-config` with keys `kv_connector`, `kv_role`, `kv_buffer_device`, `kv_connector_extra_config` (for example `{"backends":["UCX","GDS"]}`). The docs state plainly that disaggregated prefill "does not improve throughput"; it is for tuning TTFT and ITL separately and controlling tail ITL ([disagg doc](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)).
- Quantised KV cache: FP8 KV default for some models on SM100; NVFP4 KV on SM8x/SM12x/SM90 (v0.32).

### 2.5 LoRA, structured output

- LoRA ([doc](https://docs.vllm.ai/en/latest/features/lora.html)): static via `--lora-modules name=path` or JSON with `base_model_name`. Dynamic: set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, then `POST /v1/load_lora_adapter {lora_name, lora_path}` and `POST /v1/unload_lora_adapter`. `load_inplace: true` swaps an adapter of the same name without interrupting inference; the docs tie this to asynchronous RL. Resolver plugins (`lora_filesystem_resolver` with `VLLM_LORA_RESOLVER_CACHE_DIR`, enabled through `VLLM_PLUGINS`) load adapters on first request. The LoRA path is part of the prefix-cache block hash since v0.31, so adapters do not cross-contaminate caches.
- Structured output ([doc](https://docs.vllm.ai/en/latest/features/structured_outputs.html)): request field `structured_outputs` with `json`, `regex`, `choice`, `grammar`, `structural_tag`; backends xgrammar and guidance selected by `--structured-outputs-config.backend` (default `auto`). `guided_*` fields were removed in v0.12; `lm-format-enforcer` is deprecated from v0.33, removal in v0.35.

### 2.6 Weight formats and load time

`--load-format` values ([LoadConfig](https://docs.vllm.ai/api/vllm/config/load.html)): `auto`, `pt`, `safetensors`, `instanttensor` (direct I/O, pipelined prefetch, distributed loading), `ipc_cache` (map post-quantised weights from a local daemon via CUDA IPC; start the daemon with `vllm preload`), `npcache`, `dummy` (random weights; useful for preflight and profiling), `tensorizer`, `runai_streamer`, `runai_streamer_sharded`, `sharded_state`, `mistral`, `modelexpress`. `safetensors_load_strategy` is `lazy` (mmap), `eager` (read all to RAM; recommended on network filesystems), or `prefetch` (warm page cache); by default prefetch auto-enables on NFS when the checkpoint fits in about 90% of available RAM.

Numbers: Run:ai Model Streamer on vLLM 0.5.5, AWS g5.12xlarge, loaded in 14.34 s versus 47.99 s for the safetensors loader on gp3, and 4.88 s from S3 versus 37.36 s for Tensorizer; total time to ready was 35.08 s versus 66.13 s ([NVIDIA blog](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer)). These are 2024-era and directional only.

Restart accelerators added in 2026: `vllm preload` weight-cache daemon that keeps post-quantised weights in GPU memory across engine restarts (v0.30/0.31, extended to multi-node TP); experimental CRIU snapshots `vllm snapshot create/restore` (v0.31); `vllm download-kernels` for precompiled FlashInfer kernels; concurrent worker startup at TP > 1 (v0.33); frozen GC during graph capture cut capture from 12 s to 2 s (v0.30) ([releases](https://github.com/vllm-project/vllm/releases)).

Where the time goes: the "Breaking the Ice" study (MLSys 2026) finds vLLM cold start "overwhelmingly CPU-bound" except profiling and CUDA graph capture, and that loading from SSD rather than DRAM changes only the load phase, not the total ([arXiv 2606.07362](https://arxiv.org/html/2606.07362)). RunPod reported 324 s to about 91 s median cold start for a 32B FP8 Qwen on 2x H200 by persisting the compile cache, prefetching weights, limiting CUDA-graph capture sizes, and offline mode ([summary](https://www.plushcap.com/content/runpod/blog/runpod-cut-your-vllm-cold-starts-from-5-minutes-to-90-seconds-on-runpod-serverless)).

### 2.7 Health, metrics, failure behaviour

- Endpoints ([online serving](https://docs.vllm.ai/en/latest/serving/online_serving/)): `/health`, `/ping` (SageMaker), `/load` (server load), `/version`, `/metrics`. Dev-mode only (`VLLM_SERVER_DEV_MODE=1`): `/sleep`, `/wake_up`, `/is_sleeping`, `/pause`, `/resume`, `/is_paused`. Checkpoint endpoints `/checkpoint/prepare|status|restore`. Scale-out endpoints (`/inference/v1/generate` etc.) need `--enable-scale-out`.
- `/health` calls `check_health()` and returns 200, or 503 on `EngineDeadError` ([health source docs](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/serve/instrumentator/health/)). The port does not listen until the model is loaded, so "connection refused" is the loading state. A `/live` endpoint was proposed in PR #36258; merge status not confirmed (unverified).
- `EngineDeadError` is "global and therefore not recoverable". A watchdog checks every 5 s (`VLLM_WATCHDOG_TIME_S = 5.0`) and sets the server to exit unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` is set (documented for v0.10 to v0.17; current behaviour should be re-checked against source) ([launcher docs](https://docs.vllm.ai/en/v0.17.1/api/vllm/entrypoints/launcher/)). So GPU faults (Xid, CUDA error, OOM inside the engine) surface as process exit: **the recovery unit is the whole replica.**
- A watchdog that dumps stack traces from stuck engine/worker processes landed in v0.33 (#55700). A third-party report claims `/health` returned 200 while inference was deadlocked on 0.19.x ([dev.to](https://dev.to/sachincool/your-model-isnt-crashing-your-probe-is-jbc)); a generate-based probe is safer.
- OOM: steady-state OOM is rare because KV is pre-allocated; OOM appears at startup (model plus minimum KV does not fit) or from CUDA-graph and activation memory, which is why v0.29 added CUDA graph memory reservation.
- Driver mismatch: `CUDA error: the provided PTX was compiled with an unsupported toolchain` or "The NVIDIA driver on your system is too old" means the wheel's CUDA is newer than the host driver. Fix: `VLLM_ENABLE_CUDA_COMPATIBILITY=1` (uses `cuda-compat`), which is "only supported on select professional and datacenter NVIDIA GPUs" ([troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting.html)).
- Metrics, prefix `vllm:` ([metrics doc](https://docs.vllm.ai/en/latest/usage/metrics.html)). Autoscaling/routing signals: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_requests_waiting_by_reason`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions`, `vllm:admission_rejections`. Latency histograms: `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`. Throughput counters: `vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:request_success`. Cache: `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, `vllm:prompt_tokens_cached_by_source`. Integrity: `vllm:corrupted_requests` (NaNs in logits; a useful silent-corruption signal). State: `vllm:engine_sleep_state`, `vllm:lora_requests_info`. NIXL: `vllm:nixl_xfer_time_seconds`, `vllm:nixl_num_failed_transfers`. Optional `--enable-mfu-metrics` adds `vllm:estimated_flops_per_gpu_total`.

### 2.8 RL-facing features (detail in file 12)

Weight-transfer backends for RL landed in 2026: `sharded_rdt` P2P over NIXL or Ray Direct Transport, rank-local IPC weight updates, sparse checkpoint-coordinate updates (v0.29), and a sharding-aware NCCL M2N backend (v0.31) ([releases](https://github.com/vllm-project/vllm/releases)).

---

## 3. SGLang

- **Launch:** `python -m sglang.launch_server --model-path <m> --port 30000` (a newer `sglang serve` entrypoint appears in third-party docs; unverified). Key args ([server arguments](https://docs.sglang.io/advanced_features/server_arguments.html)): `--tp-size` (1), `--dp-size` (1), `--pp-size` (1), `--mem-fraction-static` (auto; fallback 0.88), `--context-length`, `--max-running-requests`, `--max-queued-requests`, `--chunked-prefill-size` (-1 disables), `--max-prefill-tokens` (16384), `--schedule-policy` (`fcfs`; also `lpm`, `random`, `dfs-weight`, `lof`, `priority`, `routing-key`), `--schedule-conservativeness` (1.0), `--quantization`, `--kv-cache-dtype` (`auto`, `bf16`, `fp8_e5m2`, `fp8_e4m3`, `nvfp4`, `fp4_mx_block16`), `--grammar-backend` (`xgrammar`, `outlines`, `llguidance`, `none`), `--attention-backend`, `--load-format`, `--watchdog-timeout` (300 s), `--enable-metrics`, `--weight-version`.
- **Caching:** RadixAttention prefix cache (radix tree over token sequences, LRU eviction); since v0.5.21 the prefix cache runs on a Rust core by default. `lpm` (longest prefix match) scheduling reorders the queue to maximise reuse. HiCache adds tiers: `--enable-hierarchical-cache`, `--hicache-ratio` (2.0, host pool relative to device pool), `--hicache-size` (GB), `--hicache-write-policy` (`write_through`, `write_back`, `write_through_selective`), `--hicache-storage-backend` (`file`, `mooncake`, `hf3fs`, `nixl`, ...); L3 storage can be attached or detached on a running server (v0.5.19) ([releases](https://github.com/sgl-project/sglang/releases)).
- **Multi-node single replica** ([doc](https://docs.sglang.io/references/multi_node_deployment/multi_node.html)): same command on each node with `--tp 16 --dist-init-addr <node0-ip>:20000 --nnodes 2 --node-rank {0,1}`. Example: Llama-3.1-405B FP16 needs 2x8 GPUs at TP=16, while the FP8 variant fits one node at `--tp 8`: quantisation removes the multi-node requirement.
- **PD disaggregation** ([doc](https://docs.sglang.io/advanced_features/pd_disaggregation.html)): `--disaggregation-mode prefill|decode`, `--disaggregation-transfer-backend mooncake|nixl`, `--disaggregation-ib-device mlx5_roce0` (shared list or per-GPU JSON map), router `python -m sglang_router.launch_router --pd-disaggregation --prefill http://... --decode http://...`. Mooncake has an EFA build; NIXL defaults to UCX (`SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` for alternatives). Failure handling is explicit: decode pings prefill every 5 s (`SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`), marks it offline after 2 failures (`..._HEARTBEAT_MAX_FAILURE`); `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` and `..._WAITING_TIMEOUT` default 300 s. Heterogeneous TP between prefill and decode is supported through a GPU staging buffer (`SGLANG_DISAGG_STAGING_BUFFER=1` on both sides, Mooncake only, non-MLA models), 2x to 5x faster than per-token transfer at high concurrency. Since v0.5.21 a PD instance can switch role on the fly.
- **LoRA:** `--lora-paths` (PATH, NAME=PATH or JSON with `pinned`), `--max-loras-per-batch` (8), `--max-lora-rank`, `--lora-target-modules`; dynamic load/unload endpoints exist in `http_server.py` (names not confirmed in this sweep; unverified). v0.5.23 adds an opt-in LoRA engine for quantised (FP8/NVFP4) MoE models.
- **Health and readiness** ([native API](https://docs.sglang.io/basic_usage/native_api.html)): `/health` (cheap), `/health_generate` (generates one token: a real GPU forward pass), `/ready` (200 only when accepting requests; 503 while starting, paused, unhealthy or shutting down), `/ping`, `/get_model_info` (includes weight version), `/server_info`, `/flush_cache`, `/update_weights_from_disk` (hot-swap weights with same architecture; flushes cache automatically).
- **Failure:** `--watchdog-timeout 300`: "the server crashes if a forward batch exceeds this", i.e. hung GPU work converts to process exit. Same replica-level recovery unit as vLLM.
- **Metrics** (`--enable-metrics`, [doc](https://docs.sglang.io/references/production_metrics.html)): `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:token_usage`, `sglang:cache_hit_rate`, `sglang:gen_throughput`, `sglang:time_to_first_token_seconds`, `sglang:e2e_request_latency_seconds`, `sglang:request_time_per_output_token_seconds`, `sglang:prompt_tokens_total`, `sglang:generation_tokens_total`.
- **Load time:** InstantTensor loader "10x faster model loads" (v0.5.22); `--startup-weight-load-mode overlap` made Qwen3-32B on H100 start 2.38x faster (v0.5.18); GLM-5.2 TP4 on 4x MI355X loads in 40.4 s (v0.5.20).
- **RL:** `return_sampling_mask` for rollouts (v0.5.20), `--weight-version`, `/update_weights_from_disk`.
- **Hardware breadth:** AMD MI300X/MI350X/MI355X (ROCm 10 images), Blackwell including RTX PRO 6000 (SM120), Google TPU7x, Apple Silicon via plain PyTorch (v0.5.22). Requires Python 3.11+.

## 4. TensorRT-LLM and Triton

- The old flow (`quantize.py`, then `trtllm-build` to a per-GPU-architecture TensorRT engine, then Triton `tensorrtllm` backend) is legacy. A March 2026 benchmark measured about 5 min quantise plus about 23 min build for Llama-3.3-70B FP8 on H100, versus about 60 s cold start for vLLM/SGLang ([Spheron](https://spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks)). Compiled engines were tied to GPU architecture, which is hostile to heterogeneous fleets. The 1.4 line removes these surfaces.
- **Launch now** ([trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)): `trtllm-serve <hf-model> --tp_size N --pp_size N --ep_size N --config config.yaml --set path=value`, with `--max_batch_size`, `--max_num_tokens`, `--kv_cache_free_gpu_memory_fraction`, `--num_serve_frontends` (default 8), `--server_role CONTEXT|GENERATION`. YAML keys include `kv_cache_config.enable_block_reuse`, `cuda_graph_config.batch_sizes`, `enable_attention_dp`. Endpoints: `/v1/*`, `/health`, `/metrics`, `/version`.
- **Multi-node** is MPI-shaped: `srun -N 2 --ntasks 16 --mpi=pmix ... trtllm-llmapi-launch trtllm-serve deepseek-ai/DeepSeek-V3 --tp_size 16 --ep_size 4 ...`. Known issues in every 1.4 rc involve MPI worker-pool hangs at TP >= 2 ([releases](https://github.com/NVIDIA/TensorRT-LLM/releases)).
- **Disaggregated serving** ([doc](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)): context and generation servers plus an orchestrator `trtllm-serve disaggregated -c disagg_config.yaml`; `cache_transceiver_config.backend: NIXL`, `kv_transfer_timeout_ms` default 60000; `TRTLLM_NIXL_KVCACHE_BACKEND=UCX|LIBFABRIC`. Context and generation may use different TP/PP.
- **Hardware list:** Blackwell (B200, GB200, B300, GB300, DGX Spark), Hopper, Ada (L20, L40/L40S), Ampere A100 ([supported hardware](https://nvidia.github.io/TensorRT-LLM/supported-hardware.html)). Consumer cards are not listed.
- Verdict for grid: highest NVIDIA-only throughput (see section 9), release-candidate cadence with known hang issues, narrower hardware list. Consume it through Dynamo if at all.

## 5. Dynamo and NIM

- **Dynamo** is the orchestration layer over the engines, not an engine: Frontend, Router (KV-aware), Planner (SLA autoscaling), KVBM (tiered KV block manager), NIXL (transfer), ModelExpress (weight streaming), Grove (K8s topology scheduling) ([README](https://github.com/ai-dynamo/dynamo/blob/main/README.md)). v1.0 shipped 16 March 2026; v1.6.x pins SGLang v0.5.21, TRT-LLM v1.3.0rc29, vLLM v0.31.0 ([releases](https://github.com/ai-dynamo/dynamo/releases)). The pinning lag (two to three engine releases) is a cost.
- **Launch without Kubernetes:** `python3 -m dynamo.frontend --http-port 8000 --discovery-backend file` plus `python3 -m dynamo.vllm --model <m> --discovery-backend file` (or `dynamo.sglang --model-path`). etcd and NATS are optional, "mainly for Slurm or other distributed setups"; the request plane is TCP; KV-aware routing does not need NATS (`--no-router-kv-events` gives prediction-based routing).
- **Fault tolerance** ([overview](https://docs.nvidia.com/dynamo/kubernetes/fault-tolerance/overview.md), [migration](https://docs.nvidia.com/dynamo/kubernetes/fault-tolerance/request-migration.md)): request migration is off by default; enable with `--migration-limit N` / `DYN_MIGRATION_LIMIT` on the Frontend (docs suggest 3). The Frontend tracks emitted tokens and replays them on a healthy worker, so the client sees no gap. It is silently skipped for `n > 1` and for guided decoding. Failed workers are inhibited locally for 5 s (`DYN_RUNTIME_INHIBITED_DURATION_SECS`). Request rejection returns HTTP 529 when all workers are busy (off by default). Graceful shutdown is on by default. Metrics: `dynamo_frontend_model_migration_total`.
- **NIM for LLMs 2.x** "adopts a one-container, one-backend philosophy" with vLLM as the sole backend (2.0 shipped vLLM 0.17.1; 2.0.7 uses 0.22.1); any `vllm serve` argument passes through, and `--dry-run` prints the resolved config ([release notes, via search](https://docs.nvidia.com/nim/large-language-models/latest/release-notes.html)). The 1.x profile system (`tensorrt_llm-a10g-bf16-tp8-pp1-throughput`, etc.) is gone. For grid, NIM adds licensing and an older vLLM, and no engine capability.

## 6. TGI, LMDeploy, llama.cpp, Ollama, MLX

- **TGI:** maintenance mode; last image tag 3.3.5; README recommends vLLM, SGLang, llama.cpp, MLX ([README](https://cdn.jsdelivr.net/gh/huggingface/text-generation-inference@main/README.md)). Third-party sources date this to 11 Dec 2025 and say the repo was archived 21 Mar 2026 ([Spheron migration post](https://www.spheron.network/blog/migrate-tgi-to-vllm-sglang-2026/); unverified against GitHub). Do not target.
- **LMDeploy:** `lmdeploy serve api_server <model> --server-port 23333 --tp N --session-len N --cache-max-entry-count <ratio> --backend turbomind|pytorch`; multi-replica through `lmdeploy serve proxy` and `--proxy-url` ([doc](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html)). No documented health or metrics endpoints on that page; PyPI quota problem blocks new wheels since April 2026. Niche.
- **llama.cpp `llama-server`** ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)): GGUF only. `-ngl` (default `auto`), `-c`, `-np/--parallel` slots (default auto), `-cb` continuous batching (on by default), `--cache-reuse N`, `--slot-save-path`, `--lora` and `--lora-init-without-apply` with `POST /lora-adapters`, `--split-mode none|layer|row|tensor` (`layer` default, i.e. pipeline-style split; `tensor` experimental), `--tensor-split 3,1` (explicit per-GPU proportions: the one engine that natively handles unequal GPUs in one process), `--rpc host:port,...` for remote backends, `--grammar`, `--json-schema`, router mode `--models-dir` / `--models-max 4`. `/health` returns 503 "Loading model" while loading and 200 `{"status":"ok"}` when ready; `/slots?fail_on_no_slot=1` returns 503 when saturated. `--metrics` exposes `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, and newer `llamacpp:prompt_seconds_total`, `llamacpp:n_busy_slots_per_decode`; `llamacpp:kv_cache_usage_ratio` was removed in build b9110 ([release note](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b9110)). Context is divided across slots, so throughput scaling under concurrency is far below vLLM-class engines.
- **Ollama** ([FAQ](https://docs.ollama.com/faq)): `OLLAMA_NUM_PARALLEL` default 1; `OLLAMA_MAX_LOADED_MODELS` default 3 x GPU count; `OLLAMA_MAX_QUEUE` 512, beyond which it returns 503; `OLLAMA_KEEP_ALIVE` 5 min; `OLLAMA_CONTEXT_LENGTH` 4096; `OLLAMA_KV_CACHE_TYPE` `f16|q8_0|q4_0`. A model that fits one GPU is placed on one GPU; otherwise it is spread across all. A developer tool, not a fleet engine.
- **MLX:** `mlx-lm` has high single-stream speed but no continuous batching; `vllm-mlx` adds it and reports up to 525 tok/s on Qwen3-0.6B on an M4 Max and about 4.3x aggregate scaling at 16 concurrent requests (self-reported, [arXiv 2601.19139](https://arxiv.org/html/2601.19139v1)). Out of scope for marketplace GPU clusters.
- **AIBrix / llm-d** stop at the engine boundary: they consume the engine's HTTP port, `/health`, `/metrics` and LoRA endpoints, and add gateways, autoscalers and KV-event routing. llm-d documents engine readiness probes separately ([llm-d readiness probes](https://llm-d.ai/docs/usage/readiness-probes)). See file 11.

---

## 7. Quantisation and GPU generation

vLLM's compatibility matrix ([doc](https://docs.vllm.ai/en/latest/features/quantization/index.html)):

| Format | Turing (T4) | Ampere (A10G, A100, 3090) | Ada (L4, L40S, 4090) | Hopper (H100/H200) | AMD GPU |
|---|---|---|---|---|---|
| AWQ | yes | yes | yes | yes | no |
| GPTQ | yes | yes | yes | yes | no |
| Marlin kernels (GPTQ/AWQ/FP8/FP4) | yes (no MXFP4) | yes | yes | yes | no |
| INT8 W8A8 | yes | yes | yes | yes | no |
| **FP8 W8A8 (llm-compressor)** | **no** | **no** | yes | yes | yes |
| bitsandbytes | yes | yes | yes | yes | no |
| GGUF | yes | yes | yes | yes | yes |

Consequences: FP8 with activation quantisation needs Ada or newer (compute capability 8.9+); Ampere fleets (A100, A10G, 3090) fall back to weight-only FP8 through Marlin, or INT4 AWQ/GPTQ. NVFP4 checkpoints run by default on compute capability 10.0+ (Blackwell). So **the same model needs a different checkpoint or quantisation flag per GPU generation**: a mixed fleet is a set of (model artifact, GPU class) pairs. InferenceX reports FP8 scoring slightly higher than FP4 on DeepSeek R1 ([SemiAnalysis](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)), and caught a DP-attention accuracy bug that throughput numbers missed: quantised and parallelised configurations need an accuracy check, not just a throughput check.

## 8. Hardware nuance for inference

**Memory arithmetic** (my derivation; check against each model's `config.json`). Weights are parameters times bytes per parameter (70B: about 140 GB BF16, about 70 GB FP8, about 35 to 40 GB INT4). KV bytes per token are 2 x layers x KV heads x head dim x bytes per element; for a Llama-3.x-70B shape (80 layers, 8 KV heads, 128 dim, BF16) that is about 0.33 MB per token, so about 2.6 GB per 8k-token sequence and about 10 GB per 32k sequence; FP8 KV halves it. Measured: Llama-3.3-70B FP8 on one H100 used 71 GB idle and 78 GB at 100 concurrent 512-in/256-out requests ([Spheron](https://spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks)), which leaves little room for long contexts; H200 (141 GB) or 2x H100 is the comfortable configuration.

| GPU class | VRAM | Sensible single-GPU model (FP8 or INT4) | Notes |
|---|---|---|---|
| T4, L4, A10G, RTX 3090/4090 | 16 to 24 GB | 7B to 14B; about 20B to 32B at INT4 with short context | L4 and 4090 are Ada (FP8 OK); A10G and 3090 are Ampere (no FP8 activations) |
| RTX 5090 | 32 GB | up to about 27B FP8 | SM120; needs CUDA 12.8+ builds ([PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)); no NVLink |
| L40S, A6000, RTX 6000 Ada | 48 GB | about 32B FP8, 70B INT4 tight | No NVLink on L40S; vLLM recommends PP over TP |
| A100 80, H100 80 | 80 GB | 70B FP8 with short context (H100); A100 needs weight-only FP8 or INT4 | 8x gives 640 GB: 405B-class at FP8 |
| RTX PRO 6000 Blackwell | 96 GB | 70B FP8 with headroom | SM120 support in SGLang v0.5.20, TRT-LLM known issues |
| H200 | 141 GB | 70B FP8 with large KV, 120B-class | Most forgiving single-GPU serving card |
| B200 / B300 | about 180 / 288 GB (per vendor guides) | 70B BF16, large MoE with NVFP4 | NVFP4, best perf per token |
| MI300X / MI355X | 192 / 288 GB (unverified here) | as above | ROCm; SGLang is the best-supported engine; upstream vLLM MI355X CI was "zero" in Feb 2026 per SemiAnalysis |

**VRAM versus throughput.** VRAM sets what fits and how much KV (concurrency x context) is available; memory bandwidth and compute set tokens per second. A model that barely fits has high single-stream speed but collapses under concurrency because the KV pool is tiny (visible as `vllm:num_preemptions` and `vllm:kv_cache_usage_perc` near 1). InferenceX treats results as an interactivity-versus-throughput curve: on B200, DeepSeek R1 cost went from about $0.56 to about $4 per million output tokens moving from 50 to 125 tok/s per user ([SemiAnalysis](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)). For RL rollouts (throughput-only) the operating point is the high-batch end of that curve, which is several times cheaper per token.

**Consumer cards and PCIe.** No NVLink on 4090/5090. Measured PCIe peer-to-peer of 19 to 22 GB/s and 8 tok/s single-stream for Llama-3.3-70B AWQ at TP=4 on 4x 4090 (with the slow non-Marlin AWQ kernel) ([Kentino case study](https://kentino.com/blogs/ai-corner/case-study-4x-rtx-4090-ai-workstation)). Community guidance is that TP=2 over PCIe is acceptable and TP>=4 becomes communication-bound ([vLLM forum](https://discuss.vllm.ai/t/combining-2-different-gpus/1609)); the official guidance is PP for NVLink-less cards. The best use of consumer cards is **one model per card, many replicas**.

**Mixing GPU types.** Across replicas: fine, provided the router weights by capacity and each GPU class gets a compatible artifact. Within one replica: TP shards the model evenly, so every rank needs the same free memory and the step runs at the slowest rank's pace; vLLM docs do not address mixed TP at all, and the forum answer is "bottlenecked to the slowest GPU" (unquantified). PP tolerates uneven splits in principle, and llama.cpp `--tensor-split` does it explicitly. Disaggregation is the one sanctioned heterogeneous pattern: prefill and decode pools may differ in TP size (SGLang staging buffer, TRT-LLM) and plausibly in GPU class, as long as KV layout and dtype match.

**Multi-node single replica and interconnect.** InferenceX numbers show what the interconnect buys: NVLink at 900 GB/s per GPU versus 50 to 100 GB/s for InfiniBand/Ethernet scale-out; multi-node disaggregated serving "generally beats single-node aggregated serving" per GPU on NVIDIA at most interactivity levels, and at 60 tok/s per user a GB200 NVL72 GPU produced just under 3x the tokens of a B200. Those results depend on RDMA-class fabric. On a marketplace with plain TCP between pods, cross-node TP pays an all-reduce per layer per token over sockets; expect it to be usable only for throughput-insensitive work (no primary benchmark found; unverified).

**CPU, RAM, disk.**
- CPU: startup is CPU-bound ([arXiv 2606.07362](https://arxiv.org/html/2606.07362)); tokenisation, detokenisation and the API server are CPU work (hence `--api-server-count`, TRT-LLM's default of 8 frontends, and SGLang's Slurm example using 18 CPUs per task for 8 GPUs). Marketplace pods with 4 to 8 vCPUs per GPU are adequate; 1 to 2 vCPUs per GPU will throttle high-QPS small-model serving (unverified threshold).
- RAM: at least the checkpoint size for `eager`/page-cache loading, plus whatever is given to KV offload (`--kv_offloading_size`, `--hicache-ratio 2.0` means host pool is 2x the device pool). `/dev/shm` must be large for multi-GPU (`--shm-size=16G`).
- Disk: local NVMe for weights and for compile/kernel caches (persisting them was one of RunPod's four cold-start fixes). Network filesystems need `safetensors_load_strategy=eager|prefetch`.

**Driver and CUDA.** vLLM v0.33 publishes only CUDA 13.0 wheels and images (12.9 dropped); SGLang retired its CUDA 12 lane in v0.5.20. Hosts with older drivers fail with the PTX/"driver too old" errors unless the forward-compat path (`VLLM_ENABLE_CUDA_COMPATIBILITY=1`) is used, which is limited to datacenter/professional GPUs. A secondary source says CUDA 13 wants the R580+ driver branch ([runaihome](https://runaihome.com/blog/cuda-no-kernel-image-available-pytorch-fix-2026/); unverified against NVIDIA docs). PyTorch has dropped Maxwell/Pascal/Volta from current wheels; Turing through Blackwell remain ([PyTorch dev notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). On a marketplace the host driver is not under grid's control, so driver version must be a placement filter. See file 06.

## 9. Benchmark evidence

| Source | Setup | Result | Trust |
|---|---|---|---|
| [Spheron, Mar 2026](https://spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks) | 1x H100 80 GB, Llama-3.3-70B FP8, vLLM 0.18.0 / TRT-LLM 1.2.0 / SGLang 0.5.9, 512 in / 256 out | Output tok/s at concurrency 100: vLLM 2,400, SGLang 2,460, TRT-LLM 2,780. p95 TTFT: 1,450 / 1,380 / 1,280 ms. Cold start: 62 s / 58 s / about 28 min (engine build) | Vendor blog, own measurement, single run design; engines now 15 releases newer |
| [SemiAnalysis InferenceX v2, Feb 2026](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs) | About 1,000 GPUs, DeepSeek R1 etc., prefix caching off, curves of tok/s/user versus tok/s/GPU | Dynamo + TRT-LLM leads on NVIDIA; SGLang disagg on MI355X FP8 competitive with SGLang on B200; MTP cut GB300 cost per million tokens about 21x at 150 tok/s/user (8k/1k); B200 SGLang throughput doubled at some points between Oct 2025 and Feb 2026 | Best available: open-source harness, continuous, but vendor-tuned configs and paywalled analysis |
| [PremAI blog](https://www.premai.io/blog/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026) | H100, Llama-3.1-8B | SGLang and LMDeploy about 16,200 tok/s versus vLLM about 12,500 | Low: unnamed source, no versions or concurrency |
| [vLLM KV offload blog](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) | H100, Llama-3.1-8B | CPU KV offload: TTFT 2x to 22x lower on hits, up to 9x throughput at high hit rate | Primary, but best-case workload |

Reading: on a single GPU with a dense model, engine choice is worth roughly 3 to 15%. Software drift is larger than that (2x in four months on some SKUs), so pinning an engine version forfeits more than choosing the "wrong" engine. Workload shape (prefix reuse, structured output, long prompts) decides between vLLM and SGLang more than raw speed.

---

## Implications for grid

1. **Model a serving replica as a homogeneous gang**: N identical GPUs on one node (TP), optionally M identical nodes (TP x PP or `--nnodes`). Reuse the training gang-scheduler and rank plumbing: vLLM's `--nnodes/--node-rank/--master-addr/--headless` and SGLang's `--nnodes/--node-rank/--dist-init-addr` are the same rendezvous shape as torchrun (file 03). Prefer these over Ray to avoid a second cluster manager.
2. **Heterogeneity lives at the fleet level.** A deployment is a set of replica classes, each with (GPU class, count per replica, artifact/quantisation, engine flags). Selection rule: FP8-W8A8 artifact for Ada/Hopper/Blackwell, weight-only FP8 or INT4 for Ampere, NVFP4 for Blackwell. Never place mixed GPU models in one TP group.
3. **Fit-first placement.** Before considering multi-node, try: smaller quantisation, FP8 KV cache, lower `--max-model-len`. Only go multi-node when the provider offers RDMA; otherwise mark the configuration as degraded and verify with `NCCL_DEBUG` that the transport is not `NET/Socket`. Provide a sizing calculator (weights + KV per token x context x concurrency + about 3 to 8 GB overhead) and validate against the engine's logged "GPU KV cache size" and "Maximum concurrency".
4. **Engine adapters, two first-class:** vLLM (default) and SGLang (prefix-heavy, DeepSeek-style MoE, AMD, PD disaggregation). An adapter must provide: launch command builder, readiness probe, deep health probe, metrics mapping, LoRA and weight-update calls, drain.
5. **Probes.** Startup: poll until port opens and `/health` is 200 (vLLM) or `/ready` is 200 (SGLang); budget minutes, scaled by checkpoint size. Liveness: do not trust `/health` alone; send a periodic 1-token generate (SGLang `/health_generate`; vLLM a `max_tokens=1` completion) with a timeout, and treat failure as replica death. Also scrape `vllm:corrupted_requests` and Xid events (file 04).
6. **Failure unit is the replica process.** Engines exit on engine death (vLLM watchdog; SGLang `--watchdog-timeout`). grid should restart fast (see 7), drain via router, and retry requests upstream. For long generations consider token-replay migration in the gateway, as Dynamo does (`--migration-limit`), noting it cannot cover guided decoding or `n > 1`.
7. **Cold start is a first-order design target**, reusing file 09: bake engine image with precompiled kernels (`vllm download-kernels`), keep weights and compile caches on local NVMe, use `instanttensor`/`runai_streamer` when pulling from object storage, limit CUDA graph capture sizes, and evaluate `vllm preload` + `--load-format ipc_cache` so that an engine crash on a healthy node restarts without reloading weights. `--load-format dummy` is a cheap preflight that exercises the whole engine path without weights.
8. **Normalise metrics** into one schema (running, waiting, KV utilisation, TTFT, ITL, tokens/s, cache hit rate, rejections) from `vllm:*`, `sglang:*`, `llamacpp:*` for file 11's router and autoscaler.
9. **RL mode is a flag set, not a different system:** throughput operating point (high `--max-num-seqs`, no latency SLO), `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` with `load_inplace`, or SGLang `/update_weights_from_disk` and `--weight-version`; run only on a private network because both the LoRA update endpoint and vLLM multi-node traffic are unauthenticated code-execution surfaces.
10. **Version policy.** Engines release every two weeks with breaking flags (for example `quantization="fp8"` to `fp8_per_tensor`, CUDA 12.9 wheels dropped, Python 3.10/3.11 floors). grid should pin engine image per deployment, record host driver version as a placement constraint, and keep flag generation versioned per engine release.
11. **Skip:** TGI, NIM (as a dependency), LMDeploy, Ollama, MLX. Keep llama.cpp as an optional adapter for CPU-only or odd-GPU nodes, since it is the only engine with explicit uneven splits. Treat Dynamo as an optional upper layer whose frontend can run with `--discovery-backend file` and no etcd/NATS, rather than something to reimplement.

## Open questions

- **Cross-node TP/PP over plain TCP:** no primary benchmark found for vLLM or SGLang at 10/25/100 GbE without RDMA. Needs an experiment on RunPod-class networking (tokens/s versus single-node, by batch size).
- **Mixed-GPU TP behaviour** in vLLM and SGLang is undocumented: does startup fail, size KV to the smallest card, or OOM? Needs a test (for example 4090 + 3090, H100 + A100).
- **Current vLLM engine-death semantics** (v0.33): is exit-on-death still default, does `/live` exist, what HTTP code do in-flight streams get? Read `vllm/entrypoints/launcher.py` at the tag.
- **`vllm preload` / `ipc_cache` and CRIU snapshot** maturity, restart times, and whether they work inside unprivileged marketplace containers (CUDA IPC and CRIU usually need privileges).
- **SGLang dynamic LoRA and tensor/distributed weight-update endpoints**: exact names and semantics were not confirmed in this sweep (hand to file 12).
- **Exact defaults** for vLLM `--max-num-seqs`, `--max-num-batched-tokens`, chunked prefill, and the "extensible KV cache" behind the new `--gpu-memory-utilization` default of 1.0 versus 0.92.
- **Driver floor for CUDA 13 images** (R580?) and how many marketplace hosts meet it; whether `VLLM_ENABLE_CUDA_COMPATIBILITY` works on consumer cards (docs say only "select professional and datacenter" GPUs).
- **Dynamo internals** not retrieved from primary docs: router cost function, KVBM tiers, health-check env vars, behaviour outside Kubernetes at scale.
- **Disaggregation without RDMA**: can NIXL/Mooncake over TCP still improve tail ITL on marketplace networks, or does KV transfer time erase the benefit?
- **A neutral, current single-node benchmark** across vLLM 0.33 / SGLang 0.5.23 / TRT-LLM 1.4 on Hopper and Ada; the InferenceX public dataset (github.com/SemiAnalysisAI/InferenceX) should be mined for per-SKU numbers.
- **AMD and non-NVIDIA marketplaces**: MI300X availability on providers and upstream vLLM ROCm status after Feb 2026.
