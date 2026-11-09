# 17 — Serving-side source checks (vLLM, SGLang, routers, mid-stream failover)

Second sweep, 2026-11-09. Closes open questions from `10-inference-engines.md`, `11-serving-fleet-orchestration.md` and `12-inference-for-rl.md` by reading source at the current release tags.

**Method.** Shallow git clones at these tags, read locally (commit, commit date):

| Short | Repo @ tag | Commit | URL prefix used below |
|---|---|---|---|
| V | vllm-project/vllm @ `v0.33.0` | `4ba787e`, 2026-10-31 | `https://github.com/vllm-project/vllm/blob/v0.33.0/` |
| S | sgl-project/sglang @ `v0.5.23` | `297e54f`, 2026-10-28 | `https://github.com/sgl-project/sglang/blob/v0.5.23/` |
| P | vllm-project/production-stack @ `vllm-stack-0.1.14` | `405a2fa`, 2026-11-06 | `https://github.com/vllm-project/production-stack/blob/vllm-stack-0.1.14/` |
| D | ai-dynamo/dynamo @ `v1.6.2` | `28f9765`, 2026-11-04 | `https://github.com/ai-dynamo/dynamo/blob/v1.6.2/` |
| L | llm-d/llm-d-router @ `v1.0.0` (cloned under its old name `llm-d-inference-scheduler`; `go.mod` says `module github.com/llm-d/llm-d-router`) | `b9393d1`, 2026-10-15 | `https://github.com/llm-d/llm-d-router/blob/v1.0.0/` |
| G | kubernetes-sigs/gateway-api-inference-extension @ `v1.6.2` | `50872e0`, 2026-09-15 | `https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/v1.6.2/` |

Paths below are relative to those prefixes (for example V:`vllm/envs.py` is `https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py`). LiteLLM was read from its docs site, not source. Nothing here was executed; behaviour under load is in "Still open".

## Summary

- **vLLM engine death is still exit-on-death by default**, via a 5 s watchdog that sets `server.should_exit`; opt out with `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1`. `/health` returns 503 on `EngineDeadError`. There is **no `/live` and no `/ready`** on the API server (PR #36258 was closed unmerged on 2026-10-29). In-flight streams get an **error JSON as an SSE `data:` chunk followed by `data: [DONE]` under HTTP 200**, so a router must parse stream bodies to detect failure.
- **vLLM multi-node flags exist as described** (`--nnodes`, `--node-rank`, `--master-addr`, `--master-port` default 29501, `--headless`). Scheduler output goes to remote ranks over a ZMQ XPUB socket carrying pickled data; tensors go over NCCL via `torch.distributed`.
- **RL weight-update routes are confirmed and have grown**: `/start_draft_weight_update`, `/weight_checker`, `/abort_requests`; vLLM now also returns per-token weight-version spans. All sit behind `VLLM_SERVER_DEV_MODE`.
- **SGLang `/health` is not cheap any more**: it is the same handler as `/health_generate` and runs a 1-token generation by default. `/ready` is 503 while paused. Dynamic LoRA routes are `/load_lora_adapter`, `/load_lora_adapter_from_tensors`, `/unload_lora_adapter`.
- **Three routers now run without Kubernetes with runtime membership change**: SGLang Model Gateway (`POST/DELETE /workers`), production-stack router (`--dynamic-config-yaml`, re-read every 10 s), and, new since May 2026, the llm-d EPP itself (`file-discovery` plugin with `watchFile`). SGLang's new slim `sgl-router` has only static `--worker-urls` outside Kubernetes and no active health check.
- **GAIE's endpoint picker moved to `llm-d/llm-d-router`.** Built-in default scorer weights are queue 2, KV-utilisation 2, prefix 3; saturation defaults are queue depth 5 and KV 0.8.
- **Dynamo migration is token replay inside the Dynamo pipeline**: append emitted token IDs to the request's `token_ids`, subtract from `max_tokens`/`min_tokens`, redispatch. It needs Dynamo workers. The same can be built against plain vLLM using its scale-out API (`/v1/chat/completions/render` → `/inference/v1/generate` → `/v1/chat/completions/derender` with client-carried `stream_state`).
- **Fast restart**: `vllm preload` + `--load-format ipc_cache` needs no privileges (Unix socket, same user, same node). CRIU snapshots need root, `--privileged`, host sysctl changes, TP=1. SGLang has an equivalent weight-cache daemon.

---

## 1. vLLM v0.33.0

### 1.1 Engine death

- `EngineDeadError` (V:`vllm/v1/engine/exceptions.py`): "Raised when the EngineCore dies. Unrecoverable." `CLIENT_MESSAGE = "The server had an error while processing your request."`
- Watchdog (V:`vllm/entrypoints/launchers/launcher.py`): `watchdog_loop` sleeps `VLLM_WATCHDOG_TIME_S = 5.0` (a local constant, not an env var) and calls `terminate_if_errored`, which sets `server.should_exit = True` when `engine.errored and not engine.is_running`, unless `envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH or engine.should_keep_api_server_alive`. The env default is `False` (V:`vllm/envs.py` line 106). `should_keep_api_server_alive` is true only for a retired elastic-EP rank (V:`vllm/v1/engine/async_llm.py` line 1389). So uvicorn shuts down and the process exits within about 5 s of engine death.
- `/health` (V:`vllm/entrypoints/serve/instrumentator/health.py`): `await client.check_health()` → 200, `except EngineDeadError` → 503. No other state is expressed. A grep of all route decorators under `vllm/entrypoints` finds no `/live`; `/ready` and `/readyz` exist only on the data-parallel supervisor app (V:`vllm/entrypoints/launchers/dp_supervisor.py`), which probes child servers every `dp_supervisor_probe_interval_s = 5.0` with `dp_supervisor_probe_failure_threshold = 3`.
- Non-streaming requests on a dead engine get HTTP 500 `InternalServerError` (V:`vllm/entrypoints/serve/exception_handling/error_response.py`).
- Streaming: the handler docstring states it directly: "If an exception is encountered in a StreamingResponse generator, the exception is not raised, since we already sent a 200 status. Rather, we send an error message as the next chunk" (V:`vllm/entrypoints/serve/exception_handling/handlers/vllm_error.py`). The chat generator yields `create_error_response(e)` then `data: [DONE]` (V:`vllm/entrypoints/openai/chat_completion/serving.py` lines 947-953, 503-507). If the process is killed (spot reclaim), the client instead sees a truncated stream with no `[DONE]`.
- Separate stack-dump watchdog: `WatchdogConfig` with `timeout = 300`, `check_interval = 10`, `dump_dir = ""` (empty disables) (V:`vllm/config/watchdog_config.py`). It dumps stacks; it does not kill.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300` bounds one model step (V:`vllm/envs.py` line 252).
- New, opt-in: `--enable-fault-tolerance` registers `POST /fault_tolerance/apply` (only instruction: `"retry"`, returns 202) and `GET /fault_tolerance/status`; it targets data-parallel engine-core loss and requires `--api-server-count=1` (V:`vllm/entrypoints/serve/fault_tolerance/api_router.py`, V:`vllm/config/parallel.py` line 530).

### 1.2 Startup, shutdown, admission

- `VLLM_ENGINE_READY_TIMEOUT_S: int = 600` (V:`vllm/envs.py` line 26). The API socket is bound before engine setup but uvicorn only serves after the engine is built (V:`launcher.py` `setup_server`), so "connection refused" remains the loading signal.
- Graceful drain exists: `shutdown_timeout: int = 0` ("Shutdown will be delayed for up to this amount of time to allow already-running requests to complete"); on SIGTERM/SIGINT the server runs `engine_client.shutdown(timeout=...)` with `mode = "abort" if timeout == 0 else "drain"` (V:`vllm/config/vllm.py` line 454, V:`launcher.py`). Default is abort.
- `--sse-keep-alive-interval` (default 0 = off) sends an SSE comment line while a stream is idle "to prevent reverse proxies/tunnels with read timeouts from closing the connection" (V:`vllm/entrypoints/common/cli_args.py` line 164). This targets exactly RunPod's 100 s proxy cut (file 11 §8).
- Admission: `--max-num-queued-reqs` and `--max-num-queued-tokens` (both default `None`) reject with HTTP 503 "so the client can retry on another instance" (V:`vllm/config/scheduler.py`).

### 1.3 Defaults

- `gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)` (V:`vllm/config/cache.py` line 215). No "extensible KV cache" default of 1.0 exists in this tree.
- `max_num_seqs` / `max_num_batched_tokens` are `None` in `EngineArgs` and resolved per device in V:`vllm/engine/arg_utils.py` (lines 2944-3040), for the OpenAI server:

| Device | `max_num_seqs` | `max_num_batched_tokens` |
|---|---|---|
| >= 160 GiB (B200/B300) | 1024 | 16384 |
| >= 70 GiB, not A100, not GB10 (H100/H200) | 1024 | 8192 |
| everything else (incl. A100) | 256 | 2048 |
| CPU | 128 x world_size | 2048 x world_size |

  `--performance-mode throughput` doubles both when not set explicitly (default mode `balanced`). The dataclass fallbacks `DEFAULT_MAX_NUM_SEQS = 128` and `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` are "mainly for convenience when testing".
- `long_prefill_token_threshold = 0`; `max_num_active_seqs = None`.

### 1.4 CUDA and driver

- `ARG CUDA_VERSION=13.0.3`, `PYTHON_VERSION=3.12`, `torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` (V:`docker/Dockerfile`); `torch==2.14.0` (V:`requirements/cuda.txt`). The release pipeline builds with `CUDA_VERSION=13.0.2` plus a `cu134` variant and `cu134-efa` (V:`.buildkite/release-pipeline.yaml`).
- Driver floor, from V:`docs/getting_started/installation/gpu.cuda.inc.md` line 346: "CUDA 13 requires an R580 or newer driver. Compatibility mode supports R535 and R570 host drivers". Compatibility mode is `VLLM_ENABLE_CUDA_COMPATIBILITY=1` (image default `0`) and "only supports select professional and datacenter NVIDIA GPUs".
- The same doc still says wheels are "compiled with CUDA 12.9 ... by default"; that text contradicts the Dockerfile and release pipeline at the same tag and looks stale. I could not open the v0.33.0 release-notes body (GitHub page did not render), so which wheel variants are on PyPI is (unverified).

### 1.5 Multi-node

- Fields in V:`vllm/config/parallel.py`: `master_addr: str = "127.0.0.1"`, `master_port: int = 29501`, `node_rank: int = 0`, `nnodes: int = 1`, all documented as "when distributed_executor_backend is mp"; "To use "mp" you must also set nnodes". Example in V:`docs/serving/parallelism_scaling.md` lines 148-158 uses `--nnodes 2 --node-rank 0 --master-addr <HEAD_NODE_IP>` and `--node-rank 1 ... --headless`.
- Control path: the leader creates a `MessageQueue` whose remote side is `context.socket(XPUB)` bound to `tcp://{connect_ip}:0` (random port), and remote readers "receive the pickled bytes" (V:`vllm/distributed/device_communicators/shm_broadcast.py` lines 526-576). Tensor path: workers call `init_device()` to form `torch.distributed` groups before message queues are set up ("multi-node setups (nnodes_within_dp > 1) require distributed groups to be initialized", V:`vllm/v1/executor/multiproc_executor.py` line 805), so activations ride NCCL (IB/RoCE if present, else sockets).
- Consequences for grid: nodes need full mutual TCP reachability on `master_port` plus ephemeral ports, and the pickle channel is a code-execution surface.

### 1.6 RL weight updates and LoRA

Registration (V:`vllm/entrypoints/launchers/api_server/routers.py`): `if envs.VLLM_SERVER_DEV_MODE:` → dev routers and `register_rl_api_routers(app)`. Routes in V:`vllm/entrypoints/rl/online/api_router.py`:

| Route | Body / query | Notes |
|---|---|---|
| `POST /pause` | query `mode` in `"abort"` (default), `"wait"`, `"keep"`; `clear_cache` (deprecated) | "In abort and wait modes new requests are rejected with HTTP 503 until /resume; keep continues to accept and queue them" |
| `POST /resume`, `GET /is_paused` | | `{"is_paused": bool}` |
| `POST /abort_requests` | `{"request_ids": [...]}`; empty aborts all | does not pause |
| `POST /init_weight_transfer_engine` | `{"init_info": {...}}` | NCCL info: `master_address`+`master_port` or `nccl_unique_id_b64`, `rank_offset`, `world_size`, `packed`, `packed_buffer_size_bytes`, `packed_num_buffers` (V:`vllm/distributed/weight_transfer/nccl_common.py`) |
| `POST /start_weight_update`, `POST /start_draft_weight_update` | none | draft = speculative-decoding draft model |
| `POST /update_weights` | `{"update_info": {...}}` | repeatable |
| `POST /finish_weight_update` | `{"weight_version": str?}` | |
| `POST /update_weight_version` | `{"new_version": str}` | |
| `GET /weight_info` | | `{"weight_version": ...}` |
| `POST /weight_checker` | `{"action": "checksum"|"reset"|"compare", "baseline": {...}?}` | real tensor checksums per worker |
| `POST /shutdown_weight_transfer_engine`, `GET /get_world_size?include_dp=` | | |

- Backends: `WeightTransferConfig.backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt", "nccl_m2n"] | str = "nccl"` (V:`vllm/config/weight_transfer.py`); the factory also registers `"modelexpress"` (V:`vllm/distributed/weight_transfer/factory.py`).
- The `nccl_unique_id_b64` rendezvous exists "for torch-free trainers (e.g. JAX)"; all ranks must enter init concurrently and do a one-element `all_reduce` handshake.
- Provenance: chat responses carry `weight_versions: list[WeightVersionSpan]` when the request sets `return_weight_versions` (V:`vllm/entrypoints/openai/chat_completion/protocol.py` lines 140, 446).
- LoRA (V:`vllm/entrypoints/serve/lora/`): routes exist only if `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set (independent of dev mode). `POST /v1/load_lora_adapter` body `LoadLoRAAdapterRequest {lora_name, lora_path, load_inplace=False, is_3d_lora_weight=False}`; `POST /v1/unload_lora_adapter` body `{lora_name, lora_int_id?}`. Hard constraint: "VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1" (V:`vllm/entrypoints/cli/serve.py` line 317). The server logs "This should ONLY be used for local development!".

### 1.7 KV events and metrics

- `KVEventsConfig` (V:`vllm/config/kv_events.py`): `enable_kv_cache_events = False`, `publisher` `"null"|"zmq"` (auto `zmq` when enabled), `endpoint = "tcp://*:5557"`, `replay_endpoint = None`, `buffer_steps = 10_000`, `hwm = 100_000`, `max_queue_size = 100_000`, `topic = ""`. CLI flag `--kv-events-config`. New `GET /kv_event_sources` returns the publisher config per DP rank (V:`vllm/entrypoints/serve/instrumentator/basic.py`), which lets a router discover the ZMQ endpoints over HTTP. The engine binds; consumers dial in.
- Metric names found by grep of `name="vllm:..."` in the tree, router-relevant subset: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_requests_waiting_by_reason`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions`, `vllm:admission_rejections`, `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, `vllm:prompt_tokens`, `vllm:prompt_tokens_cached`, `vllm:generation_tokens`, `vllm:request_success`, `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:corrupted_requests`, `vllm:engine_sleep_state`, `vllm:lora_requests_info`. `GET /load` remains.

## 2. SGLang v0.5.23

- **Entry point**: `[project.scripts] sglang = "sglang.cli.main:main"` with `serve.py` under `sglang/cli` (S:`python/pyproject.toml`), so `sglang serve` exists; `python -m sglang.launch_server` still works. Defaults `host = "127.0.0.1"`, `port = 30000` (S:`python/sglang/srt/arg_groups/fields/serving.py`). Args now live in `srt/arg_groups/fields/*.py`.
- **Multi-node** (S:`.../fields/parallel.py`): `nnodes = 1`, `node_rank = 0`, `dist_init_addr` (alias `--nccl-init-addr`, e.g. `192.168.0.2:25000`), `nccl_port` (random), `dist_timeout`. New: `gated_launch_port`, where every rank blocks after distributed init "before any sizable GPU allocation, until `POST /gate/activate`" (S:`python/sglang/srt/distributed/gated_launch.py`): a built-in warm-standby gate.
- **Health** (S:`python/sglang/srt/entrypoints/http_server.py` lines 714-800):
  - `GET /ready` → 200 iff `tokenizer_manager.is_ready()`, which is `not self.is_pause and self.is_healthy()`.
  - `GET /health` and `GET /health_generate` share one handler. 503 if `gracefully_exit` or `ServerStatus.Starting`. `/health` returns 200 without generating only when `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` is false; its default is `EnvBool(True)` (S:`python/sglang/srt/environ.py` line 344). Otherwise it submits `input_ids=[0]`, `max_new_tokens: 1` and waits up to `HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))` seconds; on timeout it sets `ServerStatus.UnHealthy` and returns 503. "If the server is running something, this request will be ignored".
  - Health, ready and metrics paths bypass API-key auth by prefix match (S:`python/sglang/srt/utils/auth.py`).
- **Watchdog** (S:`.../fields/device.py`, S:`python/sglang/srt/utils/watchdog.py`): `watchdog_timeout = 300`, `soft_watchdog_timeout = None`. On timeout: py-spy dump, sleep 5 s, `self.parent_process.send_signal(signal.SIGQUIT)`. A `SubprocessWatchdog` does the same when a subprocess crashes natively.
- **Weight updates** (routes in `http_server.py`, bodies in S:`python/sglang/srt/managers/io_struct.py`):
  - `POST /update_weights_from_disk`: `model_path`, `load_format`, `abort_all_requests=False`, `weight_version`, `is_async=False`, `torch_empty_cache=False`, `keep_pause=False`, `recapture_cuda_graph=False`, `token_step=0`, `flush_cache=True`, `manifest`. Returns `{success, message, num_paused_requests}`, 400 on failure.
  - `POST /init_weights_update_group` (`master_address`, `master_port`, `rank_offset`, `world_size`, `group_name="weight_update_group"`, `backend="nccl"`) → `POST /update_weights_from_distributed` (`names`, `dtypes`, `shapes`, `group_name`, `flush_cache=True`, `abort_all_requests=False`, `weight_version`, `load_format`, `selector` `"target"|"draft"|"all"`) → `POST /destroy_weights_update_group`.
  - `POST /update_weights_from_tensor` (`serialized_named_tensors` base64 list, `load_format`, `flush_cache`, `abort_all_requests`, `weight_version`, `selector`).
  - `POST /update_weights_from_ipc` (`zmq_handles: Dict[str, str]`, `flush_cache`, `weight_version`).
  - `POST /begin_weight_update` / `POST /end_weight_update`; `POST /update_weight_version` (`new_version`, `abort_all_requests=True`); `/weights_checker`; `/get_weights_by_name`.
  - Peer-to-peer seeding: `/init_weights_send_group_for_remote_instance`, `/send_weights_to_remote_instance`, `/remote_instance_transfer_engine_info`.
  - `--weight-version` default `"default"`.
- **Pause**: `POST /pause_generation` body `mode: "abort" | "retract" | "in_place"` (default `"abort"`); `POST /continue_generation` body `torch_empty_cache=True`. While paused `/ready` is 503.
- **LoRA**: `POST /load_lora_adapter` (`lora_name`, `lora_path`, `pinned=False`, `lora_id?`), `POST /load_lora_adapter_from_tensors`, `POST /unload_lora_adapter` (`lora_name`, `lora_id?`). Flags: `--enable-lora`, `--max-loras-per-batch` (8), `--max-loaded-loras`, `--max-lora-rank`, `--lora-target-modules`, `--lora-eviction-policy`.
- **Auth**: control routes are marked `@auth_level(AuthLevel.ADMIN_OPTIONAL)`: open if no keys are configured, otherwise accept `--api-key` or `--admin-api-key`. Only `ADMIN_FORCE` routes demand the admin key.
- **Metrics** (`--enable-metrics`, default off): router-relevant names from `srt/observability`: `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:num_paused_reqs`, `sglang:token_usage`, `sglang:kv_used_tokens`, `sglang:kv_available_tokens`, `sglang:cache_hit_rate`, `sglang:gen_throughput`, `sglang:time_to_first_token_seconds`, `sglang:inter_token_latency_seconds`, `sglang:e2e_request_latency_seconds`, `sglang:queue_time_seconds`, `sglang:num_aborted_requests_total`, `sglang:startup_time_seconds`, `sglang:weight_load_duration_seconds`.
- **KV events**: SGLang has the same `--kv-events-config` (S:`.../fields/observability.py` line 144); sgl-router's docs show `{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}`.

## 3. Routers that run without Kubernetes

| Router | Version read | Backend membership at runtime | Health | Cache-aware policy | Retry / failover | HA |
|---|---|---|---|---|---|---|
| **SGLang Model Gateway** (`sgl-model-gateway`, Rust) | crate 0.3.2 in S | `POST /workers` (body `url`, `model_id`, `priority`, `labels`, ...; 202 + job queue), `GET /workers`, `PUT`/`DELETE /workers/{worker_id}` | Active: `--health-check-endpoint` default `/health`, interval 60 s, timeout 5 s, failure threshold 3, success 2; circuit breaker (failure 10, success 3, timeout 60 s, window 120 s) | `cache_aware`: router-side approximate radix tree over **text characters**; `--cache-threshold 0.3`, `--balance-abs-threshold 64`, `--balance-rel-threshold 1.5`, `--eviction-interval 120`, `--max-tree-size 67108864`. Imbalanced iff `(max-min) > abs` and `max > rel*min`, then shortest queue | `--retry-max-retries` 5 on 408/429/500/502/503/504; queue `--queue-size 100`, `--queue-timeout-secs 60` | Not documented in README (unverified) |
| **sgl-router** (new, "Slim, KV-aware router") | crate 0.1.0 in S | Static `--worker-urls`, or Kubernetes EndpointSlices. No add/remove API found | **None** under `--worker-urls`: "there is no active health check"; circuit breaker only (`--cb-threshold 3`, `--cb-cool-down-secs 30`) | `cache_aware` from **engine KV events** (precise), `--cache-affinity-min-matched-tokens 1024`, `--affinity-mode prefer|balanced`, `--affinity-load-factor 2`; also `power_of_two` (default), `session_aware` (`x-session-id`) | `--retry-max-attempts` default 1 (off). "Once a streaming response's 2xx status has been sent, it is never retried" | "Routers are stateless ... run more router replicas behind a load balancer"; state rebuilt from engine reports; KV-tree peer bootstrap is Kubernetes-only |
| **production-stack router** (Python) | vllm-stack-0.1.14 | `--dynamic-config-yaml` / `--dynamic-config-json`, watched every 10 s; reloadable fields include `static_backends`, `static_models`, `routing_logic`, `session_key` (P:`src/vllm_router/dynamic_config.py`, `app.py`) | Opt-in `--static-backend-health-checks`; interval `--static-backend-health-check-interval 60`, timeout 10 s; sends a real POST per model (P:`src/vllm_router/utils.py` `is_model_healthy`) | `prefixaware`: router-side `HashTrie` over prompt text, "we assume that there is no eviction of prefix cache", `--prefix-min-match-length 0`. `kvaware`/`loadaware`: need an LMCache controller; `score(i) = matched_tokens(i)/prompt_tokens - beta * relative_load(i)` | `--max-instance-failover-reroute-attempts` default 0 | Single process; none found |
| **Dynamo frontend** | v1.6.2 | Workers self-register through discovery: `--discovery-backend` choices `kubernetes|etcd|file|mem`, **default `etcd`**. `file` is a directory (`DYN_FILE_KV`, default `$TMPDIR/dynamo_store_kv`) with `DEFAULT_TTL` 10 s keep-alive, so multi-host use needs a shared filesystem (D:`lib/runtime/src/storage/kv.rs`, `.../kv/file.rs`) | `DYN_HEALTH_CHECK_ENABLED`, `DYN_HEALTH_CHECK_REQUEST_TIMEOUT`, `DYN_CANARY_WAIT_TIME`; failed worker inhibited `DYN_RUNTIME_INHIBITED_DURATION_SECS` | Cost below | `--migration-limit` (default 0), `--migration-max-seq-len` | `router_replica_sync` (default false), described as advisory |
| **llm-d EPP** | v1.0.0 | `file-discovery` plugin: endpoints YAML (`name`, `address` IPv4 literal, `port`, `rankIndex`, `labels`), `watchFile: true` reconciles on write or atomic rename; max 1 MiB (L:`docs/discovery.md`) | Metrics scrape staleness (200 ms threshold) | approximate or precise (`precise-prefix-cache-producer`, dials engine ZMQ) | via Envoy | Needs Envoy v1.31+ in front (ext-proc on :9002). `InferenceObjective` and `InferenceModelRewrite` are inactive in file mode |
| **LiteLLM proxy** | docs only | `POST /model/new`, `POST /model/delete` (`{"id": ...}`), `PATCH /model/{model_id}/update`; requires `store_model_in_db` and a database | Cooldown: default `cooldown_time` 5 s; without `allowed_fails`, cools on 429, >50% failures in the minute, or 401/404/408 | none (`prompt_caching` pre-call check is provider-side) | `num_retries`, `fallbacks` | Redis for shared cooldown/usage and session affinity |

Sources: S:`sgl-model-gateway/README.md`, S:`sgl-model-gateway/src/main.rs`, S:`sgl-model-gateway/src/policies/cache_aware.rs`, S:`sgl-router/README.md`, S:`sgl-router/REFERENCE.md`, P:`src/vllm_router/parsers/parser.py`, P:`src/vllm_router/routers/routing_logic.py`, D:`components/src/dynamo/frontend/frontend_args.py`, [LiteLLM routing](https://docs.litellm.ai/docs/routing), [LiteLLM model management](https://docs.litellm.ai/docs/proxy/model_management).

production-stack `--routing-logic` choices at this tag: `roundrobin`, `session`, `kvaware`, `loadaware`, `prefixaware`, `disaggregated_prefill`, `disaggregated_prefill_orchestrated`, `priority`. `--engine-stats-interval 30`, `--request-stats-window 60` are unchanged and not reloadable.

LiteLLM `routing_strategy` values per docs: `simple-shuffle` (default), `least-busy`, `latency-based-routing`, `usage-based-routing`, `usage-based-routing-v2`, `cost-based-routing`.

**Dynamo cost function** (D:`lib/kv-router/src/scheduling/selector/reference.rs` lines 296-381; defaults in D:`lib/kv-router/src/scheduling/config.rs`). Lower is better; `router_temperature = 0.0` means argmin:

```
overlap_credit_decay  = 1 / (1 + overlap_score_credit_decay * excess_active_prefill_blocks / request_blocks)
overlap_credit_blocks = overlap_score_credit * overlap_credit_decay * device_overlap_blocks
                      + host_cache_hit_weight * host_overlap_blocks
                      + disk_cache_hit_weight * disk_overlap_blocks
                      + shared_cache_multiplier * shared_beyond_device_blocks
raw_prefill_blocks    = (worker active_prefill_tokens + this request's tokens) / block_size
logit = prefill_load_scale * max(0, raw_prefill_blocks - overlap_credit_blocks)
      + decode_cost_blocks
      + decode_active_request_weight * active_requests
```

Defaults: `overlap_score_credit = 1.0`, `overlap_score_credit_decay = 0.0`, `prefill_load_scale = 1.0`, `decode_active_request_weight = 0.0`, `host_cache_hit_weight = 0.75`, `disk_cache_hit_weight = 0.25`, `use_kv_events = true`, `router_assume_kv_reuse = true`. `overlap_score_weight` is deprecated in favour of these. The useful idea for grid: cache hits on slower tiers get fractional credit, and the unit is "blocks of prefill work left to do", which is naturally comparable across replicas.

## 4. llm-d router (former GAIE EPP) as design reference

- G:`README.md`: "The Endpoint Picker (EPP), InferenceObjective and InferenceModelRewrite APIs, and Body Based Router (BBR) packages have moved to new repositories: EPP and associated APIs: llm-d/llm-d-router". The GAIE repo keeps only the `InferencePool` API, a lightweight EPP (`cmd/lwepp`) and conformance tests.
- Config kind: `apiVersion: llm-d.ai/v1`, `kind: EndpointPickerConfig` (L:`deploy/config/epp-precise-prefix-cache-config.yaml`).
- Built-in default when no config is given (L:`pkg/epp/config/loader/defaults.go`): plugins `queue-scorer`, `kv-cache-utilization-scorer`, `prefix-cache-scorer`, `metrics-data-source`, `core-metrics-extractor`; weights `queueScorerWeight := 2.0`, `kvCacheUtilizationScorerWeight := 2.0`, `prefixCacheScorerWeight := 3.0`; `DefaultScorerWeight = 1.0` for anything unweighted. The precise-prefix sample uses prefix 2.0 / KV 1.0 / queue 1.0 with `max-score-picker` and `blockSizeTokens: 64`.
- Saturation (L:`pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/config.go`): `DefaultQueueDepthThreshold int = 5`, `DefaultKVCacheUtilThreshold float64 = 0.8`, `DefaultMetricsStalenessThreshold = 200 * time.Millisecond`.
- Objective header: `ObjectiveKey = "x-llm-d-inference-objective"` (L:`pkg/epp/metadata/consts.go`); profile header `x-llm-d-epp-profile`.
- Registered plugin types worth knowing (grep of `...Type = "..."`): scorers `queue-scorer`, `kv-cache-utilization-scorer`, `prefix-cache-scorer`, `lora-affinity-scorer`, `load-aware-scorer`, `active-request-scorer`, `running-requests-size-scorer`, `token-load-scorer`, `session-affinity-scorer`, `no-hit-lru-scorer`, `latency-scorer`, `topology-affinity-scorer`; prefix producers `approx-prefix-cache-producer`, `precise-prefix-cache-producer`, `burst-prefix-cache-producer`; pickers `max-score-picker`, `random-picker`, `weighted-random-picker`; detectors `utilization-detector`, `concurrency-detector`, `max-saturation-detector`; parsers `vllmhttp-parser`, `vllmgrpc-parser`, `sglanghttp-parser`, `openai-parser`; discovery `file-discovery`, `multicluster-file-discovery`; state sync `redis-syncer`, `local-syncer`.

## 5. Mid-stream failover

**How Dynamo does it** (D:`lib/llm/src/migration.rs`). `RetryManager` wraps the stream. `track_response` on each chunk: extends `self.request.token_ids` with the chunk's token IDs, sets `max_tokens = max_tokens.saturating_sub(output_len)` and likewise `min_tokens`, and stores a `DecoderCheckpoint` (`jail_seed`) so text withheld by the incremental detokenizer or stop-sequence filter is restored. On a migratable error it opens a new stream with the mutated request. Migratable reasons: `transport.cannot_connect`, `transport.disconnected`, `transport.connection_timeout`, `backend.cannot_connect`, `backend.disconnected`, `backend.connection_timeout`, `backend.response_timeout`, `backend.engine_shutdown`, `backend.stream_incomplete`, `backend.worker_unavailable`, `capacity.worker_overloaded`. Blocking: `request.cancelled`, `request.deadline_exceeded`, `backend.cancelled`, `capacity.exhausted`, `capacity.pool_exhausted`. Disabled for guided decoding ("FSM state is not transferable (applies to all backends)"), `n > 1`, requests pinned to a worker, and when prompt + output exceeds `--migration-max-seq-len`. Source comments list what is not carried: "generated-token penalties and thinking-token budgets are not", and `prompt_embeds` requests. This works because the Dynamo frontend owns tokenisation and detokenisation and workers are token-in/token-out; it does not apply to plain OpenAI-compatible backends.

**Against plain vLLM: three options, best first.**

1. **Scale-out API** (needs `--enable-scale-out`; otherwise routes answer 403). `POST /v1/chat/completions/render` gives prompt token IDs; `POST /inference/v1/generate` takes `token_ids` and streams tokens; `POST /v1/chat/completions/derender` turns token chunks into chat deltas, and for streaming "the client carries `stream_state` between successive calls" (V:`vllm/entrypoints/scale_out/`). A router holding prompt IDs, emitted IDs and `stream_state` can re-issue generate with `token_ids = prompt + emitted` on another replica and keep the same derender state, so tool-call and reasoning parsing continue without a seam. This is the Dynamo design using stock vLLM. Cost: one extra HTTP hop per chunk for derender, or run a `vllm launch render` sidecar.
2. **`/v1/completions` with token IDs**: render once, then stream completions with `prompt` as a token list and `return_token_ids`; continuation is prompt + emitted IDs. Exact for raw text; the router must do its own tool/reasoning parsing.
3. **`/v1/chat/completions` with `continue_final_message: true`, `add_generation_prompt: false`** and the emitted text as a trailing assistant message (mutually exclusive flags, V:`.../chat_completion/protocol.py` line 1063). Simplest, with caveats:
   - Re-tokenising emitted text may not reproduce the original token sequence.
   - Reasoning: vLLM parsers have `set_prompt_token_ids` to "start parsing in content when the prompt already closes reasoning, as with continue_final_message", but only "configs with turn boundaries" detect this (V:`vllm/parser/engine/parser_engine.py` line 235). A stream cut mid-reasoning needs the partial reasoning placed back in the template in the model-specific form.
   - Tool calls: a call cut mid-arguments has its opening marker in the prompt; I found no hook that seeds tool-parser state from the prompt, so treat mid-tool-call failover as unsupported on this path (unverified by test).
   - SGLang supports the same flag and seeds its reasoning parser from `previous_content` (S:`python/sglang/srt/parser/reasoning_parser.py` lines 100-116).

**Limits common to all three.** Constrained decoding and `n > 1` cannot be continued. A `seed` restarts the RNG stream, so the continuation is a valid sample but not the original. Repetition/presence penalties see only the new request's generated tokens unless the engine counts prompt tokens. Logprobs for already-delivered tokens stay valid; new ones are conditioned identically if token IDs are replayed exactly. In RL, a migrated trajectory may span replicas at different policy versions; check `weight_versions` spans or discard.

**Failure detection for the router**: (a) connection reset or EOF without `data: [DONE]`; (b) an SSE `data:` payload that parses as an error object (vLLM engine death under HTTP 200); (c) idle timeout, measured with `--sse-keep-alive-interval` comments excluded.

## 6. Fast restart

| Mechanism | Engine | What it saves | Privileges | Source |
|---|---|---|---|---|
| `vllm preload --model M --tensor-parallel-size N`, then `vllm serve ... --load-format ipc_cache` | vLLM | Weight load and quantisation post-processing; daemon also pre-pays the FlashInfer autotune pass. "cutting restart time from minutes ... to seconds" (docs' claim, no number) | **None beyond same user, same node.** Unix socket in a per-user `0700` dir, files `0600`, peer UID checked via `SO_PEERCRED`. Pickle protocol. My inference, not stated in the docs: engine and daemon need the same socket dir and GPU IPC view, so in practice the same container | V:`docs/features/preload.md`, V:`vllm/entrypoints/cli/preload.py` |
| Loader keys | vLLM | `mode` `zero_copy` (default; daemon must outlive engine; incompatible with sleep mode) or `copy`; `fallback: true`; `connect_timeout_s 5.0`; `state_timeout_s 300.0` | | same |
| Daemon-to-daemon seeding `--seed-listen-host/-port`, `--seed-source-host/-port`, `--seed-token` | vLLM | New node fetches autotune caches from a peer (base port example 29600) | TCP, token-authenticated, unencrypted | same |
| Initialized engine snapshots (CRIU) | vLLM | Nearly all init | "Root or passwordless sudo for CRIU"; example uses `--privileged --pid=host --ipc=host --network=host`; `kernel.io_uring_disabled=2` host-wide; one GPU, TP1, plaintext unauthenticated HTTP, same machine, remote model ID with 40-char `--revision`; "CUDA 12.x and Arm64 images omit it" | V:`docs/features/initialized_snapshots.md` |
| `tensorizer`, `runai_streamer`, `runai_streamer_sharded`, `instanttensor`, `sharded_state`, `modelexpress` | vLLM | Load step only | none | V:`vllm/config/load.py` |
| `fastsafetensors` | vLLM | Not a `--load-format` value at v0.33.0; appears only in test requirements (`fastsafetensors==0.3.3`). A plugin could add it (unverified) | | V:`requirements/test/cuda.txt` |
| Weight cache daemon: `python -m sglang.srt.weight_cache.daemon --model-path M --tp-size N`, engine `--weight-cache-mode client` (`off|daemon|client`), `--weight-cache-socket`, `--weight-cache-timeout 1800` | SGLang | Same idea (CUDA IPC handles over a Unix socket). Engine-spawned `daemon` mode does "NOT persist across restarts"; run the standalone daemon for recovery | not stated; Unix socket | S:`python/sglang/srt/weight_cache/daemon.py`, S:`.../fields/model.py` |
| `--load-format presharded` | SGLang | Dumps per-rank post-quantisation shards on first load; later loads "skip re-quantization" | none; shared FS for multi-node | S:`.../fields/model.py` |
| `--startup-weight-load-mode overlap` (default `serial`) | SGLang | Overlaps file staging with CUDA graph capture | none | same |
| `LoadFormat` enum also has `fastsafetensors` (`enable_gds` default true), `runai_streamer`, `ipc_cache`, `remote_instance`, `rdma`, `local_cached`, `flash_rl` | SGLang | | GDS needs host support | S:`python/sglang/srt/configs/load_config.py`, S:`.../model_loader/loader.py` line 691 |

Verdict for marketplace containers: the preload/weight-cache daemons are usable (they survive an engine crash, not a container or node loss, and hold the weights' VRAM for the daemon's lifetime). CRIU snapshots are not usable on RunPod-class unprivileged pods.

## Corrections to first sweep

1. **`10` §2.1**: "`--gpu-memory-utilization` ... docs now say 1.0 with extensible KV cache, otherwise 0.92". At v0.33.0 the default is 0.92 unconditionally (V:`vllm/config/cache.py`); no such mode in the tree.
2. **`10` §2.7**: "Checkpoint endpoints `/checkpoint/prepare|status|restore`". No such routes exist at v0.33.0 (grep of the tree). Snapshot is a CLI flow (`vllm snapshot`), not HTTP.
3. **`10` §2.7**: `/live` "merge status not confirmed". PR #36258 was closed unmerged on 2026-10-29 ([PR](https://github.com/vllm-project/vllm/pull/36258)); there is no `/live`. Exit-on-death is confirmed as current behaviour, not just v0.10-v0.17.
4. **`10` §2.1**: `--max-num-seqs` "(see --help)". Actual defaults are device-tiered (256 or 1024), see §1.3. This changes capacity assumptions per GPU class.
5. **`10` §3**: SGLang "`/health` (cheap)". At v0.5.23 `/health` generates a token by default, same handler as `/health_generate`. For a cheap check use `/ready`, or set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`.
6. **`10` §3**: "a newer `sglang serve` entrypoint ... unverified" is confirmed. "dynamic load/unload endpoints ... names not confirmed" are `/load_lora_adapter`, `/load_lora_adapter_from_tensors`, `/unload_lora_adapter`.
7. **`10`/`11`**: the SGLang router is now two Rust programs, `sgl-router` (new, KV-event based) and `sgl-model-gateway`; `cache_aware_zmq` was removed from sgl-router. Only the gateway has a runtime worker API and active health checks.
8. **`11` §1 table and §1.1**: GAIE and llm-d "Runs without Kubernetes? No". The EPP now lives in `llm-d/llm-d-router` and runs without Kubernetes with `file-discovery` (still needs Envoy). Weights "queue 2 / KV 2 / prefix 3" are the built-in defaults, not just a guide's example. Saturation defaults 5 and 0.8 are confirmed. The objective header is `x-llm-d-inference-objective`, not `x-gateway-inference-objective`.
9. **`11` §1.5**: production-stack routing logics are eight (list in §3), the static health-check interval is a flag (default 60 s), and backends can be changed at runtime through the dynamic config file.
10. **`10` §5 / `11`**: Dynamo `--discovery-backend file` is a filesystem directory with 10 s TTL keep-alives; it is single-host or shared-filesystem only. The default backend is `etcd`. The router cost is no longer "load + overlap" weighted by `overlap_score_weight` (deprecated); see §3.
11. **`12` §2.3**: "the only engine-level provenance for mixed-policy trajectories" is SGLang's. vLLM v0.33.0 also has `return_weight_versions` / `weight_versions` spans.
12. **`12` §2.2**: route list is incomplete; add `/start_draft_weight_update`, `/weight_checker`, `/abort_requests`; `modelexpress` is a registered backend name; NCCL init accepts `nccl_unique_id_b64`. SGLang additionally has `/update_weights_from_ipc` and `/update_weight_version`.
13. **`10` §8**: "CUDA 13 wants the R580+ driver branch (unverified)" is confirmed by vLLM's own docs; compatibility mode covers R535 and R570.
14. **`11` §1.8**: LiteLLM default cooldown is 5 s; runtime model add/remove needs `store_model_in_db` and a database.

## Implications for grid

**Serving replica contract (one engine process group per replica).**

- *Launch.* vLLM: `vllm serve M --host 0.0.0.0 --port P --shutdown-timeout T --sse-keep-alive-interval 15 --max-num-queued-reqs Q --enable-scale-out --kv-events-config '{"enable_kv_cache_events":true,"replay_endpoint":"tcp://*:5558"}'`, with explicit `--max-num-seqs`/`--max-num-batched-tokens` (defaults differ 4x between A100 and H100). Leave `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` unset so death means exit, and run under a supervisor that restarts. SGLang: `sglang serve --model-path M --host 0.0.0.0 --port P --enable-metrics --watchdog-timeout 300`, optional `--gated-launch-port` for warm standby. Multi-node replicas: vLLM `--nnodes/--node-rank/--master-addr/--master-port` (+ `--headless` on non-zero ranks); SGLang `--nnodes/--node-rank/--dist-init-addr`.
- *Placement filter.* Host driver R580+ for stock images; R535/R570 only with `VLLM_ENABLE_CUDA_COMPATIBILITY=1` on datacenter/professional GPUs.
- *Readiness.* vLLM: TCP connect → `GET /health` 200 → one `max_tokens=1` completion with a latency bound. SGLang: `GET /ready` 200 (already covers paused and unhealthy), then `/health_generate` once. grid's agent, not the engine, must publish the three-state signal (initialising / ready / draining), since vLLM has none.
- *Liveness.* vLLM: process alive and `/health` 200, plus a periodic 1-token generate (deadlock is invisible to `/health`). SGLang: `/health` already generates; give the probe a timeout above 20 s or lower `SGLANG_HEALTH_CHECK_TIMEOUT`.
- *Drain.* Mark draining in the router first, then SIGTERM. vLLM drains only if `--shutdown-timeout > 0`. SGLang returns 503 on health during `gracefully_exit`.
- *Metrics scraped.* Running, waiting, KV usage, prefix hits/queries, TTFT, ITL, rejections, corrupted requests (names in §1.7, §2), plus `/load` on vLLM. Scrape at 1-2 s for routing; llm-d treats metrics older than 200 ms as stale, production-stack's 30 s is too slow.
- *Weight-update hooks.* vLLM: `VLLM_SERVER_DEV_MODE=1`; sequence `/pause?mode=` → `/start_weight_update` → `/update_weights`* → `/finish_weight_update {weight_version}` → `/weight_checker {action: compare}` → `/resume`; verify with `/weight_info`. SGLang: `/pause_generation` → `/update_weights_from_{disk,distributed,tensor,ipc}` with `weight_version` → `/continue_generation`; verify with `/get_model_info` and `/weights_checker`. LoRA on vLLM forces `--api-server-count 1`. All of these are unauthenticated or weakly authenticated: bind the engine port to a private interface and expose only the router.

**Router requirements.**

1. Runtime add/remove of backends by API, with push registration (re-addressable endpoints). Only SGLang Model Gateway offers this over HTTP today; production-stack and the llm-d EPP do it by file rewrite, which grid's agent can drive.
2. Active health checks with a real generation, plus passive ejection (circuit breaker).
3. Scoring in the llm-d/Dynamo shape: prefix affinity + queue + KV utilisation, with a saturation gate (queue > 5 or KV > 0.8 as starting values) and capacity normalisation per replica.
4. Stream-aware failure detection (error chunk under HTTP 200, missing `[DONE]`) and retry before first token.
5. Mid-stream continuation via token replay. Prefer the vLLM scale-out path; fall back to `continue_final_message`; refuse for constrained decoding, `n > 1`, and mid-tool-call.
6. Stateless routers that rebuild from engine reports (sgl-router's model), so HA is "run two".

**Build versus adopt.** Fastest path: SGLang Model Gateway for SGLang fleets (runtime API, health checks, approximate cache-aware). For vLLM fleets: llm-d EPP + Envoy in file-discovery mode gives the best scoring and precise prefix routing, at the price of two processes and no mid-stream failover; Dynamo gives migration but requires etcd across hosts and Dynamo-wrapped workers pinned to older engine versions. None provides item 5 for plain backends, so that piece is grid's to write.

**Changed since the first sweep**: no `/live`; SGLang `/health` generates; EPP runs without Kubernetes; vLLM has drain (`--shutdown-timeout`), SSE keep-alive, `/kv_event_sources`, scale-out render/derender, weight-version spans and `/weight_checker`; restart daemons need no privileges.

## Still open (needs hands-on experiment)

- Measure vLLM engine-death timeline: time from a CUDA fault to `/health` 503, to process exit, and exactly what bytes a streaming client receives (error chunk vs truncation) for Xid, OOM and `kill -9` of a worker.
- `ipc_cache` restart time for a 32B/70B FP8 model inside an unprivileged RunPod pod, and whether CUDA IPC works there at all (some hosts restrict it); same for SGLang's weight-cache daemon.
- Prototype mid-stream failover on vLLM via render → `/inference/v1/generate` → derender: per-chunk overhead of derender, correctness for tool calls and reasoning across the seam, behaviour with speculative decoding.
- `continue_final_message` fidelity per model family (re-tokenisation drift, reasoning templates, tool-call cut points) on vLLM and SGLang.
- Whether `--sse-keep-alive-interval` actually keeps RunPod's Cloudflare proxy open past 100 s.
- KV-event transport on port-starved platforms: can ZMQ 5557/5558 be reached through RunPod TCP port mappings, and event loss at `hwm` under load.
- Cross-node vLLM `--nnodes` over plain TCP: throughput, and which ports must be open (XPUB uses a random port).
- SGLang Model Gateway HA and state loss on restart; llm-d EPP + Envoy behaviour when the endpoints file churns every few seconds.
- vLLM wheel variants actually published for v0.33.0 (release notes body not readable here) and whether compatibility mode works on consumer RTX cards.
- LiteLLM facts here are from docs, not source; verify defaults against the version deployed.
