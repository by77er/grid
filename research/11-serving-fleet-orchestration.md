# 11 — Serving fleet orchestration

Research date: 2026-11-09. Lane: fleet-level serving (routing, autoscaling, cold starts, health,
rollout, tenancy, capacity) across many heterogeneous nodes and providers. Engine internals are in
`10-inference-engines.md`; RL-loop inference is in `12-inference-for-rl.md`; provider facts are in
`07-provider-landscape.md`; image/weight staging overlaps `09-fast-bootstrap-preflight.md`.

## Summary

- The open-source serving control plane has converged on one shape: an L7 proxy (Envoy or a custom
  router) that asks a **model-aware endpoint picker** which replica to use, with the picker scoring
  replicas on queue depth, KV-cache utilisation and prefix-cache locality. The Kubernetes Gateway
  API Inference Extension (GAIE) made this a GA API (`InferencePool`,
  `inference.networking.k8s.io/v1`), and llm-d, KServe `LLMInferenceService`, and NVIDIA Dynamo
  all plug into it.
- **Prefix-cache-aware routing is the single biggest routing lever**, and the precision of the
  cache picture matters: in llm-d's published benchmark, precise (event-driven) prefix routing gave
  p90 TTFT 0.54 s against 31 s for approximate prefix routing and 93 s for random, at roughly 2x
  the output tokens/s of random, on identical hardware.
- **Autoscaling on GPU utilisation is wrong for LLM servers.** Every current stack scales on queue
  depth / in-flight requests / KV-cache saturation, usually through KEDA reading Prometheus. Only
  two systems found treat heterogeneous replica types as first-class in the scaler: llm-d's
  Workload Variant Autoscaler (cost per variant) and Dynamo's Planner (profiled SLA model).
- **Cold start is minutes, not seconds, unless engineered.** A clean vLLM cold start for an 8B
  model measured 317 s (141 s image pull + 176 s engine startup). Engine startup, not weight I/O,
  dominates once images and weights are cached, so the big wins are snapshot/restore (Modal:
  ~118 s to ~12 s) and warm standby pools, not just faster downloads.
- **Marketplace providers give no load-balancer primitive and a hostile ingress**: RunPod's HTTP
  proxy cuts connections at 100 s (Cloudflare 524), public IPs and port mappings change on restart,
  and RunPod's own serverless load balancer has no retries and a 5.5 minute per-request cap. `grid`
  must bring its own router and treat each replica as an unreliable, re-addressable endpoint.
- Serving and training want opposite things from the scheduler (independent, replaceable replicas
  versus gang-scheduled ranks) but share provisioning, image/weight staging, preflight, health
  telemetry and the capacity/stockout loop.

---

## 1. Control planes: what exists and how each is shaped

| System | Shape | Routing | Autoscaling | Runs without Kubernetes? | Notes |
|---|---|---|---|---|---|
| **GAIE** (kubernetes-sigs) | CRDs + Endpoint Picker (EPP) called by the gateway via Envoy ext-proc | Scorer plugins: queue, KV-cache utilisation, prefix cache, LoRA affinity | None (delegated) | No | `InferencePool` is `inference.networking.k8s.io/v1`, GA since v1.0.0 ([docs](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/)) |
| **llm-d** | GAIE EPP + vLLM + optional prefill/decode disaggregation | GAIE scorers plus a *precise* prefix scorer fed by vLLM KV events | Workload Variant Autoscaler (WVA) emitting a desired-replica metric consumed by KEDA/HPA | No | v0.5 (2026-02) added hierarchical KV offloading, cache-aware LoRA routing, active-active HA (per README, via search) |
| **KServe** `LLMInferenceService` | Operator that stamps out Deployment, Gateway, HTTPRoute, InferencePool, EPP | Via GAIE (v1.3.0 in KServe 0.17) | WVA + KEDA or HPA | No | v0.17, 2026-03-13 ([release](https://kserve.github.io/website/blog/kserve-0.17-release)) |
| **Ray Serve LLM** | Ray actors; proxy + per-deployment router in-process | Power-of-two-choices default; `PrefixCacheAffinityRouter` | Serve autoscaler on ongoing requests | **Yes** (any Ray cluster) | Router API alpha ([docs](https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html)) |
| **NVIDIA Dynamo** | Rust frontend + router + workers (vLLM/SGLang/TRT-LLM) + Planner | KV-aware router (load + KV overlap) | Planner: throughput / latency / SLA modes | Partly (K8s-first; local mode exists) | Has mid-stream request migration ([docs](https://docs.nvidia.com/dynamo/kubernetes/fault-tolerance/request-migration.md)) |
| **vLLM production-stack** | Helm chart: Python router + vLLM pods + Prometheus | `roundrobin`, `session`; prefix-aware and KV-aware in newer docs | KEDA on `vllm:num_requests_waiting` | Router can (static discovery) | ([router README](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/README.md)) |
| **AIBrix** | Envoy Gateway ext-proc plugin + controllers | least-request, least-kv-cache, prefix-cache, SLO variants | LLM-specific pod autoscaler + GPU optimizer | No | v0.7.0 2026-06-16 (via search); KV event sync since v0.4.0 ([docs](https://aibrix.readthedocs.io/latest/features/kv-event-sync.html)) |
| **SkyServe** | Controller VM + load balancer VM + replicas on any cloud | Simple LB policies | QPS per replica | **Yes** | Spot with on-demand fallback ([docs](https://docs.skypilot.ai/en/latest/serving/spot-policy.html)) |
| **dstack services** | Server + optional gateway VM (nginx) + replicas | Built-in LB, or delegate to SGLang Model Gateway / Dynamo router | RPS per replica | **Yes** | ([docs](https://dstack.ai/docs/concepts/services)) |
| **LiteLLM / OpenRouter** | Stateless API gateway over *other people's* endpoints | Weighted shuffle, least-busy, latency, usage | None | Yes | Auth, budgets, fallbacks; no cache awareness |

### 1.1 GAIE: the reference architecture

An `InferencePool` selects model-server pods by label, names target ports, and references an
endpoint picker service:

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
spec:
  selector: { matchLabels: { app: vllm-qwen3-32b } }
  targetPorts: [{ number: 8000 }]
  endpointPickerRef:
    name: vllm-qwen3-32b-epp
    port: { number: 9002 }
    failureMode: FailOpen
```

`FailOpen` means that if the EPP fails or does not answer, the gateway picks an endpoint itself.
One EPP serves exactly one pool; an `HTTPRoute` can reference several pools as `backendRefs`
(which is how canary weights between model versions are expressed). `endpointPickerRef` became
optional in v1.5.0
([InferencePool docs](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/)).

The EPP scrapes each pod's metrics (KV-cache utilisation, pending queue length, active LoRA
adapters) and runs a plugin pipeline of filters, scorers and a picker. Scorer names seen in
current material are `queue-scorer`, `kv-cache-utilization-scorer` and `prefix-cache-scorer`;
example weights differ between guides (queue 2 / KV 2 / prefix 3 in an Istio guide), so treat
weights as tuning parameters, not defaults (unverified against the upstream config reference). A
**saturation detector** (default `utilization-detector`) looks at queue depth and KV-cache
utilisation to decide when the pool is full
([troubleshooting guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/troubleshooting/)).
Often-quoted defaults of queue depth 5 and KV utilisation 0.8 could not be confirmed from the
official docs (unverified).

On saturation, **flow control** takes over: requests are queued and higher priority is always
served first ([API reference](https://gateway-api-inference-extension.sigs.k8s.io/reference/x-spec)).
A Datadog write-up reports that low-priority requests are shed with `429` or `503` while
normal-priority requests continue to queue
([Datadog](https://www.datadoghq.com/blog/llm-routing-kubernetes-inference-extension.md)). Priority
is declared with an `InferenceObjective`; selection by an `x-gateway-inference-objective` request
header is reported by a third-party page only (unverified).

### 1.2 llm-d and the precise prefix scorer

llm-d's contribution over stock GAIE is routing on the *actual* KV-cache contents. Each vLLM pod
publishes KVEvents on block creation and eviction; the `llm-d-kv-cache` library's `kvevents.Pool`
consumes them into a block index (block hash to pod and medium, GPU or CPU), and `kvcache.Index`
answers "what fraction of this request's prefix is resident on each pod"
([llm-d blog](https://llm-d.ai/blog/kvcache-wins-you-can-see)).

On the vLLM side the events are ZMQ: `enable_kv_cache_events` (default `False`), publisher `zmq`
or `null`, endpoint default `tcp://*:5557`, optional replay endpoint, `buffer_steps` default
10,000, high-water mark default 100,000 after which events drop
([vLLM kv_events config](https://docs.vllm.ai/en/stable/api/vllm/config/kv_events/)). Example
invocation from LMCache docs:
`--kv-events-config '{"enable_kv_cache_events": "True", "publisher": "zmq", "topic": "kv-events"}'`
([LMCache](https://docs.lmcache.ai/production/kv_cache_events.html)). Note the direction: the
*router* must be able to reach port 5557 on every replica (or replicas must push to the router),
which matters on platforms with only proxied HTTP ports (section 8).

Benchmark, llm-d blog appendix: 8 vLLM pods x 2 H100 (16 GPUs), Qwen3-32B, 150 "customers" each
with a 6,000-token shared prefix, 5 users each sending 1,200-token queries, Poisson ramp 3 to 60
QPS; ideal caching needs about 73% of cluster KV capacity.

| Scheduler | Output tok/s | TTFT p90 (s) | TTFT mean (s) | Mean vLLM wait queue |
|---|---|---|---|---|
| random | 4428.7 | 92.551 | 45.281 | 27.3 |
| load-aware | 4428.7 | 94.865 | 46.987 | 28.9 |
| approximate prefix | 6944.4 | 31.083 | 13.316 | 8.1 |
| precise prefix | 8730.0 | 0.542 | 0.298 | 0.1 |

Source: [llm-d blog](https://llm-d.ai/blog/kvcache-wins-you-can-see). Two readings: (a) load-aware
routing alone bought nothing over random on this workload, and (b) the gap between approximate and
precise is mostly a queueing cliff: the approximate scheduler leaves the fleet just over capacity,
so queues grow without bound. The 57x and 170x headlines are p90 TTFT ratios at overload, not
steady-state speedups.

### 1.3 Ray Serve LLM

Default router is power-of-two-choices: sample two replicas, send to the less busy.
`PrefixCacheAffinityRouter` (Ray 2.49+) keeps a **character-level prefix tree** per replica in a
detached actor, an approximation with no feedback from the engine. Logic: if the queue-length gap
between replicas is under `imbalanced_threshold` (default infinity), route to the replica with the
highest prefix match if the match rate is at least `match_rate_threshold` (default 0.1), otherwise
to the replica with the lowest prefix-cache utilisation; if load is imbalanced, fall back to
power-of-two. Eviction is off by default (`do_eviction=False`, `eviction_threshold_chars` 400,000,
`eviction_target_chars` 360,000, `eviction_interval_secs` 10)
([Ray docs](https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html)).

Anyscale's benchmark (DeepSeek-R1-Distill-Qwen-32B, eight 8xL4 nodes, TP=4, 512-token prefixes,
128-token suffixes and outputs, 32 concurrent per replica) reports about 60% lower TTFT and over
40% higher end-to-end throughput versus power-of-two; main results are on Anyscale's RayTurbo
runtime ([Anyscale blog](https://anyscale.com/blog/ray-serve-faster-first-token-custom-routing)).

Autoscaling: `target_ongoing_requests` (default 1 per Anyscale docs, a soft target, not a cap),
`upscale_delay_s` default 30, `downscale_delay_s` default 600, optional
`downscale_to_zero_delay_s`
([Ray advanced autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)).
For an LLM replica that batches continuously, the target must be set near the replica's useful
concurrency (tens), not 1.

Ray Serve is the one mature option that needs no Kubernetes, but it requires a Ray cluster where
all nodes can reach each other on many ports, which is a poor fit for cross-provider fleets behind
HTTP proxies (see `05-networking-interconnects.md`).

### 1.4 NVIDIA Dynamo

Components: OpenAI-compatible frontend, a KV-aware router that scores workers on load and KV-cache
overlap ([GitHub](https://github.com/ai-dynamo/dynamo)), workers wrapping vLLM / SGLang /
TensorRT-LLM, and a Planner. The Planner has three optimisation targets: `throughput` (default, no
profiling), `latency`, and `sla`, which uses regression models with TTFT and ITL targets (examples
`ttft_ms: 500.0`, `itl_ms: 50.0`); it sizes prefill and decode pools separately using profiled
curves (input length to TTFT, KV utilisation to ITL), and load-based scaling requires the KV
router ([Planner docs](https://docs.nvidia.com/dynamo/latest/components/planner)). Doc versions
disagree on details; check the version deployed.

Dynamo's most relevant feature for unreliable fleets is **request migration** (section 5.3).
dstack can already delegate routing to a Dynamo router or the SGLang Model Gateway
([dstack services](https://dstack.ai/docs/concepts/services)).

### 1.5 vLLM production-stack and AIBrix

production-stack's router README documents `--routing-logic roundrobin|session`, `--session-key
<header>`, `--service-discovery static|k8s`, `--static-backends`, `--static-models`,
`--static-aliases`, `--static-backend-health-checks` (dummy request roughly every minute; failing
backends removed), `--engine-stats-interval` (default 30 s) and `--request-stats-window` (default
60 s) ([README](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/README.md)).
Newer docs add prefix-aware and KV-aware modes
([prefix-aware tutorial](https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.8/_sources/use_cases/prefix-aware-routing.rst)).
The static-discovery mode is the piece that works without Kubernetes: a single router process and
a list of URLs.

AIBrix routes via an Envoy Gateway ext-proc plugin with a per-request strategy header; strategies
include `least-request`, `least-kv-cache`, `least-gpu-cache`, `prefix-cache` (falls back to
least-running-requests when load is imbalanced) and SLO variants
([router design](https://aibrix.readthedocs.io/latest/designs/aibrix-router.html)). The docs warn
that prefix-cache routing state is not synchronised between gateway replicas unless KV event sync
is enabled; that uses vLLM flags `--enable-kv-cache-events`, `--kv-events-publisher=zmq`,
`--kv-events-endpoint=tcp://*:5557`, `--kv-events-replay-endpoint=tcp://*:5558`
([KV event sync](https://aibrix.readthedocs.io/latest/features/kv-event-sync.html)). The paper
claims 50% throughput gain and 70% latency reduction from its distributed KV cache, by the authors'
own tests ([arXiv 2504.03648](https://arxiv.org/abs/2504.03648)).

### 1.6 SkyServe and dstack: the closest prior art to grid

**SkyServe** autoscaling: `target replicas = ceil(current_qps / target_qps_per_replica)` clamped to
`[min_replicas, max_replicas]`; `upscale_delay_seconds` default 300, `downscale_delay_seconds`
default 1200; `min_replicas: 0` scales to zero and the upscale delay is skipped when scaling from
zero ([autoscaling docs](https://docs.skypilot.ai/en/latest/serving/autoscaling.html)). Spot:
`use_spot: true` with `base_ondemand_fallback_replicas: N` (always-on on-demand floor) and
`dynamic_ondemand_fallback: true` (launch on-demand when spot is unavailable, then switch back);
it tries other regions and clouds on preemption, but "relies on the user side to retry" failed
requests ([spot policy](https://docs.skypilot.ai/en/latest/serving/spot-policy.html)).
Load-balancing policy options and per-accelerator QPS targets could not be confirmed (docs page
did not render; see Open questions).

**dstack services**: `replicas: 0..4` plus `scaling: {metric: rps, target: N}` (RPS is the only
metric; scaling requires a gateway); bearer-token auth on by default; `model:` registers an
OpenAI-compatible `/v1` endpoint; `probes` with `ready_after` consecutive successes gate traffic;
rolling updates replace one replica at a time after probes pass; rate limits (`rps`, `burst`, per
IP or per header key) return 429; `retry: on_events: [no-capacity, error, interruption]`
resubmits only the failed replica; replica groups allow different `resources` per group; the
router group must be `replicas: 1` ([dstack services](https://dstack.ai/docs/concepts/services)).

Both scale on request rate, which is a weak proxy for LLM load (a 50-token and a 50,000-token
request count the same), and neither does cache-aware routing natively.

### 1.7 Managed platforms, as far as published

- **RunPod Serverless**: two endpoint types. Queue-based (jobs, handler function) and
  load-balancing (direct HTTP to workers, URL `https://ENDPOINT_ID.api.runpod.ai/PATH`). Scaling
  is either queue delay (default threshold 4 s) or request count,
  `ceil((requestsInQueue + requestsInProgress) / scalerValue)`. Active workers default 0, max
  workers default 3, idle timeout default 5 s (console), execution timeout default 600 s. Up to
  three GPU types in priority order; with five or more workers they are spread across types
  ([endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)).
- **Modal**: per-function autoscaler with `min_containers`, `buffer_containers` (extra idle
  containers while active), `scaledown_window` (default 60 s, range 2 s to 20 min); container boot
  "about one second" excluding user initialisation
  ([cold start guide](https://modal.com/docs/guide/cold-start)).
- **Baseten**: "Multi-cloud Capacity Management" presents GPUs across 20+ clouds and regions as
  one pool ([blog](https://www.baseten.co/blog/how-we-built-multi-cloud-capacity-management/));
  sync requests get priority over background work when capacity is tight
  ([docs](https://docs.baseten.co/concepts/howbasetenworks.md)). Internals of the scheduler are
  not published in what was retrieved.
- **Together / Fireworks**: no primary architecture source found in this sweep (see Open
  questions).

### 1.8 Gateways over third-party endpoints

**LiteLLM** router strategies: `simple-shuffle` (default, weighted random), `least-busy`,
latency-based, usage-based (TPM/RPM); reliability keys `num_retries`, `fallbacks`,
`context_window_fallbacks`, `allowed_fails`, `cooldown_time`, `enable_pre_call_checks`. Fallbacks
cover all errors including 429; streams that fail before the first chunk fall back, and the docs
state mid-stream failures also get a fallback without explaining how partial output is handled
([reliability docs](https://docs.litellm.ai/docs/proxy/reliability),
[routing docs](https://docs.litellm.ai/docs/routing.md)).

**OpenRouter** default: deprioritise providers with significant outages in the last 30 s, pick
among the cheapest stable ones weighted by inverse square of price, keep the rest as fallbacks;
`sort: price|throughput|latency` or an explicit `order` disables load balancing
([provider routing](https://openrouter.ai/docs/features/provider-routing)).

These solve auth, keys, budgets and API normalisation, and are a reasonable front door for `grid`.
They do not see queue depth or KV state, so they cannot replace the replica-level router.

---

## 2. Request routing

### 2.1 Signals and policies

| Policy | Signal | Good for | Failure mode |
|---|---|---|---|
| Round-robin / random | None | Homogeneous, uncached workloads | Ignores load skew from variable request length |
| Least in-flight / power-of-two | Router-side in-flight count | Default baseline | Blind to queueing inside the engine; no cache locality |
| Queue-depth aware | `vllm:num_requests_waiting` etc. | Avoiding saturated replicas | Scrape staleness (production-stack polls every 30 s by default) |
| KV-utilisation aware | KV-cache usage gauge | Long-context workloads | Does not know *whose* prefix is cached |
| Session affinity | Header or user id hash | Multi-turn chat, agents | Hot sessions overload one replica; breaks on scale events |
| Approximate prefix | Router-side trie or hash of text/tokens | Shared system prompts, RAG, few-shot | Diverges from real cache under eviction pressure (section 1.2 table) |
| Precise prefix | Engine KV events | Fleets near KV capacity | Needs event channel from every replica to router |

All real implementations combine prefix affinity with a load guard: Ray's `imbalanced_threshold`,
AIBrix's imbalance check, GAIE's weighted sum of scorers plus saturation detector. Affinity
without a guard produces hot spots; a guard that is too tight destroys the cache hit rate.

### 2.2 Heterogeneous replicas

None of the surveyed routers has an explicit "this replica is 2.3x faster" weight. Heterogeneity
is handled implicitly: a faster replica drains its queue sooner and so wins least-queue
comparisons. That works when the router sees real engine queue depth; it works badly with
router-side in-flight counts when replicas have different optimal concurrency (an H200 replica is
happy at 64 in-flight where an L4 replica is saturated at 8). For `grid`, normalised load should be
`in_flight / capacity_i`, with `capacity_i` measured per replica at preflight or learned online
(tokens/s at target latency). Where heterogeneity does appear explicitly is in the scalers: WVA's
per-variant cost and Dynamo's profiled curves (section 3).

### 2.3 Hedging, retries and mid-stream failover

- **Retries before first token** are safe and universal: nothing has been delivered. LiteLLM,
  SkyServe (client-side) and RunPod (client-side, "at least 3 retries with 5 to 10 s delays",
  [RunPod LB docs](https://docs.runpod.io/serverless/load-balancing/overview)) all rely on this.
- **Hedging** (send a duplicate after a delay, take the first to produce a token) was not found as
  a built-in feature of any surveyed LLM router (unverified by absence). It is expensive for LLMs
  because the duplicate consumes a batch slot and KV memory for the full prefill; it is defensible
  only before first token, with cancellation of the loser, and only when the fleet has headroom.
- **Mid-stream failover** exists in Dynamo. With `--migration-limit N` (env `DYN_MIGRATION_LIMIT`,
  default 0 = off, docs suggest 3) the frontend records every generated token; if the worker dies
  or its drain grace period expires, the frontend opens a new stream on another worker with prompt
  plus generated tokens as the new prompt, and the client sees no interruption.
  `--migration-max-seq-len` caps tracked length; `DYN_RUNTIME_INHIBITED_DURATION_SECS` (default 5)
  locally blacklists the failed worker. Not migrated: `n > 1` and guided decoding (the grammar FSM
  would restart from the schema root)
  ([Dynamo request migration](https://docs.nvidia.com/dynamo/kubernetes/fault-tolerance/request-migration.md)).
  The cost is a full re-prefill on the new worker, and with sampling the continuation is a valid
  sample but not the one the dead worker would have produced.

This "replay prompt + emitted tokens as a new prefix" trick is engine-agnostic: any router that
tokenises or tracks the emitted text can do it against plain vLLM by issuing a completion with the
assistant turn pre-filled. That makes it practical for `grid` on spot capacity.

---

## 3. Autoscaling

### 3.1 Signals

| Signal | Leading or lagging | Comment |
|---|---|---|
| Engine queue length (`vllm:num_requests_waiting`) | Leading | Best single signal; non-zero for sustained periods means under-provisioned |
| In-flight per replica | Leading | What Ray Serve and RunPod "request count" use |
| KV-cache utilisation | Leading for long-context | Saturation at ~0.8+ predicts preemption/queueing inside the engine |
| Requests per second | Weak | SkyServe and dstack use it; ignores token counts |
| Tokens/s, TTFT, ITL | Lagging (SLO) | Use as the target of a model (Dynamo SLA mode), not as a raw trigger |
| GPU utilisation | Useless | A continuously batching engine sits near 100% at any load above trivial; llm-d docs say to scale on queue depth and true demand, not raw GPU utilisation ([vLLM llm-d integration](https://docs.vllm.ai/en/stable/deployment/integrations/llm-d/)) |

### 3.2 Concrete configurations

**production-stack + KEDA**: Prometheus trigger on `vllm:num_requests_waiting` with threshold 5;
`pollingInterval` 15 s, `cooldownPeriod` 360 s; scale-to-zero example adds a keepalive trigger
`sum(rate(vllm:num_incoming_requests_total[1m]) > bool 0)` so the last replica is not removed while
traffic exists; HPA scale-down behaviour 300 s stabilisation, at most 50% per 60 s; KEDA `fallback`
of 2 replicas after 3 failed metric reads
([KEDA guide](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html)).

**llm-d WVA**: saturation analyser (default) over KV-cache utilisation and queue depth, including
requests queued in the EPP before reaching any pod; experimental token-based and SLO analysers;
30 s loop; scale-up goes to the **cheapest** variant, scale-down removes the **most expensive**;
publishes `wva_desired_replicas` for KEDA/HPA; the `VariantAutoscaling` CRD is deprecated as of
llm-d 0.8.0 ([WVA docs 0.7](https://llm-d.ai/docs/0.7/architecture/advanced/autoscaling/wva)).
The accompanying paper reports 37% better effective throughput and 10x fewer request failures than
HPA ([arXiv 2603.09730](https://arxiv.org/abs/2603.09730)). In KServe 0.17 `variantCost` defaults
to 10.0 ([KServe 0.17](https://kserve.github.io/website/blog/kserve-0.17-release)). On OpenShift
AI, scale-to-zero is `spec.scaling.keda.idleReplicaCount: 0` with `minReplicas >= 1`
([Red Hat docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploy_models_using_distributed_inference_with_llm-d/autoscaling-llmd-model-deployments)).

**Asymmetry rule**: every system scales up fast and down slow (Ray 30 s / 600 s, SkyServe 300 s /
1200 s, KEDA cooldown 360 s). With multi-minute cold starts and per-second or per-minute billing
on marketplaces, the downscale delay should be at least the cold start time, and scale-up should
be triggered on the *derivative* of queue length rather than a threshold where possible.

### 3.3 Scale-to-zero

Requires something that stays up to hold the request: KEDA plus a queueing gateway (GAIE flow
control), SkyServe's load balancer, RunPod's and Modal's platform front ends. On RunPod's LB
endpoints a request waits at most 2 minutes for a worker before timing out
([RunPod LB docs](https://docs.runpod.io/serverless/load-balancing/overview)), which is shorter
than most uncached LLM cold starts, so scale-to-zero there only works with cached models and
FlashBoot or client retries.

---

## 4. Cold start anatomy and mitigation

### 4.1 Where the time goes

| Phase | Measured | Source |
|---|---|---|
| Provision / schedule | Seconds (warm pool) to minutes (new VM); provider-specific | see `07-provider-landscape.md` |
| Image pull, vLLM image (~14 GB), same-region registry, clean node | 141 s (gzip) / 130 s (uncompressed) | [Microsoft](https://techcommunity.microsoft.com/blog/linuxandopensourceblog/dissecting-llm-container-cold-start-where-the-time-actually-goes/4508831) |
| vLLM engine startup incl. weight load, Llama 3.1 8B on A100 80GB | 176 s | same |
| Total clean cold start to first token, vLLM 8B | 317 s (44% pull, 56% startup) | same |
| llama.cpp equivalent (weights in image) | 139 s gzip, 102 s uncompressed; 43 s of it is load to VRAM | same |
| Weight load only, Llama-3.1-8B, default loader vs Run:ai Model Streamer | 15.5 s vs 3.6 s | [Azure SDK blog](https://devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer/) (via search) |
| vLLM + Modal, Ministral 3 3B: full init vs GPU snapshot restore | ~118 s vs ~12 s (median) | [Modal](https://modal.com/blog/mistral-3) |

Key finding from two independent sources: **for vLLM, once the image is local, the bottleneck is
CPU-bound engine initialisation, not weight I/O.** The Microsoft study found removing gzip saved
6 s of 317 s for vLLM (1.02x) versus 1.35x for llama.cpp; lazy-pull via Nydus/FUSE pushed vLLM
startup beyond 900 s. The MLSys 2026 paper "Breaking the Ice" decomposes vLLM startup into six
steps and concludes that all but the final two (profiling, CUDA graph capture) are CPU-bound, that
faster loaders (Run:ai streamer, Tensorizer) cut checkpoint load time but barely move end-to-end
startup, and that disabling the `torch.compile` cache raised that step from 3-6 s to 11-21 s
([arXiv 2606.07362](https://arxiv.org/abs/2606.07362); step numbers via secondary summary).

Implication: on marketplace nodes with weak or oversubscribed CPUs, cold starts will be slower
than GPU class predicts; CPU model and core count belong in the placement score.

### 4.2 Mitigations

| Mitigation | Removes | Evidence / mechanism |
|---|---|---|
| Image pre-cached on host or small image + mounted engine | Pull (130-240 s) | Baseten streams Truss images so the container starts before the pull completes ([Baseten](https://docs.baseten.co/deployment/autoscaling/cold-starts.md)) |
| Host-level weight cache | Weight download | RunPod cached models: scheduler prefers hosts already holding the model, mounts at `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{hash}/`, no billing during download, one model per endpoint, all quantisations in a repo are fetched ([RunPod](https://docs.runpod.io/serverless/endpoints/model-caching)) |
| Network volume weight cache | Download from origin | Works, but RunPod notes volumes add latency and pin the endpoint to the volume's data centre ([RunPod](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)) |
| Tiered / peer weight distribution | Download from origin | Baseten Delivery Network: multi-tier weight cache, mounted not copied ([Baseten](https://docs.baseten.co/deployment/autoscaling/cold-starts.md)); peer-to-peer options are covered in `09-fast-bootstrap-preflight.md` |
| Fast loaders (Run:ai streamer, Tensorizer) | Part of load-to-GPU | ~4x on the load step; small end-to-end effect for vLLM (above) |
| Compile-cache reuse | torch.compile, CUDA graph steps | Baseten b10cache shares `torch.compile` artifacts across replicas (same doc) |
| **Process snapshot / restore** | Nearly all engine init | Modal: `enable_memory_snapshot=True` plus `experimental_options={"enable_gpu_snapshot": True}`; vLLM is started, warmed, put into sleep mode (GPU state moved to CPU memory, needs `VLLM_SERVER_DEV_MODE=1`), snapshotted in `@modal.enter(snap=True)`; a `snap=False` hook wakes it after restore. Alpha. ([Modal example](https://modal.com/docs/examples/vllm_snapshot), [blog](https://modal.com/blog/mistral-3)) |
| Paused-container resume | Everything, when it hits | RunPod FlashBoot retains worker state after spin-down; RunPod claims resume in about 200 ms; on by default in console, off in REST v2 ([blog](https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start), [settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)). Hit rate depends on traffic and is not guaranteed. |
| Warm standby pool | Everything, at a price | Modal `buffer_containers`, RunPod active workers, Baseten `min_replica`, Ray/SkyServe `min_replicas` |

Baseten's startup dashboard is a good model for what to record: four phases (acquiring resources,
pulling image, pulling weights, loading model), p50/p90/p95/p99 per phase, and a "Stalled" outcome
if not ready after 30 minutes
([Baseten](https://docs.baseten.co/deployment/autoscaling/cold-starts.md)).

Process snapshots are not portable across hosts with different GPUs, drivers or CPU features
(unverified in general; Modal controls its whole fleet). On a heterogeneous marketplace fleet the
realistic levers are cache placement, compile-cache reuse keyed by (engine version, model, GPU
type), and standby pools.

---

## 5. Health checking and failure recovery

### 5.1 Liveness versus readiness

A model server has at least four states that a single `/health` 200 conflates: process up,
weights loading, ready, and draining. Platforms that get this right:

- RunPod LB endpoints poll `/ping` on `PORT_HEALTH` (override with `HEALTH_CHECK_PATH`): `200` =
  healthy, `204` = initialising, anything else = unhealthy and removed from the pool. Cold start
  is measured from first 204 to first 200. Misconfigured ports leave workers up for 8 minutes
  returning 502 ([RunPod LB docs](https://docs.runpod.io/serverless/load-balancing/overview)).
- Dynamo's frontend returns `503` from `/health` while draining, so it leaves the ready set, but
  keeps `/live` at `200` so the liveness probe does not kill it mid-drain; workers wait for
  in-flight requests with a default 900 s bound; a vLLM engine monitor polls `check_health()`
  every 2 s and on `EngineDeadError` force-exits with `os._exit(1)` so the supervisor restarts it
  ([Dynamo graceful shutdown](https://docs.nvidia.com/dynamo/latest/user-guides/fault-tolerance/graceful-shutdown)).
- dstack requires `ready_after` consecutive probe successes before routing traffic
  ([dstack](https://dstack.ai/docs/concepts/services)).

A passing HTTP health check does not prove the engine generates correctly or at expected speed.
production-stack's static health check sends a real dummy request about once a minute
([README](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/README.md)).
For rented hardware, readiness should include a short generation with a tokens/s floor, which also
catches throttled or mis-provisioned GPUs (see `04-failure-taxonomy-recovery.md`,
`06-gpu-hardware-nuances.md`).

Kubernetes sends SIGTERM then SIGKILL after `terminationGracePeriodSeconds` (default 30 s), far
shorter than a long generation; Dynamo's docs recommend raising it (same source).

### 5.2 Spot preemption and stockouts

- RunPod interruptible pods reportedly get a **5 s** SIGTERM-to-SIGKILL window (mirrored docs and
  forum posts; a docs PR removing spot references exists, so current status is unverified). Five
  seconds is enough to flip readiness and notify the router, not to finish generations. Recovery
  therefore has to be router-side: retry or migrate.
- SkyServe's pattern is the right default: a fixed on-demand floor plus dynamic on-demand fallback
  while spot is unavailable, and automatic retry in other regions/clouds
  ([spot policy](https://docs.skypilot.ai/en/latest/serving/spot-policy.html)).
- RunPod Serverless accepts up to three GPU types in priority order precisely because single-type
  stockouts are routine ([settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)).
- Baseten's answer to stockouts is a pool spanning 20+ clouds/regions
  ([blog](https://www.baseten.co/blog/how-we-built-multi-cloud-capacity-management/)).

### 5.3 Recovery ladder

1. Connection error or 5xx before first token: retry on another replica; mark the replica suspect
   (Dynamo inhibits for 5 s; LiteLLM cools down after `allowed_fails` within `cooldown_time`).
2. Stream breaks mid-generation: replay prompt + emitted tokens on another replica (Dynamo
   migration) or surface an error for `n > 1` and guided decoding.
3. Replica fails N probes: remove from routing, terminate, replace; replacement goes through the
   full cold start, so the standby pool absorbs the gap.
4. Capacity cannot be replaced in the same SKU/region: fall back across GPU type, then region,
   then provider; the router must already cope with replicas of different speed.
5. Control-plane failure: GAIE `FailOpen` degrades to plain load balancing rather than an outage;
   llm-d v0.5 lists active-active HA for the scheduler.

---

## 6. Rolling out new model versions

- **Weighted pools**: in GAIE an `HTTPRoute` references multiple `InferencePool`s as weighted
  `backendRefs`, so blue/green and canary are gateway-level traffic splits between pools, each with
  its own EPP ([InferencePool docs](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/)).
- **Rolling replace**: dstack starts a new replica, waits for probes, then terminates an old one,
  one at a time ([dstack](https://dstack.ai/docs/concepts/services)). This needs N+1 capacity,
  which on a stocked-out GPU type may not exist; the fallback is surge on a different SKU.
- **Cache consequences**: a new model version has an empty KV cache and (for a changed engine or
  GPU type) no compile cache. Shifting traffic gradually keeps prefix-affinity sessions on the old
  pool until they end; an instant cut causes a TTFT spike equal to full prefill for every session.
- **RL case** (detail in `12-inference-for-rl.md`): weights change every few minutes, which is a
  rollout problem at much higher frequency; pool-level blue/green is too slow, and in-place weight
  reload or LoRA hot-swap with a version tag on responses is needed instead.

---

## 7. Multi-tenancy, rate limiting, auth, API surface

- **API surface**: OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models` is the
  universal contract; every stack here exposes it. Ray Serve LLM (`build_openai_app`), dstack
  (`model:`), production-stack, Dynamo, SkyServe-fronted vLLM all pass it through.
- **Auth**: dstack defaults to bearer tokens at the gateway; RunPod's pod proxy URLs are publicly
  reachable with no auth, so the application must provide its own
  ([RunPod expose ports](https://docs.runpod.io/pods/configuration/expose-ports)). Replica-level
  API keys (vLLM `--api-key`) plus a gateway that holds tenant keys is the minimum.
- **Rate limiting**: request-count limits (dstack `rps`/`burst`, 429) are easy; token-based limits
  (TPM) need the gateway to read usage from responses. LiteLLM does TPM/RPM budgets per key;
  AIBrix does per-user limits in its gateway plugin (details unverified: docs page 404 during this
  sweep).
- **Fairness under saturation**: GAIE flow control queues by priority and sheds the lowest class.
  This is the mechanism that lets one fleet serve both interactive API traffic and bulk RL rollouts:
  RL generation can be marked sheddable/low priority and absorb spare capacity.
- **Multi-model / LoRA**: GAIE and llm-d route on active LoRA adapters; one base-model pool can
  serve many tenants' adapters.

---

## 8. Provider realities for a fleet without a load balancer

RunPod pods ([expose ports docs](https://docs.runpod.io/pods/configuration/expose-ports)):

- HTTP is exposed only via `https://[POD_ID]-[INTERNAL_PORT].proxy.runpod.net`, at most 10 HTTP
  ports, Cloudflare in front, **100 s connection limit, 524 on overrun**. Streaming keeps bytes
  flowing, but a long non-streaming generation or a long prefill before the first byte will be cut.
- Direct TCP gives a public IP and a *random* external port. External port mappings change on
  every pod reset; Community Cloud public IPs can change on migration or restart. Requesting a
  port number above 70000 asks for symmetric mapping, exposed as `RUNPOD_TCP_PORT_<n>`. No UDP.
  The same port cannot be both HTTP and TCP.
- No load-balancer or stable-VIP primitive for pods.

RunPod Serverless LB endpoints do provide a front door, but with hard limits: 2 minute wait for a
worker, **5.5 minute processing timeout per request**, 30 MB payloads, no built-in retry, and
"overloaded endpoints drop requests"
([RunPod LB docs](https://docs.runpod.io/serverless/load-balancing/overview)).

Consequences:

1. `grid` needs its own router/gateway on a host with a stable address (a small on-demand VM on a
   hyperscaler, or the user's machine for development), as SkyServe and dstack both concluded.
2. Replica registration must be **push-based with re-registration**: the replica (or grid's agent
   beside it) reports its current `ip:port` and model identity; the router never assumes an
   address is stable.
3. Prefer direct TCP ports over the HTTP proxy for data-plane traffic to avoid the 100 s cut and
   the extra hop; use the proxy only as a fallback. TLS and auth are then grid's job.
4. Precise prefix routing needs a second channel (ZMQ 5557) per replica; on port-starved platforms
   an agent should relay KV events over the same outbound connection it uses for heartbeats, or
   grid should use approximate (router-side) prefix tracking.
5. Cross-provider replicas have different network latency to the router; TTFT-sensitive routing
   should include measured RTT, and the router should sit near the majority of GPU capacity.

---

## 9. Cost and capacity planning

- **Throughput per GPU must be measured, not assumed.** It depends on model, quantisation, prompt
  and output length distribution, cache hit rate and the latency SLO. As one anchor, the llm-d
  benchmark gives 8,730 output tok/s on 16 H100 for a 32B model with cache-friendly routing, about
  546 output tok/s per GPU (derived from the table in section 1.2), and half of that with naive
  routing. Routing policy is therefore a 2x cost lever on the same hardware.
- **SLO-driven sizing** (Dynamo SLA mode) is the principled approach: profile each replica type to
  get TTFT as a function of input length and ITL as a function of concurrency/KV load, then
  compute the max request rate per replica that satisfies the TTFT and ITL targets, then
  `replicas = ceil(predicted_load / per_replica_capacity)`
  ([Planner docs](https://docs.nvidia.com/dynamo/latest/components/planner)). On a heterogeneous
  fleet this needs one profile per (model, GPU type, engine config).
- **Cost-aware mix** (WVA): scale up on the cheapest variant that meets the SLO, scale down the
  most expensive first. For marketplaces the "cost" should be $ per million tokens at SLO, i.e.
  hourly price divided by profiled capacity, not hourly price.
- **Headroom for cold start**: required standby capacity is roughly peak load growth rate times
  cold start time. With a 5 minute cold start and load that can double in 5 minutes, standby must
  equal current capacity; cutting cold start to 30 s cuts standby cost proportionally.
- **Spot economics**: a fixed on-demand floor sized for the SLO-critical traffic plus spot for the
  rest (SkyServe pattern), with sheddable traffic classes (RL rollouts, batch) soaking up spot.

---

## 10. Serving versus training: what the orchestrator must do differently

| Dimension | Training (files 01-09) | Serving |
|---|---|---|
| Unit of scheduling | Gang: all ranks or nothing | Independent replica (itself possibly a small gang for TP/PP across GPUs or nodes) |
| Membership | Fixed world size, rendezvous once | Elastic, changes continuously |
| Node failure | Stops or degrades the whole job; restart from checkpoint | Loses one replica's in-flight requests; router retries |
| Heterogeneity | Slowest rank sets the pace | Each replica contributes its own throughput; router and scaler weight it |
| Network needs | East-west, high bandwidth, low latency between ranks | North-south, reachable ingress; east-west only for multi-node replicas, KV transfer, P/D disaggregation |
| Provisioning goal | All N nodes ready simultaneously | First replica ready as fast as possible, then trickle |
| State | Checkpoints | Stateless apart from KV cache (soft state) and weights version |
| Lifetime | Hours to weeks, bounded | Indefinite, with rollouts |
| Spot | Painful without elastic algorithms | Natural fit with router-side retry |
| Success metric | Time to first step, steps/s | TTFT, tokens/s at SLO, $ per token, availability |

**Shareable machinery**: provider adapters and stockout-driven fallback across SKU/region/provider;
image and weight pre-staging and caches; preflight and burn-in (GPU health, measured tokens/s or
it/s); the node agent with heartbeats and push registration; telemetry; secrets; the inventory
model that describes each node's GPU, CPU, memory, network and measured performance. A multi-node
serving replica (TP across nodes, LeaderWorkerSet in KServe 0.17) is exactly a small gang and can
reuse the training gang launcher.

**Not shareable**: the router/gateway, the autoscaler, the readiness/drain protocol and the rollout
controller are serving-only.

---

## Implications for grid

1. **Own the data plane.** Ship a router that runs on a stable host and treats replicas as
   push-registered, re-addressable endpoints. Do not depend on a provider load balancer, and do not
   route production traffic through RunPod's 100 s HTTP proxy. Reusing an existing router is
   viable: vLLM production-stack router with static discovery, SGLang Model Gateway, or Dynamo's
   frontend all run as a single process, and dstack already delegates to the latter two.
2. **Do not require Kubernetes.** GAIE/llm-d/KServe/AIBrix are the richest stacks but assume a
   cluster with pod networking, which marketplace containers cannot provide across providers. Copy
   their design (scorers, saturation detector, flow control, desired-replica metric), not their
   deployment model. Offer a GAIE-compatible mode only when the user already has a cluster.
3. **Routing default**: normalised least-load (in-flight divided by measured replica capacity)
   plus approximate prefix affinity with an imbalance guard; upgrade to precise prefix routing via
   vLLM KV events relayed through grid's node agent when the fleet runs near KV capacity.
4. **Capacity is a measured property of each replica.** Preflight should produce tokens/s at a
   reference workload, max useful concurrency and cold-start phase timings per (model, GPU type,
   engine build), feeding router weights, the autoscaler and cost ranking ($ per million tokens).
5. **Autoscale on engine queue depth and KV saturation**, with fast up / slow down, downscale
   delay no shorter than measured cold start, and a metrics-loss fallback replica count. Never on
   GPU utilisation.
6. **Engineer cold start explicitly** and record it per phase (provision, image, weights, engine
   init, warmup). Priority order for marketplaces: host/volume weight caches and image reuse,
   compile-cache reuse keyed by GPU type, a standby pool sized from load slope times cold start,
   and a preference for hosts with strong CPUs. Treat snapshot/restore as provider-specific
   (FlashBoot, Modal) rather than something grid can provide generically.
7. **Three-state health protocol** (initialising / ready / draining) separate from liveness,
   readiness gated on a real generation with a tokens/s floor, and drain-on-SIGTERM that flips
   readiness immediately because the spot window may be 5 s.
8. **Streaming-safe failover in the router**: retry before first token always; after first token,
   replay prompt plus emitted tokens on another replica, excluding `n > 1` and constrained
   decoding. This is what makes spot and flaky marketplace hosts usable for interactive serving.
9. **Capacity policy**: on-demand floor plus spot burst, ordered fallback across GPU type, region
   and provider, with priority classes so RL rollouts and batch are sheddable and API traffic is
   not (ties into `12-inference-for-rl.md`).
10. **Front door**: OpenAI-compatible API, per-tenant keys, RPM and TPM limits, 429 with
    `Retry-After`. Embedding LiteLLM as the tenant-facing gateway in front of grid's replica
    router is a reasonable split of responsibilities.
11. **Share with training**: provider adapters, inventory, staging caches, preflight, node agent.
    Keep the scheduler abstraction as "a group of nodes with a readiness contract": a training job
    is one group with all-or-nothing readiness; a serving fleet is many groups (size 1 or more)
    with independent readiness.

## Open questions

- **GAIE/llm-d exact config**: current `EndpointPickerConfig` apiVersion, plugin names, default
  scorer weights, saturation thresholds, and the header that selects an `InferenceObjective` were
  not confirmed from primary docs (several llm-d doc URLs returned 404). Needs a pass over the
  GitHub repos directly.
- **SkyServe**: `load_balancing_policy` options, readiness probe defaults, and whether
  `target_qps_per_replica` can vary by accelerator type (heterogeneous autoscaling). The YAML spec
  page did not render.
- **RunPod spot semantics today**: is the 5 s SIGTERM window still accurate, and do interruptible
  pods still exist as a product? Does Serverless ever preempt workers? Cross-check with
  `07-provider-landscape.md`.
- **FlashBoot internals and hit rate**: what state is retained, for how long, and how the hit rate
  varies with traffic; no primary measurement found.
- **Together, Fireworks**: no primary architecture material found; Baseten MCM internals only at
  summary level. A targeted sweep of their engineering blogs and docs is warranted.
- **Hedging**: no LLM-specific hedging implementation or measurement found; worth a search of
  papers (tail-latency work for LLM serving) before deciding whether grid should support it.
- **Mid-stream replay against plain vLLM/SGLang**: correctness details (chat-template handling of
  a pre-filled assistant turn, tool-call and reasoning-token streams, logprobs continuity) need a
  prototype.
- **Throughput-per-GPU tables** by model and GPU class at fixed SLOs: this file has only one
  derived anchor. Should be built from `10-inference-engines.md` sources or measured.
- **Snapshot/restore outside Modal**: CRIU with GPU support (cuda-checkpoint) on rented hosts,
  portability across GPU models and driver versions; untested here.
- **Router high availability and multi-region**: how to run more than one grid router with
  consistent prefix-affinity state (AIBrix notes its state is not synchronised without KV event
  sync), and whether geo-distributed routers are worth it.
- **Per-step vLLM startup numbers** from arXiv 2606.07362 were taken from secondary summaries; read
  the PDF for the full table and for CPU-sensitivity figures.
- **KEDA HTTP add-on / request buffering at zero** for non-Kubernetes deployments: what grid's
  router must do to hold requests while the first replica boots, and sensible client timeouts.
