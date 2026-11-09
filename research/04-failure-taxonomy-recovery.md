# 04 - Failure taxonomy and recovery for large GPU training

Research sweep date: 2026-11-09. Sibling files referenced by name: `03-rank-configuration.md` (rank/world-size mechanics), `05-networking-interconnects.md` (NCCL transport tuning), `06-gpu-hardware-nuances.md`, `07-provider-landscape.md` (spot semantics per provider), `08-heterogeneous-distributed-training.md` (algorithms that tolerate loss of workers), `09-fast-bootstrap-preflight.md` (burn-in and preflight checks).

## Summary

- **Failure is the steady state, and it scales linearly with node count.** Meta measured 6.50 (RSC-1) and 2.34 (RSC-2) hardware failures per 1,000 node-days on A100 clusters; a 1,024-GPU job has a measured MTTF of 7.9 h, and the projection for 16,384 GPUs is 1.8 h ([Kokolis et al., HPCA 2025](https://arxiv.org/html/2410.21680v2)). Llama 3 405B on 16k H100s saw 419 unexpected interruptions in 54 days (one per ~3 h), 58.7% GPU-related ([Llama 3 paper, Table 5](https://arxiv.org/html/2407.21783v3)). Gemini 2.5 reports hardware interruptions "multiple times per hour" ([Gemini 2.5 report](https://arxiv.org/html/2507.06261)).
- **Synchronous training has no partial-failure mode by default.** torchrun, Ray Train, JobSet and NVIDIA's `ft_launcher` all implement the same recovery: kill every worker, re-rendezvous, reload the last checkpoint. So the cost of one failure = (work since last checkpoint) + (detect time) + (restart time), and the three levers are checkpoint cost, detection latency and restart latency.
- **A small set of bad nodes causes most pain.** Meta: 1.2-1.7% of nodes ("lemons") touched 13% of daily jobs; removing them cut large-job failures from 14% to 4%. Delta (NCSA): one GPU produced 99% of 38,905 uncontained-memory-error events; excluding top offenders tripled MTBE (67 to 190 node-hours) ([Cui et al.](https://arxiv.org/html/2503.11901v2)). Per-node strike counting and quarantine is the highest-leverage reliability feature.
- **Hangs, not crashes, are the expensive class.** A dead peer usually shows up as every other rank blocked in a collective. Without a step-time watchdog the job burns money indefinitely. Detection needs an explicit progress heartbeat, not just process liveness.
- **Silent data corruption (SDC) is real at LLM scale and loss curves do not reliably reveal it.** Meta cites roughly one SDC-affected device per thousand; Gemini 2.5 replayed ~0.25% of steps on suspicion and 6% of those replays were genuine corruption.
- **Marketplace hardware adds provider-level failure classes** (spot kill with seconds of notice, stockout on re-provision, DC-pinned volumes, hosts going offline) on top of everything above, and public quantitative data on them is nearly nonexistent.

---

## 1. Published evidence: how often things break

| Source | System | Key numbers |
|---|---|---|
| [Meta, "Revisiting Reliability in Large-Scale ML Research Clusters" (HPCA 2025)](https://arxiv.org/html/2410.21680v2) | RSC-1 (16k A100), RSC-2 (8k A100), 11 months, 4M jobs, 150M GPU-hours | 6.50 / 2.34 failures per 1,000 node-days (30-day rolling rate on RSC-1 swung between ~2.5 and ~17.5). MTTF: 8-GPU job 47.7 days; 1,024-GPU job 7.9 h; projected 16,384 GPUs 1.8 h, 131,072 GPUs 0.23 h. Hardware failures hit 0.2% of jobs but 18.7% of runtime. Top attributed causes: InfiniBand links, filesystem mounts, GPU memory errors, PCIe errors. 43% (RSC-1) / 63% (RSC-2) of PCIe errors co-occur with Xid 79. |
| [Llama 3 paper, sec. 3.3.4](https://arxiv.org/html/2407.21783v3) | 16k H100, 54-day snapshot | 466 interruptions (47 planned, 419 unexpected); ~78% of unexpected attributed to confirmed/suspected hardware. Faulty GPU 30.1%, HBM3 17.2%, software bug 12.9%, switch/cable 8.4%, unplanned host maintenance 7.6%, GPU SRAM 4.5%, GSP 4.1%, NIC 1.7%, NCCL watchdog timeouts 1.7%, SDC 1.4%, thermal interface/sensor 1.4%; CPU and system DIMM 0.5% each. >90% effective training time; manual intervention only 3 times. |
| [Cui et al., "Characterizing GPU Resilience..." (Delta, NCSA)](https://arxiv.org/html/2503.11901v2) | 1,168 GPUs (A40/A100/H100), 855 days | 63,253 GPU errors; node MTBE 67 h; per-node availability 99.5%. Job failure probability given error: Xid 119 100%, Xid 95 97%, Xid 48 90%, Xid 74 66%, Xid 31 59%. Later versions of the paper report H100 memory MTBE 3.2x worse than A100 and ~5% overprovisioning needed at scale (from search snippet of v3/v4, not read in full). |
| [Gemini 2.5 technical report](https://arxiv.org/html/2507.06261) | Multiple 8,960-chip TPUv5p pods | Hardware interruptions multiple times per hour; elastic reconfiguration loses "tens of seconds" vs "10 or more minute delay" before; training continues at ~97% throughput while a failed slice recovers. |
| [Meta engineering blog, Jul 2025](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/) | Fleet | SDC ~1 fault per 1,000 devices vs ~1 per million for soft-error bitflips. |
| [ORNL Summit study (ICS '24)](https://www.osti.gov/biblio/2378092) | 27k V100, 2.5 years | Double-bit errors correlate with high-power operating modes and prior DBE history on the same GPU, not with temperature across the general population (abstract-level; counts not extracted). |
| ORNL Titan lifetime data ([Ostrouchov et al., via](https://arxiv.org/pdf/2303.16369)) | 30k+ K20X, ~7 years | GPU replacement dominated by two competing risks: double-bit errors and "off the bus"; failure time depends on cabinet and slot position. |

**Reconciling the Meta numbers.** 6.5 failures per 1,000 node-days implies a hardware-only node MTBF of ~154 days, which for 128 nodes (1,024 GPUs) would give ~29 h, yet the measured job MTTF is 7.9 h. The measured figure includes all failure causes seen by jobs (user/software bugs, unattributed failures), i.e. an effective per-node rate near 24 per 1,000 node-days (node MTBF ~42 days; the 42-day figure is also derived by [Glenn Lockwood's notes](https://glennklockwood.com/garden/papers/Revisiting-Reliability-in-Large-Scale-Machine-Learning-Research-Clusters)). For planning, use **MTBF_job = MTBF_node / N** with a node MTBF of 40-150 days on well-run datacenter hardware:

| Nodes (8 GPU each) | GPUs | Job MTBF @ 150-day node MTBF | Job MTBF @ 42-day node MTBF |
|---|---|---|---|
| 8 | 64 | 18.8 days | 5.3 days |
| 32 | 256 | 4.7 days | 31.5 h |
| 128 | 1,024 | 28 h | 7.9 h |
| 512 | 4,096 | 7 h | 2.0 h |
| 2,048 | 16,384 | 1.8 h | 0.5 h |

Marketplace/community hardware should be assumed worse than both columns until grid has its own telemetry (unverified; no public study found, see Open questions).

---

## 2. Failure taxonomy

Recovery vocabulary used below: **RIP** = retry in place (same processes), **RPG** = restart process group on same nodes from checkpoint, **REPL** = replace node then RPG, **SHRINK** = continue with fewer nodes, **ABORT** = stop and surface to the user (retrying cannot help).

### 2.1 GPU hardware (Xid events)

Xids appear in the kernel log as `NVRM: Xid (PCI:0000:xx:00): <code>, ...` and through DCGM (`DCGM_FI_DEV_XID_ERRORS`, field 230, reports only the *most recent* Xid, not a history, per [this DCGM comparison](https://www.netdata.cloud/guides/nvidia-gpu/nvidia-gpu-dcgm-vs-nvidia-smi-nvml/)). NVIDIA's [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html) assigns each code an "Immediate Action" resolution bucket; that bucket is the correct machine-readable input for an automated policy.

| Xid | Meaning (catalog mnemonic) | NVIDIA immediate action | Field evidence (Delta) | Transient? | grid recovery |
|---|---|---|---|---|---|
| 13 | Graphics engine exception | `RESTART_APP` | - | Usually app bug or bad kernel | RPG; ABORT if it repeats at same step |
| 31 | MMU fault (`..._MMU_ERR_FLT`) | `RESTART_APP` | 18,876 events; 59% job failure | Mostly software (illegal access), sometimes hardware | RPG; strike the node |
| 48 | Double-bit ECC (`GPU_ECC_DBE`) | `WORKFLOW_XID_48` (if followed by 63/64: drain, reset GPU) | 90% job failure | No - needs GPU reset to apply remap | REPL (or RPG after `nvidia-smi --gpu-reset` if permitted) |
| 63 | Row remap recorded (`INFOROM_DRAM_RETIREMENT_EVENT`) | `IGNORE` (remap is pending until reset) | 50% job failure | Yes after reset | Finish/RPG, reset before reuse |
| 64 | Row remap failure | `RESET_GPU`; [debug guide](https://docs.nvidia.com/deploy/gpu-debug-guidelines/) says reboot immediately | 100% job failure | No - RMA candidate | REPL, blacklist GPU |
| 74 | NVLink error | `WORKFLOW_NVLINK_ERR` (decode bit field) | 2,987 events; 66% job failure; 14% touch >=2 GPUs | Mixed; 20% lead to a state needing GPU reset | RPG once; REPL on repeat |
| 79 | GPU fallen off the bus | `RESTART_BM` (reboot bare metal) | 31 events | No - PCIe/power/thermal/riser | REPL always |
| 92 | High single-bit ECC rate | `IGNORE` | - | Leading indicator | Strike only |
| 94 | Contained ECC error | `RESTART_APP` | 100% job failure (3/3) | Yes - only the faulting process dies | RPG |
| 95 | Uncontained ECC error | `RESET_GPU` | 38,905 events, 97% job failure, 99% from one GPU | No until reset | REPL, blacklist on repeat |
| 109 | Context-switch timeout | `RESET_GPU` | - | No | REPL |
| 119 / 120 | GSP RPC timeout / GSP error | `RESET_GPU`, investigate software | 100% job failure; >99% leave GPU in error state; node reboot needed | No; often a driver bug | REPL; record driver version |
| 136 | Link training failed | `RESET_GPU` | 70 events on Delta H100s | No | REPL |
| 140 | Unrecovered ECC | `RESET_GPU` | - | No | REPL |
| 143 | GPU init error | `RESET_GPU` | - | No | REPL |
| 144-150 | NVLink5 (SAW/RLW/NETIR...) | `WORKFLOW_NVLINK5_ERR` | - | Depends on severity decode | RPG/REPL |
| 154 | "GPU recovery action changed" | Informational: reports required action (`None`, `Drain P2P`, `Drain and Reset`, `GPU Reset Required`, `Node Reboot Required`) | - | - | **Parse this first on new drivers** |

Notes:

- **Row remapping** (A100 and later) replaces legacy page retirement: up to 512 remaps for the frame buffer versus 64 retired pages before, remaps persist for the GPU's life, and a remap only takes effect after a GPU reset ([NVIDIA memory error management](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html)). Delta found remapping succeeds only ~50% of the time after a DBE and overall DBE impact is mitigated 70.6% of the time. Query with `nvidia-smi --query-remapped-rows=...` (`remapped_rows.pending`, `remapped_rows.failure`; exact field list and RMA thresholds not verified in this sweep).
- **GSP (Xid 119)**: Meta traced a wave of GSP timeouts to a driver regression fixed by a patch ([Kokolis et al.](https://arxiv.org/html/2410.21680v2)); a provider KB recommends driver >= 535.216.01 or >= 550.144.03 (from search snippet, [Tencent Cloud doc](https://www.tencentcloud.com/document/product/560/75964)). On marketplace nodes the driver is the host's and cannot be changed from inside a container, so grid can only *select* on driver version.
- **GPU reset inside a container** is typically impossible on RunPod-style pods (no privileged access to the host) (unverified per provider). Any Xid whose action is `RESET_GPU` or `RESTART_BM` therefore degenerates to **replace the node**.
- **Thermal throttling** does not raise an Xid. It appears as slow steps. Llama 3 measured a 1-2% diurnal throughput swing from temperature-driven DVFS and 1.4% of interruptions from thermal interface/sensor faults. Detect via `DCGM_FI_DEV_THERMAL_VIOLATION` (throttling duration in microseconds, [DCGM field IDs](https://docs.nvidia.com/datacenter/dcgm/3.0/dcgm-api/dcgm-api-field-ids.html)), GPU temperature, and per-rank step-time skew. Not transient on a badly cooled host; treat as a straggler (see `08-heterogeneous-distributed-training.md`) and replace at the next restart.

### 2.2 Host

| Failure | Manifestation | Detection | Transient? | Recovery |
|---|---|---|---|---|
| Host OOM killer | Worker exits 137 / `Killed`; dataloader worker "killed by signal"; dmesg `Out of memory: Killed process` (may be invisible inside a container) | Exit code 137, cgroup `memory.events` `oom_kill` counter, RSS trend | Deterministic if config-driven, intermittent if leak | RPG once; ABORT with diagnosis if it recurs at similar step count |
| `/dev/shm` too small | DataLoader bus errors / hangs ([pytorch#1595](https://www.github.com/pytorch/pytorch/issues/1595)) | Preflight `df /dev/shm` | Deterministic | Fix config (preflight, see `09-fast-bootstrap-preflight.md`) |
| Disk full (container disk, checkpoint volume) | `ENOSPC` on checkpoint write; truncated checkpoints | Free-space gauge, write-then-verify | Deterministic | Garbage-collect old checkpoints; ABORT if still full |
| Kernel/driver/userspace mismatch | `Failed to initialize NVML: Driver/library version mismatch`; CUDA init errors (unverified exact strings) | Preflight: `nvidia-smi`, `torch.cuda.is_available()`, NCCL smoke test | Deterministic per node | Exclude node at provisioning time |
| PCIe errors (AER) | Often co-occur with Xid 79 (43-63% at Meta) | dmesg AER lines, link width/speed check | No | REPL |
| Local SSD / PSU / chassis / DIMM | 0.5-0.7% each of Llama 3 interruptions; DIMM is 20.5% of Meta lemon-node root causes | Node disappears or EDAC/MCE logs | No | REPL |
| Unplanned host maintenance | 7.6% of Llama 3 interruptions; RunPod notifies by email, sometimes only after the outage starts ([RunPod docs](https://docs.runpod.io/pods/maintenance-and-outages)) | Provider API / heartbeat loss | - | REPL |

### 2.3 Network

| Failure | Manifestation | Detection | Transient? | Recovery |
|---|---|---|---|---|
| IB/RoCE link flap or bad cable | NCCL error `ibv_poll_cq` status 12 ("transport retry counter exceeded"), or a hang; 8.4% of Llama 3 interruptions; a top cause at Meta | NCCL log, `NCCL_IB_RETURN_ASYNC_EVENTS` (default 1: IB async events logged, fatal ones stop IB communication), port counters | Flap: yes; bad cable: no | RPG once; REPL if the same node is implicated twice |
| Peer death seen by survivors | All other ranks blocked in a collective; no error for tens of seconds to minutes | Step-time watchdog; PyTorch NCCL watchdog | - | RPG/REPL depending on cause of peer death |
| TCP socket transport stalls (the common case on marketplace Ethernet) | Slow or stuck collectives; connection errors at init | `NCCL_SOCKET_RETRY_CNT` (default 34) and `NCCL_SOCKET_RETRY_SLEEP_MSEC` (default 100 ms, linear backoff, ~60 s total) govern connect retries ([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)) | Often | RPG |
| Packet loss / congestion without failure | Step time variance, throughput collapse; Meta saw 50-75% bandwidth loss during bring-up without adaptive routing | Per-rank collective timing (`TORCH_NCCL_ENABLE_TIMING=1`), periodic bandwidth probe | Varies | Treat as straggler; REPL or regroup at next restart |
| Rendezvous / DNS / control plane | Workers cannot find `MASTER_ADDR` or the store; restart loops fail to re-form | Rendezvous timeout | Usually | RIP with backoff; use IPs not DNS names for rendezvous |
| Shared filesystem mount loss | Checkpoint or data reads hang (uninterruptible I/O); a top-4 cause at Meta | Mount health check with timeout, I/O latency | Varies | RPG after remount; REPL if node-local |

NCCL's IB timeout is `4.096 us * 2^NCCL_IB_TIMEOUT` per attempt; with defaults `NCCL_IB_TIMEOUT=20` (since 2.23) and `NCCL_IB_RETRY_CNT=7` that is ~30 s before an error is raised ([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)). Raising these values delays detection rather than fixing anything. Transport tuning itself belongs in `05-networking-interconnects.md`.

### 2.4 Software

| Failure | Manifestation | Detection | Transient? | Recovery |
|---|---|---|---|---|
| CUDA OOM | `torch.OutOfMemoryError: CUDA out of memory` on one rank; others then hang or abort | Exception text; `DCGM_FI_DEV_FB_USED` near capacity | Deterministic for fixed shapes; intermittent with variable sequence lengths or fragmentation | RPG at most once; ABORT if repeat. On heterogeneous clusters the smallest-VRAM node sets the limit (see `06-gpu-hardware-nuances.md`) |
| NaN / Inf loss | Loss becomes NaN; may be numerics, bad data, or SDC | Loss/grad-norm finite check each step; `TORCH_NCCL_NAN_CHECK` checks collective inputs ([PyTorch NCCL env vars](https://docs.pytorch.org/docs/2.8/_sources/torch_nccl_environment_variables.md.txt)) | Depends | Roll back to last good checkpoint and replay; if deterministic replay reproduces it: ABORT (model/data bug); if not: suspect SDC, bisect nodes |
| Deadlocked dataloader | GPU utilization 0%, main process at ~0% CPU; fork-inherited locks / thread pools in workers ([pytorch#75147](https://github.com/pytorch/pytorch/issues/75147)) | Step watchdog + `py-spy dump --pid` stack capture | Often intermittent | RPG; prefer `multiprocessing_context="spawn"`/`forkserver` |
| Hung collective / desync | All ranks alive, no progress; ranks disagree on which collective is next | PyTorch NCCL watchdog and flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE>0`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, `TORCH_NCCL_DESYNC_DEBUG`) | Depends | RPG; ABORT if same step repeats |
| Software/dependency bug | 12.9% of Llama 3 interruptions | Same stack trace across restarts | Deterministic | ABORT after N identical failures |
| Crash loop | Meta: one 1,024-GPU job requeued 35 times, triggering 548 preemptions of other jobs | Restart counter with no step progress | - | ABORT; restart budget must be tied to *progress*, not wall-clock |

PyTorch specifics: `TORCH_NCCL_ASYNC_ERROR_HANDLING` defaults to 3 (tear down the process without aborting the communicator); `TORCH_NCCL_ENABLE_MONITORING=1` starts a monitor thread that aborts the process if the ProcessGroupNCCL watchdog itself stops heart-beating for `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` ([PyTorch NCCL env vars](https://docs.pytorch.org/docs/2.8/_sources/torch_nccl_environment_variables.md.txt)). The numeric defaults (heartbeat timeout, the per-collective process-group timeout, commonly cited as 8 and 10 minutes) are **not stated in the docs** and were not verified against source in this sweep. Llama 3 logged "NCCL Watchdog Timeouts" as its own root-cause category (1.7%), and Meta relies on the flight recorder plus automatic timeouts to localize stalls.

### 2.5 Provider-level (marketplace specific)

| Failure | Manifestation | Detection | Transient? | Recovery |
|---|---|---|---|---|
| Spot / interruptible preemption | RunPod: SIGTERM then SIGKILL about 5 s later (a [RunPod docs PR](https://github.kakarurec.workers.dev/runpod/docs/pull/626/files/8e5aebdb3740725f1f0d080d1f112ba8267e20ee) and [forum answer](https://answeroverflow.com/m/1262811070603464704); third-party pages disagree; treat as "seconds"). Vast.ai: outbid or displaced by an on-demand rental, instance is *stopped/paused* and may resume later ([Vast docs](https://docs.vast.ai/documentation/instances/choosing/instance-types)) | SIGTERM handler; provider API status | Capacity may return | REPL or SHRINK; 5 s is too short for a multi-GB checkpoint, so rely on periodic checkpoints |
| Pod eviction / host offline | Pod vanishes; container disk lost (RunPod: data only on container storage is lost when a pod is "interrupted, restarted, stopped, or terminated") | Control-plane heartbeat + provider API reconciliation | No | REPL |
| Stockout on re-provision | Replacement request fails or queues; the specific SKU/region is gone | API error / timeout | Hours to days | Fallback ladder: other SKU, other DC, other provider; else SHRINK; else park the job with checkpoint safe |
| Network volume constraints | RunPod network volumes must be attached at pod deploy time and cannot be detached later without deleting the pod; volumes pin compute to their datacenter ([RunPod network volume docs](https://docs.runpod.io/storage/network-volumes.md)) | - | - | Keep the canonical checkpoint in object storage, not only on a DC-pinned volume |
| Provider control-plane or DC outage | API errors; many nodes lost at once (correlated failure) | API health, multi-node simultaneous heartbeat loss | Yes | Wait with backoff; do not burn restart budget |
| Billing / quota / credit exhaustion | Pods stopped | API | Deterministic | ABORT with clear message |

### 2.6 Silent data corruption

- **Manifestation.** Either a NaN that spreads through the all-reduce to the whole cluster, or (worse) a plausible-looking wrong gradient that passes every check ([Meta](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/)). The [Harvard/AWS study](https://arxiv.org/html/2502.12340v1) of 15 nodes that failed fleet tests found per-node mismatch frequency ranging from zero to 2.89e-2, loss curves nearly identical to healthy nodes while weights drifted to different optima, and one fine-tune ending at 0% accuracy. ABFT checksummed matmul did not detect it reliably and gives false positives under bf16.
- **Rate.** Meta: ~1 per 1,000 devices; 6 of 419 Llama 3 interruptions. Google expected SDC to affect Gemini 1.0 training "every week or two" ([Gemini 1.0 report](https://arxiv.org/pdf/2312.11805)).
- **Detection that works in practice.** Recomputation: Gemini 2.5 immediately replays any step with suspicious metrics using lightweight deterministic replay, compares per-device intermediate checksums to localize the bad accelerator, and removes it within minutes; ~0.25% of steps replayed, 6% of replays genuine. Meta describes shadow nodes that repeat random steps, "reductive triage" (binary search on smaller clusters), deterministic training and hyper-checkpointing. A [2026 fault-injection study](https://arxiv.org/html/2604.00726v1) also concludes recomputing the latest step on detection is an effective mitigation (from search snippet).
- **Recovery.** Roll back to a checkpoint that predates the corruption, exclude the device, resume. This requires keeping more than one checkpoint generation.

---

## 3. Detection stack

| Layer | Signal | Catches | Latency |
|---|---|---|---|
| Node agent (out of band from training) | dmesg/`/dev/kmsg` Xid parse, DCGM or NVML polls (Xid, remapped rows, ECC counts, temperature, thermal violation), disk free, mount probe, `/dev/shm` | GPU hardware, host | Seconds. Meta runs health checks every 5 minutes and attributes a failure if a check fails within 10 min before / 5 min after the job dies |
| Control-plane heartbeat | Agent to coordinator liveness | Node loss, eviction, partitions | 10-30 s typical |
| Training progress heartbeat | Step counter + step duration per rank | Hangs, deadlocks, stragglers, throttling | Must be phase-aware: NVIDIA's `ft_launcher` uses per-section timeouts, e.g. `--ft-rank-section-timeouts setup:900,step:120,checkpointing:600 --ft-rank-out-of-section-timeout=180` ([NVRx usage guide](https://nvidia.github.io/nvidia-resiliency-ext/fault_tolerance/usage_guide.html)) |
| Framework watchdog | PyTorch ProcessGroupNCCL watchdog, flight recorder; NCCL RAS (`NCCL_RAS_ENABLE=1` default, listens on `localhost:28028`, queryable job health since NCCL 2.24) | Hung collectives, desync, which rank is the culprit | Minutes with defaults |
| Stack capture on stall | `py-spy dump` on every rank before killing (unverified against a primary source, standard practice) | Distinguishes dataloader deadlock vs NCCL hang vs I/O hang | On demand |
| Numerical guards | Finite loss/grad-norm, grad-norm spike vs running stats, optional step replay | NaN, some SDC | One step |
| Provider API reconciliation | Pod status vs desired state | Preemption, eviction, stockout | Poll interval |

Important container caveat: inside an unprivileged marketplace container `dmesg` is often unreadable and DCGM's host engine may not be installable (unverified per provider). NVML event APIs (`nvmlEventSetWait` with Xid critical-error events) work from user space and are the portable fallback (unverified in this sweep).

---

## 4. Recovery mechanisms

### 4.1 Checkpoint/restart cadence (Young/Daly)

Young's first-order optimum is `tau = sqrt(2 * delta * M)` where `delta` is checkpoint write cost and `M` the mean time to interrupt; Daly's 2006 refinement adds higher-order terms and shows restart time does not move the optimum materially ([Daly 2006](https://graal.ens-lyon.fr/~abenoit/CR02/papers/daly.pdf), [summary](https://blog.acolyer.org/2015/06/10/a-higher-order-estimate-of-the-optimun-checkpoint-interval-for-restart-dumps/)). Meta uses the same form, `dt* = sqrt(2 * w_cp / (N_nodes * r_f))`, and found it within ~5% of Monte Carlo. Wasted fraction at the optimum is roughly `sqrt(2 * delta / M)` plus `restart_time / M`.

| Job MTBF (M) | Blocking checkpoint cost (delta) | Optimal interval | Approx. overhead (checkpoint + rework) |
|---|---|---|---|
| 28 h | 5 min | 130 min | 8% |
| 7.9 h | 5 min | 69 min | 15% |
| 7.9 h | 30 s | 22 min | 5% |
| 1.8 h | 5 min | 33 min | 30% |
| 1.8 h | 10 s | 6 min | 6% |

(Computed here from the formula.) This is why Meta concludes that reaching 0.9 "effective training time ratio" on ~12k GPUs requires either cutting the failure rate from 6.5 to ~1 per 1,000 node-days **or** cutting checkpoint write time to ~10 s. Typical Meta parameters: checkpoint write ~5 min, restart overhead 5-20 min; hourly checkpoints lose 30 min per interruption on average. **Cheap checkpoints and fast restart matter more than a lower failure rate, and both are under grid's control.**

### 4.2 Async and sharded checkpointing

- **Sharding.** Each rank writes its own shard in parallel (PyTorch Distributed Checkpoint, Megatron `torch_dist`). Llama 3 per-GPU checkpoint state ranged 1 MB to 4 GB and the bursty writes saturated the storage fabric.
- **Async.** `torch.distributed.checkpoint.async_save` stages state to CPU then writes in the background; `AsyncCheckpointerType.PROCESS` moves the write into a separate process to avoid GIL contention (default is `THREAD`) ([DCP docs](https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html)). PyTorch reported background checkpoint processing falling from ~436 s to ~67 s on a 1,856-GPU H200 TorchTitan Llama3-70B run using process-based async save plus cached save plans ([PyTorch blog](https://pytorch.org/?p=3455); 67 s figure from search snippet).
- **NVIDIA Resiliency Extension** ships an async checkpointing module with a non-blocking finalize call per iteration ([NVRx async checkpoint guide](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/async/usage_guide.html)). [DataStates-LLM](https://arxiv.org/html/2406.10707v1) copies state during forward/backward passes, claiming up to 4x checkpoint throughput.
- **Consistency.** A checkpoint is valid only when every shard plus a manifest has landed. Write to a temporary name, fsync, then publish a manifest atomically; restart always uses the newest *complete* generation. Keep at least 2-3 generations for SDC/NaN rollback.
- **Two tiers.** Frequent cheap checkpoints to node-local NVMe or a peer (survives process failure, not node loss) and less frequent uploads to object storage (survives everything). This is classic multi-level checkpointing; no LLM-specific primary source was found in this sweep (unverified).

### 4.3 TorchElastic / torchrun

`torchrun --nnodes=MIN:MAX --max-restarts=N --rdzv-backend=c10d --rdzv-endpoint=host:29400 --rdzv-id=JOB` ([docs](https://docs.pytorch.org/docs/2.15/elastic/run.html)):

- On any worker failure, "if k<=n workers fail all workers are stopped and restarted up to max_restarts."
- On membership change (node leaves or joins) existing workers are stopped, a new worker group is formed, and all workers restart with new `RANK` and `WORLD_SIZE`. Scale events and failures share the same restart budget.
- Surviving workers are killed immediately, so only checkpointed state survives. "RANK is NOT stable." `TORCHELASTIC_RESTART_COUNT` and `TORCHELASTIC_MAX_RESTARTS` are exposed to the script.
- Agent or node failure is left to the job manager. Every node must run the same `LOCAL_WORLD_SIZE`, which is a hard limit for heterogeneous clusters (see `03-rank-configuration.md`).
- The `c10d` backend's store lives on one node (single point of failure; `etcd` backends are in maintenance mode and slated for removal).

### 4.4 NVIDIA Resiliency Extension (`ft_launcher`)

A torchrun replacement adding per-rank monitor processes, section timeouts for hang detection, and **in-job restart**: on failure or timeout the worker group is terminated, health checks and a barrier rendezvous run, hot spares (launchers that joined with no workers) are promoted so the world size stays fixed, and workers reload the latest complete checkpoint. `--max-restarts` defaults to 0; in-process restart is deprecated in favour of in-job restart ([NVRx usage guide](https://nvidia.github.io/nvidia-resiliency-ext/fault_tolerance/usage_guide.html), [release notes](https://nvidia.github.io/nvidia-resiliency-ext/release-notes.html)). This is the closest existing open-source design to what grid needs at the launcher layer.

### 4.5 Ray Train

`RunConfig(failure_config=FailureConfig(max_failures=N))`: default 0 (no recovery), -1 unlimited. On failure Ray Train shuts down *all* workers, acquires new nodes if needed and restarts from the latest checkpoint reported through `ray.train.report`; driver failures are recovered by relaunching with the same `storage_path` and `name` and do not count toward `max_failures` ([fault tolerance guide](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)). Elastic mode: `ScalingConfig(num_workers=(min, max))` requests max, starts at min, restarts with fewer workers after a failure, and re-checks for scale-up every `elastic_resize_monitor_interval_s` (default 60 s) ([elastic training guide](https://docs.ray.io/en/latest/train/user-guides/elastic-training.html)). Every resize is still a full worker-group restart.

### 4.6 Slurm

- `--requeue` / `--no-requeue` and `JobRequeue` (default 1): a batch job is requeued after node failure ([sbatch man page](https://slurm.schedmd.com/sbatch.html), [SchedMD ticket 1889](https://support.schedmd.com/show_bug.cgi?id=1889)). `RequeueExit` / `RequeueExitHold` requeue on chosen exit codes. A restart counter env var (`SLURM_RESTART_COUNT`) is commonly used but was not verified in this sweep.
- `-k, --no-kill`: "Do not automatically terminate a job if one of the nodes it has been allocated fails." The allocation survives so new steps can be launched on the remaining nodes; the *step* that lost a node still dies. This is the Slurm primitive for shrink-and-continue.
- `--signal=[B:]sig@time` gives a pre-timeout warning (default 60 s when no time is given), useful for a final checkpoint.

### 4.7 Kubernetes

- **Job `podFailurePolicy`** (GA in 1.31): ordered rules matching `onExitCodes` or `onPodConditions`, with actions `Ignore` (not counted against `backoffLimit`), `FailJob`, `FailIndex`, `Count`. The standard pattern is `Ignore` on the `DisruptionTarget` condition (preemption, eviction) and `FailJob` on exit codes that signal non-retriable bugs; requires `restartPolicy: Never` ([Kubernetes docs](https://kubernetes.io/docs/tasks/job/pod-failure-policy)).
- **JobSet `failurePolicy`**: `maxRestarts` plus ordered `rules` with actions `FailJobSet`, `RestartJobSet` (default; recreates all child Jobs), `RestartJobSetAndIgnoreMaxRestarts`, `RestartJob`, `RestartJobAndIgnoreMaxRestarts`, filtered by `targetReplicatedJobs` and `onJobFailureReasons` (`BackoffLimitExceeded`, `DeadlineExceeded`, `PodFailurePolicy`) ([JobSet docs](https://jobset.sigs.k8s.io/docs/tasks/failure_policy/)).

The transferable idea is the **classification**: infrastructure disruptions should not consume the retry budget; application errors should fail fast.

### 4.8 MPI ULFM and NCCL shrink

- **ULFM** (Open MPI 5.x, enabled at run time with `mpirun --with-ft ulfm`): failures surface as `MPIX_ERR_PROC_FAILED`; `MPIX_Comm_revoke` interrupts all ranks, `MPIX_Comm_shrink` builds a communicator of survivors, `MPIX_Comm_agree` is a fault-tolerant consensus ([Open MPI ULFM docs](https://docs.open-mpi.org/en/v5.0.x/features/ulfm.html)). It gives *forward* recovery without killing survivors, but the application must rebuild its own state.
- **NCCL** now exposes the same shape: `ncclCommShrink` creates a new communicator excluding given ranks, with `NCCL_SHRINK_ABORT` to abort outstanding operations after an error, and `ncclCommGetAsyncError` for polling ([NCCL comms API](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/comms.html); AMD's [RCCL fault tolerance guide](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/fault-tolerance.html) documents a revoke-then-shrink flow). No evidence was found that PyTorch `ProcessGroupNCCL` uses it yet (unverified), so in practice PyTorch jobs still restart.

### 4.9 Elastic shrink/grow versus fixed world size, and hot spares

| Strategy | What survives a node loss | Cost per failure | Who does it | Constraint |
|---|---|---|---|---|
| Fixed world size, wait for replacement | Checkpoint only | Rework + restart + **provision time** (Gemini: "10 or more minute delay") | Default everywhere | Stockout blocks the job |
| Fixed world size + hot spares | Checkpoint only | Rework + restart | NVRx `ft_launcher`; Gemini 1.0 hot standbys | Pay for idle spares; Delta estimates ~20% overprovisioning at 40 min recovery vs ~5% at 5 min recovery for an 800-GPU job |
| Shrink to fewer DP replicas, grow back later | Checkpoint only (torchrun/Ray) or in-memory state (Gemini slice elasticity, ULFM) | Tens of seconds (Gemini) to a full restart (torchrun/Ray) | TorchElastic `MIN:MAX`, Ray elastic, Gemini 2.5 | Only the data-parallel dimension can shrink; TP/PP/EP groups need a fixed shape. Global batch or LR must be re-derived |

For sharded-model parallelism (tensor/pipeline), losing one member invalidates the whole model replica, so the unit of elasticity is a *replica group*, not a node. Rank-to-group mapping is in `03-rank-configuration.md`.

---

## 5. Failures expected to be more common on marketplace hardware

No public measurement study of marketplace GPU reliability was found; the following is reasoned from the mechanisms above and provider docs, and is **unverified** unless cited.

1. **Preemption and host-initiated loss** dominate on spot/interruptible tiers, with notice measured in seconds on RunPod. Vast pauses rather than deletes.
2. **Stockout on re-provision.** Replacement of the same SKU in the same datacenter is not guaranteed; with DC-pinned volumes this couples storage to capacity.
3. **Host-controlled driver stack.** GSP-class driver bugs (Xid 119) cannot be patched from a container; grid can only filter on driver version and quarantine.
4. **No GPU reset or reboot rights.** Every `RESET_GPU` / `RESTART_BM` Xid becomes node replacement.
5. **PCIe / power / cooling quality variance** on community hosts (risers, consumer chassis) points at Xid 79, thermal throttling and stragglers. Vast advises picking hosts with a high reliability score for multi-day jobs ([Vast docs](https://docs.vast.ai/documentation/instances/choosing/instance-types)); RunPod Community Cloud is peer-hosted with no SLA (third-party figure of ~98% uptime is weakly sourced).
6. **Lemon recycling.** A node grid releases after an Xid 79 may be handed straight back by the provider on the next request. Persistent per-machine identity (GPU UUID/serial, host ID) is needed to blacklist across sessions.
7. **TCP-only inter-node networking** on most pods makes NCCL socket timeouts and bandwidth variance, rather than IB errors, the relevant network class.
8. **Ephemeral container disk** means the checkpoint path must never depend on a node surviving.

---

## Implications for grid

1. **Make the failure classifier a first-class component.** Every worker death or stall is mapped to one of {infra-transient, infra-node-bad, provider, app-deterministic, numeric} from exit code, signal, last Xid (or Xid 154's recovery action), stderr signature and provider status. Policy then follows the tables above: infra and provider failures do not consume the user's retry budget (JobSet's `...AndIgnoreMaxRestarts` pattern); the same stack trace at the same step twice aborts.
2. **Tie the restart budget to progress.** Reset the counter only after N successful steps beyond the last failure. This prevents the 35-requeue crash loop Meta documented.
3. **Per-node strike list with persistent identity.** Record GPU UUIDs/serials and host IDs; quarantine on any `RESET_GPU`/`RESTART_BM`-class Xid, on a second network implication, or on repeated straggling. Persist across runs and consult at provisioning. Expect 1-2% of nodes to cause a double-digit share of failures.
4. **Step-time watchdog with phase-aware timeouts** (setup / step / checkpoint / eval), auto-calibrated from observed step times (e.g. a multiple of the rolling p99), running in a sidecar per node so it survives a wedged Python process. On trigger: capture `py-spy` stacks and NCCL flight-recorder dumps on all ranks, *then* kill. Set PyTorch's `TORCH_NCCL_ENABLE_MONITORING`, `TORCH_NCCL_TRACE_BUFFER_SIZE` and `TORCH_NCCL_DUMP_ON_TIMEOUT` by default.
5. **Budget for restart, not just for failure rate.** Target numbers: detect < 60 s, re-rendezvous + reload < 3 min, blocking checkpoint cost < 30 s. At 128 nodes this keeps overhead near 5% even with a pessimistic 8 h job MTBF. Compute the checkpoint interval from Young/Daly using grid's *measured* MTBF for the current cluster and the measured checkpoint cost, and adapt it during the run.
6. **Own the checkpoint protocol.** Sharded per-rank writes, async (process-based) upload, atomic manifest, at least three retained generations, canonical copy in object storage rather than a DC-pinned network volume, plus an optional fast local tier. Integrate with DCP/Megatron `torch_dist` rather than inventing a format.
7. **Support three replacement strategies, chosen per job:** (a) hot spares (default small percentage, e.g. 2-5%, tuned to observed failure rate), (b) shrink the data-parallel dimension and grow back when capacity returns, (c) wait for replacement. A stockout fallback ladder (SKU, datacenter, provider) feeds (c). The unit of shrink is a whole model-replica group.
8. **Do not rely on preemption notice.** With ~5 s of SIGTERM warning, the handler should only flush logs and mark the node as leaving so peers fail fast instead of waiting 30+ s for an NCCL timeout.
9. **Fail fast on peer loss.** When the control plane learns a node is gone, proactively SIGTERM all ranks rather than letting survivors sit in a collective until transport timeouts expire.
10. **Preflight is the cheapest reliability feature** (details in `09-fast-bootstrap-preflight.md`): driver version filter, remapped-row/ECC state, PCIe link width, `/dev/shm`, disk, a short matmul determinism check against golden output, and a pairwise NCCL bandwidth test. Reject nodes before they join the world.
11. **Cheap SDC defence:** finite checks and grad-norm anomaly detection every step; on anomaly, deterministic replay of the step before deciding; an optional low-rate shadow recompute of a random step on a different node; node bisection tooling for when a run goes bad. Requires deterministic data order keyed by step and rank-independent seeding.
12. **Design the launcher around full-group restart now, leave room for in-place shrink later.** Today every mainstream stack restarts all workers on membership change. `ncclCommShrink` and ULFM show where this is going; grid's rank/membership layer should not assume restart is the only option.
13. **Telemetry as product.** Log every failure with class, node identity, provider, SKU, driver version and time-to-recover. This yields the marketplace MTBF data that nobody publishes and directly improves placement.

## Open questions

- **Marketplace failure rates.** No public per-provider MTBF, preemption-rate or stockout-duration data was found. Needs grid's own measurement or a second sweep of community reports (Discords, forums) and provider status histories.
- **Exact preemption semantics per provider.** RunPod's ~5 s SIGTERM-to-SIGKILL figure comes from a docs PR and a forum post, and RunPod appears to have been removing spot references from its docs. Vast pause/resume behaviour for multi-node jobs, and Lambda/CoreWeave/Crusoe/Nebius notice periods, need verification (coordinate with `07-provider-landscape.md`).
- **What is observable from inside an unprivileged pod:** dmesg, DCGM host engine, NVML Xid event subscription, `nvidia-smi --gpu-reset`, PCIe AER, per provider. Needs hands-on testing.
- **PyTorch watchdog defaults** (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, process-group collective timeout, whether monitoring is on by default) must be read from `ProcessGroupNCCL.hpp` for the PyTorch version grid targets.
- **Is `ncclCommShrink` reachable from PyTorch,** and what would in-place shrink of a DDP/FSDP job require (optimizer state, sampler re-partitioning)? Also the exact NCCL version that introduced it and its behaviour over the socket transport.
- **NVRx `ft_launcher` outside Slurm:** can it run on bare pods with grid supplying rendezvous and spares, or should grid reimplement the rank-monitor design?
- **Row-remap RMA thresholds and the exact `nvidia-smi --query-remapped-rows` fields** were not retrieved from NVIDIA's memory-error-management pages.
- **H100/H200/B200-era failure data.** The Delta v3/v4 paper (H100 memory MTBE 3.2x worse than A100, ~5% overprovisioning) was read only via snippets; Blackwell-specific Xids (121, 144-150) have no field studies yet.
- **Restart-time decomposition at 100+ nodes** on TCP networks (rendezvous, NCCL init, checkpoint download) has no published numbers; it determines whether the 3-minute target is realistic.
- **SDC prevalence on older or consumer GPUs** (no ECC on some consumer cards) and the real cost of deterministic replay in PyTorch (nondeterministic CUDA ops) deserve a dedicated sweep.
- **Multi-level checkpointing** (local NVMe / buddy node / object store) for LLM training: no primary LLM-scale source found.
- **Slurm details** (`SLURM_RESTART_COUNT`, behaviour of `--no-kill` with `srun` steps under `KillOnBadExit`) were not verified against the man pages.
