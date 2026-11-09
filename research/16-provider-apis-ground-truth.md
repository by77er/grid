# 16 - Provider APIs: ground truth from specs, SDK source and issue trackers

Research date: 2026-11-09. Second-sweep file; builds on `07-provider-landscape.md`,
`02-multicloud-provisioners.md`, `09-fast-bootstrap-preflight.md`, `01-orchestrators-schedulers.md`.

Method. OpenAPI documents and raw source files were downloaded and parsed locally; identifiers
are copied from them. GitHub's API was unreachable, so **commit SHAs are not recorded**: source
is cited as `master`/`main` served by raw.githubusercontent.com on 2026-11-09. Issue threads
were read through a summarising fetcher (paraphrase plus short quotes; numbers and dates as
shown). Text in RunPod docs addressed to AI agents was ignored.

## Summary

- **RunPod now has one supported control plane: REST v2.** REST v1 "will be retired on
  November 15, 2026" (six days after this was written) and GraphQL "in early 2027".
  runpod-python 2.2.0 (released 2026-11-09), runpodctl, SkyPilot `master` and dstack `master`
  have all moved pod, template, registry and volume calls to `/v2`.
- **RunPod spot is gone, not merely de-emphasised.** The migration guide lists "Spot Pods |
  `interruptible`, `podRentInterruptable`, `podBidResume` | None", and the Terraform 2.0
  migration page says spot pods "were retired by the platform. The APIs stopped accepting
  bids". `grid` should treat RunPod as on-demand only.
- **Stockout on RunPod has no machine-readable code.** It is HTTP `400` with a human `detail`,
  the same status as a cross-field rule violation. dstack hit exactly this ambiguity on
  clusters (issue #3910). Lambda (`instance-operations/launch/insufficient-capacity` vs
  `global/quota-exceeded`) and Nebius (`available` vs `limit` per fabric) do separate the cases.
- **Port and IP discovery needs a per-pod `GET`.** `Pod.runtime` (which carries
  `ports[{private, public, type, ip}]`) and `ssh.direct` are documented as null in
  `GET /v2/pods` list responses. dstack polls each pod; SkyPilot `master` reads both from the
  list, which by the spec can never become ready.
- **RunPod cluster membership is not editable through the API.** No scale route exists, and
  member pods "cannot be modified or deleted via the pod endpoints" (`409`). Grow is
  console-only; shrink by API is not possible short of deleting the cluster.
- **Neither SkyPilot nor dstack does multi-node on plain RunPod pods.** SkyPilot still declares
  `MULTI_NODE` unsupported; dstack does it only through `POST /v2/clusters` with
  `type: "TRAINING"`, with pod counts hard-coded to 2..8 in its catalogue.
- **Fast-fabric clusters that are fully API-creatable today**: RunPod (`POST /v2/clusters`),
  Together (`POST /v1/compute/clusters`), Nebius (gpu-cluster then VMs), Crusoe (IB partition
  then VMs), SF Compute (IB partition, subnet, order, instances; public preview). Not
  API-creatable: Lambda 1-Click Clusters (console reservation in weeks), Vast RDMA clusters
  (sales), Shadeform (no cluster route), Prime Intellect Slurm clusters (only onto a team's
  already-dedicated hardware).

---

## 1. RunPod

### 1.1 API generations and deadlines

| Surface | Status on 2026-11-09 | Source |
|---|---|---|
| REST v2, `https://api.runpod.io/v2`, spec `info.version` `2.0.0` | Current | https://api.runpod.io/v2/openapi.json |
| REST v1, `https://rest.runpod.io/v1`, spec `info.version` `0.1.0` | "deprecated and will be retired on **November 15, 2026**" | https://docs.runpod.io/api-reference/overview.md |
| GraphQL, `https://api.runpod.io/graphql` | "deprecated and will be retired in early 2027" | https://docs.runpod.io/sdks/graphql/configurations.md |

dstack issue #4167 (opened 2026-08-19) records RunPod's schedule as: rate limits on v1 and
GraphQL from September 17, v1 returning `410 Gone` from November 15, GraphQL `410` in early
2027 (https://github.com/dstackai/dstack/issues/4167). The `410` behaviour is from that issue,
not from a RunPod page I opened (unverified).

What v2 removed, per the migration guide
(https://docs.runpod.io/api-reference-v2/migrate-from-v1.md):

- Spot (`interruptible`, `podRentInterruptable`, `podBidResume`): "None".
- Multi-GPU-type fallback (`gpuTypeIds` list), `countryCodes`, `minDownloadMbps`,
  `minUploadMbps`, `minDiskBandwidthMBps`, `supportPublicIp`, pod `reset`, savings plans.
- "Host machine details | `machineId`, `podHostId` | `dataCenterId` only".
- Server-side list filters: "`GET /v2/pods?desiredStatus=RUNNING` ignores the filter and
  returns every Pod". Unknown *body* fields are rejected with `422`; unknown *query*
  parameters are silently ignored.
- Account balance has no v2 route: "Keep using the GraphQL API for this"
  (`myself { clientBalance }`).

Silent behaviour changes in the same guide: `ports` has no default (v1 defaulted to
`8888/http,22/tcp`); a pod without `mounts` has no persistent storage (v1 gave a 20 GB
`/workspace`); `templateId` is copied at create time rather than linked; `dockerStartCmd`
(array) became `args` (single string).

### 1.2 Create a pod: exact request (`POST /v2/pods`, schema `CreatePodRequest`)

`unevaluatedProperties: false`, so any unknown field is a `422`.

| Field | Type and constraints (from the spec) |
|---|---|
| `name` | required string, `minLength` 1. Not unique. |
| `image` | required unless `templateId` is set |
| `templateId` | template resolved at create time; body fields override, `env` merged per key |
| `cloud` | `SECURE` (default) or `COMMUNITY` |
| `gpu.id` | GPU type id, e.g. `"NVIDIA H100 80GB HBM3"` (the console's "H100 SXM"; SkyPilot's `GPU_NAME_MAP` notes "displayed as SXM at the console but the ID from the API appears as HBM") |
| `gpu.count` | integer 1..1024, default 1; real ceiling is `GpuType.maxCount.{secure,community}` |
| `gpu.allowedCudaVersions` / `gpu.minCudaVersion` | `major.minor`; mutually exclusive (`400`); exact match, "a version no machine reports yields a capacity error rather than a fallback" |
| `gpu.minRamPerGpu` (1..65536 GB), `gpu.minVcpuCountPerGpu` (1..4096) | placement filters, not requests |
| `cpu` | alternative to `gpu`; exactly one must be set |
| `dataCenterIds` | array of ids such as `US-TX-3`; omit to let the scheduler choose |
| `globalNetworking` | boolean, default false; needs NVIDIA GPU and a data center with `globalNetwork: true` |
| `ports` | array of `"<port>/<protocol>"`, example `["8888/http","22/tcp"]`; protocols `http`, `tcp` |
| `env` | object of string to string |
| `args` | single string: a bare shell string is treated as CMD, or JSON `{"entrypoint":[...],"cmd":[...]}` |
| `entrypoint`, `cmd` | exec-form arrays, encoded into `args` |
| `disk` | container disk GB, minimum 1, "ephemeral, wiped on restart" |
| `mounts.persistent` | `{size >= 10, path}`; host-pinned; deprecated; exclusive with the two below |
| `mounts.network` | up to 4 of `{volumeId, path}`, all in the pod's data center, `path` has no default |
| `mounts.global` | at most 1 `{volumeId, path?}` |
| `registry` | registry credential id for private images |
| `startSsh` | injects `PUBLIC_KEY` from account keys (`PUT /v2/account/ssh-keys`); "with none registered the flag does nothing" |
| `startJupyter` | injects `JUPYTER_PASSWORD` |

Not present anywhere in the v2 spec (text search of the 386 kB document): `interruptible`,
`spot`, `privileged`, an idempotency key, a machine/host id, country codes, bandwidth filters.

Request size limit: 102400 bytes for the pod create *after template expansion* (`413`).

Symmetric ports (request a port number above 70000, read `RUNPOD_TCP_PORT_70000`) are
documented only in prose (https://docs.runpod.io/pods/configuration/expose-ports.md); the
spec's `ports` item is an unconstrained string, so whether v2 accepts `"70000/tcp"` is
untested. Environment variables seen in docs: `RUNPOD_POD_ID`, `RUNPOD_PUBLIC_IP`,
`RUNPOD_TCP_PORT_22`, `RUNPOD_POD_HOSTNAME`, `RUNPOD_DC_ID`, `RUNPOD_GPU_COUNT`,
`RUNPOD_CPU_COUNT`, `RUNPOD_VOLUME_ID`.

### 1.3 Pod response: addresses and ports (`Pod` schema)

- `status`: `PROVISIONING | STARTING | RUNNING | EXITED | ERROR | TERMINATED`; `actions`
  lists the transitions valid now.
- `runtime.ports[]`: `PodRuntimePort {private:int, public:int|null, type:string, ip:string|null}`.
  `runtime` is "Null when the pod is not RUNNING, **and when listing pods with `GET /v2/pods`;
  fetch a single pod by ID**".
- `ssh.proxy` / `ssh.direct`: each `{host, port, username, command}`. `direct` is null unless
  `22/tcp` is in `ports` and a public port is assigned; "Also null when listing pods". The
  proxy "carries an interactive shell only - SCP, SFTP, rsync, and port forwarding need
  `direct`".
- `globalNetworking {enabled, ip, internalDns}` with `internalDns` = `<podId>.runpod.internal`.
- `cluster {id, rank, role, ip}` on cluster members (section 1.5).
- `dataCenterId`, `cudaVersion` (host-reported), `cost` (hourly rate, returned in every
  status), `gpu.vcpuCount`, `gpu.memory`, `createdAt`, `startedAt`.
- No host or machine id. The Terraform provider still reads `machine_id` through GraphQL
  (https://docs.runpod.io/integrations/terraform/migration.md), so a per-host scorecard
  (`09`, implication 10) has no v2 key after GraphQL retires.

`GET /v2/pods`: cursor-paginated, `limit` 1..1000 (default 1000), `includeClusterPods`
(default false), and `filters` (URL-encoded JSON `{"tags":[{"key","values"}]}`). Tags can be
*filtered and read* through the API but I found no route or body field that *sets* them; the
tagging page describes console-only assignment and says tagging "is available to organization
accounts only" (https://docs.runpod.io/organizations/orgs-resource-tagging.md). So ownership
marking must use `name`.

Other routes: `POST /v2/pods/{id}/action` `{action: start|stop|restart|terminate}` (`409` if
invalid for the state); `PATCH /v2/pods/{id}` (`409` for cluster members);
`GET /v2/pods/{id}/logs` as Server-Sent Events (`source`, `tail` 0..5000, `since`), which
shows image pull and container start without SSH.

### 1.4 Errors, stockout and rate limits

Error body (`ErrorResponse`, `application/problem+json`): `{title, status, detail, errors[]?}`.
There is no error `code` field.

The `createPod` description gives the retry table verbatim: `422` "Fix the request. Never
retry."; `400` "either it breaks a cross-field rule, or this GPU and data center combination
could not be placed ... Try your next candidate."; `402` "Insufficient balance ... Stop";
`403` "Your account cannot access the requested pool ... Skip this candidate"; `429` "Back off
using `Retry-After`"; `5xx` "Retry the same candidate with backoff". And the key admission:
"`400` covers both ... because capacity exhaustion currently carries no machine-readable code
of its own - only a human-readable `detail`. ... if *every* candidate returns `400`, read the
last `detail` as a problem with the request rather than as absent capacity."

Quirk recorded in dstack's client: "As of 2026-09-18, v2 API returns 400 with `"detail":
"Registry auth not found"` instead of 404" (`api_client.py`). So `400` is overloaded three
ways: rule violation, stockout, and some not-found cases.

Rate limits (`RateLimitHeader`, `RateLimitPolicyHeader` in the spec;
https://docs.runpod.io/api-reference-v2/rate-limits.md):

- `RateLimit-Policy: "minute";q=60;w=60, "hour";q=3000;w=3600, "day";q=50000;w=86400`
  (illustrative values) and `RateLimit` with remaining `r` and reset seconds `t` per window.
- "Windows are fixed and aligned to the clock, not sliding." "Limits are per user, and every
  `/v2` route counts toward the same limits; creating pods has no separate limit."
- Headers are "omitted for rate-limit-exempt callers", so their absence is a state to handle.
- The spec lists `RateLimit` headers on the `201` of `POST /v2/pods` but not on the `201` of
  `POST /v2/clusters`, nor on cluster `GET`/`DELETE` success responses. Whether that is a
  documentation gap or real is untested.

Stock signals: `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD|CLUSTER|SERVERLESS&count=N&cloud=...`
returns `availability` in `NONE|LOW|MEDIUM|HIGH`, `dataCenters[{id,name,availability}]`,
`cudaVersions[{version, available}]`, `minVcpuCount`, `minRam`. `product=CLUSTER` exists, so
cluster stock can be read separately from pod stock. `count` here is a GPU count; there is no
parameter for "N pods of M GPUs", so the catalogue cannot answer "can I get 8 nodes".

### 1.5 Clusters (`POST /v2/clusters`, schema `CreateClusterRequest`)

Body = `BaseContainerConfig` (`image`, `args`/`entrypoint`/`cmd`, `disk`, `env`, `ports`) plus:

- `name` (required), `type` (required) with enum **`APPLICATION | TRAINING | SLURM | RAY`**.
- `compute` (required): `{gpuTypeId, gpuCountPerPod (>=1, bounded by GpuType.maxCount), podCount (2..250)}`.
- `templateId`: "the only private-image path for clusters (a bare `registry` property on this
  body is rejected)"; mutually exclusive with `image`, `args`, `entrypoint`, `cmd`, `disk`,
  `env`, `ports`. dstack accordingly raises "Runpod clusters with private registries are not
  supported".
- `dataCenterIds`, `startSsh`, `startJupyter`.
- `mounts` (`ClusterMounts`): `persistent` **or** one `network` volume; `global` is a `422`.
- No `gpu.minCudaVersion`, RAM or vCPU filter and no `globalNetworking` field in this schema.

Response (`Cluster`): `id`, `name`, `type`, `compute`, `dataCenterId` ("omitted until at least
one pod is placed"), `pods {total, byStatus}`, `network {cidr, vxlanId, vxlanPort}` ("omitted
until the network is provisioned"), `primary {podId, status, sshEndpoint "host:port"}`.
Members come from `GET /v2/clusters/{id}/pods` as full `Pod` objects with
`cluster {id, rank (null until assigned), role (SLURM_CONTROLLER|SLURM_COMPUTE|RAY_HEAD|RAY_WORKER, absent for TRAINING/APPLICATION), ip (omitted until assigned)}`.
`PATCH` only renames. `DELETE` returns `204` and terminates all members. There is no
status field on the cluster itself, no scale route, and no documented partial-fill behaviour.

Scaling: the docs page describes a console "Scale cluster" stepper and says "To downsize your
cluster, we recommend terminating the pods on the nodes you want to remove"
(https://docs.runpod.io/instant-clusters/scale-clusters.md). The v2 spec contradicts the
second half for API users: member pods "cannot be modified or deleted via the pod endpoints".

Self-serve size: docs table still says "2-8 nodes (16-64 GPUs)", larger "up to 512 GPUs" via
sales, and "All accounts have a default spending limit. To deploy larger clusters, contact
help@runpod.io" (https://docs.runpod.io/instant-clusters.md). The schema maximum of 250 is a
validation bound, not an entitlement.

### 1.6 Volumes

`POST /v2/network-volumes` `{name, size (10..4096 GB), dataCenter, type?: STANDARD|HIGH_PERFORMANCE}`;
`dataCenter` and `type` are immutable; size is grow-only. Which tiers a data center supports
is `DataCenter.networkVolumeTypes`.

Attach semantics conflict between two primary sources. The spec's `UpdatePodMounts` says a
network or global volume "can be mounted, moved, or unmounted after creation" via
`PATCH /v2/pods/{id}` (`remove: true` to unmount, up to 8 entries per request). The migration
guide says "The mount type is fixed at creation. You can't change or clear mounts ... Create a
new Pod to change its storage." The likely reconciliation is that the *kind* (persistent vs
network) is fixed while individual volumes can change, but that needs a probe.

### 1.7 Container capabilities, shm, cgroup, spend limit, spot notice

| Topic | Primary statement found | Verdict |
|---|---|---|
| Privileged / capabilities | None. The words `privileged`, `capabilit`(ies as Linux caps) and `cgroup` do not occur in the v2 spec; docs pages (105 non-serverless pages text-searched) say only "Docker Compose is not supported, and you cannot run your own Docker instance" and offer a Bazel tutorial to "Emulate a Docker-in-Docker workflow". | No official statement exists. |
| FUSE | SkyPilot #8592 (opened 2026-01-15): `mknod /dev/fuse: operation not permitted`, "fuse device not found". Maintainer on 2026-01-16: "Without certain permissions, it might be not possible to do so on RunPod". No RunPod reply; closed by the stale bot 2026-10-04 as not planned. SkyPilot code: "Mounting object stores is not supported on RunPod ... use `mode: COPY`". | Unavailable in practice; never confirmed by RunPod. |
| TUN, ulimits, memlock | Nothing found outside the Instant Cluster statement already in `07` (memlock limit removed where RDMA is present). | Probe. |
| `/dev/shm` | Conflicting. A docs `df -h` sample shows `shm 24G /dev/shm` next to `tmpfs 252G` (https://docs.runpod.io/pods/troubleshooting/storage-full.md), suggesting shm scales with the host. The Ray+vLLM cluster guide says "The default (64 MB) is too small" and tells the reader to set `--shm-size` in "Docker run options", for which no API field exists (https://docs.runpod.io/instant-clusters/ray-vllm.md). | Probe; do not rely on shm for NCCL or dataloaders until measured. |
| cgroup version | SkyPilot #8293 (opened 2025-12-12): one node exposed cgroup v1 `/sys/fs/cgroup/cpuset/cpuset.cpus`, another only cgroup v2 `/sys/fs/cgroup/cpuset.cpus`; Ray saw 192 host CPUs for a 2-vCPU pod. Fixed in SkyPilot PR #8998 (2026-03-05); `get_cluster_info` now passes `num-cpus` from the pod's `vcpuCount`. | Hosts are mixed v1/v2; `nproc`-style detection is unreliable. Use `gpu.vcpuCount` from the API. |
| `chown` | SkyPilot #8549 (opened 2026-01-12): `rsync: chown ... failed: Invalid argument (22)` on upload; fixed by `--no-owner --no-group` (PR #8556). | File ownership changes can fail on pod filesystems. |
| Spend limit | "default spend limit of $80 per hour across all resources"; "Spending limits increase automatically over time based on your account history" (https://docs.runpod.io/accounts-billing/billing.md). `runpodctl user` prints balance, hourly spend and spend limit. | Confirmed; the auto-increase is new information. No v2 route returns it. |
| Spot notice | Moot: spot is retired (section 1.1). | - |

---

## 2. How existing tools drive RunPod

### 2.1 SkyPilot (`master`, 2026-11-09)

Files: `sky/adaptors/runpod.py`, `sky/provision/runpod/{instance,utils,volume}.py`,
`sky/provision/runpod/api/{commands,pods}.py`, `sky/clouds/runpod.py`
(https://raw.githubusercontent.com/skypilot-org/skypilot/master/sky/provision/runpod/instance.py).

- Transport: `_REST_BASE = 'https://api.runpod.io/v2'`, `_MAX_RETRIES = 3`, `_TIMEOUT = 30`
  ("Pod creation can take well over 10s and a POST is not retried on timeout"),
  `_MAX_RETRY_AFTER_SECONDS = 30`, `_IDEMPOTENT_METHODS = {GET, PUT, DELETE}`. "A POST is only
  repeated on 429, which RunPod returns before processing the request."
- Sequence in `run_instances`: wait while any pod named `<cluster>-head|-worker` is
  `PROVISIONING`/`STARTING`; delete stale `EXITED`/`ERROR` pods ("an exited or errored pod
  keeps billing its disk"); then a **sequential** `for _ in range(to_start_count)` of
  `POST /pods`; then an unbounded `while True` poll every `POLL_INTERVAL = 5` s until every
  pod is `RUNNING` with an `ssh_port`. Failure states `('EXITED','ERROR','TERMINATED')` raise.
  There is still no overall timeout.
- Create body: `{name, image, disk, ports, args, startSsh: True, templateId?, dataCenterIds?,
  mounts.network?, gpu{id,count,minVcpuCountPerGpu: 4,minRamPerGpu: <GPU VRAM GB>}, cloud}`.
  `args` is a base64-wrapped bootstrap ("Use base64 to deal with the tricky quoting issues
  caused by runpod API") that `apt install`s `openssh-server rsync curl patch` at boot.
  Ports: `22/tcp`, user ports as `/tcp`, Ray dashboard and Ray port as `/http`; comment "Port
  8081 is occupied for nginx in the base image".
- Port discovery (`_parse_pod`): reads `runtime.ports` and `ssh.direct` from the **list**
  response, with the comment "RunPod only publishes the port mappings once the pod is
  RUNNING, and even then the ports may take a while to be assigned". The v2 spec says both
  are null in list responses. Either the spec is stricter than the service or this path
  cannot complete; I could not run it.
- `query_ports`: "RunPod ports sometimes take a while to be ready", `QUERY_PORTS_TIMEOUT_SECONDS = 30`.
- Addresses: "RunPod pods have no private network; the public IP is the only address other
  nodes can reach" (`internal_ip = external_ip`).
- Spot: module docstring says "Spot pods have no v2 endpoint yet and still go through the
  GraphQL API"; `create_spot_pod` sends `podRentInterruptable` with `bidPerGpu`. Given the
  retirement in 1.1 this path is presumably dead (unverified by execution).
- Features declared unsupported in `sky/clouds/runpod.py`: `STOP`, `MULTI_NODE` ("Multi-node
  not supported yet, as the interconnection among nodes are non-trivial on RunPod."),
  `CUSTOM_DISK_TIER`, `CUSTOM_NETWORK_TIER`, `STORAGE_MOUNTING`, `HIGH_AVAILABILITY_CONTROLLERS`,
  `CUSTOM_MULTI_NETWORK`, `LOCAL_DISK`. One network volume only. No use of `/v2/clusters` or
  `globalNetworking` anywhere in these files.
### 2.2 dstack (`master`, 2026-11-09)

Files: `src/dstack/_internal/core/backends/runpod/{compute,api_client}.py`
(https://raw.githubusercontent.com/dstackai/dstack/master/src/dstack/_internal/core/backends/runpod/compute.py),
gpuhunt `src/gpuhunt/providers/runpod.py`.

- Secure Cloud pods: `POST /v2/pods` with `cloud="SECURE"`, `data_center_ids=[offer.region]`,
  `entrypoint=["/bin/sh"]`, `cmd=["-c", ...]`, one port `f"{DSTACK_RUNNER_SSH_PORT}/tcp"`,
  `env={"RUNPOD_POD_USER": "0"}`. Per-GPU CPU and RAM floors are rounded **down** ("rounding
  up may exclude the host the offer is based on").
- Community Cloud pods still use GraphQL `podFindAndDeployOnDemand`: "Deprecated GraphQL API
  is used for Community Cloud, as API v2 does not support country codes". Issue #4400 (opened
  2026-10-21, open) tracks this and the gpuhunt catalogue as the remaining GraphQL users,
  noting GraphQL "has already been rate limited".
- Spot: `raise ComputeError("Runpod spot offers are not supported")`.
- Port/IP discovery (`update_provisioning_data`): `GET /v2/pods/{id}` per pod; returns early
  while `runtime` or `runtime.ports` is null; for cluster members it also waits for
  `cluster.ip` ("May not be assigned yet"); hostname and SSH port come from the entry whose
  `private == DSTACK_RUNNER_SSH_PORT`.
- Clusters (`run_jobs`): `create_cluster(cluster_type="TRAINING", gpu_type_id, gpu_count_per_pod,
  pod_count, data_center_ids=[one], ...)`, then `list_cluster_pods`; if `len(pods) != pod_count`
  it deletes the cluster ("dstack does not track the cluster until run_jobs() returns") and
  logs "Please delete it manually to avoid unexpected charges" if that delete fails.
- Cluster offers come from gpuhunt via GraphQL `gpuTypes(input: {cluster: true})` reading
  `clusterPrice`, `nodeGroupDatacenters`, `lowestPrice{stockStatus,minVcpu,minMemory}` with
  `gpuCount: 8`, and are tagged `pod_counts=list(range(2, 9))`, i.e. the 2..8 limit is a
  constant in the catalogue, not read from RunPod. Comment: "Runpod returns no CPU and memory
  if the offer is out of stock. Out of stock appears to be different from no capacity".
- Orphan cleanup: `list_resource_groups` lists pods and parses an owner reference out of the
  pod **name** (`parse_provider_ref_from_name`), confirming name-as-tag is the workable scheme.
- Other notes in code: `MAX_RESOURCE_NAME_LEN = 60` ("Undocumented but names of len 60 work");
  "Lowercase regions are accepted in the API but they break Runpod in several ways".

### 2.3 runpod-python 2.2.0 and runpodctl (`main`, 2026-11-09)

`runpod/api/ctl_commands.py` calls `/v2/...` and raises `ValueError("REST API v2 does not
support: ...")` for `support_public_ip`, `country_code`, `min_download`, `min_upload`; it has
no cluster helper (https://raw.githubusercontent.com/runpod/runpod-python/main/runpod/api/ctl_commands.py).
runpodctl `pod create` has `--wait`, `--wait-for ssh|running` and `--wait-timeout` (default
`10m`), no spot flag, and no cluster command in its README
(https://raw.githubusercontent.com/runpod/runpodctl/main/cmd/pod/create.go). RunPod's own
wait budget for one pod is ten minutes.

---

## 3. User-reported problems (issue trackers)

| Issue | Date | What happened | Relevance |
|---|---|---|---|
| dstack #3910 | 2026-05-27 | Two-pod H100 cluster failed with "Error creating cluster - cluster creation failed"; looked like stockout, was caused by sending `minCudaVersion: "12.8"`; fixed in PR #3911 by dropping the field | Stockout and bad request are indistinguishable on clusters |
| dstack #1976 | 2024-11-07 | Pods stuck provisioning: RunPod system log "Error response from daemon: layer does not exist"; SSH "Connection refused" reproduced without dstack; separately a no-capacity retry loop on stale offers; an 8x H100 spot pod stuck while on-demand worked | Image-layer and sshd failures after a successful create; needs per-node deadline |
| dstack #1740 | 2024-09-30, closed 2026-04-22 | Many spot offers undeployable at the listed price; error "There are no longer any instances available with the request specifications"; closing note says Secure Cloud availability improved | Catalogue is advisory |
| dstack #4400 / #4167 | 2026-10-21 / 2026-08-19 | GraphQL retirement and rate limiting; v1 retirement | API churn risk |
| SkyPilot #7336 | 2025-09-23 | `Launched instance dfpb14js8oc6z8.` then endless "Waiting for instances to be ready: (0/1)"; pod absent from the dashboard (a spot pod, likely reclaimed at once); closed stale | Unbounded wait; create success does not imply the pod exists a second later |
| SkyPilot #7318 | 2025-09-23 | No auto-recovery after a pod was terminated; cluster stuck `STARTING`; closed stale | State mapping must cover every terminal status |
| SkyPilot #8820, #8158 | 2026-02-08, 2025-12-02 | "Connection timed out during banner exchange"; "Network connection issue during setup, this is likely due to the reboot of the instance"; both closed stale | `RUNNING` with a mapped port does not mean sshd answers |
| SkyPilot #9546; dstack #2993 | 2026-05-06; 2025-08-17 | "Invalid container image name." (PR #9550); "Failed to find data center, please try again." for a Community region on a Secure-only volume | Server-side validation with unhelpful text |
| SkyPilot #8293, #8549, #8592 | see 1.7 | cgroup v1/v2 mix, `chown` EINVAL, FUSE denied | Container environment |
| SkyPilot #10518 | 2026-08-20, open | Vast: `launch()` does `del ports` ("ports is currently unused"); only port 8080 reachable via the SSH gateway; direct mappings require offers with `direct_port_count > 0`; hosts without direct ports report `direct_port_start/end = 65535/-1` and `ports: null` | Vast inbound ports are a per-offer property |

Searches of both trackers for RunPod plus multi-node, `num_nodes`, "instant cluster" or
"global networking" returned no user reports of multi-node RunPod runs (SkyPilot has only
#5344, "Multi-node support error message is not shown", closed 2025-04-29; dstack has #2802,
"Support RunPod Instant Clusters", opened 2025-06-16 and closed 2025-10-29 with no
discussion). The absence is itself a finding: there is essentially no public field experience
of multi-node on RunPod through these tools. RunPod's Discord mirrors and Reddit were not
mined in this sweep.

---

## 4. Other providers: can a fast-fabric cluster be created purely by API?

| Provider | Answer | Call sequence | Capacity signal | Stockout / quota / rate limit |
|---|---|---|---|---|
| **Vast.ai** | TCP overlay yes; RDMA no ("Need RoCE or Infiniband? Submit a cluster request", A100/H100/H200 only) | `search offers 'cluster_id!=None ...'` -> `POST /overlay/` `{cluster_id, name}` (`vastai create overlay CLUSTER_ID NAME`) -> `PUT /asks/{offer_id}/` with `env "-n NAME"` per instance (or `PUT /overlay/` `{name, instance_id}` = `join overlay`). `create cluster SUBNET MANAGER_ID` and `join cluster` are host-side commands. Overlay is Docker templates only; interface is `eth0` | Offer list itself (`rentable`, `rented`, `verified`, `cluster_id`, `direct_port_count`, `inet_down`, `min_bid`, `cuda_vers`, `driver_version`) | Interruptible = `price` (bid) on the create body, `PUT /instances/bid_price/{id}/` to change. No notice period found in the pages read |
| **Lambda** (API 1.10.0) | No. `POST /api/v1/instance-operations/launch` takes `region_name`, `instance_type_name`, `ssh_key_names` (exactly one), `file_system_names`, `user_data` (cloud-init), `tags`, `firewall_rulesets`; there is no cluster field and no `quantity` field. 1-Click Clusters are reserved in the console: "Enter the duration in weeks" | n/a | `GET /api/v1/instance-types` -> `regions_with_capacity_available[]` (boolean per region) | Distinct codes: `instance-operations/launch/insufficient-capacity`, `global/quota-exceeded`, `global/rate-limited` (with `Retry-After`). "one request per second"; launch "one request per 12 seconds, or five requests per minute". Statuses include `preempted` |
| **Nebius** | Yes | `gpu-cluster create --infiniband-fabric <fabric>` -> create each VM with the cluster id ("You can assign a GPU cluster only when creating a VM") | `nebius capacity resource-advice list --parent-id <tenant>`: per (platform, preset, region, **fabric**) with `on_demand`, `preemptible`, `reserved` each giving `available`, `limit`, `availability_level` (`LOW`/`MEDIUM`/`HIGH`/`LIMIT_REACHED`), `data_state`, `effective_at` | `available` vs `limit` separates stock from quota. Preemptible: "`SIGTERM` signal 60 seconds before stopping the VM" |
| **Crusoe** | Yes | `crusoe networking ib-networks list` -> `ib-partitions create --name --ib-network-id` -> create VMs into the partition; "maximum of five InfiniBand partitions on the same InfiniBand network" | Capacities endpoint cited in `07` was not re-opened (unverified) | Spot is "by contacting Crusoe Cloud sales", so not self-serve |
| **Together** | Yes | `GET /v1/compute/regions` (inventory; "contains no prices") -> `POST /v1/compute/clusters` -> poll `GET` (`phase_transitions`, `first_ready_at`, per-node `ib_hca_type`, `ib_hca_count`, `public_ipv4`). Fields include `num_gpus`, `num_preemptible_gpus`, `auto_scale`, `auto_scale_max_gpus`, `capacity_pool_id`, reservation times, and acceptance-test switches (`gpu_burn_duration`, `nccl_single_node_skipped`, `nccl_multi_node_skipped`, `dcgm_diag_skipped`, `storage_skipped`). Non-interactive CLI create requires an existing `--volume` | Regions route; exact response shape not parsed here | Preemptible nodes: cordon, `TogetherPreemptionNotified` event and SIGTERM at T+0, node removed at T+5:00 (hard maximum); replacements are provisioned automatically |
| **SF Compute** (public preview) | Yes | `sf infiniband-partitions create --name --fabric <zone>.fabN` -> `sf subnets create` -> `sf pools create` -> `sf buy --pool --sku --count --start now --duration 4h [--max-rate --yes]` -> per node `sf instances create --pool --sku --infiniband-partition --subnet --private-ip`. SKUs with RDMA type `infiniband`; fabrics do not interconnect | Public orderbook routes: market depth, quote, fills, windows; `Get account limits` | The order is prepaid for its full duration and starts when filled. "SSH becomes available ... which can take up to 10 minutes" |
| **Prime Intellect** | Partly | `GET /api/v1/availability/multi-node` -> `POST /api/v1/pods/` with `cloudId`. `POST /api/v1/slurm-clusters/{team_id}/deploy` (`primeClusterId`, `numNodes`, `gpusPerNode`) and `/resize` exist but only deploy "onto one of the team's dedicated clusters" | `GpuAvailability`: `stockStatus` (`Available|Low|Medium|High|Unavailable`), `provisioningTime` (minutes), `interconnect` (Gbps), `interconnectType`, `isSpot`, `prepaidTime` | Slurm deploy: "'not enough nodes' or a port conflict is an immediate error" |
| **Shadeform** | No | `POST /v1/instances/create` (`cloud`, `region`, `shade_instance_type`, `rental_type: on_demand|spot`, `queue_window{start,end}`); `Instance.cluster_id` is read-only | `GET /v1/instances/types` with `availability[]` per region and rental type; `GET /v1/instances/queue` | `queue_window` turns a stockout into a bounded wait server-side |

Sources: Vast CLI `vast.py` (https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py)
and https://docs.vast.ai/multi-node-training-using-torch-nccl.md; Lambda
https://cloud.lambda.ai/api/v1/openapi.json and https://docs.lambda.ai/public-cloud/1-click-clusters/;
Nebius https://docs.nebius.com/compute/clusters/gpu.md,
https://docs.nebius.com/compute/virtual-machines/capacity-advisor.md,
https://docs.nebius.com/compute/virtual-machines/preemptible.md; Crusoe
https://docs.crusoe.ai/networking/infiniband/managing-infiniband-networks.md and
https://docs.crusoe.ai/usage-billing/spot.md; Together
https://docs.together.ai/reference/clusters-create.md,
https://docs.together.ai/docs/gpu-clusters-management.md,
https://docs.together.ai/docs/preemptible-compute.md; SF Compute
https://docs.sfcompute.com/preview/guides/multi-node-training.md and
https://docs.sfcompute.com/preview/networking/infiniband-partitions.md; Prime Intellect
https://api.primeintellect.ai/openapi.json; Shadeform https://api.shadeform.ai/v1/openapi.json.

---

## Corrections to first sweep

1. **`07` sections 1.2 and 1.7, implication 3 ("Use REST v2 exclusively except for spot
   (v1)") and implication 8.** Outdated. Spot pods are retired platform-wide and REST v1
   stops on 2026-11-15. There is no spot path at all; the 5-second SIGTERM figure no longer
   describes anything `grid` can buy on RunPod.
2. **`07` section 1.2 ("polling must therefore use `GET /v2/pods` list calls, not per-pod
   gets").** Half right. List gives `status` cheaply, but `runtime` (port map, IP) and
   `ssh.direct` are null there. Readiness needs one `GET /v2/pods/{id}` per pod, or for
   clusters `GET /v2/clusters/{id}/pods`, whose runtime population is undocumented.
3. **`07` section 1.3 ("removal is by terminating pods") and the `elastic_remove:
   terminate-pod` flag.** By the v2 spec, cluster members cannot be deleted through pod
   endpoints. Through the API a cluster is fixed-size.
4. **`07` section 1.6 ("attach only at pod creation").** The v2 spec allows mounting, moving
   and unmounting network and global volumes after creation; the migration guide disagrees.
   Mark as contested, not as fact.
5. **`07` section 1.2 table ("Python SDK ... Thin wrappers").** runpod-python 2.2.0 is now a
   v2 client that rejects the removed fields; runpodctl's reference no longer has any bid flag.
6. **`07` spend limit.** Still $80/hour by default, but limits also "increase automatically
   over time based on your account history"; support is not the only route.
7. **`07` comparison table, SF Compute row.** The `h100i`/`h100v`, Kubernetes-by-default
   description is superseded by SKUs, pools, orders, instances, subnets and InfiniBand
   partitions (public preview).
8. **`07` comparison table, Together row ("reclaim any time").** Preemptible nodes get a
   drain of up to five minutes.
9. **`07` comparison table, Nebius row.** Preemptible notice is documented (60 s). The
   capacity advisor reports `preemptible` availability per InfiniBand fabric, which suggests
   but does not prove that preemptible VMs can join a gpu-cluster.
10. **`07` Lambda row.** Rate limits confirmed from the current spec (1 req/s; launch 1 per
    12 s or 5 per minute). The launch body has no `quantity`, so N instances need N launch
    calls: at least 12 s per node.
11. **`02` ("SkyPilot's RunPod loop ...", "dstack calls `create_cluster`" described against
    the SDK/GraphQL).** Both are now REST v2 clients. SkyPilot's `MULTI_NODE` gate is
    unchanged and no flag lifts it; it uses neither clusters nor global networking. dstack's
    2..8 limit is a constant in gpuhunt.
12. **`07` section 1.5 / `09` open question on FUSE.** SkyPilot #8592 contains no RunPod
    statement and was closed by a bot; cite it as a user report only.

---

## Implications for grid

### Provider-driver interface (sketch)

Operations, each justified by something a spec forces:

| Operation | Why it must exist |
|---|---|
| `capabilities() -> Caps` | Static per driver, refined per offer (table below) |
| `account_preflight() -> {balance?, spend_cap_per_h?, rate_policy?, quotas?}` | RunPod `402` and spend cap; Lambda `global/quota-exceeded`; Nebius `limit`; SF Compute account limits. Fields are optional because RunPod v2 exposes neither balance nor cap |
| `list_offers(filter) -> [Offer{sku, location, fabric?, price, stock: level|count|bool, quota?, provisioning_eta?, max_gpus_per_node, cuda/driver, inbound_ports?}]` | RunPod enum per data center and product; Nebius count plus limit per fabric; Lambda boolean; Vast and Prime Intellect offer rows; SF Compute order book |
| `create_fabric(spec) -> FabricHandle` / `delete_fabric` | Nebius gpu-cluster, Crusoe and SF Compute IB partitions, Vast overlay. No-op on RunPod pods; folded into `create_group` on RunPod and Together |
| `create_node(offer, fabric?, launch_spec, owner_tag) -> NodeHandle` | One call per node on RunPod pods, Lambda, Vast, Nebius, Crusoe, SF Compute |
| `create_group(offer, n, launch_spec, owner_tag) -> GroupHandle` | Atomic N-node products: RunPod `/v2/clusters`, Together clusters |
| `list_owned(owner_prefix) -> [NodeSummary{id, name, status}]` | Cheap bulk status; must paginate (RunPod 1000 per page) and include group members (`includeClusterPods=true`) |
| `describe_node(id) -> NodeDetail{status, public_ip, port_map, private_ip, fabric_ip, rank_hint, vcpus, cost}` | RunPod's list/get split; `cluster.ip` and `rank` arrive late and are nullable |
| `tail_logs(id, since)` (optional) | RunPod SSE logs expose image-pull and container failures such as "layer does not exist" |
| `terminate_node(id)`, `terminate_group(id)` | Must treat not-found as success (dstack does); RunPod sometimes returns `400` for not-found |
| `resize_group(id, n)` (optional) | Together, Prime Intellect; absent on RunPod |
| `watch_preemption(id)` (optional) | Together event, Nebius SIGTERM, Lambda `preempted`, Shadeform `status_details`; absent on RunPod |

Launch spec must carry both an exec-form entrypoint/cmd and a cloud-init/user-data form,
because container providers take the first (RunPod `entrypoint`/`cmd`, Vast `onstart`) and VM
providers the second (Lambda `user_data`, SF Compute `--cloud-init`).

Capability flags added or sharpened by this sweep (extending the table in `07`):

| Flag | Values seen |
|---|---|
| `stockout_signal` | `coded` (Lambda), `count_and_quota` (Nebius), `ambiguous_400` (RunPod), `offer_disappears` (Vast), `queueable` (Shadeform `queue_window`), `order_unfilled` (SF Compute) |
| `ownership_marker` | `name_only` (RunPod: tags read-only via API), `tags` (Lambda, Shadeform), `label` (Vast) |
| `bulk_status` / `bulk_ports` | RunPod: yes / no |
| `group_atomic`, `group_resizable_api`, `group_member_deletable` | RunPod: unknown-partial / no / no; Together: yes / yes / n/a |
| `launch_rate` | RunPod: shared per-user windows from headers; Lambda: 5 per minute |
| `host_identity` | RunPod v2: none (GraphQL only until 2027); Vast: `machine_id` |
| `cpu_count_trustworthy_in_guest` | RunPod: false (cgroup v1/v2 mix) |
| `spot` | RunPod: none; Vast: bid; Nebius: 60 s; Together: 300 s drain; Lambda: status only; Crusoe: sales |
| `api_stability` | RunPod: v1 ends 2026-11-15, GraphQL early 2027 |

Error taxonomy every driver maps into:

| Class | Controller action | RunPod v2 | Lambda |
|---|---|---|---|
| `Stockout` | next candidate; cool down this (sku, location) | `400` on create when other candidates succeed | `instance-operations/launch/insufficient-capacity` |
| `InvalidRequest` | fail the launch, never retry | `422`; `400` on every candidate | `global/invalid-parameters` |
| `QuotaOrSpend` | stop, surface to the user | `402`; spend cap (shape unknown) | `global/quota-exceeded` |
| `Forbidden` | skip candidate | `403` | `global/forbidden`, `global/account-inactive` |
| `RateLimited` | sleep `Retry-After`; safe to resend POST | `429` | `global/rate-limited` |
| `Transient` | backoff; for POST, **list first** | `5xx`, timeouts | `global/internal-error`, `provider/*` |
| `Conflict` | re-read state | `409` | `global/conflict` |
| `NotFound` | success for deletes | `404`, sometimes `400` | `global/object-does-not-exist` |
| `Unknown400` | RunPod only: classify by comparing against a canary candidate | - | - |

Design consequences:

1. **RunPod stockout classification must be empirical.** Keep a per-launch tally: a `400`
   that differs across candidates is `Stockout`; a `400` identical on all is `InvalidRequest`.
   Validate request shape once against a cheap, plentiful candidate before a large launch.
   Never send `gpu.minCudaVersion`-style filters to clusters (dstack #3910).
2. **Creates are non-idempotent and names are the only tag.** Use a deterministic name
   (`grid-<run>-<slot>-<attempt>`, at most 60 characters, uppercase data-center ids) and, after
   any timeout on POST, list by name before retrying, as SkyPilot's "POST is only repeated on
   429" rule implies.
3. **Two-tier polling.** One paginated list per tick for status, then `describe_node` only
   for pods that just turned `RUNNING`, until `port_map` is non-null. Budget both against the
   shared minute window read from `RateLimit`.
4. **Every wait gets a deadline.** RunPod's own CLI uses 10 minutes; SkyPilot's unbounded loop
   produced #7336. A node that is `RUNNING` with ports but whose agent has not phoned home
   within the budget is culled (`09`).
5. **Treat a RunPod cluster as an immutable island.** No grow or shrink by API, no private
   image except via template, one volume. Repair means a new island or pods outside it.
6. **Do not build on GraphQL-only data** (host id, balance, Community country placement,
   cluster price and `nodeGroupDatacenters`) beyond early 2027; isolate any such call behind
   an optional capability.
7. **Derive CPU counts from the API, not the guest**; avoid `chown`-dependent file sync.
8. **Borrow Shadeform's `queue_window`**: a bounded standing request per candidate instead of
   a tight retry loop that burns rate limit.

## Still open (needs hands-on experiment)

| Gap | Probe on real pods |
|---|---|
| Does `GET /v2/pods` really return `runtime: null`, and does `GET /v2/clusters/{id}/pods` include `runtime.ports`? | Create one pod and one 2-node cluster; diff list vs get |
| Actual `RateLimit-Policy` for a normal account; are headers present on cluster routes | Read headers on each route |
| Exact `detail` strings for stockout vs rule violation vs spend-cap breach vs concurrent-pod cap | Provoke each; record status and `detail` |
| Is `POST /v2/clusters` all-or-nothing; time from `201` to all `cluster.ip` and `rank` assigned; behaviour at `podCount` 9+ without sales | Create at 2, 8, 9 nodes; poll `pods.byStatus` |
| Can a cluster member be terminated (expect `409`); what happens to rank and `WORLD_SIZE` after console scale-up | Try `DELETE /v2/pods/{member}`; scale in console |
| Post-create volume attach (`UpdatePodMounts`) vs "fixed at creation" | `PATCH` a running pod to add and remove a network volume |
| Does v2 accept symmetric ports (`"70000/tcp"`) and how many per pod; do mappings survive `restart` | Create with 70000..70015; restart; compare |
| Capabilities: `capsh --print`, `/dev/fuse`, `/dev/net/tun`, `ulimit -l`/`-n`, cgroup version, `/dev/shm` size, `dmesg`, DCGM; Secure vs Community vs cluster member | T0 probe script across several data centers |
| Time to `RUNNING`, to port map non-null, to sshd answering, by image size and data center | Timed launches with a phone-home agent |
| Is GraphQL `podRentInterruptable` truly refused now (SkyPilot still calls it) | One call with a bid |
| Global networking and public-IP TCP throughput between pods | `iperf3` over `*.runpod.internal` and over mapped ports |
| Vast overlay: bandwidth on `eth0`, how many hosts per `cluster_id` are rentable at once, interruptible notice | Rent two hosts in one cluster; measure; outbid one |
| Together and SF Compute end-to-end create latency and failure shape on no capacity | One small on-demand cluster each |
| Nebius: can a preemptible VM be created with a gpu-cluster id | One create call |

Not covered: RunPod Discord/Reddit, Voltage Park, CoreWeave, hyperscalers, Crusoe's capacities endpoint, and the response schema of Together's regions route.
