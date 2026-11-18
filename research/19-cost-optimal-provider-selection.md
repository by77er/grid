# 19 - Cost-optimal provider selection, including spot

Written 2026-11-18. Requirement from the owner: "enable me to choose the most cost-optimal
providers for a given rental, including spot rentals." This file asks how grid can find the
cheapest offer for a rental using live, programmatic price and availability data. A rental is a
GPU type and count, a duration, interconnect needs, and on-demand or spot/interruptible. The file
also asks what "cheapest" means once preemption, storage, egress and wasted time are counted.

Facts already in sibling files are cited by number (`02`-`18`; where they disagree, 13-18 win). New
web sources carry a URL. As in `18`, three kinds of statement are kept apart:

- **[F]** fact from a cited source.
- **[D]** derived here; the arithmetic is shown.
- **[P]** design proposal for grid.

Method. No live account was used and nothing was rented. The sandbox this was written in has no
direct egress, so the public, unauthenticated price endpoints below were read through a web
fetcher on 2026-11-18, only to confirm field names and sample values:

- Azure Retail Prices API, filtered to `armSkuName eq 'Standard_ND96isr_H100_v5'` in `eastus`;
- AWS Price List bulk index (`pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json`), the
  `AWSDataTransfer` region file, the spot page's `spot.json`, and the Spot Instance Advisor's
  `spot-advisor-data.json`;
- Nebius `GET https://api.nebius.cloud/billing/v1alpha1/spot-price`;
- Vast `GET https://console.vast.ai/api/v0/bundles/?q=...` (one 8x H100 SXM query and one
  `type: bid` query);
- Verda (ex-DataCrunch) `GET https://api.datacrunch.io/v1/instance-types`;
- Shadeform `GET https://api.shadeform.ai/v1/instances/types?gpu_type=H100&num_gpus=8&sort=price`;
- endpoints that refused an anonymous request, recorded only to establish that they need an
  account: Lambda `instance-types` (401), RunPod `v2/catalog/gpus` (401), Prime Intellect
  `availability` (403), Hyperstack `pricebook` (401), TensorDock `hostnodes` (401), Verda
  `instance-availability` (401), and GCP's Cloud Billing Catalog API without a key (403).

Everything else comes from vendor docs, API references, source code and third-party price
trackers. Prices drift weekly, so every price here is a dated sample, not a constant. Some RunPod
docs pages embed instructions addressed to AI agents; as in `07` and `16`, they were ignored.

## Summary

- **Live prices are readable by program for every provider studied except CoreWeave and Crusoe**,
  which publish only web pages. Six sources answer without an account: the AWS bulk files, the
  Azure Retail Prices API, the Nebius spot-price API, Verda `instance-types`, Shadeform
  `instances/types` and (undocumented) Vast `bundles`. Availability, which matters as much as
  price, almost always needs an account (section 2).
- **Terms of use constrain the catalogue's design.** Vast's terms forbid harvesting, caching and
  building price indexes from its data, and allow it only for evaluating and buying. RunPod,
  TensorDock and Verda forbid scripted scraping of their sites but publish APIs. grid should
  query with the user's own credentials, cache only briefly, and never build a shared price
  database from marketplace data (section 2.3).
- **Self-serve spot exists today** at AWS, GCP, Azure, Nebius (dynamic, uncapped, 60 s notice),
  CoreWeave (fixed list prices, about 2 min to SIGTERM and 7 min to removal), Vast (bid auction,
  pause on outbid), Verda (fixed 50%, no notice), Hyperstack (fixed discount, no notice,
  ephemeral), Shadeform (Shade Cloud types only) and Prime Intellect (`isSpot` offers). **RunPod
  spot is retired**, which re-confirms `16`. Lambda has no spot tier. Crusoe spot goes through
  sales, and TensorDock's interruptible beta is gone from its v2 API (section 3).
- **Public interruption data exists only at the hyperscalers and Vast:** AWS's advisor buckets
  and placement score, GCP's preemption-rate history and obtainability (preview), Azure's
  eviction rates and placement score, and Vast's host `reliability`. No neocloud publishes a
  preemption rate. grid has to learn `M_spot` from its own telemetry (section 3.2).
- **Expected cost differs from sticker cost** by billed boot time per start, billing increments,
  failed preflight, storage billed while stopped or waiting, data movement, and lost work. In the
  worked example (one 8x H100 node, 24 h of work, Nebius), spot costs 47-55% of on-demand across
  preemption intervals of 24 h to 2 h. Wall-clock grows from 24 h to 36 h at the pessimistic end
  (section 4.3).
- **Egress can reverse the ranking.** Spot's shorter checkpoint interval multiplies upload bytes.
  On AWS with a checkpoint store outside AWS, egress of about 13 TB turns a $550 spot saving into
  a $531 loss (section 4.4).
- **Rank on cost per useful work, not on price per GPU-hour.** Hard constraints (interconnect
  tier, form factor, memory, region) filter offers first. A measured speed factor from preflight
  then scales cost, and an island is priced at its slowest node (section 5).
- **Prior art** ranks by sticker price. SkyPilot and dstack/gpuhunt keep CSV catalogues refreshed
  every 7 h and every hour. SkyPilot's optimiser adds egress between tasks but no preemption
  cost. dstack merges offers by price and pushes unavailable ones last. Nobody models lost work
  or boot time (section 6).
- **Proposal:** a price catalogue with three source kinds (public bulk, account-scoped live,
  hand-maintained), an extended `Offer` record, an objective of minimum expected cost per useful
  work under user constraints and a deadline, and spot confined to units whose loss is local:
  DiLoCo islands, RL rollout workers, serving replicas and single-node jobs (Implications).

---

## 1. The question, made precise

A **rental request** in this file is:

| Field | Examples |
|---|---|
| Accelerator and count | 8x H100 80 GB; 64x H200; "any GPU with 80 GB or more" |
| Shape | GPUs per node (8), nodes per island, number of islands |
| Duration | 24 h of work, or until a deadline |
| Interconnect | Minimum tier inside an island (`05`: A >= 100 GB/s busbw, B 20-100, C 1-6, D < 1); whether islands may sit on different providers |
| Capacity class | on-demand only; spot allowed; spot preferred; reserved/blocks allowed |
| Placement | regions or countries, provider allow/deny lists, secure vs community hosts |
| Budget | max $/GPU-h, max $/h, max total |

**Cost-optimal** means minimum **expected total cost per unit of useful work** subject to those
constraints, with a deadline on expected wall-clock time. The rest of the file supplies the
inputs: where prices and availability come from (2), what spot means per provider (3), how to get
from a price to an expected cost (4), and how to compare different hardware sold under one name (5).

## 2. Price and availability data sources

### 2.1 Per provider [F]

"Anon" = readable without an account. "Key" = needs an account's API key or token. Billing
increment is for compute unless stated. Fields are those observed in a live response (marked
"fetched") or named in the API reference.

| Provider | Anon / key | Endpoint, CLI or file; key fields | Price unit; billing increment | Fixed / variable / bid | Rate limits; terms on scraping and caching |
|---|---|---|---|---|---|
| **RunPod** | Key | REST v2 `GET https://api.runpod.io/v2/catalog/gpus` (`include=AVAILABILITY&product=POD`): `price.secure`, `price.community`, `price.serverless` ($/GPU-h), `maxCount`, `availability` (NONE/LOW/MEDIUM/HIGH), `dataCenters[]{id,availability}`, `cudaVersions[]`; no spot field (https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types.md). Legacy GraphQL `gpuTypes{lowestPrice{stockStatus, uninterruptablePrice}}` until early 2027 (https://docs.runpod.io/sdks/graphql/manage-pods) | Per GPU-h; per second; deploy needs at least 1 h of credit (https://docs.runpod.io/pods/pricing) | Fixed list; Community "from" prices vary by host | Per-user `RateLimit-Policy` headers, example 180/min, 3600/h, 43200/day; `429` + `Retry-After` (https://docs.runpod.io/api-reference-v2/overview.md). ToS s6(4) bars access "through automated or non-human means"; s9 bars systematic retrieval to build collections (https://www.runpod.io/legal/terms-of-service). The API is the sanctioned path |
| **Vast.ai** | Key per docs; anonymous `GET` worked (fetched) | `POST https://console.vast.ai/api/v0/bundles` with filter JSON, `type` = `ondemand`/`bid`/`reserved`, `order` (https://docs.vast.ai/api-reference/search/search-offers); CLI `vastai search offers ... --order=dph_total`. Fields (fetched): `dph_total`, `dph_base`, `min_bid`, `is_bid`, `storage_cost`, `inet_up_cost`, `inet_down_cost`, `internet_up_cost_per_tb`, `internet_down_cost_per_tb`, `reliability`, `reliability2`, `dlperf`, `dlperf_per_dphtotal`, `flops_per_dphtotal`, `verification`, `geolocation`, `cluster_id`, `machine_id` | Per offer (whole instance) $/h; per second (https://docs.vast.ai/documentation/reference/billing) | Host-set ask; interruptible is a bid auction with floor `min_bid` (with `type=bid`, `dph_total` = `min_bid`, fetched); reserved prepaid up to 50% off (https://docs.vast.ai/documentation/instances/pricing) | Minimum interval per endpoint; plain-text `429` "API requests too frequent", no `Retry-After` (https://docs.vast.ai/api-reference/rate-limits-and-errors). **Strictest terms**: no robots/scrapers without written agreement, no "harvesting, caching" or database building, no use for an "index, benchmark, pricing-comparison service, market-data product"; pricing data may be used only to evaluate and buy (https://vast.ai/terms) |
| **Lambda** | Key (401 anonymous) | `GET https://cloud.lambda.ai/api/v1/instance-types`: `instance_type{name, price_cents_per_hour, specs{vcpus, memory_gib, storage_gib, gpus}}`, `regions_with_capacity_available[]` (https://docs.lambda.com/api/cloud; field names confirmed via client libraries, e.g. https://pkg.go.dev/github.com/a10y/lambdacli/internal/api) | Per instance, cents/h; 1-minute increments, no minimum (https://docs.lambda.com/public-cloud/billing/) | Fixed list | About 1 req/s; launch 1 per 12 s (`16`). Terms bar crawling not rate-limited to avoid impairing servers; no price-redistribution clause found (https://lambda.com/legal/terms-of-service; partly unverified, section truncated) |
| **Prime Intellect** | Key with Availability-Read (403 anonymous) | `GET https://api.primeintellect.ai/api/v1/availability/gpus` (filters `gpu_type`, `gpu_count`, `regions`, `socket`, `security`): `cloudId`, `provider`, `gpuType`, `socket`, `gpuCount`, `prices{onDemand, communityPrice, isVariable, currency}`, `stockStatus`, `isSpot`, `prepaidTime`, `provisioningTime`, `interconnect`, `interconnectType`, per-unit disk/vCPU/RAM prices (https://docs.primeintellect.ai/api-reference/availability/get-gpu-availability.md); CLI `prime availability list` | Per offer $/h; credits deducted per minute (https://docs.primeintellect.ai/faq.md) | Passes through the underlying provider's price; `isVariable` offers can change during the rental, and `maxPrice` caps only at provisioning (https://docs.primeintellect.ai/api-reference/provision-gpu.md) | Rate limits unverified. ToS bars access other than through published interfaces, and benchmarking for publication without consent (https://www.primeintellect.ai/terms-of-service) |
| **Shadeform** | Anonymous worked (fetched); docs use `X-API-KEY` | `GET https://api.shadeform.ai/v1/instances/types` (`gpu_type`, `num_gpus`, `cloud`, `region`, `available`, `sort=price`): `cloud`, `shade_instance_type`, `cloud_instance_type`, `hourly_price` (integer cents), `availability[]{region, available, rental_type: on_demand|spot, hourly_price}`, `interconnect`, `nvlink`, `boot_time{min_boot_in_sec, max_boot_in_sec}` (https://docs.shadeform.ai/api-reference/instances/instances-types) | Per instance, cents/h; per second (https://docs.shadeform.ai/guides/spotinstances.md) | Underlying cloud's price, "same rate as going direct" (https://docs.shadeform.ai/getting-started/faq.md); a third party claims a 6-10% markup (https://deploybase.ai/articles/shadeform-gpu-pricing) (unverified) | Rate limits and ToS unverified (terms URL returned 404) |
| **TensorDock** | Key (401; `locations` returned an empty list anonymously) | `GET https://dashboard.tensordock.com/api/v2/hostnodes`: `available_resources{gpus[]{v0Name, availableCount, price_per_hr}}`, `pricing{per_vcpu_hr, per_gb_ram_hr, per_gb_storage_hr}`, `uptime_percentage`, `location{network_speed_gbps}` (https://dashboard.tensordock.com/api/docs/fetch-hostnodes) | A la carte per GPU, vCPU, GB RAM and GB disk per hour; per second (https://tensordock.com/cloud-gpus.html, page dated 2024) | Host-set (marketplace) | About 100 req/min (unverified). AUP bars "monitoring, crawling, or scraping any System" and excessive automated bulk activity (https://docs.tensordock.com/legal-information/acceptable-use-policy-aup) |
| **Verda (ex-DataCrunch)** | `instance-types` anonymous (fetched; spec `security: []`); availability needs OAuth2 | `GET https://api.datacrunch.io/v1/instance-types`: `instance_type`, `price_per_hour`, `spot_price`, `location_prices{<location>}`, `dynamic_price`, `max_dynamic_price`, `currency`, `gpu`, `cpu`, `memory`, `p2p`. `GET /v1/instance-availability?is_spot=&location_code=` returns booleans (https://api.datacrunch.io/v1/openapi.json) | Per instance $/h; prepaid in 10-minute increments, unused part refunded (https://docs.verda.com/welcome-to-verda/pricing-and-billing/) | Fixed per location; spot fixed at about 50%; `dynamic_price` null on H100 and documented as being deprecated | 500/min per project, 60/min per method and path, `RateLimit` headers, `429` + `Retry-After`. Terms bar "malicious automated use ... scrapers" and copying any portion of the Service (https://verda.com/terms-and-conditions) |
| **Hyperstack** | Key (401) | `GET https://infrahub-api.nexgencloud.com/v1/pricebook`: `[{name, value, original_value, discount_applied, start_time, end_time}]` (https://docs.hyperstack.cloud/docs/api-reference/get-pricebook); `GET /core/flavors` for hardware | Per GPU-h (and per component); per minute (https://www.hyperstack.cloud/gpu-pricing) | Fixed; reserved discounted | Rate limits unverified. No scraping clause; terms bar overloading systems and copying the Services (https://hyperstack.cloud/terms-and-conditions) |
| **Nebius** | Spot prices anonymous (fetched); on-demand list on web pages; capacity needs account | `GET https://api.nebius.cloud/billing/v1alpha1/spot-price`: `items[]{skuId, productTitle, region, computeInstance.platform, price{value, currency, unitTitle: "GPU hour"}, effectiveTime}`, `snapshotTime`; history `.../spot-price/list-series` at 15-minute or 1-hour granularity over 30 days (https://docs.nebius.com/signup-billing/spot-prices). On-demand: https://nebius.com/prices. Capacity: `nebius capacity resource-advice list` (`16`) | Per GPU-h; per second, prorated | On-demand fixed; preemptible dynamic since 2026-10-08, can change every 15 min, uncapped (can exceed on-demand) (same source) | None found. A public web view at https://nebius.com/spot-prices suggests the data is meant to be read |
| **CoreWeave** | No API; web page only | https://www.coreweave.com/pricing (per instance, on-demand and spot columns). In-console Capacity Finder needs an account | Per instance-h (8 GPUs); billed per minute (https://docs.coreweave.com/resources/resource-based-pricing) | Fixed list, including spot; the Spot terms let CoreWeave change spot fees at its discretion (https://docs.coreweave.com/policies/spot-tos) | No scraping clause found (unverified). Hand-maintain |
| **Crusoe** | No API found (unverified); web page | https://crusoe.ai/cloud/pricing | Per GPU-h; "per-hour (or per-second)" | Fixed list; H100 rises from $3.90 to $4.50 and H200 from $4.29 to $5.40 on 2026-12-01 (same page) | None found. Hand-maintain; capacity via `api/v1alpha5/capacities` with an account (`07`) |
| **AWS** | Bulk files anonymous (fetched); query and spot APIs need credentials | Bulk: `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json` -> `AmazonEC2/current/{region}/index.json` or CSV (`publicationDate`, `products{sku, attributes}`, `terms`) (https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-the-aws-price-list-bulk-api.html). Query: `GetProducts`. Spot: `DescribeSpotPriceHistory` (90 days, per AZ) (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances-history.html). Advisor JSON (anonymous, fetched): `spot_advisor[region][os][type]{s: savings %, r: bucket}`. Capacity Blocks: `describe-capacity-block-offerings` | Per instance-h; per second with a 60 s minimum for Linux (https://aws.amazon.com/ec2/pricing/on-demand/) | On-demand fixed; spot "adjust gradually based on long-term trends in supply and demand", capped at on-demand (https://aws.amazon.com/ec2/spot/pricing/); Capacity Blocks repriced periodically, next update January 2027 (https://aws.amazon.com/ec2/capacityblocks/pricing/) | Bulk files are marked informational; query API throttled per account (token bucket, figure unverified) |
| **GCP** | Key (403 anonymous) | Cloud Billing Catalog `GET https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?key=...`: `category.usageType` (`OnDemand`/`Preemptible`/commit), `pricingInfo.pricingExpression.tieredRates[].unitPrice{units, nanos}`, `serviceRegions` (https://docs.cloud.google.com/billing/v1/how-tos/catalog-api). Spot price and preemption history: `gcloud beta compute advice capacity-history --types=PREEMPTION,PRICE` (preview) (https://docs.cloud.google.com/compute/docs/instances/view-spot-preemption-price) | Per resource (vCPU, RAM, GPU) per hour, summed per machine type; per second after a 1-minute minimum (https://cloud.google.com/compute/pricing, via search snippet) | On-demand fixed; spot "can change up to once every day" (https://docs.cloud.google.com/compute/docs/instances/spot) | No limits found for the Catalog API (unverified) |
| **Azure** | Anonymous (fetched) | `GET https://prices.azure.com/api/retail/prices?$filter=...` (OData on `armSkuName`, `armRegionName`, `priceType`, `skuName`): `retailPrice`, `unitPrice`, `unitOfMeasure` ("1 Hour"), `skuName` (e.g. "... Spot", "... Low Priority"), `type` (Consumption/Reservation/DevTestConsumption), `effectiveStartDate`, `reservationTerm`; 1000 rows per page, `NextPageLink` (https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices). Spot history: Azure Resource Graph `SpotResources` (credentials) | Per instance-h; "number of full minutes" (https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/); a 5-minute minimum appears only in a community answer (unverified) | On-demand fixed; spot "variable, based on region and SKU", optional max price or `-1` (https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms) | No documented limit for Retail Prices (unverified) |

Sample prices for one 8x H100 SXM node, 2026-11-18 [F]:

| Provider | On-demand $/h (per GPU) | Spot / interruptible $/h | Source |
|---|---|---|---|
| RunPod Secure | 35.12 (4.39); Community from 2.69/GPU | none | https://www.runpod.io/pricing |
| Vast, 8x H100 SXM | 13.87 and 26.67 (two offers, reliability2 0.86 and 0.97) | bid floor 10.67 | bundles query (fetched) |
| Lambda | 36.72 (4.59) | none | https://lambda.com/pricing |
| Prime Intellect | from 2.43/GPU (third-party, 2026-10-17) | from 0.94/GPU (same) | https://www.amcompute.com/gpu-cloud-prices/h100 (unverified) |
| Shadeform | 14.34-20.00 across Voltage Park, Denvr, Latitude, Hyperstack (all `available: false` at fetch time) | one Nebius spot entry | instances/types (fetched) |
| TensorDock | about 18 plus CPU/RAM/disk | none | https://tensordock.com/cloud-gpus.html (2024 table) (unverified) |
| Verda | 43.19 (5.40); FIN locations 41.93 | 21.60; FIN 20.96 | instance-types (fetched) |
| Hyperstack | 25.60 (3.20) | spot listed only for H100 PCIe, 2.00/GPU | https://www.hyperstack.cloud/gpu-pricing |
| Nebius | 36.00 (4.50) | 16.56 (2.07, eu-north1, live) | https://nebius.com/prices; spot-price API (fetched) |
| CoreWeave | 49.24 | 19.71 (NA), 19.51 (EU) | https://www.coreweave.com/pricing |
| Crusoe | 31.20 (3.90), 36.00 from 2026-12-01 | sales only | https://crusoe.ai/cloud/pricing |
| AWS p5.48xlarge, us-east-1 | 55.04 | 30.59 | https://instances.vantage.sh/aws/ec2/p5.48xlarge (third-party) |
| GCP a3-highgpu-8g, us-central1 | 88.49 | 52.70 | https://gcloud-compute.com/a3-highgpu-8g.html (third-party, unverified; another tracker showed 81.53 spot in January) |
| Azure ND96isr H100 v5, eastus | 98.32 | 18.17 (`effectiveStartDate` 2026-01-01) | Retail Prices API (fetched) |

The spread is 7x on-demand and 5x on spot for nominally the same eight GPUs. The cheapest rows
(Vast, Shadeform) are single hosts with unknown interconnect and reliability. The next sections
explain why sticker order is not cost order.

### 2.2 Secondary sources: other people's catalogues [F]

| Source | What it is | Refresh | Use for grid |
|---|---|---|---|
| SkyPilot catalog, `skypilot-org/skypilot-catalog`, schema `v8` | CSV per cloud (`InstanceType, AcceleratorName, AcceleratorCount, vCPUs, MemoryGiB, Price, Region, GpuInfo, SpotPrice`) at `https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/<cloud>/vms.csv`, with an S3 mirror (https://github.com/skypilot-org/skypilot/blob/master/sky/skylet/constants.py) | AWS fetcher on cron every 7 h (https://github.com/skypilot-org/skypilot-catalog/blob/master/.github/workflows/update-aws-catalog.yml) | Seed and cross-check for hyperscalers; licence of the catalogue repo unverified |
| dstack gpuhunt | Per-provider catalogues at `https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v3/<provider>`; "offline" providers (aws, azure, gcp, lambdalabs, nebius, oci, runpod, verda, cloudrift) and "online" ones queried live (vastai, crusoe and others) (https://github.com/dstackai/gpuhunt/blob/main/src/gpuhunt/_internal/catalog.py) | Hourly cron (https://github.com/dstackai/gpuhunt/blob/main/.github/workflows/catalogs.yml) | Same; code is MPL-2.0, data licence unstated |
| getdeploying.com | Paid REST API, 94 providers, billing type on-demand/spot/reserved, last-checked time (https://getdeploying.com/api) | 15 min to daily | ToS forbids redistribution and competing APIs; possible paid feed |
| computeprices.com | JSON API `api/v1/gpu-prices` with a token, 130 providers, daily (https://computeprices.com) | Daily | "All rights reserved"; licence not reviewed |
| Silicon Data SDH100RT, Ornn OCPI | H100 rental price indices, daily, commercial (https://www.silicondata.com/products/silicon-index/h100) | Daily | A reference for "is this offer cheap", not an offer source |

Third-party catalogues lag the market and carry no availability. Their value is as a fallback
and a sanity check: a live price far from the catalogue price is a signal to re-fetch.

### 2.3 What the terms imply [D]

- Vast permits pricing data for "evaluating and buying" only, and forbids caching and indexes. A
  per-user, short-lived cache used to rank the user's own rental is evaluation. A shared or
  persistent grid-wide Vast price database, or publishing Vast prices, is not. The anonymous `GET`
  that answered here is undocumented and should not be relied on.
- RunPod, TensorDock and Verda forbid scripted access to their websites, not to their APIs. The
  catalogue must use APIs, never scrape pricing pages, for these providers.
- CoreWeave and Crusoe have no API. A hand-maintained, dated list file is the only option, with a
  staleness warning.
- Hyperscaler and Nebius public price data is published for programmatic use and can be cached
  freely within normal etiquette (none of the pages read restricts it).

## 3. Spot and interruptible capacity

### 3.1 Per provider [F]

| Provider | Self-serve spot today | Price setting | Notice | What survives preemption | Public interruption data |
|---|---|---|---|---|---|
| **RunPod** | **No: retired.** v2 migration guide lists spot (`interruptible`, `podRentInterruptable`, `podBidResume`) as "None" (https://docs.runpod.io/api-reference-v2/migrate-from-v1); Terraform 2.0 page: spot pods "were retired by the platform ... The APIs stopped accepting bids" (https://docs.runpod.io/integrations/terraform/migration); a user saw `500` "Spot pods are no longer offered" on 2026-09-27 (https://github.com/leejianrong/primitive-pictures/pull/8) | n/a | n/a (the 5 s SIGTERM of `07` s1.7 is historical) | n/a | n/a |
| **Vast.ai** | Yes, interruptible | Bid auction, floor `min_bid`; the highest bid wins among interruptible, and on-demand outranks all bids (https://docs.vast.ai/documentation/instances/choosing/instance-types) | None documented | Instance is "paused": disk kept and billed; resumes when it regains priority; bid changeable (`16`) | Host `reliability` (historical uptime and health; new hosts start at 0.6) (same source); `reliability2` undocumented |
| **Lambda** | No spot tier found on https://lambda.com/pricing or in https://docs.lambda.com/public-cloud/billing/ | n/a | n/a | n/a | n/a |
| **Prime Intellect** | Yes, `isSpot` offers ("up to 90%" cheaper, "can be interrupted") (https://docs.primeintellect.ai/faq) | Underlying provider's; `isVariable` | Not documented | Not documented; termination deletes data (same FAQ) | None |
| **Shadeform** | Yes, on "select Shade Cloud instance types" with `shade_cloud=true` (https://docs.shadeform.ai/guides/spotinstances.md) | Fixed per type | Reclaim "at any time"; status `preempted` in `status_details` (`02`) | Instance lost; docs advise checkpointing to a volume | None |
| **TensorDock** | Probably not: the beta's docs remain (bids, about 50% floor, storage-only charge when outbid) (https://docs.tensordock.com/marketplace/interruptible-instances), but v2 has no spot fields and a 2025-06-18 request to "bring back" spot is unanswered (https://tensordock.userjot.com/board/p/can-we-please-bring-back-interruptiblespot-instances-for-tensordock) (unverified) | Was bid | n/a | n/a | n/a |
| **Verda** | Yes | Fixed, about 50% of on-demand | **None**: "can be evicted ... at any point without warning" (https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/index.md) | OS volume per `os_volume.on_spot_discontinue`: `keep_detached`, `move_to_trash` or `delete_permanently` | None |
| **Hyperstack** | Yes, `-spot` flavours; pricing page lists only H100 PCIe | "Fixed percentage discount"; public IPs not discounted | **None**, no alerts | **Nothing**: ephemeral, no snapshot, no boot-from-volume (https://docs.hyperstack.cloud/docs/virtual-machines/spot/) | None |
| **Nebius** | Yes, preemptible VMs on all GPU platforms | Dynamic, every 15 min; published price is the minimum; uncapped, can exceed on-demand; policy "max price" or "follow spot price" (https://docs.nebius.com/signup-billing/spot-prices) | SIGTERM 60 s before stop (https://docs.nebius.com/compute/virtual-machines/preemptible) | VM stopped, not deleted; volumes kept; dynamic public IP released | Price history only; no rate |
| **CoreWeave** | Yes, spot node pools for all types except GH200/GB200/GB300 | Fixed list; can change at CoreWeave's discretion (https://docs.coreweave.com/policies/spot-tos) | Cordon plus label `qos.coreweave.cloud/pending-preemption` at T+0; drain and SIGTERM at T+2 min (30 s default grace); removed by T+7 min (https://docs.coreweave.com/platform/capacity-plans/spot-node-pools) | Node lost; storage behaviour undocumented (unverified) | None |
| **Crusoe** | No, via sales or support (https://docs.crusoe.ai/usage-billing/spot) | Unpublished | A 7-day notice was offered "for a limited time" at launch in 2025 (https://www.crusoe.ai/resources/blog/crusoe-cloud-now-offers-spot-pricing-access-powerful-gpus-on-crusoe-cloud-up-to-90-off); current terms unverified | Unverified | None |
| **AWS** | Yes, including p5 | Variable, smoothed, capped at on-demand; optional max price (https://aws.amazon.com/ec2/spot/pricing/) | 2 min via IMDS `spot/instance-action` and EventBridge, best effort; rebalance recommendation earlier (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html) | Terminate, stop or hibernate; EBS kept (and billed) on stop; no charge if AWS interrupts in the first hour (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/billing-for-interrupted-spot-instances.html) | Spot Instance Advisor buckets (<5%, 5-10%, 10-15%, 15-20%, >20%), public JSON; Spot placement score 1-10 (credentials) (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-placement-score.html) |
| **GCP** | Yes; A4X, A4X Max and A5X excluded; GPU spot VMs also preempted at host maintenance (https://docs.cloud.google.com/compute/docs/instances/spot) | Variable, up to once a day | 0 s by default or 120 s if configured, then up to 30 s best-effort shutdown (same page) | STOP (default) or DELETE; persistent disks kept and billed; no charge if preempted in the first minute | `preemptionRate` per day over 30 days and `obtainability`/`estimatedUptime` via `advice` APIs (preview, credentials) (https://docs.cloud.google.com/compute/docs/instances/view-vm-availability) |
| **Azure** | Yes | Variable; max price or `-1` (https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms) | 30 s via Scheduled Events `Preempt`; `SpotRebalanceRecommendation` up to 15 min ahead (preview) (https://learn.microsoft.com/en-us/azure/virtual-machines/linux/scheduled-events) | Deallocate (default; disks kept and billed, quota held) or Delete; ephemeral OS disks lost | Eviction rate per SKU and region (portal, last 7 days; Resource Graph `SpotResources` 28 days); Spot Placement Score High/Medium/Low (https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/spot-placement-score) |

Related discounted but not preemptible options [F]: AWS Capacity Blocks (prepaid, 1-182 days,
p5 at $47.757/instance-h, not cancellable) and GCP DWS flex-start (on-demand-style billing, at
most 7 days) and calendar mode (paid for the whole reservation). They fit the "reserved/blocks
allowed" capacity class of section 1, not spot.

### 3.2 Turning interruption data into a rate [D]

Section 4 needs `M_spot`, a mean time between preemptions per node. No source gives it directly.

- **AWS advisor buckets** do not define their window or denominator on the advisor page, which
  says only that the average "has historically been <5%" and that actual rates "depend on
  point-in-time available capacity" (https://aws.amazon.com/ec2/spot/instance-advisor/). Read as
  "fraction of instances interrupted per month", <5% would give M_spot above 720/0.05 = 14,400 h.
  That is far beyond what GPU spot users report, so the bucket is usable as a ranking signal, not
  as a rate (unverified interpretation). p5.48xlarge sat in bucket 0 (<5%) in us-east-1 and
  us-west-2 on 2026-11-18; p4d.24xlarge in bucket 3 (15-20%) (advisor JSON, fetched).
- **GCP `preemptionRate`** is a daily fraction (0-1) of preempted VMs over 30 days. If it means
  "fraction preempted per day", then `M_spot ~ 24 h / rate`; a rate of 0.3 gives 80 h. It needs
  credentials and is in preview.
- **Azure eviction rate** is given in bands per SKU and region. Treat it as GCP's figure, with
  the band edges as bounds.
- **Vast `reliability`** mixes host uptime and health, not outbids. Outbid risk depends on
  the margin between the user's bid and other renters' demand, which no field exposes.
- **Everyone else**: no data. Start from a pessimistic prior (`M_spot` = 8 h per node, the
  middle row of 4.3) and update from grid's own telemetry per (provider, region, class).

Prior over observed counts [P]: put a Gamma prior with shape a and rate b (in node-hours) on
the preemption rate per node-hour, with a = 1 and b = 8 h for the 8 h prior. After k preemptions
in T node-hours, the posterior mean rate is `(a + k)/(b + T)`, so grid's estimate of M is
`(b + T)/(a + k)`. Telemetry outweighs the prior after a few dozen node-days.

---

## 4. Total cost of a rental [D]

### 4.1 Why the sticker price is not the cost

The sticker price is the per-hour rate an offer lists. Six things separate it from what a
rental costs.

1. **Billed time before useful work.** Billing starts at create or at image pull, not at the
   first step (`07` s1.9). Provisioning, image pull, data staging and preflight take minutes
   per start (`09` s1). On spot, this happens again after every preemption.
2. **Minimum increments.** Per-second billing makes a culled spare cost cents. Hourly
   rounding or a 60 s or 5 min minimum makes it cost a full increment (`09` s2). Section 2
   gives each provider's increment.
3. **Failed nodes.** Nodes that fail preflight are billed until culled. Over-provisioned spares
   (10-20%, `09` s2) are billed until the cull.
4. **Storage.** Volumes are billed for the whole wall-clock time, including while stopped and
   while waiting for spot capacity. Some providers charge more for a stopped volume: RunPod
   volume disk costs $0.20/GB-month stopped against $0.10 running (`07` s1.6), and Vast storage
   is "billed continuously while your instance exists, regardless of running state"
   (https://docs.vast.ai/guides/instances/pricing).
5. **Data movement.** Hyperscalers charge for internet egress, and Vast charges per byte in
   both directions at a rate each host sets (same source). Nebius, CoreWeave, Crusoe and RunPod
   charge nothing for egress (section 2).
6. **Lost work.** A preemption or failure loses the work since the last durable checkpoint,
   plus the restart time. This depends on the checkpoint interval and on whether the preemption
   notice is long enough to save state (`04` s4.1, `18` s5.1).

### 4.2 The formula

Symbols, per node class i in the rental (one class per island; `06` s5 `class_key`):

| Symbol | Meaning | Source of the number |
|---|---|---|
| p_i | compute price, $ per node-hour, including bundled CPU/RAM/container disk | catalogue (section 2) |
| n_i | nodes needed | job spec |
| W0 | useful work, in node-hours on a reference node class | job spec or estimate |
| s_i | measured speed relative to the reference (1.0 = reference) | preflight T1/T4 (`09` s6.3), prior per class before measurement |
| M | mean time between interruptions of the job, combining hardware (`04` s1) and preemption | prior, then grid telemetry |
| delta, P, R | blocking checkpoint time, background persist time, billed restart time | `18` s5.1, preflight |
| N | preemption notice, seconds | section 3 |
| t_b | billed time from create to first step, per start | `09` s1, measured |
| q, t_f | probability a node fails preflight; billed time before it is culled | prior, then telemetry |
| k, t_c | spare nodes and the billed time until they are culled | acquisition policy (`09` s2) |
| b, m | billing increment and minimum charge | section 2 |
| V, r_run, r_stop | volume size; $/GB-hour while running and while stopped | section 2 |
| A | unbilled wait for capacity after each preemption | telemetry |
| B_in, B_out, c_in, c_out | bytes moved in and out; $/GB each way | job spec; section 2 |

**Interruption rate.** A synchronous group stops when any node stops, so job MTBF is node MTBF
divided by the node count (`04` s1):

```
1/M = n * (1/M_hw_node + 1/M_spot_node)
```

`M_spot_node` is infinite on on-demand capacity. Below, `M_hw = M_hw_node/n` and
`M_spot = M_spot_node/n` are the job-level values, and `1/M = 1/M_hw + 1/M_spot`.

**Overhead fraction o** (`18` s5.1, extended here for notice):

```
salvage  = 1 if a full save fits in the notice (delta + P <= N) and the target survives the
           preemption; otherwise 0
o(tau)   = delta/tau
         + (tau/2 + P) * (1/M_hw + (1 - salvage)/M_spot)       rework since the last durable copy
         + R/M                                                 restart, billed
         + salvage * (delta + P)/M_spot                        emergency save in the notice
tau*     = max( sqrt(2 * delta * M_rework), P ),  where 1/M_rework = 1/M_hw + (1 - salvage)/M_spot
```

**Billed node-hours per class.** `K = 1 + W/M_spot` is the expected number of starts. Restarts
are already inside `o` through `R`, which is set to the billed boot time `t_b`, so only the first
start's boot is added separately. `ceil_b(x) = max(m, b * ceil(x/b))` applies the billing
increment to a billed segment. Each of the K segments loses on average `b/2` to rounding when
the provider bills in whole increments (zero for per-second billing).

```
W      = W0/s_i * (1 + o)                                      useful work plus overhead, hours
H_i    = n_i * ( ceil_b(t_b) + W + K * b/2 )                   first boot, work, rounding per segment
       + n_i * K * q/(1 - q) * ceil_b(t_f)                     preflight failures, replaced at each start
       + k   * ceil_b(t_c)                                     culled spares
```

**Expected cost:**

```
E[C] = sum_i p_i * H_i                                        compute
     + sum_vol V * ( r_run * T_run + r_stop * T_stop )        storage, T_stop includes K*A
     + B_in * c_in + B_out * c_out                            data movement
     + fixed fees (reservation holding fees, IP addresses)
```

**Cost per useful work** is `E[C] / W0`, in dollars per reference node-hour. Offers are ranked on
it, not on `p_i` (section 5).

Three properties follow from the formula.

- **Spot discount against preemption.** Spot beats on-demand while
  `p_spot * (1 + o_spot) < p_od * (1 + o_od)` (ignoring storage and boot). The R/M term
  dominates `o_spot` once M is a few hours, so fast restart is worth as much as cheap
  checkpoints (`04` s4.1).
- **Notice only helps when a save fits in it.** A 5 s notice salvages nothing. A 2-minute
  notice salvages a job whose staged save plus persist takes under 2 minutes. When `delta + P`
  fits in N, the rework term for preemptions vanishes and only restart remains. This makes
  payload size (`18` s2) a spot-pricing lever.
- **Group size multiplies preemption.** n spot nodes in one synchronous group interrupt n times
  as often. Spot pays off most where the unit of preemption is small and independent: single
  nodes, DiLoCo islands, rollout workers (Implications).

### 4.3 Worked example: one 8x H100 SXM node, 24 hours of useful work

Job: 24 node-hours of useful training (W0 = 24 h, s = 1) on one 8x H100 SXM node. A single
node needs no inter-node fabric, so every offer in 2.1 is feasible. Both options are on one
provider, Nebius, so that hardware class, storage and egress are held equal and only the spot
terms differ. The same overhead is then applied to other providers' prices for comparison.

Inputs:

| Input | Value | Basis |
|---|---|---|
| On-demand price | 8 x $4.50 = $36.00/h | Nebius H100 on-demand list (section 2) |
| Preemptible price | 8 x $2.07 = $16.56/h | Nebius spot-price API, H100 eu-north1, effective 2026-11-18T14:31Z (section 2) |
| Payload | 8B-parameter model, full training state 13.5 GB x 8 = 108 GB | `18` s2 |
| delta | 10 s (async staging to host RAM) | `18` s5.1 |
| P | 120 s (108 GB at about 0.9 GB/s to the store) | assumption |
| R = t_b | 900 s billed per (re)start: provision, image, staging, preflight | `09` s1 ranges |
| M_hw | 1,000 h (node MTBF of about 42 days) | `04` s1 |
| M_spot | 24 h, 8 h, 2 h (scenarios; Nebius publishes no rate) | assumption, section 3 |
| Notice | 60 s SIGTERM (Nebius); delta + P = 130 s > 60 s, so salvage = 0 | section 3 |
| q, t_f | 5%, 0.1 h | assumption |
| Storage | 1 TiB network SSD at $0.071/GiB-month = $0.0996/h, billed for wall-clock | Nebius pricing (section 2) |
| A | 0.5 h unbilled wait for capacity per preemption | assumption |
| Egress | $0 (Nebius egress is free) | section 2 |

On-demand (M = M_hw = 1,000 h = 3.6M s):

```
tau* = sqrt(2 * 10 * 3,600,000) = 8,485 s (141 min)
o    = 10/8,485 + (4,243 + 120)/3,600,000 + 900/3,600,000
     = 0.00118 + 0.00121 + 0.00025 = 0.26%
W    = 24 * 1.0026 = 24.06 h
H    = 0.25 (initial boot) + 24.06 + 1 * 0.05/0.95 * 0.1 (preflight) = 24.32 h
C    = 24.32 * $36.00 = $875.5 compute + 24.32 h * $0.0996 = $2.4 storage = $877.9
```

Preemptible, M_spot = 8 h (1/M = 1/8 + 1/1,000, so M = 7.94 h = 28,571 s):

```
tau* = sqrt(2 * 10 * 28,571) = 756 s (12.6 min)
o    = 10/756 + (378 + 120)/28,571 + 900/28,571
     = 0.0132 + 0.0174 + 0.0315 = 6.22%
W    = 24 * 1.0622 = 25.49 h;  preemptions = 25.49/8 = 3.19;  K = 4.19
H    = 0.25 + 25.49 + 4.19 * 0.05/0.95 * 0.1 = 25.76 h
C    = 25.76 * $16.56 = $426.6 compute
     + (25.76 + 3.19 * 0.5) h * $0.0996 = 27.36 * 0.0996 = $2.7 storage
     = $429.4
```

| Scenario | tau* | o | Preemptions | Billed h | Wall h | Expected cost | Against on-demand |
|---|---|---|---|---|---|---|---|
| On-demand, $36.00/h | 141 min | 0.26% | 0 | 24.32 | 24.3 | $877.9 | 1.00 |
| Preemptible $16.56/h, M_spot 24 h | 21.7 min | 2.75% | 1.03 | 24.92 | 25.4 | $415.2 | 0.47 |
| Preemptible $16.56/h, M_spot 8 h | 12.6 min | 6.22% | 3.19 | 25.76 | 27.4 | $429.4 | 0.49 |
| Preemptible $16.56/h, M_spot 2 h | 6.3 min | 19.5% | 14.3 | 29.00 | 36.2 | $483.9 | 0.55 |

Re-check: the values were recomputed in a short script with the same inputs. The on-demand
row reproduces `04` s4.1's shape: at M = 1,000 h, overhead is negligible.

Readings:

- **The break-even spot price** at M_spot = 8 h is `(877.9 - 2.7) / (25.76 * 8) = $4.25` per
  GPU-hour, 94% of on-demand. Nebius spot is uncapped and "can exceed on-demand" (section 3), so
  the catalogue must re-check the live price before each start and set a max price.
- **Wall-clock is the hidden cost.** At M_spot = 2 h the job is still 45% cheaper but takes 36 h
  instead of 24 h, and the first-order formula is at the edge of its validity (`18` s5.1). A
  deadline constraint has to be part of the ranking (Implications).
- **Restart dominates.** At M_spot = 8 h, `R/M` (3.2%) is half the overhead. Halving t_b by
  keeping the image cached on the preserved boot disk (Nebius stops rather than deletes, section 3)
  is worth about as much as halving the checkpoint cost.

Same overheads applied to other providers' list prices from 2.1, compute only (storage and
egress differ by provider and are left out here):

| Provider, 8x H100 | On-demand $/h | Spot $/h | On-demand 24.32 h | Spot 25.76 h (M_spot 8 h) | Ratio |
|---|---|---|---|---|---|
| Nebius | 36.00 | 16.56 (live, eu-north1) | $875 | $427 | 0.49 |
| CoreWeave HGX H100 | 49.24 | 19.71 (NA list) | $1,197 | $497 (with salvage, below) | 0.41 |
| AWS p5.48xlarge, us-east-1 | 55.04 | 30.59 (third-party tracker) | $1,338 | $788 | 0.59 |
| Azure ND96isr H100 v5, eastus | 98.32 | 18.17 (API, effective 2026-01-01) | $2,391 | $468 | 0.20 |

CoreWeave's preemption cordons the node, sends SIGTERM 2 minutes later after a 30 s default grace
period, and removes it within 7 minutes (section 3). If grid triggers a save on the cordon event,
`delta + P = 130 s` fits, so salvage = 1. Then:

```
tau* = 141 min (set by M_hw only)
o    = 0.26% + (900 + 130)/28,800 = 0.26% + 3.58% = 3.84%
H    = 0.25 + 24 * 1.0384 + preflight 0.022 = 25.19 h
cost = 25.19 * $19.71 = $496.6
```

The salvage cuts overhead from 6.2% to 3.8%. That is less than the gap between CoreWeave's and
Nebius's spot prices, so Nebius still wins this example. Azure's spot row looks strongest, but its
price sample is ten months old in the API, and ND H100 v5 spot capacity is not verified (section 3).
Sticker prices alone would rank Nebius spot first and Azure second. The expected-cost ranking agrees
here only because the overheads were held equal, which is the point of the next section.

### 4.4 Data movement: where egress decides

Take the job of 4.3 with every checkpoint uploaded in full (108 GB = 0.108 TB) and a 2 TB dataset
staged once per start.

```
on-demand: 24.06 h * 60 / 141.4 min = 10.2 checkpoints  ->  1.10 TB uploaded
spot 8 h:  25.49 h * 60 / 12.6 min  = 121.4 checkpoints ->  13.1 TB uploaded
```

Spot's optimal interval is 11x shorter, so it uploads 12x the bytes. Who pays for those bytes
depends on where the store is:

| Where the bytes go | On-demand | Spot (M_spot 8 h) | Basis |
|---|---|---|---|
| Nebius, CoreWeave, Crusoe, RunPod, Lambda, Hyperstack (free egress) | $0 | $0 | section 2 |
| Vast host at the observed ~$4.00/TB up (`inet_up_cost` 0.0039 $/GB), $2.67/TB down | 1.10 x 4.00 + 2 x 2.67 = $9.7 | 13.1 x 4.00 + 4.19 x 2 x 2.67 = $74.8 | bundles fetch; rates vary per host |
| AWS to an object store outside AWS, $0.09/GB | 1,100 x 0.09 = $99 | 13,100 x 0.09 = $1,180 | `$0.09/GB` first tier (unverified; 100 GB/month free) |
| AWS to S3 in the same region | $0 transfer (S3 storage and request fees instead) | $0 transfer | data transfer section of https://aws.amazon.com/ec2/pricing/on-demand/ |

AWS p5 with an external store [D]: on-demand $1,338 + $99 = $1,437 against spot $788 + $1,180 =
$1,968. The $550 compute saving becomes a $531 loss. Keeping the store in-region, or shrinking
the payload with lossless deltas or weights-only intermediate saves (`18` s2, s5.4), restores
the saving. On Vast the bytes are cheap at the sampled host, but host-set rates mean the
catalogue must read `inet_up_cost`/`inet_down_cost` per offer rather than assume them. This is the
same zero-egress argument `14` made for the cross-island relay, now applied to checkpoints.

---

## 5. Comparability: one SKU name, many machines

### 5.1 What differs under one name [F]

| Difference | Effect | Source |
|---|---|---|
| H100 SXM vs PCIe vs NVL | SXM: 700 W, 3.35 TB/s HBM, NVLink 900 GB/s; PCIe: 300-350 W, 2 TB/s, no NVSwitch; NVL: 350-400 W, 3.9 TB/s, NVLink bridge between card pairs | NVIDIA H100 page https://www.nvidia.com/en-us/data-center/h100/ ; `06` s2.1 |
| Host power cap | H100 throughput 8% lower at 500 W, 18% at 400 W, 41% at 300 W against 700 W | `06` s2.9 |
| CPU and RAM per GPU | 20 vCPU per H100 SXM on RunPod against 6 per RTX 4090; dataloader starvation | `06` s2.4 |
| Intra-node topology | NVSwitch vs PCIe P2P vs no P2P (consumer cards) | `06` s2.2 |
| East-west fabric | Tier A-D (`05`); TCP instead of RDMA costs 10-57x on FSDP | `05` |
| Host health | Remapped rows, degraded PCIe links, repeat-offender hosts | `06` s5, `09` s6.3 |
| Host driver and CUDA | Decides which image runs at all | `06` s3.2 |

Provider SKU strings collapse these. Vast reports `gpu_name` with separate fields for host
bandwidth and a `dlperf` score (section 2). Prime Intellect reports `socket` (PCIe/SXM) and
`interconnect`. RunPod's v2 catalogue reports neither form factor nor host id (`16`).

### 5.2 Normalisation [P]

Three steps, in order.

1. **Hard constraints filter, not price penalties.** Interconnect tier, form factor where TP
   needs NVLink, GPU memory, CUDA/driver floor, region and residency, and security
   (secure/community) are pass/fail. An offer that fails one is not "cheaper with a penalty";
   it is infeasible. A PCIe node cannot hold an 8-way TP group at NVSwitch speed at any price
   (`06` s2.1), and a TCP-only offer cannot join a synchronous FSDP island at all (`05`).
2. **Speed factor s per class.** Before a node is measured, s comes from a prior table keyed by
   `class_key` (`06` s5). Examples: H100 SXM at 700 W = 1.0; H100 PCIe about 0.6-0.75 for
   compute-bound training, from the TDP and HBM ratios (unverified, no independent
   training benchmark found); a node with `power.limit` 400 W = 0.82 (`06` s2.9). After preflight
   T1 (matmul GFLOPS) and optionally T4 (real steps), s is replaced by the measured ratio for that
   node and for that host's history (`09` s6.3).
3. **Rank on cost per useful work.** Use `E[C]/W0` from section 4, with `W0/s_i` hours. Two
   offers at $2.00 and $2.40 per GPU-hour with s = 0.8 and s = 1.0 cost $2.50 and $2.40 per
   reference GPU-hour: the dearer sticker wins.

For a synchronous island the slowest rank sets step time (`08`, `09` s2), so the island's s is the
**minimum** over its nodes, not the mean. Offers are therefore compared at the island level. Four
s = 1.0 nodes and one s = 0.8 node give an island at 0.8; dropping the slow node (cull on rank,
`09` s2) can lower cost per useful work even after paying for its billed minutes.

### 5.3 Measurement feeds the catalogue [P]

Preflight already produces `measured.tokens_per_s_ref` and similar fields per node (`06` s5). Two
uses:

- **Within a run:** cull the slowest spares, and price replacements with the measured s of the
  host class they will join.
- **Across runs:** store s, preflight pass/fail and observed preemptions per (provider, region,
  class_key, host id where exposed). This is grid's own reliability and speed history, the
  marketplace analogue of Vast's `reliability` score. It updates the priors for s, q and M_spot
  that section 4 needs. Keying on host id is possible on Vast (`machine_id`) and not on RunPod v2
  (`16`).

---

## 6. Prior art: how others collect and rank prices [F]

### 6.1 SkyPilot

- **Catalogue collection.** One fetcher script per cloud in `sky/catalog/data_fetchers/`
  (https://github.com/skypilot-org/skypilot/tree/master/sky/catalog/data_fetchers): `fetch_aws.py`,
  `fetch_gcp.py`, `fetch_azure.py`, `fetch_lambda_cloud.py`, `fetch_vast.py`, `fetch_runpod.py`,
  `fetch_nebius.py`, `fetch_shadeform.py`, `fetch_verda.py`, `fetch_mithril.py` and others. The AWS
  fetcher reads the public price list. The output is a CSV per cloud in
  `skypilot-org/skypilot-catalog`, schema `v8`, regenerated by GitHub Actions (AWS every 7 hours).
  Clients re-download a catalogue older than `_PULL_FREQUENCY_HOURS = 7`
  (https://github.com/skypilot-org/skypilot/blob/master/sky/catalog/aws_catalog.py), with an S3
  mirror used when GitHub returns `429` (https://github.com/skypilot-org/skypilot/blob/master/sky/catalog/common.py).
- **Optimiser.** `sky/optimizer.py` minimises `OptimizeTarget.COST` (default) or `TIME`
  (https://github.com/skypilot-org/skypilot/blob/master/sky/optimizer.py,
  https://github.com/skypilot-org/skypilot/blob/master/sky/utils/common.py). Chains are solved by
  dynamic programming and general DAGs by an ILP (`pulp`/CBC). Egress between tasks on different
  clouds is part of the objective (`_egress_cost` calls each cloud's `get_egress_cost`). The
  per-node cost is the catalogue's hourly price times an estimated runtime (how runtime is
  estimated was not traced; unverified). No code path read has a term for boot time, failed
  nodes or preemption rework.
- **Spot.** `use_spot: true` in resources selects the `SpotPrice` column
  (https://docs.skypilot.co/en/latest/reference/yaml-spec.html). Managed jobs recover from
  preemption by relaunching, with strategy `FAILOVER` (same region first) or `EAGER_NEXT_REGION`
  (the default, moves on at once) (https://github.com/skypilot-org/skypilot/blob/master/sky/jobs/recovery_strategy.py).
  The choice between spot and on-demand is the user's; the optimiser compares only sticker
  prices once `use_spot` is set. `sky show-gpus` prints on-demand and spot prices per cloud and
  region (https://docs.skypilot.co/en/latest/reference/cli.html).
- **Research.** "Can't Be Late" (NSDI '24) mixes spot and on-demand to meet a deadline with
  "Uniform Progress", using SkyPilot's spot traces
  (https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao, https://github.com/skypilot-org/spot-traces).
  This is the policy grid's deadline constraint needs (Implications 6).

### 6.2 dstack and gpuhunt

- **Collection.** `gpuhunt` (https://github.com/dstackai/gpuhunt, MPL-2.0) has one module per
  provider in `src/gpuhunt/providers/`. "Offline" providers (aws, azure, gcp, lambdalabs, nebius,
  oci, runpod, verda, cloudrift) are pre-built hourly into S3 catalogues. "Online" providers
  (vastai, crusoe, digitalocean, and others) are queried live at request time
  (https://github.com/dstackai/gpuhunt/blob/main/src/gpuhunt/_internal/catalog.py). The split
  matches the volatility of each source: marketplaces live, list prices cached.
- **Ranking.** `query()` merges providers with `heapq.merge` keyed on price, with a code comment
  that the result "is not strictly sorted by the price". dstack's server merges backend offers by
  `offer.price` and then stable-sorts unavailable offers last
  (https://github.com/dstackai/dstack/blob/master/src/dstack/_internal/server/services/offers.py).
- **User knobs.** `spot_policy: spot | on-demand | auto` (default on-demand), `max_price` in
  $/h, and `dstack offer --gpu ... --spot-auto --max-price ... --group-by gpu,backend,region --json`
  (https://dstack.ai/docs/reference/cli/dstack/offer/). Where `spot_policy` and `max_price` are
  applied server-side was not traced (unverified).

### 6.3 Prime Intellect and Shadeform

- **Prime Intellect** aggregates many providers (RunPod, Lambda, Hyperstack, Nebius, Vultr and
  its own) behind one availability call. Each row carries price, `stockStatus`,
  `provisioningTime`, `interconnect` and `isSpot` (section 2.1). With Shadeform's `boot_time`, it
  is one of only two sources that give a provisioning-time estimate per offer, which feeds `t_b`
  directly. How it orders results server-side is undocumented (unverified).
- **Shadeform** resells the underlying clouds' on-demand capacity at their price (FAQ) and adds
  `sort=price`, per-region `available` flags, `boot_time` ranges, and spot entries on its own
  Shade Cloud types. `queue_window` turns a stockout into a bounded wait (`16`).

### 6.4 Aggregator sites and indices

Section 2.2 lists them. None is both open and live. The open datasets are the SkyPilot and gpuhunt
catalogues. getdeploying sells API access under terms that forbid redistribution
(https://getdeploying.com/api); computeprices sells token access under terms not reviewed here.
The Full Stack Deep Learning CSVs (https://fullstackdeeplearning.com/cloud-gpus/) stopped in
October 2023. gpulist.ai is a cluster-listing marketplace, not a price feed, and
cloud-gpus.com exposes no API (both from a search, no API page found).

### 6.5 What nobody does [D]

Every tool surveyed ranks on the sticker price, sometimes with availability as a tie-breaker.
None estimates boot or provisioning time billed before work, failed-preflight waste, the
checkpoint and rework overhead of spot, the rise in upload bytes that spot causes (4.4), or
measured per-host speed. SkyPilot's egress term and Can't Be Late are the closest pieces, and
both are worth reusing.

---

## Corrections and refinements to earlier files

- **`16` s1.1 and summary, "RunPod spot is gone"**: re-confirmed on 2026-11-18 from the v2
  migration guide, the Terraform 2.0 page and a user-reported `500` "Spot pods are no longer
  offered". REST v1 was retired on 2026-11-15 as scheduled (https://docs.runpod.io/release-notes).
  `07` s1.7's 5 s notice is historical only.
- **`07` s2 table, AWS spot "2-minute notice (from memory)"**: verified, with a rebalance
  recommendation earlier and no charge if AWS interrupts in the first hour.
- **`07` s2 table, GCP spot "30 s notice (from memory)"**: outdated. Notice is 0 s by default or
  120 s if configured, followed by up to 30 s of best-effort shutdown. Spot prices can change daily
  (not every 30 days), and GCP now publishes preemption-rate history in preview.
- **`07` s2 table, Azure spot "at least 30 s"**: verified. A rebalance recommendation up to 15
  minutes ahead is in preview.
- **`07` s2 table, CoreWeave "Spot offered per aggregators (unverified)"**: CoreWeave documents
  spot node pools with fixed list prices, a cordon-then-SIGTERM sequence at T+2 min and removal by
  T+7 min.
- **`07` s2 table, Hyperstack "Spot VMs (unverified)"**: they exist as `-spot` flavours, with no
  notice and nothing preserved.
- **`07` s2 table, TensorDock spot "Was offered"**: the v2 API has no spot fields; treat it as
  unavailable (unverified).
- **`16` capability table, `spot` row**: extend to RunPod none; Vast bid, pause, notice
  undocumented; Nebius 60 s, **dynamic uncapped price since 2026-10-08**; CoreWeave cordon + 2 min;
  AWS 120 s; GCP 0 or 120 s; Azure 30 s; Verda 0 s; Hyperstack 0 s, ephemeral; Shadeform and Prime
  Intellect undocumented; Crusoe sales; **Lambda none** (the `preempted` status in `16` does not
  imply a self-serve spot product; none was found).
- **`06` s1.5 and `07` s1.8 prices**: drifted within nine days. RunPod Secure H100 SXM $3.99 ->
  $4.39, Lambda H100 SXM $3.99 -> $4.59/GPU-h (8x), and Crusoe H100 rises to $4.50 on 2026-12-01.
  Treat every price in 01-18 as a dated sample. Ranking must use live catalogue data.
- **`02` Shadeform "resells ... at provider price"**: Shadeform's FAQ still says so; one third
  party claims a 6-10% markup (unverified). Compare against a direct quote where grid holds both
  accounts.
- **`18` s5.1 interval formula**: extended in 4.2 so that a preemption whose notice covers
  `delta + P` costs an emergency save and a restart, not rework.

## Implications for grid

All [P].

### 1. A price and offer catalogue component

One component, `catalogue`, owns every price and availability read. Provider drivers (`16`)
expose `list_offers`; the catalogue calls them, adds public and hand-maintained sources, and
caches.

| Source kind | Providers | Credentials | Refresh | Cache |
|---|---|---|---|---|
| Public bulk or list APIs | AWS bulk files and spot advisor JSON; Azure Retail Prices; Nebius spot-price; Verda `instance-types` | none | List prices daily. Nebius spot every 15 min while a plan or run uses it. AWS advisor daily | On disk, per source, with `fetched_at` and the source's own `publicationDate`/`effectiveTime`; may be shared between a user's runs |
| Account-scoped live queries | RunPod v2 catalogue, Vast `bundles`, Lambda `instance-types`, Prime Intellect `availability`, Shadeform `instances/types`, TensorDock `hostnodes`, Hyperstack `pricebook`, Verda `instance-availability`, Nebius capacity advice, AWS `DescribeSpotPriceHistory` and placement score, GCP Catalog and `advice`, Azure Resource Graph and placement score | the user's own keys | At plan time, then **again immediately before each create**, because offers are advisory (`16`, dstack #1740) | In memory for the plan; TTL 60 s for marketplace offers (Vast, RunPod stock, Prime Intellect, Shadeform), 15-30 min for placement scores (Azure's advice), never persisted for Vast (section 2.3) |
| Hand-maintained | CoreWeave, Crusoe; scheduled changes (Crusoe 2026-12-01); fallbacks | none | By pull request; each entry has `as_of` and `source_url` | In the repository; warn when older than 30 days |
| Optional seed | SkyPilot catalogue v8, gpuhunt S3 | none | Only to fill gaps and to sanity-check | Cross-check only, after confirming licences |

Rules:

- **Rate budgets are shared with the control plane.** RunPod's per-user window and Lambda's
  1 req/s cover creates and catalogue reads alike (`16`). The catalogue gets a fixed share and
  backs off first, so pricing never starves acquisition.
- **Never scrape web pages** for providers whose terms forbid it (RunPod, TensorDock, Verda,
  Vast). Use APIs only, under the user's account.
- **Never publish or pool marketplace prices** across users. grid ships no hosted price service
  built from Vast data.
- **Record the billed rate.** After create, read the node's own cost field (RunPod `cost`, `07`)
  and compare it with the quoted price. A mismatch is logged and the run is re-priced.

### 2. Offer record fields to add to `16`'s `list_offers`

`16` sketches `Offer{sku, location, fabric?, price, stock, quota?, provisioning_eta?,
max_gpus_per_node, cuda/driver, inbound_ports?}`. Add:

| Field | Type / values | Filled from |
|---|---|---|
| `capacity_class` | `on_demand` / `spot` / `bid` / `reserved` / `block` | every source |
| `price` | `{amount, currency, unit: gpu_h / instance_h / component_h, includes: [gpu, cpu, ram, disk]}` | normalised to $ per node-hour for ranking |
| `price_kind` | `fixed` / `variable` (Nebius, AWS, GCP, Azure spot; Prime Intellect `isVariable`) / `bid` (Vast `min_bid`) | section 2.1 |
| `max_price_supported` | bool, plus whether the cap holds for the rental's life (Nebius, AWS, Azure: yes; Prime Intellect: at provisioning only) | section 3.1 |
| `billing` | `{increment_s, minimum_s, prepaid_increment_s?, first_period_refund?}` (AWS spot first hour, GCP spot first minute) | section 2.1, 3.1 |
| `storage_rates` | per volume type: `{running, stopped}` $/GB-h | `07` s1.6; section 4.1 |
| `transfer_rates` | `{in, out}` $/GB, `per_host: bool` (Vast `inet_*_cost`) | section 2.1 |
| `preemption` | `{notice_s, signal (SIGTERM / IMDS / scheduled event / k8s cordon / none), on_preempt: stop / delete / pause, survives: [boot_disk, volume, ip], rate_prior, rate_source}` | section 3 |
| `hardware` | `{form: sxm / pcie / nvl, gpu_mem_gb, power_cap_w?, vcpu_per_gpu, ram_per_gpu_gb, interconnect_kind, interconnect_gbps}` | Prime Intellect `socket`/`interconnect`, Shadeform `interconnect`/`nvlink`, Vast fields, instance type tables |
| `perf_hint` | provider score (Vast `dlperf`) and grid's own measured `s` for this class or host, if any | section 5 |
| `host_id` | where exposed (Vast `machine_id`) | for the strike list (`04`) and speed history |
| `boot_eta_s` | Prime Intellect `provisioningTime`, Shadeform `boot_time`, else grid telemetry | section 6.3 |
| `quote` | `{fetched_at, valid_until, source, effective_at}` | the catalogue |

### 3. Ranking objective and user constraints

**Objective.** Minimise expected cost per useful work, `E[C]/W0` (section 4.2). An alternative
target, minimum expected wall-clock under a budget, mirrors SkyPilot's `COST`/`TIME` switch.

**Constraints the user sets**, all optional:

```yaml
rental:
  gpus: {type: H100, count: 64, form: [sxm], min_mem_gb: 80}
  islands: {nodes_per_island: 4, min_interconnect: B, cross_provider: allowed}
  work: {hours: 24}                    # or deadline: 2026-11-20T12:00Z
  capacity: {spot: allowed}            # never | allowed | preferred | only
  budget: {max_per_gpu_h: 3.00, max_per_h: 200, max_total: 4000}
  placement: {regions: [eu-*, us-*], providers_deny: [], security: secure}
  risk: {max_preemptions_per_day: 6, min_host_reliability: 0.95}
  data: {dataset_gb: 2000, checkpoint_gb: 108, store: r2}   # sets transfer cost and P
```

**Algorithm.**

1. Filter on hard constraints (section 5.2).
2. Price each feasible offer with the formula of 4.2, using live price, priors for s, M and t_b,
   and the job's checkpoint numbers from `18`.
3. Assemble islands. Each island is one provider, one class and one capacity class.
4. Rank candidate plans by expected cost, rejecting those whose expected wall-clock misses the
   deadline.
5. Feed the top N plans to acquisition as the candidate ladder `07` and `16` already require, so
   a stockout falls through to the next plan, not to a retry loop.

`grid plan` prints, for the top plans, sticker $/h, expected total, the breakdown of 4.2
(compute, boot, preflight waste, rework, storage, transfer), expected wall-clock, and the
sensitivity to M_spot halving and to the spot price rising to its cap, as `18` s6.4 does for
checkpoint policy.

### 4. Where spot fits in the island model

The formula's `n * 1/M_spot` term says spot should be used only where one preemption costs one
unit, not the whole job.

| Pool | Spot? | Why |
|---|---|---|
| Synchronous training island (FSDP/TP inside a fabric) | On-demand by default; spot only for small islands (1-2 nodes) with a large M_spot and a save that fits the notice | A preemption stalls every rank in the island (`04`, `13`) |
| Whole DiLoCo islands across providers (`08`, `14`) | **Yes**, the main training use | The outer loop tolerates a missing island for a round. A preempted island rejoins from the last outer step. Each island is homogeneous and on one provider's spot pool |
| RL rollout and actor workers (`12`) | **Yes** | Stateless apart from the policy version; a lost worker loses only its in-flight rollouts |
| Serving replicas (`11`) | Yes, behind mid-stream failover (`17`) | Replay moves a cut stream to another replica |
| Single-node jobs, evals, data preprocessing | Yes | n = 1 keeps M large |
| Hot spares (`04` s4.9) | Possibly | A preempted spare costs nothing in lost work. A spare that is too often gone is useless, so this is gated on M_spot |
| Rank 0, coordinator, learner island, checkpoint store | **Never** | `03`, `12`, `14` |

Mixing spot and on-demand nodes inside one synchronous island is refused at plan time, because
it pays the spot interruption rate at the on-demand price.

### 5. Use the notice

Subscribe to every provider's preemption signal (`watch_preemption` in `16`): AWS IMDS, the GCP
120 s option (set it, since the default is 0 s), Azure scheduled events, Nebius SIGTERM and the
CoreWeave cordon label. On the signal, trigger an emergency save when `delta + P` fits in the
notice, and choose the payload so that it fits (weights and optimizer shards to a local
surviving volume, or a smaller codec; `18` s5.4). Where the notice is 0 (Verda, Hyperstack) the
offer is priced with salvage = 0 and a short interval.

### 6. Deadlines and mixed capacity

Start on spot and switch to on-demand when slack runs out, following Can't Be Late's Uniform
Progress policy. This is a later feature. The first version only needs the deadline as a filter.

### 7. Price drift during a rental

For variable-price offers (Nebius spot, Prime Intellect `isVariable`), set a max price where the
provider honours it for the rental's life, and re-price running islands from the catalogue every
refresh. If an island's expected remaining cost on the current offer exceeds the best
alternative by more than the cost of migrating (`15` contract), propose migration at the next
checkpoint.

### 8. Learn the priors

Store `t_b`, preflight pass rate, measured s, preemption count and node-hours per (provider,
region, class_key, capacity_class, host where known). The Gamma update of 3.2 replaces the
pessimistic defaults. This shares its store with the strike list (`04`) and preflight history
(`09`).

## Open questions

- **Owner decision:** should the default objective be minimum cost or minimum wall-clock within a
  budget? And may grid choose spot without being asked (`spot: allowed` as default), or only
  when told?
- What are real `M_spot` values for GPU spot at Nebius, CoreWeave, Verda, Vast (outbid rate at a
  given margin over `min_bid`) and the hyperscalers' H100 pools? Only grid's own telemetry or a
  probe campaign will tell. AWS's advisor bucket has no documented window.
- Can a Nebius preemptible VM join a gpu-cluster (`16`), and can CoreWeave spot nodes carry IB?
  If not, multi-node spot islands with a fast fabric exist only on the hyperscalers.
- Is Vast's anonymous `GET /bundles` intended? Would Vast agree in writing to grid caching
  offers per user for longer than a plan's life?
- Licences of the SkyPilot catalogue repository and the gpuhunt S3 data, if grid wants them as a
  seed.
- Does Shadeform add a markup over direct prices? Answer by comparing quotes on accounts held at
  both.
- How large is the speed gap between H100 SXM, PCIe and NVL nodes for the job classes grid
  targets, and between power-capped and uncapped hosts? This sets the default s priors of 5.2.
  Preflight T1/T4 on a few hosts would settle it.
- What are `t_b` distributions per provider (`09` open questions)? They are the largest
  per-start term in spot overhead.
- Do spot instances on AWS and Azure for 8x H100 actually obtain capacity at the advisor's or
  API's prices? Azure's spot row in the retail API dates from 2026-01-01.
- Should prices be normalised to one currency and include VAT for EU providers (Verda, Nebius
  EU)? The responses read did not state VAT treatment.
