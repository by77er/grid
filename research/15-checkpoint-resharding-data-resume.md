# 15 - Checkpoint resharding and data resume across cluster shapes

Second-sweep file. Question: what decides whether a training job can resume on a *different* cluster
shape (world size, TP/PP/DP split, GPU type, provider), and what should grid require of jobs?

Method: source was read on 2026-11-09 from shallow clones / raw files at the commits below. Paths are
relative to each repo; URLs are `https://github.com/<repo>/blob/main/<path>`.

| Repo | Commit / version read | Latest PyPI release |
|---|---|---|
| `pytorch/pytorch` | `main`, `version.txt` = `2.16.0a0` | torch 2.15.0 (2026-10-28) |
| `NVIDIA/Megatron-LM` | `d2b0e35c` (2026-11-09), `package_info.py` 0.21.0 | megatron-core 0.20.0 (2026-10-29) |
| `deepspeedai/DeepSpeed` | `6f96b4a`, `version.txt` 0.19.10 | deepspeed 0.19.9 |
| `pytorch/torchtitan` | `8a33dbf0` (2026-11-09) | n/a |
| `huggingface/accelerate` | `f47439c`, 1.16.0.dev0 | 1.15.0 |
| `huggingface/transformers` | `main`, 5.23.0.dev0 | 5.22.2 |
| `Lightning-AI/pytorch-lightning` | `84df182`, 2.7.0.dev | lightning 2.6.6 |
| `mosaicml/streaming` | `1381ca1`, 0.14.0.dev0 | 0.13.0 (2025-07-15) |
| `Lightning-AI/litdata` | `b18848c`, 0.2.76 | 0.2.76 |
| `webdataset/webdataset` | `e0953f9`, 1.0.2 | 1.0.2 (2025-06-19) |
| `huggingface/datasets` | `d3e4a3f` | 5.1.0 |
| `meta-pytorch/data` (torchdata) | `5a4e2a6` | 0.11.0 (2025-02-20) |
| `NVIDIA/nvidia-resiliency-ext` | `3e712dd` | 0.7.0 |
| `coreweave/tensorizer`, `huggingface/safetensors` | `main` README / `serialization.py` | 2.12.1 / 0.8.0 |

Not opened in this sweep: Lance's sampler/resume code, JAX/Orbax, Ray Train checkpoint code, s5cmd's
own benchmarks. Anything said about them is marked (unverified).

## Summary

- **Reshardable checkpoints exist in exactly one design family**: tensors stored under a
  fully-qualified name with a *global shape* plus a list of chunks (offset, size, file, byte range).
  PyTorch DCP does this for `DTensor`/`ShardedTensor`; Megatron `torch_dist` and `fsdp_dtensor` are
  built on DCP; DeepSpeed gets there only through an offline conversion (`ds_to_universal.py`).
  Everything keyed by rank (`mp_rank_XX`, `zero_pp_rank_N`, `rng_state_{i}.pth`,
  `random_states_{i}.pkl`, `dp_rank_{k}` dataloader state) does not reshard.
- **Weights reshard more easily than optimizer state.** Megatron's default optimizer sharding
  (`dp_reshardable`) tolerates only DP changes; TP/PP changes need
  `--dist-ckpt-optim-fully-reshardable` at *save* time. DeepSpeed's `elastic_checkpoint` repartitions
  fp32 weights only, "and only without optimizer states".
- **RNG state is never resharded.** Megatron explicitly drops it when TP/PP or world size changes;
  HF Trainer and Accelerate silently skip a missing per-rank file. Exact (bitwise) resume across
  shapes is not available anywhere; grid should promise "sample-exact", not "bit-exact".
- **The data pipeline is the weakest link.** torchtitan's loader raises on DP change; HF `datasets`
  state is relative to a rank's shard subset; WebDataset has no resume API. World-size-independent
  resume is implemented by Mosaic StreamingDataset (`num_canonical_nodes`), LitData (elastic resume,
  `state_version: 2`) and Megatron's built-in sampler (global `consumed_train_samples`).
- **Effective batch**: Megatron and torchtitan define the step in global units
  (`global_batch_size`, `num_tokens_per_train_step`) and derive accumulation from DP, so a DP change
  keeps tokens-per-step constant if divisibility holds. Accelerate/Trainer define the step
  per-device, so world size silently changes batch size and LR schedule progress.
- **Speed**: the blocking part of an async DCP save is the GPU-to-host copy (6-14 s for 7B-13B
  models per IBM); persisting is bounded by per-node write bandwidth (~1 GB/s with `torch.save`,
  ~10 GB/s node peak on Polaris). Full Adam state is roughly 13-14 GB per billion parameters.
- **Integrity**: DCP and safetensors carry no checksums. Megatron `--verify-integrity` writes a
  SHA-256 `integrity.json` (torch_dist only); tensorizer stores per-tensor CRC32 and SHA-256.

## 1. PyTorch Distributed Checkpoint (DCP)

Files: `torch/distributed/checkpoint/{state_dict_saver,state_dict_loader,default_planner,planner_helpers,filesystem,fsspec_filesystem,staging,resharding,_extension}.py`.

**On-disk format** (`filesystem.py`). `_metadata_fn = ".metadata"`, `DEFAULT_SUFFIX = ".distcp"`; each
rank writes files prefixed `__{rank}_` (`_StoragePrefix(f"__{self.rank}_")`); the coordinator
(rank 0, "Rank 0 is assumed to be the coordinator rank") pickles a `Metadata` object to
`.metadata.tmp` and renames it to `.metadata` in `finish()`. Presence of `.metadata` is therefore the
commit marker. `FileSystemWriter` defaults: `single_file_per_rank=True`, `sync_files=True`,
`thread_count=1`, `per_thread_copy_ahead=10_000_000`, `overwrite=True`,
`serialization_format=SerializationFormat.TORCH_SAVE` (alternative `SAFETENSORS`). An experimental
`_extensions` argument accepts stream transforms; the only registered one is `ZStandard`
(`registry_name` `"stream.zstd"`). I found no checksum or hash anywhere in `filesystem.py`.

**Resharding mechanism.** `Metadata.state_dict_metadata[fqn]` is a `TensorStorageMetadata` holding the
global `size` and a list of `ChunkStorageMetadata(offsets, sizes)`. On load,
`create_default_local_load_plan()` checks only that `md.size` equals the destination's *global*
shape ("Size mismatch between saved ... and current") and then calls
`create_read_items_for_chunk_list()`, which "applies the resharding algorithm and computes the reads
needed to satisfy `local_chunks`" by intersecting saved chunks with local chunks
(`resharding._shards_get_overlap_region_wrt_saved_tensor`). Nothing in the plan refers to the saving
world size. Consequences:

- Any `DTensor` (FSDP2 parameters, TP-sharded DTensors) or `ShardedTensor` loads into any number of
  ranks and any sharding dimension, provided FQN and global shape match. FSDP2 optimizer state
  reshards the same way because its `state_dict()` values are DTensors (this is what torchtitan
  relies on; I did not re-read FSDP2's optimizer path, so treat as verified-by-usage).
- Plain tensors and non-tensor objects are written by every rank and deduplicated by key on the
  coordinator (`dedup_save_plans`, `DefaultSavePlanner(dedup_save_to_lowest_rank=False)`), and
  non-tensor data is loaded with `torch.load()`. **Per-rank state stored under one shared key is
  silently collapsed to a single rank's copy**; per-rank state must be keyed by rank, and then it
  cannot be resharded.
- `load(..., no_dist=True)` reads a multi-rank checkpoint in a single process (used by tools).
- `DefaultLoadPlanner(allow_partial_load=False)` raises "Missing key in checkpoint state_dict".
- Docstring warning: "There are no guarantees of Backwards Compatibility across PyTorch versions for
  saved state_dicts." A cross-provider migration must pin the torch version or re-export.

**Rank-local mode.** `save(..., use_collectives=False)` skips cross-rank coordination and writes a
per-rank `__{rank}.metadata`; the loader falls back to it when global metadata is missing ("Global
metadata is not found. Falling back to rank local metadata") and sets `use_collectives = False`. The
docstring says it "will change the format of the saved checkpoint and may not be backward
compatible". Rank-local metadata describes only that rank's chunks, so this mode trades away
resharding for independence from collectives.

**Async save.** `async_save(..., async_checkpointer_type=AsyncCheckpointerType.THREAD, async_stager=None)`.
- Requires a CPU backend: "A CPU backend must be enabled for async save; try initializing process
  group with 'cpu:gloo,cuda:nccl'". torchtitan builds its process group as
  `f"{device_type}:{backend},cpu:gloo"`. For grid this means **Gloo (TCP) connectivity between all
  ranks is a prerequisite for async DCP**, on top of NCCL.
- Default stager is `DefaultStager(StagingOptions(False, False, False, False))`. `StagingOptions`
  fields: `use_pinned_memory`, `use_shared_memory`, `use_async_staging`, `use_non_blocking_copy`
  (dataclass defaults all `True`). With async staging, `async_save` returns
  `AsyncSaveResponse(staging_completion, upload_completion)`; otherwise a `Future`.
- `AsyncCheckpointerType.PROCESS` uses `_ProcessBasedAsyncCheckpointExecutor`; plan caching is
  `DefaultSavePlanner(enable_plan_caching=True)`.
- Cost: one full host-RAM copy of the checkpointed state per rank (pinned/shared when enabled);
  blocking time is the GPU-to-CPU copy. IBM measured 6-14 s of visible downtime for 7B-13B models
  and 148.8 s -> 6.3 s for a 7B model
  ([PyTorch blog, 2024](https://pytorch.org/blog/reducing-checkpointing-times/)). The 436 s -> 67 s
  background-time figure at 1,856 GPUs cited in file 04 was again seen only as a search snippet
  ([pytorch.org/?p=3455](https://pytorch.org/?p=3455)).
- `staging._ReplicationStager` (marked experimental) replicates a rank's state dict to partner ranks
  over `PGTransport` and persists it under `storage_dir`: an in-tree buddy-checkpoint primitive.

**Storage backends.** `FsspecWriter`/`FsspecReader` (`fsspec_filesystem.py`; the underscore module
is deprecated) resolve any fsspec URL with `url_to_fs(path, **kwargs)`, so `s3://`, `gs://` etc. work
if `s3fs`/`gcsfs` is installed. Stated assumptions: "The checkpoint path is an empty or non-existing
directory" and "File creation is atomic". `rename` maps to `fs.rename` (copy+delete on S3, but only
for the small metadata file); `os.fsync` is attempted and downgraded to a warning ("fsync not
supported for this stream, relying on flush()"). There is also an `hf_storage.py` writer producing
safetensors, used by torchtitan's HF export. The 2023 IBM post confirms S3-backed DCP in production
and that object-store eventual consistency initially broke multi-node saves
([PyTorch blog](https://pytorch.org/blog/performant-distributed-checkpointing/)).

## 2. Megatron-LM

Files: `megatron/training/config/training_config.py`, `megatron/training/checkpointing.py`,
`megatron/training/arguments.py`, `megatron/core/optimizer/distrib_optimizer.py`,
`megatron/core/dist_checkpointing/*`, `tools/checkpoint/*`, `megatron/training/datasets/data_samplers.py`.

`ckpt_format: Literal["torch", "torch_dist", "torch_dcp", "fsdp_dtensor"] = "torch_dist"`. **`zarr`
is no longer a choice on main**, and `megatron/core/dist_checkpointing/strategies/` contains only
`torch.py`, `fully_parallel.py`, `nvrx.py`, `checkpointable.py`, `common.py` (no zarr/tensorstore).

| Format | Layout | Model weights | Optimizer | Constraints (asserts) |
|---|---|---|---|---|
| `torch` (legacy) | `iter_NNNNNNN/mp_rank_{tp:02d}[_{pp:03d}]/model_optim_rng.pt` | TP and PP must match: `check_checkpoint_args` compares `tensor_model_parallel_size` and `pipeline_model_parallel_size` when `checkpoint_version >= 3.0 and not use_dist_ckpt` | distributed-optimizer state gathered to DP rank 0 over Gloo | cannot be async; needs Gloo groups |
| `torch_dist` | DCP files + Megatron metadata | reshardable over TP/PP/DP (ShardedTensor -> DCP) | depends on `distrib_optim_sharding_type` (below) | default |
| `torch_dcp` | plain DCP | via DTensor | via DTensor | "only tested with FSDP"; asserts `tensor_model_parallel_size <= 1`, `pipeline_model_parallel_size <= 1` |
| `fsdp_dtensor` | DCP, one DTensor per parameter | via DTensor | sharding type `fsdp_dtensor` | "only tested with Megatron FSDP" (`--use-megatron-fsdp` requires it) |

**Optimizer sharding types** (`DistributedOptimizer.sharded_state_dict` docstring):
- `dp_reshardable` (default for torch_dist): "Since the format relies on the internal
  DistributedOptimizer structure, it allows checkpoint resharding only in DP dimension." The save
  drops the trailing "bucket padding to some DP multiple" precisely "because that might change
  during DP resharding".
- `fully_reshardable` (`--dist-ckpt-optim-fully-reshardable`, default `False`): gathers all buffers
  on DP rank 0 at save and writes a canonical per-parameter representation; "fully reshardable
  (TP/PP/EP/DP)". `--distrib-optim-fully-reshardable-mem-efficient` uses Gloo and one rank.
- `fsdp_dtensor`; deprecated `dp_zero_gather_scatter`, `fully_sharded_model_space`.
  `checkpoint_fully_reshardable_formats = {'fully_reshardable', 'fully_sharded_model_space'}`.
- On load with changed TP/PP and a non-fully-reshardable optimizer: `RuntimeError('(TP, PP) mismatch
  after resume (...): not supported for DistributedOptimizer with sharding type ...')`.
- `--no-ckpt-fully-parallel-save` "Makes DistributedOptimizer checkpoint non-reshardable".
- Local (non-persistent) checkpoints reject `fully_reshardable`: "Don't use
  '--dist-ckpt-optim-fully-reshardable' when saving local checkpoints."

**RNG.** `get_rng_state` gathers per-DP-rank states into a `ShardedObject` keyed with shape
`(pp_size, tp_size, dp_cp_size)`. On load: RNG is restored only if `ckpt_tp_pp == run_tp_pp and
ckpt_world_size == run_world_size`; otherwise it prints "(TP, PP) mismatch after resume ...: RNG
state will be ignored" or "Job sharding has changed: RNG state will be ignored".

**Other state.** `generate_state_dict` stores `args`, `checkpoint_version` (3.1), `iteration`,
`optimizer`, `opt_param_scheduler`, `rerun_state_machine`, `rng_state`, `run_config`,
`num_floating_point_operations_so_far`, train-state metadata. Rank 0 writes
`latest_checkpointed_iteration.txt`. `save_tokenizer_assets: bool = True` copies tokenizer files into
the checkpoint "to make checkpoints self-contained and portable". External dataloaders with a
`save_state` method are saved as `train_dataloader_dprank{rank}.pt` unless the iterable sets
`is_save_state_rank_independent = True` (then only `train_dataloader_dprank000.pt`).
`override_ckpt_iteration` resets `consumed_train_samples` to replay from a chosen iteration.

**Tools.** `--ckpt-convert-format {torch,torch_dist}` + `--ckpt-convert-save` (in-job conversion);
`tools/checkpoint/convert.py` (`--model-type {GPT,BERT}`, `--loader`, `--saver` plugins; TP/PP
re-split of legacy checkpoints through the saver is (unverified), I did not read `saver_core.py`);
`tools/checkpoint/checkpoint_inspector.py` (click commands `inspect`, `print-tensor`,
`convert-torch-dist-to-fsdp-dtensor`, `modify-state-dict`, `compare-two-checkpoint`,
`print-torch-dcp-in-json`); `tools/checkpoint/weighted_merge.py`, which is "metadata-driven", builds
no model, runs CPU-only over a Gloo group, and "launching multiple ranks reshards the read/write work
through DCP" (`docs/user-guide/features/checkpoint-merge.md`). The last is a working example of an
offline, GPU-less checkpoint transformer of the kind grid could run on a cheap CPU node.

**Speed/robustness knobs.** `async_save` (distributed formats only; requires
`use_persistent_ckpt_worker`; `async_strategy: Literal["nvrx"]`), `async_ckpt_cpu_priority=10`,
`async_ckpt_io_priority=3`, `ckpt_fully_parallel_load` (default `False`),
`ckpt_assume_constant_structure`, `dist_ckpt_workers`, `ckpt_pg_tensors_cache_path` ("Only safe when
the config and world size match the run that created the cache ... no existence/validity checks").
Multi-level: `non_persistent_save_interval`, `non_persistent_ckpt_type: Literal["global", "local",
"in_memory"]`, `non_persistent_local_ckpt_dir`, and `replication` / `replication_jump` /
`replication_factor=2` (requires `non_persistent_ckpt_type == 'local'`). NVRx docs: replication uses
`CliqueReplicationStrategy`, and on load the replication parameters "(i.e. `world_size`,
`--replication-jump` and `--replication-factor`) must be the same as during save"
(`docs/source/checkpointing/local/usage_guide.rst`). So the fast local tier is **shape-locked**; only
the global tier is portable. Object storage goes through NVIDIA's Multi-Storage Client
(`megatron/core/msc_utils.py`, `maybe_msc`).

**Integrity.** `verify_integrity: bool = False`: "hash checkpointing files during save and validate
their integrity during load"; implementation in `dist_checkpointing/validation.py` with
`INTEGRITY_FNAME = 'integrity.json'`, `_HASH_ALGORITHM = 'sha256'`, `_READ_CHUNK_SIZE = 1 << 20`;
"only supported with torch_dist checkpoint format".

## 3. DeepSpeed

- Native ZeRO checkpoints are per-rank: `zero_pp_rank_{dp_rank}_mp_rank_{mp:02d}_optim_states.pt`,
  `mp_rank_{NN}_model_states.pt` (`runtime/engine.py`).
- `zero_optimization.elastic_checkpoint` (stage 1/2, default `False`): on DP-size change,
  `stage_1_and_2.py` raises `ZeRORuntimeException` unless loading weights only: "Only fp32 weights
  saved with zero_optimization.elastic_checkpoint=true can be repartitioned, and only without
  optimizer states." The shrink/expand optimizer path is present only as commented-out pseudocode.
- **Universal Checkpoint** is a three-step offline flow (`docs/_tutorials/universal-checkpointing.md`):
  normal `engine.save_checkpoint`; `python deepspeed/checkpoint/ds_to_universal.py --input_folder ...
  --output_folder ...` (`--num_extract_workers` 4, `--num_merge_workers` 2, `--keep_temp_folder`,
  `--no_strict`, `--inject_missing_state`); then config `{"checkpoint": {"load_universal": true}}`
  (constant `LOAD_UNIVERSAL_CHECKPOINT = "load_universal"`; tag `latest_universal`). Output is one
  directory per parameter under `zero/<name>/` with `fp32.pt`, `exp_avg.pt`, `exp_avg_sq.pt`.
  It covers DP world-size change, AutoTP layouts (via `UNIVERSAL_CHECKPOINT_INFO`) and AutoEP
  `autoep_size` change; "combined AutoEP + AutoTP topology changes are not documented by this path".
  The conversion is a full read-and-rewrite of the checkpoint, so it costs a second copy of storage
  and minutes-to-hours of CPU I/O at hundreds of GB (no timing published in the tutorial).

## 4. torchtitan

`torchtitan/components/checkpointer/{base,dcp}.py`, `components/data/loader.py`, `trainer.py`.

- State keys: `MODEL`, `OPTIMIZER="optimizer"`, `LR_SCHEDULER="lr_scheduler"`,
  `DATALOADER="dataloader"`, `TRAIN_STATE="train_state"`, `EMA="ema"`; all pushed through DCP, so
  model and optimizer reshard. `ModelWrapper` flattens the per-stage modules of a PP job into one FQN
  namespace, which is what makes PP-split changes loadable (inference from the code; README states a
  seed checkpoint "can be loaded from another job on an arbitrary number of GPUs via DCP
  resharding").
- Config (`BaseCheckpointManager.Config`): `folder="checkpoint"`, `interval=500`,
  `keep_latest_k=10`, `load_step=-1`, `exclude_from_loading=[]`, `initial_load_path`,
  `last_save_model_only=True`, `export_dtype`; `CheckpointManager.Config.async_mode:
  Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"`. `async_with_pinned_mem` uses
  `AsyncCheckpointerType.PROCESS` plus a persistent `DefaultStager` and waits for the previous
  staging future before the next save.
- **The dataloader blocks elastic resume.** `loader.py` (a "GrainDataLoader") saves
  `{"version": 1, "dp_world_size": ..., f"dp_rank_{dp_rank}": iterator_state}` and on load raises
  `ValueError("cannot resume after changing the effective data-parallel degree")`. The README's
  documented escape hatch for changing "the number of GPUs" is
  `exclude_from_loading=["dataloader", "lr_scheduler"]`, i.e. data position is discarded.

## 5. Accelerate, HF Trainer, Lightning

- **Accelerate** (`checkpointing.py`): `save_state` writes model/optimizer/scheduler files once plus
  per-process `random_states_{process_index}.pkl` (which also carries `step`) and sampler state only
  for `SeedableRandomSampler`. RNG load is wrapped in `try/except Exception`, so a missing file for
  a new rank is skipped. `skip_first_batches(dataloader, num_batches)` re-iterates and skips; it
  counts *per-process batches*, so the same number means a different global sample offset at a
  different world size.
- **HF Trainer** (`trainer.py`): `_load_rng_state` looks for `rng_state_{process_index}.pth` when
  `world_size > 1`, else `rng_state.pth`, and logs "reproducibility is not guaranteed" if absent.
  Resume position is `steps_trained_in_current_epoch = global_step % num_update_steps_per_epoch`
  times `gradient_accumulation_steps`, replayed with `skip_first_batches` unless
  `ignore_data_skip`. `compare_trainer_and_checkpoint_args` only warns, and only about
  `logging_steps`, `eval_steps`, `save_steps`; nothing detects a world-size change, and
  `num_update_steps_per_epoch` itself changes with it.
- **Lightning** (`checkpoint_connector.dump_checkpoint`): a single rank-0 file with `epoch`,
  `global_step`, `pytorch-lightning_version`, `state_dict`, `loops`, `callbacks`,
  `optimizer_states`, `lr_schedulers`. Weights and optimizer are unsharded, hence world-size
  independent under DDP, but the whole state passes through one process. Data position is restored
  only for loaders that implement `state_dict`/`load_state_dict`
  (`CombinedLoader._state_dicts()` -> `loops` state `"combined_loader"`), which is the hook LitData
  and Mosaic loaders use.

## 6. Optimizer, LR schedule, accumulation, effective batch, RNG

- **Megatron**: `global_batch_size` defaults to `micro_batch_size * batch_parallel_size`;
  `num_micro_batches = global_batch_size // (micro_batch_size * data_parallel_size)` with an assert
  on divisibility (`core/num_microbatches_calculator.py`). If the user pins `--global-batch-size`,
  a DP change only changes accumulation; schedules indexed by iteration stay valid.
- **torchtitan**: `training.num_tokens_per_train_step` (default `-1` = per-rank tokens x DP);
  `gradient_accumulation_steps = num_tokens_per_train_step // (num_tokens_per_dp_rank * dp_degree)`,
  raising if not divisible. Pinning it gives constant tokens per step across DP sizes.
- **Accelerate**: `AcceleratedScheduler.step` calls the wrapped scheduler `num_processes` times per
  optimizer step unless `split_batches=True` ("the training dataloader batch size was multiplied by
  `num_processes`"). A saved scheduler `_step_count` therefore means a different amount of training
  at a different world size.
- **HF Trainer**: total batch is `per_device_train_batch_size * world_size * grad_accum`; no
  rescaling on resume.
- Adam moments are per-parameter and shape-independent, so they are valid after any resharding. What
  is *not* invariant: the gradient noise scale if the effective batch changes, and any
  per-micro-batch statistic (norm layers, per-rank loss scaling). If the batch must change, the
  usual linear-scaling/warmup heuristics apply and the run is no longer the same experiment; grid
  should record it as a new segment rather than pretend continuity (design judgment, not sourced).
- Rule for a job: define **step, LR schedule and data cursor in global units (tokens or samples)**,
  and require `tokens_per_step % (tokens_per_rank_microbatch * dp) == 0` for every DP grid offers.
  grid can enforce this by only offering shapes whose DP divides the configured global batch.
- RNG: keep model-level randomness (dropout) seeded from `(seed, global_step, logical_stream)`
  rather than restored per-rank generator state; no framework above reshards generator state.

## 7. Data loaders: is resume independent of world size?

| Loader | Resume API | Depends on rank count? |
|---|---|---|
| Megatron built-in (`data_samplers.py`) | global `consumed_samples` passed to `build_pretraining_data_loader(dataset, consumed_samples)` | `MegatronPretrainingSampler` (sequential): no, each global step takes `micro_batch_size * data_parallel_size` consecutive indices and a rank slices them. `MegatronPretrainingRandomSampler` with `data_sharding=True`: **yes**, `bucket_size`/`start_idx` are per DP rank; with sharding off the permutation is global (`torch.randperm` seeded by epoch) and striped `[rank::dp]`, so independent |
| Mosaic `StreamingDataset` | `StreamingDataLoader.state_dict()` -> `{'epoch', 'sample_in_epoch', 'num_canonical_nodes', 'shuffle_seed', 'initial_physical_nodes'}`; `load_state_dict` | No, by design: partitioning (`partition/orig.py`, "elastically deterministic sample order") is over `num_canonical_nodes`. Constraint: "Either canonical or physical nodes must be evenly divisible by the other". Default `num_canonical_nodes` is `64 * physical nodes` for `py1s`/`py2s`, else the initial node count. Last release is July 2025 |
| LitData `StreamingDataset` | `StreamingDataLoader.state_dict()` / `load_state_dict()`; state includes `num_samples_yielded`, `world_size`, `num_workers`, `batch_size`, `seed`, `sample_in_epoch`, `num_canonical_nodes`, `initial_world_size`, `state_version: 2`, `resume_mode` | No for plain `StreamingDataset`: `streaming/elastic.py`, "The epoch is a 1D stream of `(chunk_index, item_index)` that does not depend on `world_size` or `num_workers`. Resume drops a prefix (`sample_in_epoch`) and restripes the suffix". README: "`num_workers` and `world_size` may change"; but "`CombinedStreamingDataset` and `ParallelStreamingDataset` resume only with the same `world_size`, `num_workers`, and `batch_size`" |
| HF `datasets` streaming | `IterableDataset.state_dict()` / `load_state_dict()`; base state `{"shard_idx", "shard_example_idx", "type"}` | **Yes.** `split_dataset_by_node` assigns `shard_data_sources(num_shards=world_size, index=rank, contiguous=False)` when `num_shards % world_size == 0`, else keeps 1 of every `world_size` examples; state is relative to that subset. Docstring: shuffle-buffer contents "are lost when resuming" |
| torchtitan | `state_dict` keyed `dp_rank_{k}` + `dp_world_size` | **Yes, hard error** (section 4) |
| WebDataset | none (no `state_dict`/resume in `src/webdataset`) | **Yes**: `split_by_node` is `islice(src, rank, None, world_size)`, `split_by_worker` likewise |
| torchdata | `StatefulDataLoader.state_dict()/load_state_dict()` (`snapshot_every_n_steps=1`), `torchdata.nodes` `BaseNode.reset(initial_state)`/`state_dict()` | State is per process (`_num_yielded`, worker snapshots); world-size independence is whatever the sampler provides. Last PyPI release Feb 2025 |
| Lance | not read | (unverified) |

The portable pattern is identical in the three that work: **a deterministic global order as a pure
function of `(dataset manifest, seed, epoch)`, one integer cursor, and striping of the suffix onto
the current `(world_size, workers, batch_size)`**. For pre-tokenized LLM corpora this is cheap to
implement directly (global token offset into a manifest of shards).

## 8. Checkpoint speed, tiers, deltas, integrity

**Size.** DataStates-LLM Table I (DeepSpeed, DP=1): 7B = 13 GB FP16 parameters + 82 GB FP32
optimizer state; 13B = 25 GB + 148 GB ([arXiv 2601.16956](https://arxiv.org/html/2601.16956v1)).
That is ~13.5 GB per billion parameters for full Adam state, so a 70B run is ~950 GB and a
model-only bf16 export ~140 GB (extrapolated here).

**Measured rates** (same paper, ALCF Polaris, A100): per-GPU pinned D2H/H2D peak 25 GB/s; node-level
peak write ~10 GB/s; `torch.save` flush "only about 1 GB/s"; serialization ~22% of `torch.save`
checkpoint time; 7B single-rank host-to-file 16.1 s (DeepSpeed) vs 11.5 s (TorchSnapshot) vs 3.8 s
(DataStates-LLM); host cache 80 GB/node staged to Lustre. It does not benchmark DCP, and
"differential checkpointing, compression" are listed as future work. IBM: 30B model with optimizer
and dataloader state "in just over 3 minutes" with sync DCP. tensorizer README: local-file medians
~1.9-2.0 GiB/s for `torch.load()`, 2.25 GiB/s tensorizer, ~4.6 GiB/s `plaid_mode`; HTTP streaming
~0.9-1.1 GiB/s; "~5GB/s on a 40GbE network". s5cmd: only relative speedups over aws-cli were found
(search snippets, not opened), no GB/s (unverified).

**Time to move a 1 TB checkpoint** (computed here; assumes shards spread evenly, per-node rate is
the bottleneck):

| Nodes writing in parallel | 0.2 GB/s per node (1.6 Gbit/s WAN) | 1 GB/s per node | 5 GB/s per node |
|---|---|---|---|
| 8 | 10.4 min | 2.1 min | 25 s |
| 32 | 2.6 min | 31 s | 6 s |
| 128 | 39 s | 8 s | 2 s |

Sharded parallel upload makes object storage viable even on slow uplinks *if every node uploads its
own shard*; funnelling through rank 0 (legacy Megatron optimizer gather, `fully_reshardable`,
Lightning) forfeits this.

**Multi-level.** Now sourced (file 04 listed it as unverified): Megatron non-persistent local
checkpoints with NVRx clique replication, and DCP's `_ReplicationStager`. Both are world-size-locked.
The portable tier must be the global object-store checkpoint.

**Incremental/delta.** No framework above ships delta checkpoints. Research only: ExCP reports ~70x
compression via weight-residual + joint weight/momentum pruning and quantization (lossy, 410M-7B
models); Check-N-Run reports 6-17x write-bandwidth reduction for recommendation models (both from
search snippets, unverified). For dense Adam-trained LLMs every tensor changes every step, so
lossless deltas buy little; the practical lever is `SerializationFormat` + zstd (`stream.zstd`) or
dropping to a model-only export.

**Integrity.** DCP: none. safetensors: 8-byte header length + JSON header + raw bytes, no checksum
(README format section). Megatron: SHA-256 manifest, opt-in. tensorizer: per-tensor CRC32 and SHA-256
written by default (`include_crc32=True`, `include_sha256=True`), checked with
`TensorDeserializer(..., verify_hash=True)` raising `HashMismatchError`. Object stores verify
transport checksums per part but not that the content is what the job meant to write (general
knowledge, unverified). grid should hash at write time, in the uploader, per file.

## 9. State that must travel besides weights

Collected from what Megatron, torchtitan, LitData and Mosaic actually persist, plus what none do:

1. Optimizer state (all moments, fp32 master weights where separate), grad-scaler state, EMA copies.
2. LR scheduler state expressed in global steps; global step; tokens/samples consumed.
3. Data cursor: `(epoch, sample_in_epoch)` + `shuffle_seed` + `num_canonical_nodes` (or equivalent)
   + **dataset manifest hash** (shard list with sizes/digests) + tokenizer hash/assets
   (Megatron `save_tokenizer_assets`).
4. Full resolved config and the *logical* model architecture, separately from parallelism config
   (Megatron's `use_checkpoint_args` vs `use_mp_args_from_checkpoint_args` split is the model).
5. Code identity: git SHA + image digest + `torch.__version__` (DCP has no cross-version guarantee;
   `.metadata` is a pickle).
6. Precision recipe (bf16/fp8 scaling state; Megatron persists TE `_extra_state` for delayed
   scaling).
7. Tracker identity: W&B run id (resume with `id=..., resume="allow"`/`"must"`, per W&B docs,
   unverified here), metric step offset.
8. Seeds and, if same-shape, per-rank RNG state (optional; ignored on reshard).
9. Checkpoint manifest: file list, sizes, SHA-256, saving shape (world size, TP/PP/DP/EP),
   framework format and version, commit marker.

## Corrections to first sweep

- **03, table in 6.6**: "DeepSpeed ZeRO optimizer partitions ... world size change needs an elastic
  or universal checkpoint path (unverified)". Now verified and narrower: `elastic_checkpoint` allows
  repartitioning of fp32 weights only, "and only without optimizer states"; resuming *with*
  optimizer state at a different DP size requires the offline Universal Checkpoint conversion.
- **03, open question on Megatron** names `torch_dist` / `torch_dcp` / `fsdp_dtensor`: confirmed as
  the current set with `torch`; the brief's `zarr` format is not selectable on main. `torch_dcp` is
  asserted to TP<=1 and PP<=1, so it is not a general resharding path. The decisive flag was missing
  from 03: optimizer state is DP-reshardable by default and TP/PP-reshardable only when saved with
  `--dist-ckpt-optim-fully-reshardable`.
- **03, 6.6 "LR and batch scaling by WORLD_SIZE"**: more specific. Megatron and torchtitan do *not*
  scale by world size if the global batch is pinned; Accelerate steps the scheduler `num_processes`
  times per step; HF Trainer multiplies per-device batch by world size.
- **04, 4.2 "Two tiers ... no LLM-specific primary source was found (unverified)"** and the matching
  open question: sources exist (Megatron `--non-persistent-ckpt-type local` + `--replication`, NVRx
  `LocalCheckpointManager`/`CliqueReplicationStrategy`, DCP `_ReplicationStager`). Caveat to add:
  the local tier requires identical `world_size` and replication parameters on load.
- **04, 4.2 "Async"**: correct on `THREAD` default and `PROCESS`; missing that `async_save`
  hard-requires a CPU (Gloo) backend in the process group, and that Megatron's async save requires
  `--use-persistent-ckpt-worker`.
- **04, 4.2 "Consistency. ... Write to a temporary name, fsync, then publish a manifest atomically"**:
  DCP already does the manifest part (`.metadata.tmp` -> rename) but has `overwrite=True` by default
  and no content hashes; on fsspec object stores fsync degrades to `flush()` with a warning.
- **09, 4.3 "Sharded, per-rank checkpoints fetched with ranged parallel GETs let each rank start as
  soon as its own shard lands (inference)"**: true only for same-shape resume. After a reshape, a
  rank's `ReadItem`s span chunks in several saved `__{rank}_*.distcp` files, so per-rank staging
  must be driven by the DCP load plan (or the whole checkpoint staged), not by rank number.
- **README/03 on torchtitan**: torchtitan is not elastically resumable as shipped; its dataloader
  state check raises on any DP change.

## Implications for grid

**Proposed portable checkpoint contract (PCC).** A job is "portable" if every *durable* checkpoint
it publishes satisfies:

1. **Logical addressing.** Every tensor (weights, optimizer moments, EMA) is stored by FQN with its
   global shape; no file or key is named by rank, TP index or PP stage. Acceptable today: DCP with
   DTensor/ShardedTensor state (torchtitan, FSDP2), Megatron `torch_dist` saved with
   `--dist-ckpt-optim-fully-reshardable`, Megatron `fsdp_dtensor`, DeepSpeed Universal, any
   unsharded single-writer format (safetensors, Lightning `.ckpt`).
2. **Global-unit progress.** Step, LR schedule, and accumulation are defined by tokens (or samples)
   per optimizer step; the job declares the set of DP sizes for which divisibility holds.
3. **Shape-free data cursor.** `(manifest_hash, seed, epoch, global_sample_offset)` and a loader that
   restripes from it (Mosaic, LitData plain `StreamingDataset`, Megatron sampler without
   `data_sharding`, or a grid-supplied token-offset sampler).
4. **Sidecar manifest** `grid.ckpt.json` written last: format + framework versions, saving shape,
   file list with sizes and SHA-256, config, code SHA, image digest, tokenizer/data hashes, tracker
   run id, `tokens_seen`. The manifest is the commit marker; grid never trusts a directory without it.
5. **Rank-local state is optional.** RNG and loader-worker state may be saved but the job must start
   correctly without them.
6. **Self-test.** The job passes `grid ckpt verify`: save at shape A, load at shape B (different DP
   and, if claimed, TP/PP) on a small config, compare a forward-pass loss and optimizer-state norms.

**What grid does with a PCC job:** shrink, grow, change GPU type (subject to memory and precision
support), and move provider, by (a) reading the manifest, (b) choosing a shape whose DP divides the
global batch, (c) staging files according to the load plan, (d) relaunching. grid owns the two-tier
policy: frequent shape-locked local/buddy checkpoints for same-shape restart, and a slower cadence
of PCC checkpoints to object storage uploaded by each node in parallel with hashes computed in the
uploader.

**What grid offers jobs that do not meet it** (tiers, to be shown to the user at submit time):

- *Tier B, same-shape only*: rank-keyed checkpoints (legacy Megatron, ZeRO, torchtitan with its
  loader, Accelerate/Trainer). grid restores the identical world size and rank->shard mapping,
  replacing failed nodes like-for-like and holding spares; cross-provider moves are allowed only to
  an identical shape.
- *Tier C, offline conversion*: grid runs the framework's converter on a CPU node
  (`ds_to_universal.py`, Megatron `--ckpt-convert-format` / `checkpoint_inspector.py`
  `convert-torch-dist-to-fsdp-dtensor`, or a DCP `no_dist=True` re-save), then resumes. Costs a
  second copy and conversion time; data position may be lost.
- *Tier D, weights-only warm start*: load model weights, re-initialise optimizer and data cursor
  (torchtitan `exclude_from_loading`, Megatron `--no-load-optim --no-load-rng`, DeepSpeed
  `load_module_only`). grid must label this in run metadata as a discontinuity.

Other concrete consequences:

- Gloo/TCP reachability between all ranks is needed for async DCP and for several Megatron paths;
  preflight should test it.
- Pin `torch` and framework versions across the migration, or export through safetensors first.
- Use `overwrite=False`-style discipline: one immutable directory per step, manifest last, keep 2-3
  generations (Megatron and torchtitan `keep_latest_k=10` already do the latter).
- Never route a multi-hundred-GB checkpoint through rank 0 on a marketplace uplink; prefer formats
  whose save is fully parallel, and budget the `fully_reshardable` gather cost only on the durable
  cadence.

## Still open (needs hands-on experiment)

- DCP save/load throughput to S3-compatible stores (R2, B2, RunPod S3 API) through `FsspecWriter`
  vs local NVMe + s5cmd/rclone upload, at 100 GB-1 TB, with `thread_count` and
  `single_file_per_rank` varied. No public numbers found.
- Cost of Megatron `--dist-ckpt-optim-fully-reshardable` (gather on DP rank 0) at 100+ GB of
  optimizer state, with and without `--distrib-optim-fully-reshardable-mem-efficient`, and whether a
  `dp_reshardable` checkpoint can be converted offline instead.
- Actual TP-change and PP-change resume in torchtitan (DTensor TP, `ModelWrapper` over PP stages)
  with optimizer state; verified here only by reading, not by running.
- Whether DCP resharded loads from object storage issue ranged reads or whole-file downloads through
  fsspec, and the resulting egress volume when a small cluster loads a large cluster's checkpoint.
- Loss-curve continuity after DP change with pinned tokens-per-step (accumulation changes
  micro-batch composition) for Megatron and torchtitan; and after TP change (different reduction
  order, bf16).
- Mosaic `num_canonical_nodes` divisibility in practice for irregular marketplace shapes (e.g. 7 or
  13 nodes), and LitData elastic resume correctness with shuffle and multiple workers at scale.
- Async-save host-RAM headroom on marketplace nodes (one extra pinned copy of per-rank state) and
  interaction with container memory limits.
- SHA-256 cost at checkpoint scale (single-core SHA-256 is roughly 0.5-2 GB/s depending on CPU
  extensions, unverified) versus xxh3/CRC32C + object-store checksums.
- Lance, Ray Train, Orbax/JAX and s5cmd were not read; `tools/checkpoint/convert.py` TP/PP re-split
  options were not verified.
