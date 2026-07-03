# Development notes

Deferred improvements and verification steps (not user-facing usage — see `README.md` for that).

## Known gaps / deferred improvements (see `rule/train.md`)

- [ ] **Stateful dataloader resume** (rule 1.2). A mid-epoch checkpoint currently
  resumes by restarting the epoch from batch 0, replaying already-seen samples.
  Needs a stateful loader (e.g. `torchdata.StatefulDataLoader`) so the sampler
  position is saved/restored, not just epoch/step.
- [ ] **Distributed validation sharding** (efficiency). `TorchrunTrainer` only
  shards the *train* loader; valid/test run the full set on every rank
  (N× redundant compute). Shard valid + `all_reduce` the metrics.
- [ ] **Remove dead legacy file** `torch_jaekwon/model/diffusion/ddpm/ddpm_loss_vlb.py`
  — imported nowhere and has broken old-layout imports (`torch_jaekwon.GetModule`,
  `torch_jaekwon.Util.*`, `torch_jaekwon.Model.*`); it cannot import.
- [ ] **Mixed precision (AMP)** — the trainer has no autocast / `GradScaler`
  support; add if bf16/fp16 training is wanted (rule 1.1 covers scaler state on resume).

## LightningTrainer — verify before trusting on a real experiment

`torch_jaekwon/train/trainer/lightning_trainer.py` is implemented but **compile-checked
only (never run)**. When you first use it for a real experiment (GPU node, never the login
node), confirm the following — a scratch smoke script (trivial model + dataset + a `run_step`)
is the quick way, but the checks apply equally to your real run:

Single-GPU (automatic optimization — the common case):
- [ ] loss decreases and training completes — confirms adapter wiring, `run_step` delegation,
  optimizer, and dataloaders all fire.
- [ ] EMA (`use_ema=true`): EMA updates each step, validation runs with EMA weights, and EMA
  state is saved inside `last.ckpt`.
- [ ] **Resume**: train → kill after ≥1 checkpoint → resume; confirm it restores and continues
  epoch/step (not from scratch). Verify `load_train()`'s `<dir>/last.ckpt` assumption matches
  where `ModelCheckpoint` actually wrote.

Multi-GPU DDP (single node):
- [ ] runs to completion on all ranks; only global rank-0 writes checkpoints (no shared-FS
  race); val metrics reduced via `sync_dist`.
- [ ] multi-optimizer / manual optimization: watch for DDP reducer errors ("parameter marked
  ready twice") from multiple `manual_backward` + `retain_graph`; may need
  `strategy=ddp_find_unused_parameters_true`.

Multinode (SLURM, the cluster path — see `ntd/script/run.sh`):
- [ ] Lightning SLURM auto-detect, global-rank sampling, single global rank-0 checkpoint.

Real-model gotchas to confirm:
- [ ] Effective batch = per-GPU batch × world_size (Lightning does **not** divide, unlike
  `TorchrunTrainer`) — re-tune LR/batch.
- [ ] `run_step` reading `self.device`/`self.global_step` works (synced each step; the batch is
  already device-moved, so `.to(self.device)` is a redundant no-op).
- [ ] FSDP/`ModelParallelStrategy`: checkpoint format + inject `data_parallel_size=$TOTAL_GPU`
  via `--set train.class_meta.args.strategy.args.data_parallel_size=$TOTAL_GPU`.
- [ ] `precision=bf16-mixed` numerics OK.

Known LightningTrainer gaps to implement if hit:
- [ ] `no_sync` during accumulation in manual mode (multinode all-reduce efficiency).
- [ ] nested dict-of-dicts models; EMA + multi-model.
- [ ] structural cleanup: lift the inner adapter to module level (independently testable) and
  reduce the `_sync_state` attribute-mutation bridge.

## Code audit (2026-07-03)

Static audit across all subsystems (vendored `external/` excluded). The core / data /
inference / util highlights were re-verified directly; **model / train / gan items are
audit-reported with line cites — confirm before fixing.** No code was run (login-node rule).

### Regression from the recent refactor — fix first
- [ ] `init_project.py:4,29` imports & uses `CLASS_DIRS`, which was removed from `path.py`
  during the dotted-`path` refactor → `ImportError`, project init is broken. Give
  `init_project.py` its own scaffold dir list (don't reintroduce `CLASS_DIRS` into `path.py`).

### Correctness bugs / crashes
- [ ] `controller.py:64` — `config_dict['cli'].get('train_resume_path', default)` never uses the
  default (`vars(args)` always has the key = `None`); `-r`/`--resume` without `--train_resume_path`
  → `None + "/train_checkpoint.pth"` `TypeError`. Use `... or f"{...}"`.
- [ ] `controller.py:63` — `os.path.splitext(relpath(...))` crashes when config isn't under
  `./config` (`relpath` returns `None`). Fall back to basename.
- [ ] `train/trainer/trainer.py:467-475` — base `backprop` calls `self.optimizer.step()` /
  `.zero_grad()` unconditionally, but `self.optimizer` can be a dict → `AttributeError`. Base can't
  do multi-optimizer. Iterate over dict values.
- [ ] `train/trainer/gan_trainer.py:14` — `super().__init__(model_class_name=...)`: base has no such
  param / no `**kwargs` → `TypeError`; class is unconstructable. Also `backprop`=`pass` returns
  `None` → `run_epoch` does `.detach()` on it → crash. GANTrainer is a non-functional stub.
- [ ] `util/util.py:42` — low-RAM guard formats `log_dict['available_ram_mb']` but the key is
  `ram_available_mb` → `KeyError` exactly when RAM is low (runs per-batch in the inferencer).
- [ ] `evaluate/metric/voice.py:189,192` — `get_sispnr` calls undefined `util_audio.energy_unify` /
  `pow_p_norm`; `'sispnr'` is in the default `metric_list` → `AttributeError` on the default eval path.
- [ ] `data/dataset/balanced_multi_dataset.py:39` — per-dataset seed built from a fresh
  `RandomState(random_seed)` each loop iter → identical seed for every dataset
  (`is_random_seed_per_dataset` is a no-op). Build one seed RNG before the loop.
- [ ] `model/multihead_attention.py:40` — `values = self.projection_key(values)` (should be
  `projection_value`) → silent wrong-weights bug; `projection_value` is unused.
- [ ] `model/diffusion/ddpm/ddpm.py:218` — `apply_model(..., is_cond_unpack=...)` but base
  `apply_model` dropped that param → positional/keyword collision → `TypeError`; breaks base DDPM
  sampling and every sampler (DDIM/PNDM/flash). Restore the param.
- [ ] `model/activation/snake.py` — `Snake` has no `forward` → `NotImplementedError` when used.
- [ ] `model/audio_module/Filter/Filter.py:40-50` — `kaiser_sinc_filter1d` leaves `filter` unbound
  on the `cutoff==0` branch → `UnboundLocalError`.
- [ ] `util/util_torch.py:41` — `freeze_param` sets `model.train = lambda self: self`; later
  `model.train()` → `TypeError` / silent no-op. Use `lambda *a, **k: model`.

### Cuda-seed at import (same class as the trainer.py fix already applied)
- [ ] `data/dataset/balanced_multi_dataset.py:13` (and `controller.py` runtime seed on CPU) —
  `int(torch.cuda.initial_seed()/2**32)` as a default forces CUDA at import / errors on CPU.
  Default to `None`, compute lazily.

### Dead / broken — stale old-layout (CamelCase) imports
These import from a pre-refactor `torch_jaekwon.Model.*` / `Util.*` / `GetModule` layout that no
longer exists → `ModuleNotFoundError`, so they're unusable. Fix imports to lowercase/relative, or
delete if abandoned:
- `evaluate/metric/sound.py`; `train/loss/MultiScaleSpectralLoss.py`;
  `model/diffusion/ddpm/ddpm_loss_vlb.py`; `model/diffusion/sampler/{DDIM,PNDM,DpmSolverForDDPM}.py`;
  `model/diffusion/ddpm/ddpm_learning_variances.py` (also calls missing `predict_start_from_noise`);
  `model/audio_module/{alias_free1d, Resample/UpSample1d, Resample/DownSample1d,
  Filter/LowPassFilter1d, Filter/.../MicrophoneEQ, FeatureExtract/ConstantQTransform}.py`.

### Other dead code
- [ ] `train/trainer/trainer.py:498` `metric_update` — dead duplicate of `update_metric`; delete.
- [ ] `train/trainer/trainer.py:512` `load_module` — dead and broken for dict/DDP models.
- [ ] `util/util_video.py` — `UtilVideo` referenced but never defined → `NameError`.
- [ ] `util/util_audio.py:256,293` — `analyze_dataset`/`resample_dataset` use nonexistent keys
  (`file_path` vs `walk`'s `file_abspath`) and wrong `walk(dir_name=)` param → `TypeError`/`KeyError`.
- [ ] `util/smoothing_function.py:57-63` — module-level scratch code runs at import (side effects,
  forces torch/matplotlib). Move under `__main__` or delete.
- [ ] `data/preprocess/torchrun_preprocessor.py:89` — template `ExampleDataset.__getitem__` never
  returns (always `None`); `data/preprocess/preprocessor.py:34` — `NotImplementedError` inside an
  empty `for` loop → silently returns `None` instead of raising.

### Footguns / inconsistencies
- [ ] `instantiate.py:8` — `'path' not in class_meta and isinstance(class_meta, dict)` checks
  membership before the type guard → `TypeError` on non-dict nested values. Reorder `isinstance` first.
- [ ] `train/trainer/trainer.py:375,379,392,397` — `print_and_log`/`log_every_epoch` in `fit()` are
  outside the `is_main_process()` guard → duplicated logs per rank under DDP.
- [ ] `train/logger/logger.py:119` — wandb path calls `wandb.log` once per metric → fragments the
  step history; batch into one `wandb.log(..., step=...)`.
- [ ] `train/trainer/trainer.py:295` — lr-scheduler init mutates the caller's config dict in place
  (injects a live optimizer object into `args`).
- [ ] `inference/inferencer.py:141` — `ckpt_name="last"` picks by lexical sort (wrong for numbered
  names; `IndexError` if none); `:123` per-batch `get_resource_usage(verbose=False)` can `sys.exit(1)`.
- [ ] `evaluate/evaluator/evaluator.py` — `result_dir_path=None` → writes into a dir literally named `None`.
- [ ] `data/dataset/balanced_multi_dataset.py:32,83` — partial `sampling_schedule_dict` → `KeyError`
  mid-iteration; `__len__` (finite) contradicts the infinite `__iter__`.
- [ ] `path.py:62,73` — path containment via `str.startswith` false-positives on sibling prefixes
  (`/a/bc` vs `/a/b`). Use `os.path.commonpath` or a trailing-sep compare.

### Efficiency
- [ ] `data/dataset/dataset.py` — eager-loads every pickle into RAM in `__init__` (duplicated per
  worker); lazy-load in `__getitem__`.
- [ ] `util/util_data.py:119` — `copy.deepcopy` of an array that's never mutated → doubles peak memory.
