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
