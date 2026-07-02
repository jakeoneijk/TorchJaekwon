# Torch Jaekwon
This is my personal repository for the efficiency of my research.

## Set up
* Clone the repository.
    ```
    git clone https://github.com/jakeoneijk/TorchJaekwon
    ```

* Install TorchJaekwon
    ```
    source install_torchjk.sh
    ```

* Init project based on torch_jaekwon.
    ```
    python init_project.py -d target_path
    ```

## Usage
### Basic
```shell
> python main.py

arguments:
    --stage STAGE_NAME
    #STAGE of the system. choices = ['preprocess', 'train', 'inference', 'evaluate']
```

## TODO

Known gaps / deferred improvements (see `rule/train.md`):

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

