# Parallel task running

Run embarrassingly-parallel GPU work — preprocess shards, batched inference, feature
dumps — as many **independent 1-GPU workers racing a shared task list**. No ranks, no
inter-process communication, crash-safe and resumable.

## How it works (30 seconds)

- You split the work into **tasks** (one shard / one utterance / one file).
- Each task has a **done-marker** on disk. A worker *claims* a task with an atomic
  `mkdir`, processes it, and publishes the marker **last**. That gives you, for free:
  - many identical workers split the list with zero coordination (first to `mkdir` wins);
  - a killed worker leaves only a temp dir, never a half-written "done" task;
  - rerun = resume — only unfinished tasks remain.
- The driver submits one **wave** of workers; you rerun until 0 are left (the local
  backend loops for you).

### Control flow: `run_parallel_tasks.sh` → `tj_submit_wave` → `run_one_worker.sh`

```
[login]  run_parallel_tasks.sh                         DRIVER  (generic; no GPU)
           wipe orphan temps -> count leftover -> (if >0) submit one wave
                                                     |
                                                     v   calls the contract function:
         tj_submit_wave <job> <njobs> <hours> <module> [app args]
           sbatch --array  (cluster)   |   background procs (local)   <- BACKEND: the ONLY
                                                     |                    cluster-specific piece
                                                     v   launches N of, one per GPU:
[worker] run_one_worker.sh -m <module> -p <python> -r <repo> -- [app args]   WORKER (generic)
           set env/caches  ->  python -m <module> run [app args]
                                  -> ParallelTaskProcessor.run()   (claims + processes tasks)
```

Three layers, clean separation:
- **`run_parallel_tasks.sh` (driver)** — generic orchestration on the login node: wipe →
  count → submit one wave. Knows *nothing* about your cluster.
- **`tj_submit_wave` (backend/contract)** — the **only cluster-specific piece**, supplied by
  your `env_setup.sh` (or a shipped `backends/*.sh`). Turns "run a wave of N workers" into
  actual launches — `sbatch --array` on a cluster, background processes locally.
- **`run_one_worker.sh` (worker)** — generic per-GPU entry: sets caches/identity, then
  `python -m <module> run`, which races the claim list.

Per-run **app args flow straight through** all three: driver's `-- …` tail → `tj_submit_wave`
trailing args → worker's `-- …` → `python -m <module> run …`. The generic layers never
interpret them; only your module does.

## How work is split across workers (no sharding)

The surprising part: **there is no sharding step.** Workers are *not* assigned slices
(no "worker 0 gets items 0–9"). Every worker runs the same code, calls `list_tasks()`,
and sees the **whole** list. They avoid colliding with two filesystem checks per task:

```python
for task in self.list_tasks():                             # everyone sees the FULL list
    if self.is_task_done(task):                 continue   # 1. output exists -> skip (resume)
    if not self.claim(self.tmp_dir_path(task)): continue   # 2. another worker owns it -> skip
    self.process_task(task, self.tmp_dir_path(task))       # won it -> do it
```

`claim()` is just an atomic `mkdir` — **the temp dir IS the claim:**

```python
def claim(path):
    try:    os.makedirs(path); return True    # only ONE process can create a given dir
    except FileExistsError:    return False   # it already exists -> someone else got it
```

`mkdir` is atomic on the filesystem, so if two workers race for the same task exactly one
wins; that single operation is the *only* coordination between workers. They walk the same
list and whoever reaches an unclaimed task first does it. Example — 3 workers, 6 tasks:

```
A: mkdir .tmp_0 ✓ (task 0)        ...A free: .tmp_1✗ .tmp_2✗ .tmp_3 ✓ (task 3)
B: .tmp_0 ✗ → .tmp_1 ✓ (task 1)   ...B free: → .tmp_5 ✓ (task 5)
C: .tmp_0 ✗  .tmp_1 ✗ → .tmp_2 ✓  ...C free: → .tmp_4 ✓ (task 4)
```

Each does ~2 — not by assignment but by racing. This **self-balances** (a slow worker
just claims fewer) and needs no ranks, which is why the same worker runs unchanged as 1
local process or 40 cluster jobs. So "which data does a worker process?" — it doesn't know
in advance; it claims the next not-done, not-taken task, first-to-`mkdir` wins.

## Try the demo first (30 seconds, no GPU/scheduler)

A complete runnable example lives in `examples/`:

```bash
TJ_BACKEND=local bash examples/launch.example.sh -j demo   # runs 8 trivial tasks locally
TJ_BACKEND=local bash examples/launch.example.sh -j demo   # rerun -> "0 leftover" (resume)
```

`examples/` is also the copy-paste starting point for your own setup:
`example_task.py` (a minimal subclass), `env_setup.example.sh` (the contract, with the
slurm/local switch), `launch.example.sh` (the thin launcher).

Prefer to see the bare mechanism with no driver/backend at all?
`simple_local_run.sh <module> <n>` just launches N `python -m <module> run` processes
(GPU-pinned, separate caches) and waits — the claims alone keep their work disjoint. It's
the "you don't actually need any of the machinery" version; the driver/backend only add
GPU pinning, resume-by-wave, and crash recovery on top.

## Use it in 2 steps

### 1. Write a subclass

```python
# src/preprocess/my_task.py
import os, shutil
from torch_jaekwon.util.parallel.parallel_task_processor import ParallelTaskProcessor

class MyTask(ParallelTaskProcessor):
    def list_tasks(self):                  # ALL units of work, any type, stable order
        return [...]
    def is_task_done(self, task):          # True iff the done-marker exists
        return os.path.exists(out_path(task))
    def tmp_dir_path(self, task):          # per-task temp dir; co-locate with output
        return os.path.join(out_dir(task), f".tmp_{task_id(task)}")
    def process_task(self, task, tmp_dir): # tmp_dir already created (the claim).
        ...                                # write outputs INTO tmp_dir, then atomically
        ...                                # rename them into place — done-marker LAST —
        shutil.rmtree(tmp_dir)             # and remove tmp_dir on success.

    # optional: setup() runs ONCE per worker before the loop (e.g. load a model).

if __name__ == "__main__":
    MyTask.main()                          # exposes subcommands: run | count | wipe
```

Keep `__init__` / `list_tasks` / `is_task_done` / `tmp_dir_path` **import-light** (no
torch) so `count` / `wipe` stay cheap on the login node. Do heavy imports inside
`process_task` (or `setup`).

### 2. Run it

```bash
# Cluster: submits a wave of jobs. Rerun until it reports 0 leftover.
TJ_CLUSTER_ENV=/path/to/your/env_setup.sh \
  bash "$TJ_PKG/util/parallel/run_parallel_tasks.sh" -M src.preprocess.my_task -j mytask

# Local (no scheduler): one worker per GPU, loops to completion in one command.
TJ_BACKEND=local TJ_CLUSTER_ENV=/path/to/your/env_setup.sh \
  bash "$TJ_PKG/util/parallel/run_parallel_tasks.sh" -M src.preprocess.my_task
```

Run from your repo root so `-M src.preprocess.my_task` resolves. Most projects wrap this
in a one-line launcher that sets `TJ_CLUSTER_ENV` — copy the pattern from
`examples/launch.example.sh`.

Driver flags: `-M <module>` (required) · `-j <job_name>` · `-t <hours>` · `-m <max_workers_per_wave>`
· `-- <app args>` (optional; see below).

### Passing per-run args to your module (optional)

Anything after `--` on the driver is forwarded verbatim to `python -m <module> {run,count,wipe} <app args>`,
so a module can take **argparse** config instead of hardcoding it:

```bash
... run_parallel_tasks.sh -M src.inference.my_infer -- --config a.yaml --ckpt step-150000.ckpt
```

Each job carries its own args (no shared state), so multiple configs run concurrently. The
generic layer never interprets them. One rule: **no spaces/commas in any value** — args
word-split through the scheduler.

## The cluster contract

`$TJ_CLUSTER_ENV` (your project's `env_setup.sh`) must define:

- `TJ_PYTHON` — interpreter used on the login node and inside jobs.
- `tj_submit_wave <job> <njobs> <hours> <module> [app args...]` — spawn `njobs` independent
  workers, each running `run_one_worker.sh -m <module> -p <TJ_PYTHON> -r <TJ_REPO> -- [app args...]`
  (named flags, so adding knobs never shifts positions). (Don't pass comma-bearing values —
  SLURM splits `--export` on commas.)

Backends are swappable there: the default is your cluster's `sbatch` implementation;
`TJ_BACKEND=local` sources the shipped `backends/local.sh` (background processes, one per
GPU, sets `TJ_WAVE_BLOCKS=1` so the driver loops until done). **A new environment = one
new `backends/*.sh`; the driver, worker, and Python stay unchanged.**

## Files

| File | Role |
|------|------|
| `parallel_task_processor.py` | `ParallelTaskProcessor` base — the claim/resume/CLI logic. Subclass this. |
| `run_parallel_tasks.sh` | Driver (login entry point): wipe → count → submit one wave. |
| `run_one_worker.sh` | Worker: one per GPU, runs `python -m <module> run`. |
| `backends/local.sh` | No-scheduler backend (local background processes). |
| `examples/` | Runnable demo + copy-paste templates (task, contract, launcher). |
