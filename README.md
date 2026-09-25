# Lightweight Coding Agent Training Pipeline

Day 1 implements a minimal ReAct coding agent. It reads a buggy calculator,
writes a minimal fix, executes tests, consumes the observation, prints a unified
diff, and saves the full trajectory to `trajectory.json`.

## Quick start

```bash
cd /home/lyq/coding-agent-training
python3 -m venv coding-agent
source coding-agent/bin/activate
python agent/main.py
```

The deterministic `demo` backend has no third-party dependency, which keeps the
demo reproducible. To use the actual open-source model backend:

```bash
pip install -r requirements.txt
python agent/main.py --backend Qwen/Qwen2.5-Coder-1.5B-Instruct
```

The first Hugging Face run downloads model weights. A GPU is recommended; the
1.5B model can run slowly on CPU. Before rerunning the demo, restore the exercise
bug with `return a + b  # BUG: should subtract` in `examples/calculator.py`.

## Day 1 architecture

```text
bug report -> policy (Demo / Qwen2.5-Coder)
                    |
              ReAct action parser
                    |
       +------------+-------------+
       |            |             |
   read_file    write_file      run_test
       |            |             |
       +------ observation <-------+
                    |
          next step or final patch
                    |
             trajectory.json
```

Security is intentionally simple but explicit: file access is confined to the
project workspace, test commands use an allowlist, subprocesses do not invoke a
shell, and every run has a timeout. Production agents need stronger isolation
(usually a disposable container per task).


## Day 2: training-data pipeline

Convert raw Agent trajectories into filtered JSONL:

    python data/collector.py trajectory.json --output data/train.jsonl

The pipeline rejects trajectories whose final test failed, whose patch is empty,
or whose patch changes more than three files. Earlier failed tests are retained
when the final test passes, because successful self-correction is useful
supervision. The limit is configurable with --max-modified-files.

Run its tests with:

    python -m unittest discover -s data -p 'test_*.py'


## Day 2: LoRA SFT

Train Qwen2.5-Coder on the filtered trajectory dataset:

    CUDA_VISIBLE_DEVICES=0 python training/train_sft.py

The script expands each trajectory into next-action policy examples, masks prompt
tokens from the loss, injects LoRA into q/k/v/o attention projections, trains
with Hugging Face Trainer, and saves a PEFT adapter plus tokenizer under
outputs/qwen2.5-coder-agent-lora. Use --resume-from-checkpoint to resume.


## Day 3: evaluation

Run the reproducible three-condition benchmark:

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python evaluation/run_evaluation.py

Each task runs in an isolated temporary workspace. Source files are writable,
tests are evaluator-owned, and the final suite is always run independently.
Outputs are evaluation/result.json (full traces and metrics) and
evaluation/REPORT.md (tables, findings, limitations, and next experiments).


## Version 2: SWE-bench Lite task data

Download SWE-bench Lite and create a deterministic 50-task Agent subset:

    HF_ENDPOINT=https://hf-mirror.com python scripts/load_swebench.py --num-tasks 50

All raw data, Hugging Face caches, task JSON files, and future repository
checkouts are kept under `/data_local/lyq/data_coding_agent`. No repository is
cloned by this loader and no SWE-bench data is written into the Git project.


## Version 2: prepare SWE-bench repositories

Prepare repositories at the exact task base commits:

    python scripts/prepare_repo.py --num-tasks 10

The manager keeps one partial bare mirror per GitHub repository and creates
commit-specific detached worktrees under
`/data_local/lyq/data_coding_agent/swebench/repos/<owner__repo>/<base_commit>`.
It checks Git >= 2.25, free disk, data-directory permissions, task schemas,
checkout commit identity, and discovers requirements/pyproject/setup/tox files.
Repeated valid workspaces are skipped. Use `--instance-id ID`, `--dry-run`, or
`--refresh-mirrors` for targeted operation. Repository dependencies are only
discovered in this phase; installation belongs to the later environment manager.


## Version 2: run a repository task

Create a task JSON outside the repository with `task_id`, `repo_path`,
`problem_statement`, and an exact `test_command`, then run:

    python agent/main.py --backend swebench --task-file /data_local/lyq/data_coding_agent/tasks/task.json

Use `--model-name` to select the base model and `--adapter-path` for an optional
PEFT adapter. The backend builds a bounded, issue-ranked Git file index, exposes
only read_file/write_file/the task-specific run_test command, forces final test
validation, and atomically stores unique rollout files under
`/data_local/lyq/data_coding_agent/trajectories/raw/`. The original calculator
demo remains available as `python agent/main.py`.


## Version 2: batch SWE-bench rollouts

Run a resumable batch with one shared model instance:

    python scripts/rollout_swebench.py --num-tasks 10

Each task runs in an isolated local shared clone. Atomic completion markers are
written as `trajectories/raw/task_xxxxx.json` with status, success/fail/error,
token usage, step count, trajectory, patch, and test result. Existing terminal
records are skipped. Use `--retry-errors` to archive and retry errors,
`--start-index` to shard, `--test-command-map` for per-instance/repo commands,
and `--dry-run` to inspect scheduling without loading the model. The fallback
`python -m pytest -q` is only a smoke command; benchmark-quality runs should use
a prepared command map and task-specific environments.
