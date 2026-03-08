# Parallel Sweeping

This document describes the practical way to run PufferLib Protein sweeps on multiple GPUs today.

## Current Limitation

`puffer sweep` does not run one shared sweep across multiple GPUs.

- Each sweep process owns its own in-memory `Protein` state.
- Suggestions and observations are not shared across processes.
- Running 8 sweep processes means running 8 independent sweeps.

Because of that, parallel sweeping should be treated as a portfolio search, not as one coordinated search.

## Recommended Approach

Run one sweep process per GPU.

- Pin each process with `CUDA_VISIBLE_DEVICES`.
- Keep runs in the same `wandb` project and group.
- Use different tags per sweep so results are easy to segment.
- Give different sweeps different initial configs or slightly different sweep bounds.

This gives better coverage than launching 8 identical sweeps.

## Why Portfolio Sweeps Work Better

Each sweep uses the current config unchanged for its first run, then Protein adapts from its own observations.

That means parallel advantage comes from:

- different starting points
- different local sweep ranges
- different exploration biases

If all 8 sweeps are identical, they still differ somewhat from randomization, but a meaningful fraction of the search becomes redundant.

## What To Vary Across Sweeps

The most useful per-sweep differences are:

- initial `train.*` values
- `sweep.train.total_timesteps.min/max`
- `sweep.train.gamma.min/max`
- `sweep.train.gae_lambda.min/max`
- `sweep.train.clip_coef.min/max`
- `sweep.train.vf_clip_coef.min/max`
- `sweep.train.vf_coef.min/max`
- `sweep.train.prio_beta0.min/max`

Keep the environment invariants fixed when required by experiment rules.

## Operational Pattern

Typical workflow:

1. Define a small portfolio of sweep variants.
2. Launch one sweep per GPU.
3. Let each sweep run for a modest number of runs, such as `20` to `30`.
4. Pool results across all sweeps.
5. Validate the strongest candidates with the standard variance rule.

## Command Pattern

Example shape:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/puffer sweep puffer_single_snake_v1 \
  --wandb \
  --wandb-project puffer_snake \
  --wandb-group single_snake_v1_parallel \
  --tag protein_gpu0 \
  --max-runs 25
```

Launch one command like this per GPU, changing GPU id, tag, and any per-sweep overrides.

## Logging

Recommended conventions:

- one shared `wandb` project
- one shared `wandb` group for the batch
- one unique tag per sweep variant
- one log file per process in `/tmp/` or a dedicated run directory

Example:

```bash
... --tag protein_exploit_a > /tmp/protein_exploit_a.log 2>&1 &
```

## Result Handling

After the batch:

- rank all runs by wall-clock time to first `environment/score >= 50`
- do not pick winners from single-run outliers alone
- validate top candidates with at least 3 replication runs
- require the existing variance rule before promoting a new default

## When To Build Something More Advanced

The current approach is good enough when:

- you want immediate multi-GPU throughput
- you can tolerate independent sweep state
- you are comfortable validating candidates afterward

A centralized coordinator is worth building only if you want one shared Protein model across all GPUs.
