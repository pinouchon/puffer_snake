# Single Snake V1 Parallel Sweep Portfolio 01

This document defines the recommended 8-sweep portfolio for `puffer_single_snake_v1`.

## Goal

Use 8 independent Protein sweeps in parallel, one per GPU, with intentionally different starting points and local ranges so the combined search covers more ground than 8 identical sweeps.

Common settings for all sweeps:

- project: `puffer_snake`
- group: `single_snake_v1_parallel`
- environment: `puffer_single_snake_v1`
- policy: `SingleSnakeV1Policy`
- one GPU per sweep

## Portfolio

### 1. `exploit_a`

Purpose: baseline exploit sweep around the current valid winner `y5j8ojni`.

Use the config file defaults as-is.

### 2. `exploit_b`

Purpose: tighter exploit sweep around the winner, focused on the strongest PPO region.

Overrides:

- `--train.total-timesteps 24000000`
- `--train.learning-rate 0.002144894941836089`
- `--train.gamma 0.9942040637696649`
- `--train.gae-lambda 0.948`
- `--train.clip-coef 0.175`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.952`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.9944`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.178`

### 3. `exploit_c`

Purpose: exploit near `y5j8ojni` while giving value loss and prioritization more room.

Overrides:

- `--train.total-timesteps 24000000`
- `--train.learning-rate 0.002144894941836089`
- `--train.gamma 0.9942040637696649`
- `--train.gae-lambda 0.948`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.80`
- `--sweep.train.vf-clip-coef.min 0.13`
- `--sweep.train.vf-clip-coef.max 0.17`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.44`

### 4. `fast_unnctkhe`

Purpose: search around the fastest high-variance run `unnctkhe`.

Overrides:

- `--train.total-timesteps 24000000`
- `--train.learning-rate 0.0021366739992030766`
- `--train.gamma 0.994`
- `--train.gae-lambda 0.948`
- `--train.clip-coef 0.16103181044718265`
- `--train.vf-clip-coef 0.13741701475481574`
- `--train.vf-coef 0.62`
- `--train.max-grad-norm 0.5019923732050768`
- `--train.prio-alpha 0.543523076161162`
- `--train.prio-beta0 0.38`

### 5. `fast_3ikcl10k`

Purpose: search around another fast but unstable candidate with a slightly different PPO region.

Overrides:

- `--train.total-timesteps 24000000`
- `--train.learning-rate 0.0020697376379905192`
- `--train.gamma 0.9941192753047684`
- `--train.gae-lambda 0.948`
- `--train.clip-coef 0.16868696611041994`
- `--train.vf-clip-coef 0.1377193364409552`
- `--train.vf-coef 0.62`
- `--train.max-grad-norm 0.44823668342706047`
- `--train.prio-alpha 0.525801845366934`
- `--train.prio-beta0 0.38`

### 6. `alt_jodgn1lq`

Purpose: keep coverage on the alternate basin represented by `jodgn1lq`.

Overrides:

- `--train.total-timesteps 20334987`
- `--train.learning-rate 0.0019880473771005758`
- `--train.gamma 0.9943237958961503`
- `--train.gae-lambda 0.9679465936885896`
- `--train.clip-coef 0.15958503401838242`
- `--train.vf-clip-coef 0.17128509493544697`
- `--train.vf-coef 0.7876720441505313`
- `--train.max-grad-norm 0.5082057271897793`
- `--train.prio-alpha 0.44993004234366496`
- `--train.prio-beta0 0.4137899039304431`
- `--sweep.train.total-timesteps.min 20000000`
- `--sweep.train.total-timesteps.max 24000000`
- `--sweep.train.gae-lambda.min 0.955`
- `--sweep.train.gae-lambda.max 0.970`

### 7. `long_horizon_push`

Purpose: test whether the new winner region improves with more total timesteps.

Overrides:

- `--train.total-timesteps 28000000`
- `--train.learning-rate 0.002144894941836089`
- `--train.gamma 0.9942040637696649`
- `--train.gae-lambda 0.948`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 36000000`

### 8. `broad_control`

Purpose: hedge against overfitting to the current local basin.

Use the config file sweep defaults, but with a unique tag so it can be analyzed separately.

## Example Commands

All commands follow the same pattern. Example for GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/puffer sweep puffer_single_snake_v1 \
  --wandb \
  --wandb-project puffer_snake \
  --wandb-group single_snake_v1_parallel \
  --tag exploit_a \
  --max-runs 150 \
  --train.stats-log-interval 250000 \
  > /tmp/exploit_a.log 2>&1 &
```

Example for `alt_jodgn1lq`:

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/puffer sweep puffer_single_snake_v1 \
  --wandb \
  --wandb-project puffer_snake \
  --wandb-group single_snake_v1_parallel \
  --tag alt_jodgn1lq \
  --max-runs 150 \
  --train.total-timesteps 20334987 \
  --train.learning-rate 0.0019880473771005758 \
  --train.gamma 0.9943237958961503 \
  --train.gae-lambda 0.9679465936885896 \
  --train.clip-coef 0.15958503401838242 \
  --train.vf-clip-coef 0.17128509493544697 \
  --train.vf-coef 0.7876720441505313 \
  --train.max-grad-norm 0.5082057271897793 \
  --train.prio-alpha 0.44993004234366496 \
  --train.prio-beta0 0.4137899039304431 \
  --sweep.train.total-timesteps.min 20000000 \
  --sweep.train.total-timesteps.max 24000000 \
  --sweep.train.gae-lambda.min 0.955 \
  --sweep.train.gae-lambda.max 0.970 \
  --train.stats-log-interval 250000 \
  > /tmp/alt_jodgn1lq.log 2>&1 &
```

## Batch Execution Guidance

- use one GPU per sweep
- use `150` runs per sweep
- keep a unique log file per sweep
- keep all sweeps in one `wandb` group
- pool results after the batch and validate the top candidates

## Expected Benefit

This portfolio should cover:

- the current validated winner basin
- nearby exploit regions with tighter PPO bounds
- fast but unstable basins worth re-testing
- the alternate `jodgn1lq` basin
- slightly longer-run candidates
- one broad hedge sweep
