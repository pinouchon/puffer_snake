# Single Snake V1 Parallel Sweep Portfolio 04

This document defines the next recommended 8-sweep portfolio for `puffer_single_snake_v1`.

## Goal

Use 8 independent Protein sweeps in parallel, one per GPU, while reopening training geometry around the best known PPO regions.

Compared with Portfolios 01, 02, and 03, this portfolio changes the search shape in one important way:

- it reopens `env.num_envs`
- it reopens `train.bptt_horizon`
- it reopens `train.minibatch_size`

The intent is to test whether the later PPO gains interact with a better geometry than the currently baked `4096 / 32 / 8192` setup.

Common settings for all sweeps:

- project: `puffer_snake`
- group: `single_snake_v1_parallel`
- environment: `puffer_single_snake_v1`
- policy: `SingleSnakeV1Policy`
- one GPU per sweep

Suggested run budget:

- `max-runs = 80`

## Current Anchors

Primary low-`gae_lambda` anchor: `gj71u10i`

- `env.num_envs = 4096`
- `train.bptt_horizon = 32`
- `train.minibatch_size = 8192`
- `train.total_timesteps = 30080548`
- `train.learning_rate = 0.0021585990625102125`
- `train.gamma = 0.9940711103165775`
- `train.gae_lambda = 0.944`
- `train.clip_coef = 0.1758868095026977`
- `train.vf_clip_coef = 0.15470946293822643`
- `train.vf_coef = 0.6108216342676894`
- `train.max_grad_norm = 0.5087626236773275`
- `train.prio_alpha = 0.5551867470045929`
- `train.prio_beta0 = 0.38`

Alternate mid-`gae_lambda` anchor: `gw5h6eqs`

- `env.num_envs = 4096`
- `train.bptt_horizon = 32`
- `train.minibatch_size = 8192`
- `train.total_timesteps = 32000000`
- `train.learning_rate = 0.0019566116239649518`
- `train.gamma = 0.99435`
- `train.gae_lambda = 0.9631289549746254`
- `train.clip_coef = 0.17558733874130295`
- `train.vf_clip_coef = 0.1307952891967335`
- `train.vf_coef = 0.6`
- `train.max_grad_norm = 0.54`
- `train.prio_alpha = 0.5119379370227983`
- `train.prio_beta0 = 0.48588364562005204`

## Geometry Envelope

Use these as the broad geometry bounds for this portfolio:

- `env.num_envs = 3072 .. 6144`
- `train.bptt_horizon = 16 .. 64`
- `train.minibatch_size = 4096 .. 16384`

Tighter sweeps should favor:

- `env.num_envs = 3584 .. 5120`
- `train.bptt_horizon = 24 .. 40`
- `train.minibatch_size = 6144 .. 12288`

## Portfolio

### 1. `geom_low_h_core`

Purpose: exploit the current low-`gae_lambda` winner while reopening geometry in the most promising direction: shorter horizons.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--train.clip-coef 0.1758868095026977`
- `--train.vf-clip-coef 0.15470946293822643`
- `--train.vf-coef 0.6108216342676894`
- `--train.max-grad-norm 0.5087626236773275`
- `--train.prio-alpha 0.5551867470045929`
- `--train.prio-beta0 0.38`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3584`
- `--sweep.env.num-envs.max 5120`
- `--sweep.train.bptt-horizon.min 16`
- `--sweep.train.bptt-horizon.max 32`
- `--sweep.train.minibatch-size.min 6144`
- `--sweep.train.minibatch-size.max 12288`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00208`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.99425`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.950`
- `--sweep.train.clip-coef.min 0.172`
- `--sweep.train.clip-coef.max 0.178`

### 2. `geom_mid_h_core`

Purpose: test whether the current PPO winner prefers a slightly longer horizon once geometry is reopened.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3584`
- `--sweep.env.num-envs.max 5120`
- `--sweep.train.bptt-horizon.min 24`
- `--sweep.train.bptt-horizon.max 48`
- `--sweep.train.minibatch-size.min 6144`
- `--sweep.train.minibatch-size.max 12288`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00208`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.99425`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.952`

### 3. `geom_high_env_core`

Purpose: test whether the low-`gae_lambda` basin improves with more native env parallelism than `4096`.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 4096`
- `--sweep.env.num-envs.max 6144`
- `--sweep.train.bptt-horizon.min 16`
- `--sweep.train.bptt-horizon.max 40`
- `--sweep.train.minibatch-size.min 8192`
- `--sweep.train.minibatch-size.max 16384`
- `--sweep.train.total-timesteps.min 28000000`
- `--sweep.train.total-timesteps.max 36000000`
- `--sweep.train.learning-rate.min 0.00208`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.9943`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.950`

### 4. `mid_lambda_geom_core`

Purpose: reopen geometry around the strongest mid-`gae_lambda` winner without changing that basin too aggressively.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0019566116239649518`
- `--train.gamma 0.99435`
- `--train.gae-lambda 0.9631289549746254`
- `--train.clip-coef 0.17558733874130295`
- `--train.vf-clip-coef 0.1307952891967335`
- `--train.vf-coef 0.6`
- `--train.max-grad-norm 0.54`
- `--train.prio-alpha 0.5119379370227983`
- `--train.prio-beta0 0.48588364562005204`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3584`
- `--sweep.env.num-envs.max 5120`
- `--sweep.train.bptt-horizon.min 24`
- `--sweep.train.bptt-horizon.max 40`
- `--sweep.train.minibatch-size.min 6144`
- `--sweep.train.minibatch-size.max 12288`
- `--sweep.train.total-timesteps.min 28000000`
- `--sweep.train.total-timesteps.max 36000000`
- `--sweep.train.learning-rate.min 0.00193`
- `--sweep.train.learning-rate.max 0.00203`
- `--sweep.train.gamma.min 0.9942`
- `--sweep.train.gamma.max 0.9945`
- `--sweep.train.gae-lambda.min 0.956`
- `--sweep.train.gae-lambda.max 0.968`

### 5. `mid_lambda_geom_wide`

Purpose: keep the mid-`gae_lambda` basin alive with a wider geometry envelope and slightly wider PPO freedom.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0019566116239649518`
- `--train.gamma 0.99435`
- `--train.gae-lambda 0.9631289549746254`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3072`
- `--sweep.env.num-envs.max 6144`
- `--sweep.train.bptt-horizon.min 16`
- `--sweep.train.bptt-horizon.max 48`
- `--sweep.train.minibatch-size.min 4096`
- `--sweep.train.minibatch-size.max 16384`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 38000000`
- `--sweep.train.learning-rate.min 0.00192`
- `--sweep.train.learning-rate.max 0.00206`
- `--sweep.train.gamma.min 0.99415`
- `--sweep.train.gamma.max 0.99465`
- `--sweep.train.gae-lambda.min 0.956`
- `--sweep.train.gae-lambda.max 0.970`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.72`
- `--sweep.train.prio-beta0.min 0.44`
- `--sweep.train.prio-beta0.max 0.50`

### 6. `broad_local_geom_low_lambda`

Purpose: keep one broader local sweep over the low-`gae_lambda` family while allowing geometry interactions to emerge.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3072`
- `--sweep.env.num-envs.max 6144`
- `--sweep.train.bptt-horizon.min 16`
- `--sweep.train.bptt-horizon.max 64`
- `--sweep.train.minibatch-size.min 4096`
- `--sweep.train.minibatch-size.max 16384`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 38000000`
- `--sweep.train.learning-rate.min 0.00200`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.99395`
- `--sweep.train.gamma.max 0.99435`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.952`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.78`

### 7. `broad_local_geom_mixed`

Purpose: keep a hedge sweep spanning both the low- and mid-`gae_lambda` basins with geometry open.

Suggested overrides:

- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3072`
- `--sweep.env.num-envs.max 6144`
- `--sweep.train.bptt-horizon.min 16`
- `--sweep.train.bptt-horizon.max 64`
- `--sweep.train.minibatch-size.min 4096`
- `--sweep.train.minibatch-size.max 16384`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 38000000`
- `--sweep.train.learning-rate.min 0.00195`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9939`
- `--sweep.train.gamma.max 0.99465`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.968`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.82`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.50`

### 8. `incumbent_geom_control`

Purpose: keep one relatively tight control sweep around the current live winner while allowing modest geometry movement.

Suggested overrides:

- use the current config file defaults as the starting point
- `--sweep.sweep-only total_timesteps, learning_rate, gamma, gae_lambda, clip_coef, vf_clip_coef, vf_coef, max_grad_norm, prio_alpha, prio_beta0, num_envs, bptt_horizon, minibatch_size`
- `--sweep.env.num-envs.min 3584`
- `--sweep.env.num-envs.max 4608`
- `--sweep.train.bptt-horizon.min 24`
- `--sweep.train.bptt-horizon.max 40`
- `--sweep.train.minibatch-size.min 6144`
- `--sweep.train.minibatch-size.max 12288`

## Batch Execution Guidance

- use one GPU per sweep
- use `80` runs per sweep
- keep a unique log file per sweep
- keep all sweeps in one `wandb` group
- pool results after the batch and validate the top candidates

## Expected Benefit

This portfolio should:

- test whether `4096 / 32 / 8192` is still the best geometry after PPO refinement
- test whether lower horizons like `16` or `24` improve wall-clock convergence
- test whether larger `env.num_envs` values outperform `4096`
- keep both the low- and mid-`gae_lambda` basins alive
- preserve one control sweep so geometry reopening does not completely dilute search efficiency
