# Single Snake V1 Parallel Sweep Portfolio 05

This document defines the next recommended 8-sweep portfolio for `puffer_single_snake_v1` on the `12x12` board.

## Goal

Use 8 independent Protein sweeps in parallel, one per GPU, while keeping the same overall portfolio method as the earlier sweep docs and updating the search space to match the latest `12x12` findings.

Primary objective:

- minimize wall-clock time to first `environment/score > 120`

Common settings for all sweeps:

- project: `puffer_snake`
- group: `single_snake_v1_parallel_12x12`
- environment: `puffer_single_snake_v1`
- one GPU per sweep
- fixed env constraints:
  - `env.width = 12`
  - `env.height = 12`
  - `env.reward_food = 1.0`
  - `env.reward_step = -0.003`
  - `env.reward_death = -1.0`
  - observation space unchanged

Suggested run budget:

- `max-runs = 80`

## Current Anchors

Primary near-winner: `p12_m192_ent0`

- `policy_name = SingleSnakeV1Policy`
- `env.num_envs = 3072`
- `env.max_episode_steps = 12000`
- `train.total_timesteps = 180000000`
- `train.bptt_horizon = 16`
- `train.minibatch_size = 8192`
- `policy.cnn_channels = 32`
- `policy.hidden_size = 192`
- `train.learning_rate = 0.0011`
- `train.gamma = 0.9967`
- `train.gae_lambda = 0.992`
- `train.clip_coef = 0.15`
- `train.vf_clip_coef = 0.12`
- `train.vf_coef = 0.5`
- `train.update_epochs = 2`
- `train.ent_coef = 0.0`

Strong alternate basin: `c01_old_y2_ep12k`

- `policy_name = SingleSnakeV1Policy`
- `env.num_envs = 3072`
- `env.max_episode_steps = 12000`
- `train.total_timesteps = 180000000`
- `train.bptt_horizon = 16`
- `train.minibatch_size = 8192`
- `policy.cnn_channels = 32`
- `policy.hidden_size = 128`
- `train.learning_rate = 0.00115`
- `train.gamma = 0.9966`
- `train.gae_lambda = 0.991`
- `train.clip_coef = 0.15`
- `train.vf_clip_coef = 0.12`
- `train.vf_coef = 0.5`
- `train.update_epochs = 2`

Score-ceiling hedge: `h08_c24_m224_180m_a11_ep12k`

- `policy.cnn_channels = 24`
- `policy.hidden_size = 224`
- remaining train settings match the conservative `a11 / g9967 / l992 / ep12k` basin

## Search Envelope

Keep the dominant geometry fixed unless explicitly noted:

- `env.num_envs = 3072`
- `train.bptt_horizon = 16`
- `train.minibatch_size = 8192`
- `train.update_epochs = 2`

Primary numeric search ranges:

- `train.total_timesteps = 140000000 .. 220000000`
- `train.learning_rate = 0.0010 .. 0.0012`
- `train.gamma = 0.9965 .. 0.9969`
- `train.gae_lambda = 0.991 .. 0.993`
- `train.clip_coef = 0.14 .. 0.16`
- `train.vf_clip_coef = 0.10 .. 0.14`
- `train.vf_coef = 0.45 .. 0.55`
- `env.max_episode_steps = 10000 .. 14000`

Model search ranges:

- `policy.cnn_channels = 24 .. 40`
- `policy.hidden_size = 128 .. 224`

Discrete hedge dimensions for selected sweeps:

- `train.ent_coef = 0.0` or small positive default
- `train.anneal_lr = False` or `True`
- `train.prio_alpha` / `train.prio_beta0` around the current defaults

## Portfolio

### 1. `m192_ent0_core`

Purpose: exploit the current best near-winner directly.

Suggested overrides:

- `--policy-name SingleSnakeV1Policy`
- `--env.num-envs 3072`
- `--env.max-episode-steps 12000`
- `--train.total-timesteps 180000000`
- `--train.bptt-horizon 16`
- `--train.minibatch-size 8192`
- `--policy.cnn-channels 32`
- `--policy.hidden-size 192`
- `--train.learning-rate 0.0011`
- `--train.gamma 0.9967`
- `--train.gae-lambda 0.992`
- `--train.clip-coef 0.15`
- `--train.vf-clip-coef 0.12`
- `--train.vf-coef 0.5`
- `--train.update-epochs 2`
- `--train.ent-coef 0.0`

Sweep focus:

- `total_timesteps`
- `learning_rate`
- `gamma`
- `gae_lambda`
- `clip_coef`
- `vf_clip_coef`
- `vf_coef`
- `max_episode_steps`

### 2. `m192_ent0_clip_value`

Purpose: search the `ent_coef = 0` basin more tightly around clip/value terms, since that was the clearest improvement in the last batch.

Suggested emphasis:

- `clip_coef = 0.14 .. 0.16`
- `vf_clip_coef = 0.10 .. 0.13`
- `vf_coef = 0.45 .. 0.52`
- `learning_rate = 0.00105 .. 0.00115`
- `gamma = 0.9966 .. 0.9968`
- `gae_lambda = 0.9915 .. 0.9925`

### 3. `m192_ent0_timing`

Purpose: test whether the `ent0` basin wants shorter or longer budget/episode settings than the current `180M / 12000`.

Suggested emphasis:

- `total_timesteps = 140M .. 220M`
- `max_episode_steps = 10000 .. 14000`
- keep the PPO terms close to the current anchor

### 4. `old_y2_core`

Purpose: keep the old `32x128` y2 basin alive, since it still produced the best raw screen in the latest batch.

Suggested overrides:

- `--policy.cnn-channels 32`
- `--policy.hidden-size 128`
- `--train.learning-rate 0.00115`
- `--train.gamma 0.9966`
- `--train.gae-lambda 0.991`
- `--train.clip-coef 0.15`
- `--train.vf-clip-coef 0.12`
- `--train.vf-coef 0.5`
- `--env.max-episode-steps 12000`

Sweep focus:

- `total_timesteps`
- `learning_rate`
- `gamma`
- `gae_lambda`
- `max_episode_steps`

### 5. `old_y2_vs_ent0_bridge`

Purpose: bridge the gap between the old `32x128` y2 basin and the new `32x192 ent0` basin.

Suggested search:

- `policy.hidden_size = 160 .. 192`
- `ent_coef = 0.0 .. small positive`
- `learning_rate = 0.00108 .. 0.00116`
- `gamma = 0.9966 .. 0.9968`
- `gae_lambda = 0.991 .. 0.992`

This sweep exists to test whether the apparent split between the two basins is really a smooth path.

### 6. `model_capacity_hedge`

Purpose: search the best non-winning model-size directions from the last batch.

Suggested search:

- `policy.cnn_channels = 24 .. 32`
- `policy.hidden_size = 192 .. 224`
- keep train settings close to `a11 / g9967 / l992 / ep12k`

This should include the `h08`-style `24x224` region.

### 7. `optimizer_hedge`

Purpose: keep a hedge sweep on optimizer-side variants that looked acceptable in screens but did not yet validate.

Suggested search:

- `anneal_lr`
- `clip_coef = 0.14 .. 0.16`
- `vf_clip_coef = 0.10 .. 0.13`
- `vf_coef = 0.45 .. 0.52`
- `prio_alpha` / `prio_beta0` in a narrow band around the defaults

This should not be broad. It is a hedge, not the main search direction.

### 8. `broad_local_control`

Purpose: keep one broader local sweep across all currently interesting 12x12 basins, without reopening geometry.

Suggested search:

- `policy.hidden_size = 128 .. 224`
- `policy.cnn_channels = 24 .. 32`
- `learning_rate = 0.0010 .. 0.0012`
- `gamma = 0.9965 .. 0.9969`
- `gae_lambda = 0.991 .. 0.993`
- `clip_coef = 0.14 .. 0.16`
- `vf_coef = 0.45 .. 0.55`
- `max_episode_steps = 10000 .. 14000`
- `total_timesteps = 140M .. 220M`

This sweep is the hedge against overfitting too early to the `p12` basin.

## Allocation

Recommended emphasis:

- 3 sweeps on the `m192 ent0` basin
- 2 sweeps on the old `y2` basin and the bridge to it
- 1 sweep on model capacity
- 1 sweep on optimizer/refinement hedges
- 1 broad local control sweep

## Recommended Next Step

Use this portfolio if the next move is a large parallel sweep batch for `12x12`.

If the next move is a smaller or more conservative step, validate `p12_m192_ent0` further first.
