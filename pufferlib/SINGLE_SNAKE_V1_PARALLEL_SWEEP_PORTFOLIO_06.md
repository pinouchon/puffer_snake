# SINGLE_SNAKE_V1 Parallel Sweep Portfolio 06

This portfolio follows the partial results from `Portfolio 05 v2` on the `12x12` setup. The goal remains:

- primary: minimize wall-clock time to first `environment/score > 120`
- secondary: raise the score ceiling after crossing `120`

## Fixed Settings

These stay fixed across the portfolio:

```ini
env.width = 12
env.height = 12
env.reward_food = 1.0
env.reward_step = -0.003
env.reward_death = -1.0
observation space unchanged
vec.backend = PufferEnv
vec.num_envs = 1
env.num_envs = 3072
train.bptt_horizon = 16
train.minibatch_size = 8192
train.update_epochs = 2
```

## Why Portfolio 06

Portfolio 05 narrowed the search space materially:

- fastest basin: `old_y2_vs_ent0_bridge`
- strongest reliability basin: `m192_ent0_clip_value`
- strongest surprise basin: `model_capacity_hedge`

Portfolio 06 should therefore:

1. exploit the bridge basin harder
2. dedicate more budget to the clip/value-tuned `32x192, ent=0` basin
3. keep a focused model-capacity branch alive
4. keep one `old_y2` control sweep
5. drop broad local and optimizer-only sweeps

## Recommended 8-Sweep Portfolio

### 1. `bridge_fast_core`

Purpose: exploit the current fastest basin.

- `policy.hidden_size: 160..180`
- `train.ent_coef: 0.00008..0.00016`
- `train.learning_rate: 0.00110..0.00113`
- `train.gamma: 0.99672..0.99680`
- `train.gae_lambda: 0.99165..0.99190`

### 2. `bridge_fast_wide`

Purpose: keep the bridge basin slightly wider in case the winner sits near the edge.

- `policy.hidden_size: 160..192`
- `train.ent_coef: 0.0..0.00020`
- `train.learning_rate: 0.00109..0.00114`
- `train.gamma: 0.99668..0.99680`
- `train.gae_lambda: 0.99150..0.99195`

### 3. `clip_value_core`

Purpose: exploit the strongest reliable basin.

- `policy.cnn_channels = 32`
- `policy.hidden_size = 192`
- `train.ent_coef = 0.0`
- `train.learning_rate: 0.00112..0.00114`
- `train.gamma: 0.99673..0.99678`
- `train.gae_lambda: 0.99168..0.99180`
- `train.clip_coef: 0.141..0.147`
- `train.vf_coef: 0.47..0.50`
- `train.vf_clip_coef: 0.105..0.112`
- `train.max_grad_norm: 0.43..0.48`

### 4. `clip_value_wide`

Purpose: allow a slightly broader PPO-tuning neighborhood around the reliable `m192` basin.

- `policy.cnn_channels = 32`
- `policy.hidden_size = 192`
- `train.ent_coef = 0.0`
- `train.learning_rate: 0.00110..0.00115`
- `train.gamma: 0.99670..0.99680`
- `train.gae_lambda: 0.99160..0.99190`
- `train.clip_coef: 0.140..0.150`
- `train.vf_coef: 0.46..0.51`
- `train.vf_clip_coef: 0.10..0.115`
- `train.max_grad_norm: 0.42..0.50`

### 5. `capacity_core`

Purpose: exploit the best larger-model branch.

- `policy.cnn_channels: 30..32`
- `policy.hidden_size: 200..216`
- `env.max_episode_steps: 12000..13000`
- `train.total_timesteps: 155M..185M`
- `train.learning_rate: 0.00105..0.00109`
- `train.gamma: 0.99666..0.99674`
- `train.gae_lambda: 0.99160..0.99175`

### 6. `capacity_wide`

Purpose: keep capacity exploration alive without reopening the entire model space.

- `policy.cnn_channels: 28..32`
- `policy.hidden_size: 192..224`
- `env.max_episode_steps: 11000..14000`
- `train.total_timesteps: 150M..210M`
- `train.learning_rate: 0.00104..0.00110`
- `train.gamma: 0.99662..0.99676`
- `train.gae_lambda: 0.99155..0.99185`

### 7. `old_y2_control`

Purpose: keep the stable older basin as a hedge and comparison line.

- `policy.cnn_channels = 32`
- `policy.hidden_size = 128`
- `train.learning_rate: 0.00111..0.00118`
- `train.gamma: 0.99658..0.99668`
- `train.gae_lambda: 0.9909..0.9914`
- `env.max_episode_steps: 11000..13000`
- `train.total_timesteps: 150M..210M`

### 8. `bridge_clip_mix`

Purpose: explicitly combine the fastest bridge basin with the successful clip/value tuning from the `m192` branch.

- `policy.hidden_size: 176..192`
- `train.ent_coef: 0.0..0.00012`
- `train.learning_rate: 0.00110..0.00114`
- `train.gamma: 0.99670..0.99680`
- `train.gae_lambda: 0.99160..0.99190`
- `train.clip_coef: 0.141..0.148`
- `train.vf_coef: 0.47..0.50`
- `train.vf_clip_coef: 0.105..0.115`

## What Portfolio 06 Should Answer

1. Can the bridge basin keep its speed advantage while improving hit rate?
2. Is the `32x192, ent=0` tuned branch actually the best stable basin?
3. Does larger model capacity improve ceiling without giving up too much wall-clock speed?
4. Does bridge-plus-clip-value beat both parent basins?

## What To Deprioritize

These were not good enough in Portfolio 05 to deserve dedicated budget in 06:

- `broad_local_control`
- `optimizer_hedge`
- plain `m192_ent0_core`
- timing-only `m192_ent0_timing`

## Recommended Next Step

Run Portfolio 06 as another parallel 8-sweep batch, then validate:

- the best bridge candidate
- the best clip/value candidate
- the best capacity candidate
- the best bridge/clip mixed candidate
