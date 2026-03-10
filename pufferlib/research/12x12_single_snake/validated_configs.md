# Validated Configs

## Current Winner

### `y2_lstm_128_ent0`

Validated in:

- [manual12x12_lstm40_validate12.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/references/manual12x12_lstm40_validate12.md)

Result:

- hit rate: `3/3`
- `t120`: `91.91s`, `104.34s`, `119.67s`
- median `t120`: `104.34s`
- max best score: `130.55`

Config:

- `policy_name = SingleSnakeV1Policy`
- `rnn_name = SingleSnakeV1LSTM`
- `env.num_envs = 3072`
- `env.max_episode_steps = 12000`
- `policy.cnn_channels = 32`
- `policy.hidden_size = 128`
- `train.total_timesteps = 220000000`
- `train.learning_rate = 0.00115`
- `train.update_epochs = 2`
- `train.bptt_horizon = 16`
- `train.minibatch_size = 8192`
- `train.clip_coef = 0.15`
- `train.ent_coef = 0.0`
- `train.gae_lambda = 0.991`
- `train.gamma = 0.9966`
- `train.max_grad_norm = 0.5`
- `train.vf_clip_coef = 0.12`
- `train.vf_coef = 0.5`

## Important Runner-Ups

### `bridge_lstm_192_ent0`

Validation outcome:

- hit rate: `1/3`
- one hit at `96.33s`
- max best score: `129.99`

Interpretation:

- strong upside
- not reproducible enough yet

### `y2_lstm_128_g9968_l9920`

Validation outcome:

- hit rate: `1/3`
- one hit at `107.02s`
- max best score: `125.50`

Interpretation:

- not good enough versus the winner

### `y2_lstm_160_clip14`

Validation outcome:

- hit rate: `1/3`
- one hit at `117.08s`
- max best score: `123.83`

Interpretation:

- best screening result from the 40-run search
- did not survive validation

## Earlier Validated Signals

### 180s matched-time validation

Earlier `180s` validation results that shaped the LSTM direction:

- `y2_lstm_128`
  - `t120 = 96.54s`
- `y2_lstm_128_long`
  - `t120 = 105.00s`
- `bridge_lstm_192`
  - `t120 = 106.60s`

Those results were important in moving the search toward:

- `y2`
- `LSTM`
- conservative PPO

