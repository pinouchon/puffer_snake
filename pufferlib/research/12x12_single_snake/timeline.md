# Timeline

## Phase 1: Initial 12x12 Search

Early 12x12 searches established the main training geometry:

- `env.num_envs = 3072`
- `train.bptt_horizon = 16`
- `train.minibatch_size = 8192`
- `train.update_epochs = 2`

At this point the best basin was still plain CNN-based, but it was unstable under replication.

## Phase 2: LSTM Becomes Dominant

The architecture exploration showed that:

- `LSTM` repeatedly helped on `12x12`
- `GRU` did not catch up
- residual and higher-capacity plain variants were slower or less reliable

This shifted the search frontier toward:

- `y2_lstm_128`
- `y2_lstm_128_long`
- `bridge_lstm_170_ent0`
- `bridge_lstm_192`

## Phase 3: 180s Matched-Time Validation

The first strong matched-time validation was the `180s` batch.

Main outcome:

- `y2_lstm_128`
  - `t120 = 96.54s`
  - best score `125.68`
- `y2_lstm_128_long`
  - `t120 = 105.00s`
  - best score `127.39`
- `bridge_lstm_192`
  - `t120 = 106.60s`
  - best score `127.80`

Important negative result:

- `bridge_lstm_170_ent0` looked strong in longer screening runs
- but it did not hold up in the stricter matched-time validation

## Phase 4: 40-Run Short LSTM Search

The `185s` short-run LSTM batch was the most focused search around the LSTM frontier.

Main screening outcome:

- `y2_lstm_160_clip14`
  - `t120 = 80.42s`
  - `best_score = 130.44`

Strong runner-ups:

- `y2_lstm_128_ent0`
  - `t120 = 107.89s`
  - `best_score = 125.78`
- `bridge_lstm_192_ent0`
  - `t120 = 108.23s`
  - `best_score = 128.51`
- `y2_lstm_128_g9968_l9920`
  - `t120 = 109.52s`
  - `best_score = 128.07`

## Phase 5: 3x Validation Of The Top Short-Run Candidates

The 12-run validation batch changed the winner again by separating speed from reproducibility.

Validated outcome:

- `y2_lstm_128_ent0`
  - hit rate `3/3`
  - `t120 = 91.91s`, `104.34s`, `119.67s`
  - median `104.34s`

Near-miss challengers:

- `bridge_lstm_192_ent0`
  - hit rate `1/3`
  - one strong run at `96.33s`
- `y2_lstm_128_g9968_l9920`
  - hit rate `1/3`
- `y2_lstm_160_clip14`
  - hit rate `1/3`

Result:

- the fastest screened config was not the best validated config
- the current winner is the reproducible `y2_lstm_128_ent0`

