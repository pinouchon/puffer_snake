# Reference: manual12x12_lstm40_3min

Source artifacts:

- local results: `/home/vast/puffer_snake/pufferlib/experiments/manual12x12_lstm40_3min/`

## Objective

- focused 40-run LSTM search
- target per-run wall-clock of about `185s`
- optimize time to first `score > 120`

## Best By Goal

1. `y2_lstm_160_clip14`
   - `t120 = 80.42s`
   - `best_score = 130.44`
2. `y2_lstm_128_ent0`
   - `t120 = 107.89s`
   - `best_score = 125.78`
3. `bridge_lstm_192_ent0`
   - `t120 = 108.23s`
   - `best_score = 128.51`
4. `y2_lstm_128_g9968_l9920`
   - `t120 = 109.52s`
   - `best_score = 128.07`

## Best By Score

1. `y2_lstm_160_clip14`
   - `best_score = 130.44`
2. `bridge_lstm_192_ent0`
   - `best_score = 128.51`
3. `y2_lstm_128_g9968_l9920`
   - `best_score = 128.07`

## Main Takeaways

- The strongest screening result shifted to `y2_lstm_160_clip14`.
- `bridge_lstm_192_ent0` emerged as the best bridge-family challenger.
- `y2_lstm_128_ent0` remained a strong practical candidate even when it did not win the screen.
- This batch created the exact shortlist used for the next 3-rep validation.

