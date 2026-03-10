# Reference: manual12x12_parallel50_v2

Source artifacts:

- local results: `/home/vast/puffer_snake/pufferlib/experiments/manual12x12_parallel50_v2/`

## Objective

- fastest wall-clock time to first `environment/score > 120`
- broad 50-run parallel search on `12x12`

## Best By Goal

1. `y2_lstm_128_long`
   - `t120 = 90.50s`
   - `best_score = 134.25`
2. `bridge_lstm_170_ent0`
   - `t120 = 100.94s`
   - `best_score = 132.42`
3. `y2_lstm_128`
   - `t120 = 108.92s`
   - `best_score = 131.62`
4. `bridge_plain_170_long`
   - `t120 = 110.34s`
   - `best_score = 133.78`
5. `bridge_lstm_192`
   - `t120 = 112.68s`
   - `best_score = 127.96`

## Main Takeaways

- `LSTM` clearly entered the frontier.
- `y2` and `bridge` became the two strongest 12x12 basins.
- The stable geometry was:
  - `3072 / 16 / 8192 / update_epochs=2`
- This batch established the modern 12x12 search space used later.

