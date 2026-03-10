# Reference: manual12x12_validate180

Source artifacts:

- local results: `/home/vast/puffer_snake/pufferlib/experiments/manual12x12_validate180/`

## Objective

- compare leading 12x12 candidates at a strict matched wall-clock cap of `180s`

## Best By Goal

1. `y2_lstm_128`
   - `t120 = 96.54s`
   - `best_score = 125.68`
   - `score_at_180s = 117.17`
2. `y2_lstm_128_long`
   - `t120 = 105.00s`
   - `best_score = 127.39`
   - `score_at_180s = 122.62`
3. `bridge_lstm_192`
   - `t120 = 106.60s`
   - `best_score = 127.80`
   - `score_at_180s = 121.60`

## Important Negative Result

- `bridge_lstm_170_ent0` did **not** reach `>120` within `180s`

This mattered because that config had looked much better in looser screen conditions.

## Main Takeaways

- `y2_lstm_128` and `y2_lstm_128_long` were the strongest matched-time candidates at this stage.
- `bridge_lstm_192` stayed relevant.
- This batch narrowed the search toward the LSTM-heavy frontier.

