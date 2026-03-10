# Reference: manual12x12_lstm40_validate12

Source artifacts:

- local results: `/home/vast/puffer_snake/pufferlib/experiments/manual12x12_lstm40_validate12/`

## Objective

- 3-rep validation of the top four candidates from the short 40-run LSTM batch
- same matched wall-clock cap of `185s`

Candidates:

- `y2_lstm_160_clip14`
- `y2_lstm_128_ent0`
- `bridge_lstm_192_ent0`
- `y2_lstm_128_g9968_l9920`

## Outcome

Current validated winner:

- `y2_lstm_128_ent0`
- hit rate `3/3`
- `t120 = 91.91s`, `104.34s`, `119.67s`
- median `104.34s`
- max best score `130.55`

Runner-ups:

- `bridge_lstm_192_ent0`
  - hit rate `1/3`
  - one hit at `96.33s`
  - max best score `129.99`
- `y2_lstm_128_g9968_l9920`
  - hit rate `1/3`
  - one hit at `107.02s`
- `y2_lstm_160_clip14`
  - hit rate `1/3`
  - one hit at `117.08s`

## Main Takeaway

The fastest screened candidate did not reproduce.

The best validated practical config was:

- `y2_lstm_128_ent0`

That is why the live 12x12 INI now points to this configuration.

