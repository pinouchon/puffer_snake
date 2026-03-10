# Architecture Notes

## What Worked

### LSTM

`LSTM` was the clear architecture improvement on `12x12`.

It repeatedly improved:

- wall-clock time to `>120`
- late-run score ceiling
- robustness of the `y2` basin

Most of the important 12x12 winners or near-winners were:

- `y2_lstm_128`
- `y2_lstm_128_long`
- `y2_lstm_128_ent0`
- `bridge_lstm_170_ent0`
- `bridge_lstm_192`

### Moderate model size changes

Some capacity changes helped, but only when they stayed near the strongest basin.

Examples:

- `hidden_size = 160` with clip/value tuning had one exceptional screen win
- `hidden_size = 192` helped in the `bridge` family

The lesson is:

- modest capacity changes can help
- but basin and recurrent structure matter more than just making the model bigger

## What Did Not Become The Main Direction

### GRU

`GRU` never became competitive with `LSTM`.

It sometimes trained, but it did not beat the strongest LSTM basins on either:

- short-run `t120`
- or practical score ceiling

### Plain larger models

Larger plain CNNs sometimes improved score ceiling, but they did not become the best wall-clock path to `>120`.

### Residual variants

Residual policies could reach respectable score ceilings, but they were too slow.

Example from the 50-run parallel search:

- `residual_clip_192`
  - best score `133.64`
  - `t120 = 712.75s`

That is far from the practical frontier.

## Architecture Frontier At The Latest State

The strongest current architectural shortlist is:

1. `y2_lstm_128_ent0`
2. `bridge_lstm_192_ent0`
3. `y2_lstm_128_g9968_l9920`

The strongest currently validated one is still:

- `y2_lstm_128_ent0`

## Open Architecture Questions

The next architecture questions worth revisiting are:

- whether a more structured recurrent policy can stabilize `bridge_lstm_192_ent0`
- whether a slightly larger `y2` LSTM can match the screen speedups without losing reproducibility
- whether recurrent state handling or sequence usage inside the training loop can be improved further

