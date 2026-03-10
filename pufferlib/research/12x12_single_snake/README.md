# 12x12 Single Snake Research

This folder is the tracked summary of the `12x12` `puffer_single_snake_v1` experiments.

It exists to preserve:

- the current validated config
- the best screening results
- the important validation outcomes
- the main architecture and hyperparameter lessons

It intentionally does **not** store raw artifacts such as checkpoints, JSONL logs, or full run directories. Those remain in local ignored paths under `experiments/`.

## Current Validated Winner

Current validated short-run winner:

- `y2_lstm_128_ent0`
- objective: fastest wall-clock time to first `environment/score > 120`
- matched validation cap: `185s`
- validation hit rate: `3/3`
- validation `t120`: `91.91s`, `104.34s`, `119.67s`
- validation median `t120`: `104.34s`
- max best score in validation: `130.55`

Current active config file:

- [single_snake_v1.ini](/home/vast/puffer_snake/pufferlib/pufferlib/config/ocean/single_snake_v1.ini)

Exact training command:

```bash
puffer train puffer_single_snake_v1
```

The INI currently resolves to:

```bash
puffer train puffer_single_snake_v1 \
  --vec.backend PufferEnv \
  --vec.num-envs 1 \
  --env.num-envs 3072 \
  --env.width 12 \
  --env.height 12 \
  --env.reward-food 1.0 \
  --env.reward-step -0.003 \
  --env.reward-death -1.0 \
  --env.max-episode-steps 12000 \
  --policy-name SingleSnakeV1Policy \
  --rnn-name SingleSnakeV1LSTM \
  --policy.cnn-channels 32 \
  --policy.hidden-size 128 \
  --train.total-timesteps 220000000 \
  --train.learning-rate 0.00115 \
  --train.update-epochs 2 \
  --train.bptt-horizon 16 \
  --train.minibatch-size 8192 \
  --train.clip-coef 0.15 \
  --train.ent-coef 0.0 \
  --train.gae-lambda 0.991 \
  --train.gamma 0.9966 \
  --train.max-grad-norm 0.5 \
  --train.vf-clip-coef 0.12 \
  --train.vf-coef 0.5
```

## Current Best Screened But Unvalidated Result

Best screened short-run result from the 40-run LSTM search:

- `y2_lstm_160_clip14`
- `t120 = 80.42s`
- `best_score = 130.44`

This did **not** hold up in the 3-rep validation:

- hit rate `1/3`

So it is a promising but unstable direction, not the current default.

## Key Takeaways

- `LSTM` is the only architecture change that has consistently improved the 12x12 frontier.
- The best geometry remains:
  - `env.num_envs = 3072`
  - `train.bptt_horizon = 16`
  - `train.minibatch_size = 8192`
  - `train.update_epochs = 2`
- The strongest reproducible basin is currently `y2 + LSTM + ent_coef=0`.
- `bridge_lstm_192_ent0` is the strongest current challenger.
- `GRU` and larger plain/residual variants did not become the best practical direction.

## Important References

- [timeline.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/timeline.md)
- [validated_configs.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/validated_configs.md)
- [architecture_notes.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/architecture_notes.md)
- [manual12x12_parallel50_v2.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/references/manual12x12_parallel50_v2.md)
- [manual12x12_validate180.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/references/manual12x12_validate180.md)
- [manual12x12_lstm40_3min.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/references/manual12x12_lstm40_3min.md)
- [manual12x12_lstm40_validate12.md](/home/vast/puffer_snake/pufferlib/research/12x12_single_snake/references/manual12x12_lstm40_validate12.md)

