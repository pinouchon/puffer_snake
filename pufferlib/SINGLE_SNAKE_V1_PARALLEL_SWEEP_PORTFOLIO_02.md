# Single Snake V1 Parallel Sweep Portfolio 02

This document defines the next recommended 8-sweep portfolio for `puffer_single_snake_v1`.

## Goal

Use 8 independent Protein sweeps in parallel, one per GPU, with a more focused search than Portfolio 01.

This portfolio is based on what the 150-run portfolio learned:

- `y5j8ojni` is still the best valid config
- `4a70fw4b` was the closest valid challenger
- `exploit_c` was the strongest basin by median sweep performance
- `bqi93gel` and `ejmd89gg` produced strong one-off runs but failed validation

Common settings for all sweeps:

- project: `puffer_snake`
- group: `single_snake_v1_parallel`
- environment: `puffer_single_snake_v1`
- policy: `SingleSnakeV1Policy`
- one GPU per sweep
- `max-runs = 80`

## Portfolio

### 1. `incumbent_control`

Purpose: anchor sweep around the current valid winner `y5j8ojni`.

Use the current config file defaults as-is, with a unique tag.

### 2. `exploit_c_tight`

Purpose: exploit the strongest basin from the previous portfolio.

Center on `gj71u10i` / `exploit_c`.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--train.clip-coef 0.1758868095026977`
- `--train.vf-clip-coef 0.15470946293822643`
- `--train.vf-coef 0.6108216342676894`
- `--train.max-grad-norm 0.5087626236773275`
- `--train.prio-alpha 0.5551867470045929`
- `--train.prio-beta0 0.38`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 32000000`
- `--sweep.train.learning-rate.min 0.00212`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.99395`
- `--sweep.train.gamma.max 0.99420`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.950`
- `--sweep.train.clip-coef.min 0.172`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.70`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.41`

### 3. `exploit_c_value`

Purpose: stay in the `exploit_c` basin but let value-learning terms move more.

Suggested overrides:

- `--train.total-timesteps 30080548`
- `--train.learning-rate 0.0021585990625102125`
- `--train.gamma 0.9940711103165775`
- `--train.gae-lambda 0.944`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 32000000`
- `--sweep.train.vf-clip-coef.min 0.14`
- `--sweep.train.vf-clip-coef.max 0.17`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.78`
- `--sweep.train.max-grad-norm.min 0.46`
- `--sweep.train.max-grad-norm.max 0.53`

### 4. `challenger_4a70fw4b_tight`

Purpose: search the closest valid challenger region more aggressively.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0020376389282354282`
- `--train.gamma 0.9943496536327361`
- `--train.gae-lambda 0.944`
- `--train.clip-coef 0.17516723755621605`
- `--train.vf-clip-coef 0.12`
- `--train.vf-coef 0.6`
- `--train.max-grad-norm 0.54`
- `--train.prio-alpha 0.4`
- `--train.prio-beta0 0.4367188893205707`
- `--sweep.train.total-timesteps.min 28000000`
- `--sweep.train.total-timesteps.max 32000000`
- `--sweep.train.learning-rate.min 0.00200`
- `--sweep.train.learning-rate.max 0.00208`
- `--sweep.train.gamma.min 0.99420`
- `--sweep.train.gamma.max 0.99445`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.950`
- `--sweep.train.clip-coef.min 0.172`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.68`

### 5. `challenger_4a70fw4b_long`

Purpose: test whether the `4a70fw4b` region improves with slightly more budget.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0020376389282354282`
- `--train.gamma 0.9943496536327361`
- `--train.gae-lambda 0.944`
- `--sweep.train.total-timesteps.min 30000000`
- `--sweep.train.total-timesteps.max 36000000`

### 6. `fast_unnctkhe_hedge`

Purpose: keep one sweep in the fast low-`gae_lambda` basin represented by `wq4tgc8p`.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.00217077224852588`
- `--train.gamma 0.9943510059037891`
- `--train.gae-lambda 0.9454393741805938`
- `--train.clip-coef 0.1605159951175878`
- `--train.vf-clip-coef 0.12`
- `--train.vf-coef 0.6`
- `--train.max-grad-norm 0.5143174752948377`
- `--train.prio-alpha 0.56`
- `--train.prio-beta0 0.38`

### 7. `mid_lambda_hedge`

Purpose: test whether the mid-`gae_lambda` basin can be made stable.

Center on the `bqi93gel` family, but narrower and more conservative.

Suggested overrides:

- `--train.total-timesteps 23999999`
- `--train.learning-rate 0.002139703141524258`
- `--train.gamma 0.9946828115670393`
- `--train.gae-lambda 0.9648722344893248`
- `--train.clip-coef 0.1743620874595081`
- `--train.vf-clip-coef 0.14490595811710133`
- `--train.vf-coef 0.7295451457385429`
- `--train.max-grad-norm 0.54`
- `--train.prio-alpha 0.475766838476095`
- `--train.prio-beta0 0.4003644274596563`
- `--sweep.train.total-timesteps.min 22000000`
- `--sweep.train.total-timesteps.max 28000000`
- `--sweep.train.gamma.min 0.9944`
- `--sweep.train.gamma.max 0.9948`
- `--sweep.train.gae-lambda.min 0.958`
- `--sweep.train.gae-lambda.max 0.966`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.176`

### 8. `broad_local_control`

Purpose: keep one broader sweep covering both promising local basins.

Suggested overrides:

- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00200`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9939`
- `--sweep.train.gamma.max 0.9948`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.966`
- `--sweep.train.clip-coef.min 0.160`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.82`

## Batch Execution Guidance

- use one GPU per sweep
- use `80` runs per sweep
- keep a unique log file per sweep
- keep all sweeps in one `wandb` group
- pool results after the batch and validate the top candidates

## Expected Benefit

This portfolio should:

- spend more of the budget in the two strongest validated regions
- keep one hedge on the fast low-`gae_lambda` basin
- keep one hedge on the mid-`gae_lambda` basin
- keep one broader local sweep to avoid premature over-tightening
