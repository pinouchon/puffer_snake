# Single Snake V1 Parallel Sweep Portfolio 03

This document defines the next recommended 8-sweep portfolio for `puffer_single_snake_v1`.

## Goal

Use 8 independent Protein sweeps in parallel, one per GPU, while keeping coverage over all currently interesting basins.

Compared with Portfolio 02, this portfolio deliberately widens several sweeps instead of only tightening further. The intent is to:

- keep the strongest `exploit_c_tight` basin
- keep the strong `broad_local_control` basin
- keep the `4a70fw4b` challenger basin
- keep the mid-`gae_lambda` hedge basin
- re-open some parameter ranges around those basins to avoid premature local overfitting

Common settings for all sweeps:

- project: `puffer_snake`
- group: `single_snake_v1_parallel`
- environment: `puffer_single_snake_v1`
- policy: `SingleSnakeV1Policy`
- one GPU per sweep

Suggested run budget:

- `max-runs = 80`

## Portfolio

### 1. `incumbent_wide`

Purpose: keep one sweep anchored on the current valid winner `y5j8ojni`, but allow it to move more than the plain control.

Suggested overrides:

- `--train.total-timesteps 24000000`
- `--train.learning-rate 0.002144894941836089`
- `--train.gamma 0.9942040637696649`
- `--train.gae-lambda 0.948`
- `--train.clip-coef 0.175`
- `--train.vf-clip-coef 0.14689315344961612`
- `--train.vf-coef 0.62`
- `--train.max-grad-norm 0.4948932777819545`
- `--train.prio-alpha 0.5483760094269113`
- `--train.prio-beta0 0.38`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00205`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.9945`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.958`

### 2. `exploit_c_core`

Purpose: keep the strongest basin from Portfolio 02 with only moderate widening.

Center on `ln5vj5cm`.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0021737817700424986`
- `--train.gamma 0.9942`
- `--train.gae-lambda 0.9443245587164626`
- `--train.clip-coef 0.17421207146880233`
- `--train.vf-clip-coef 0.175`
- `--train.vf-coef 0.6230356746724286`
- `--train.max-grad-norm 0.43795925585790585`
- `--train.prio-alpha 0.56`
- `--train.prio-beta0 0.3801332212093722`
- `--sweep.train.total-timesteps.min 28000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00212`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.9943`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.950`
- `--sweep.train.clip-coef.min 0.172`
- `--sweep.train.clip-coef.max 0.178`

### 3. `exploit_c_wide`

Purpose: widen the same basin more aggressively, especially on value and priority terms.

Suggested overrides:

- `--train.total-timesteps 32000000`
- `--train.learning-rate 0.0021737817700424986`
- `--train.gamma 0.9942`
- `--train.gae-lambda 0.9443245587164626`
- `--sweep.train.total-timesteps.min 26000000`
- `--sweep.train.total-timesteps.max 36000000`
- `--sweep.train.learning-rate.min 0.00208`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.99395`
- `--sweep.train.gamma.max 0.9945`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.956`
- `--sweep.train.vf-clip-coef.min 0.13`
- `--sweep.train.vf-clip-coef.max 0.175`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.82`
- `--sweep.train.max-grad-norm.min 0.42`
- `--sweep.train.max-grad-norm.max 0.54`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.46`

### 4. `broad_local_core`

Purpose: preserve the broad local basin that performed nearly as well as `exploit_c_tight`.

Center on `eshyj26o`.

Suggested overrides:

- `--train.total-timesteps 25430087`
- `--train.learning-rate 0.0020576558454914414`
- `--train.gamma 0.9940952431477217`
- `--train.gae-lambda 0.952188681839592`
- `--train.clip-coef 0.17294891464337706`
- `--train.vf-clip-coef 0.12594663390889763`
- `--train.vf-coef 0.7215826807729899`
- `--train.max-grad-norm 0.5253099897131324`
- `--train.prio-alpha 0.46840401798742426`
- `--train.prio-beta0 0.40635679860597285`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 32000000`
- `--sweep.train.learning-rate.min 0.00200`
- `--sweep.train.learning-rate.max 0.00214`
- `--sweep.train.gamma.min 0.9940`
- `--sweep.train.gamma.max 0.9945`
- `--sweep.train.gae-lambda.min 0.948`
- `--sweep.train.gae-lambda.max 0.960`

### 5. `broad_local_wide`

Purpose: widen the broad-local basin so it can span both the lower- and mid-`gae_lambda` regions.

Suggested overrides:

- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 36000000`
- `--sweep.train.learning-rate.min 0.00200`
- `--sweep.train.learning-rate.max 0.00218`
- `--sweep.train.gamma.min 0.9939`
- `--sweep.train.gamma.max 0.9948`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.966`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.85`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.50`

### 6. `challenger_4a70fw4b_wide`

Purpose: keep the closest validated challenger basin, but reopen it beyond the tight version from Portfolio 02.

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
- `--sweep.train.total-timesteps.max 36000000`
- `--sweep.train.learning-rate.min 0.00198`
- `--sweep.train.learning-rate.max 0.00210`
- `--sweep.train.gamma.min 0.99415`
- `--sweep.train.gamma.max 0.99455`
- `--sweep.train.gae-lambda.min 0.944`
- `--sweep.train.gae-lambda.max 0.952`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.72`

### 7. `mid_lambda_core`

Purpose: preserve the mid-`gae_lambda` hedge basin that keeps producing strong one-off runs.

Center on `l1s39dkb`.

Suggested overrides:

- `--train.total-timesteps 28000000`
- `--train.learning-rate 0.001966237571846939`
- `--train.gamma 0.9944303070471072`
- `--train.gae-lambda 0.9621996589822355`
- `--train.clip-coef 0.176`
- `--train.vf-clip-coef 0.12385802162259296`
- `--train.vf-coef 0.9`
- `--train.max-grad-norm 0.54`
- `--train.prio-alpha 0.41794193234352717`
- `--train.prio-beta0 0.410979074855771`
- `--sweep.train.total-timesteps.min 24000000`
- `--sweep.train.total-timesteps.max 32000000`
- `--sweep.train.learning-rate.min 0.00195`
- `--sweep.train.learning-rate.max 0.00206`
- `--sweep.train.gamma.min 0.99435`
- `--sweep.train.gamma.max 0.9948`
- `--sweep.train.gae-lambda.min 0.958`
- `--sweep.train.gae-lambda.max 0.966`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.176`

### 8. `mid_lambda_wide`

Purpose: widen the mid-`gae_lambda` basin rather than abandoning it because of earlier variance failures.

Suggested overrides:

- `--sweep.train.total-timesteps.min 22000000`
- `--sweep.train.total-timesteps.max 34000000`
- `--sweep.train.learning-rate.min 0.00195`
- `--sweep.train.learning-rate.max 0.00214`
- `--sweep.train.gamma.min 0.9943`
- `--sweep.train.gamma.max 0.99485`
- `--sweep.train.gae-lambda.min 0.956`
- `--sweep.train.gae-lambda.max 0.966`
- `--sweep.train.clip-coef.min 0.166`
- `--sweep.train.clip-coef.max 0.178`
- `--sweep.train.vf-coef.min 0.60`
- `--sweep.train.vf-coef.max 0.90`
- `--sweep.train.prio-beta0.min 0.38`
- `--sweep.train.prio-beta0.max 0.48`

## Batch Execution Guidance

- use one GPU per sweep
- use `80` runs per sweep
- keep a unique log file per sweep
- keep all sweeps in one `wandb` group
- pool results after the batch and validate the top candidates

## Expected Benefit

This portfolio should:

- keep the currently strongest `exploit_c` basin
- keep the broad local basin that remains competitive
- preserve the `4a70fw4b` basin without overcommitting to the long version
- spend real budget on the mid-`gae_lambda` basin instead of discarding it
- widen enough ranges that the next batch can discover neighboring stable regions, not just re-sample the current exact winners
