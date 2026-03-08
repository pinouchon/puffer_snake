# Auto Experiment Rules

These rules define how automatic training experiments should be run for `puffer_single_snake_v1`.

## Goal

Optimize for the fastest wall-clock time to reach:

- `environment/score >= 50`

Primary metric:

- wall-clock time to first `score >= 50`

Secondary metrics:

- agent steps to first `score >= 50`
- score at matched wall-clock times
- score at matched agent-step counts

## Hard Constraints

These settings must not be changed:

- `env.width = 8`
- `env.height = 8`
- `env.reward_food = 1.0`
- `env.reward_step = -0.003`
- `env.reward_death = -1.0`

Other fixed constraints:

- use a single GPU only
- keep `train.total_timesteps <= 100_000_000`

## Allowed Changes

The following may be changed during experiments:

- vectorization backend and env count
- learning rate and LR schedule
- minibatch size
- horizon / batch geometry
- update epochs
- optimizer
- entropy coefficient and other RL hyperparameters
- policy architecture
- training-loop implementation details
- logging and profiling settings

## Measurement Rules

- Use dense periodic stats logging so runs can be compared without relying on terminal output alone.
- Prefer JSONL logs written from `train.stats_log_interval` and `train.stats_log_path`.
- Compare candidates at matched wall-clock times first.
- Use a longer confirmation run only for the most promising candidates.
- Do not declare a new winner from a very short run unless it is clearly ahead on the score-vs-time curve.

## Variance Rule

- A candidate config is not valid from a single fast run alone. It must be replicated.
- A valid replicated config must reach `environment/score >= 50` in every replication run used for validation.
- Use at least 3 replication runs for a winner check.
- The primary replicated score is the median wall-clock time to first `score >= 50`.
- A replicated config is considered to have reasonable variance only if:
  - coefficient of variation of time-to-50 is `<= 10%`
  - worst run time-to-50 is no more than `20%` slower than the best run in the validation set
- If a candidate is faster on a best-case run but fails the variance rule, it does not replace the current best config.

## Experiment Workflow

1. Start from the current best known single-GPU configuration.
2. Change only a small number of variables per experiment.
3. Run multiple short experiments in parallel when hardware allows, but each run must use one GPU.
4. Eliminate weak directions quickly.
5. Re-run the best candidate long enough to measure first time to `score >= 50`.
6. Replicate the best candidate and compare median time-to-50 plus variance against the current best.
7. Only bake a new default into config after the replicated candidate beats the current best and satisfies the variance rule.

## Experiment Log Format

- Keep a high-level experiment index in [EXPERIMENT_LOG.md](/home/vast/puffer_snake/pufferlib/EXPERIMENT_LOG.md).
- Keep environment-specific round logs in `experiments/reports/`.
- Log one round entry per search batch or validation batch, not one entry per individual command.
- Each round entry should contain:
  - `Round`: stable round identifier such as `tune5` or `tune7-validation`
  - `Date`
  - `Objective`
  - `Incumbent`: config being compared against at the start of the round
  - `Changes Tested`: short list of knobs explored
  - `Key Results`: small table with the most relevant candidates and time-to-50 results
  - `Validation`: replication summary if validation was run
  - `Decision`: whether the incumbent changed
  - `Artifacts`: links to the main JSONL and stdout files for the round
- Prefer summarizing only the strongest candidates and final decisions. Raw per-run details should stay in the JSONL and stdout artifacts.

## Preferred Search Order

Try easier, lower-risk wins before invasive code changes:

1. backend / vectorization shape
2. batch geometry
3. learning rate and scheduler
4. minibatch size
5. entropy / exploration settings
6. model size
7. optimizer changes
8. training-loop code changes

## Current Supported Single-GPU Recipe

Current live supported configuration:

- `vec.backend = PufferEnv`
- `vec.num_envs = 1`
- `env.num_envs = 4096`
- `policy_name = SingleSnakeV1Policy`
- `train.total_timesteps = 30_080_548`
- `train.learning_rate = 0.0021585990625102125`
- `train.anneal_lr = False`
- `train.gamma = 0.9940711103165775`
- `train.gae_lambda = 0.944`
- `train.update_epochs = 1`
- `train.clip_coef = 0.1758868095026977`
- `train.vf_coef = 0.6108216342676894`
- `train.vf_clip_coef = 0.15470946293822643`
- `train.max_grad_norm = 0.5087626236773275`
- `train.bptt_horizon = 32`
- `train.minibatch_size = 8192`
- `train.prio_alpha = 0.5551867470045929`
- `train.prio_beta0 = 0.38`
- `train.recompute_advantages_per_minibatch = False`

Validated result:

- 5-run replicated time-to-50 runs: `15.116s`, `15.085s`, `16.752s`, `16.552s`, `16.868s`
- median time-to-50: about `16.552s`
- all validation runs reached `score >= 50` before `100_000_000` timesteps

Runner-ups:

- previous 3-run winner `gw5h6eqs`: `14.981s`, `15.510s`, `16.996s`
  - 3-run median: about `15.510s`
  - later 5-run revalidation did not satisfy the variance rule
- previous baked winner `y5j8ojni`: `16.401s`, `15.467s`, `16.214s`, `16.102s`, `20.484s`
  - 5-run median: about `16.214s`
  - did not satisfy the variance rule on 5-run revalidation
- closest alternative valid 5-run candidate `7vqfwun5`: `17.442s`, `17.417s`, `16.837s`, `17.984s`, `17.145s`
  - 5-run median: about `17.417s`
  - satisfied the variance rule, but was slower than the current winner

Archived experimental winner:

- tag: `single-snake-policies-archive`
- config included `policy_name = SingleSnakeV1HybridHeuristicPolicy` and `env.report_interval = 128`
- replicated time-to-50 runs: `18.79s`, `20.60s`, `19.27s`
- median time-to-50: about `19.27s`

## Notes

- `torch.compile` should be treated as optional and only kept if it works cleanly on the active GPU architecture.
- Higher throughput is not enough by itself; wall-clock convergence is the real objective.
- Single-GPU results are the source of truth for this task, even if multi-GPU training is available.
