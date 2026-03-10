# Auto Experiment Rules 12x12

These rules define how automatic training experiments should be run for `puffer_single_snake_v1` on the `12x12` board.

## Goal

Optimize for the fastest wall-clock time to reach:

- `environment/score > 120`

Primary metric:

- wall-clock time to first `score > 120`

Secondary metrics:

- agent steps to first `score > 120`
- best score reached within a fixed training budget
- score at matched wall-clock times
- score at matched agent-step counts

## Hard Constraints

These settings must not be changed:

- `env.width = 12`
- `env.height = 12`
- `env.reward_food = 1.0`
- `env.reward_step = -0.003`
- `env.reward_death = -1.0`
- observation space must remain unchanged

Other fixed constraints:

- use a single GPU per run
- default `env.max_episode_steps = 10_000`

## Allowed Changes

The following may be changed during experiments:

- vectorization backend and env count
- learning rate and LR schedule
- minibatch size
- horizon / batch geometry
- update epochs
- optimizer
- entropy coefficient and other RL hyperparameters
- policy architecture and connectivity
- training-loop implementation details
- logging and profiling settings
- total training budget
- episode cap if there is a strong reason, but changes should be explicit and justified against the default `10_000`

## Measurement Rules

- Use dense periodic stats logging so runs can be compared without relying on terminal output alone.
- Prefer JSONL logs written from `train.stats_log_interval` and `train.stats_log_path`.
- Compare candidates at matched wall-clock times first.
- If no run reaches `>120`, rank runs by highest score reached, then by wall-clock time to intermediate thresholds such as `>80` or `>100`.
- Use longer confirmation runs only for the most promising candidates.
- Do not declare a new winner from a very short run unless it is clearly ahead on the score-vs-time curve.

## Variance Rule

- A candidate config is not valid from a single fast run alone. It must be replicated.
- A valid replicated config must reach `environment/score > 120` in every replication run used for validation.
- Use at least 3 replication runs for a winner check.
- The primary replicated score is the median wall-clock time to first `score > 120`.
- A replicated config is considered to have reasonable variance only if:
  - coefficient of variation of time-to-120 is `<= 10%`
  - worst run time-to-120 is no more than `20%` slower than the best run in the validation set
- If no candidate reaches `>120` reliably, the incumbent remains the config with the strongest demonstrated score ceiling and best score-vs-time profile.

## Experiment Workflow

1. Start from the current best known 12x12 single-GPU configuration.
2. Change only a small number of variables per experiment.
3. Run multiple experiments in parallel when hardware allows, but each run must use one GPU.
4. Eliminate weak directions quickly.
5. Re-run the best candidate long enough to measure first time to `score > 120`.
6. Replicate the best candidate and compare median time-to-120 plus variance against the current best.
7. Only bake a new default into config after the replicated candidate beats the current best and satisfies the variance rule.

## Experiment Log Format

- Keep a high-level experiment index in [EXPERIMENT_LOG.md](/home/vast/puffer_snake/pufferlib/EXPERIMENT_LOG.md).
- Keep environment-specific round logs in `experiments/reports/` when the results are mature enough to preserve.
- Log one round entry per search batch or validation batch, not one entry per individual command.
- Each round entry should contain:
  - `Round`
  - `Date`
  - `Objective`
  - `Incumbent`
  - `Changes Tested`
  - `Key Results`
  - `Validation`
  - `Decision`
  - `Artifacts`

## Preferred Search Order

Try easier, lower-risk wins before invasive code changes:

1. episode cap and training budget
2. backend / vectorization shape
3. batch geometry
4. learning rate and scheduler
5. entropy / exploration settings
6. model size
7. optimizer changes
8. policy connectivity changes
9. training-loop code changes

## Current 12x12 Starting Point

Current starting configuration in [single_snake_v1.ini](/home/vast/puffer_snake/pufferlib/pufferlib/config/ocean/single_snake_v1.ini):

- `vec.backend = PufferEnv`
- `vec.num_envs = 1`
- `env.num_envs = 4096`
- `env.width = 12`
- `env.height = 12`
- `env.max_episode_steps = 10_000`
- `policy_name = SingleSnakeV1Policy`
- `policy.cnn_channels = 32`
- `policy.hidden_size = 128`

This is only a starting point, not a validated 12x12 winner.

## Notes

- For `12x12`, it is acceptable to use larger total training budgets than were used for `8x8`.
- Higher throughput is not enough by itself; wall-clock convergence to `>120` is the real objective.
- If no run reaches `>120`, prefer directions that improve score ceiling over directions that only improve early low-score learning.
