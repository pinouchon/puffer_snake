'''Single-agent snake with fixed map size for fast pufferlib training.'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.single_snake import binding

MAP_WIDTH = 10
MAP_HEIGHT = 10
VISION = 5
NUM_SNAKES = 1
NUM_FOOD = 2


class SingleSnake(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=MAP_WIDTH, height=MAP_HEIGHT,
            num_snakes=NUM_SNAKES, num_food=NUM_FOOD,
            vision=VISION, leave_corpse_on_death=False,
            reward_food=1.0, reward_corpse=0.0, reward_death=-1.0, reward_step=-0.01,
            use_potential_shaping=True, potential_shaping_coef=0.05,
            report_interval=128, max_snake_length=1024, max_episode_steps=400,
            render_mode='human', buf=None, seed=0):
        if num_envs is None:
            num_envs = 1
        num_envs = int(num_envs)
        if num_envs < 1:
            raise pufferlib.APIUsageError('num_envs must be >= 1')

        width = num_envs * [MAP_WIDTH]
        height = num_envs * [MAP_HEIGHT]
        num_snakes = num_envs * [NUM_SNAKES]
        num_food = num_envs * [NUM_FOOD]
        vision = VISION
        leave_corpse_on_death = num_envs * [leave_corpse_on_death]

        max_area = MAP_WIDTH * MAP_HEIGHT
        self.max_snake_length = max(1, min(max_snake_length, max_area))
        self.report_interval = report_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=3, shape=(MAP_HEIGHT, MAP_WIDTH), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.tick = 0
        self.board_width = MAP_WIDTH
        self.board_height = MAP_HEIGHT

        self.cell_size = int(np.ceil(1280 / max(MAP_WIDTH, MAP_HEIGHT)))

        super().__init__(buf)
        c_envs = []
        offset = 0
        for i in range(num_envs):
            ns = NUM_SNAKES
            obs_slice = self.observations[offset:offset+ns]
            act_slice = self.actions[offset:offset+ns]
            rew_slice = self.rewards[offset:offset+ns]
            term_slice = self.terminals[offset:offset+ns]
            trunc_slice = self.truncations[offset:offset+ns]
            # Seed each env uniquely: i + seed * num_envs
            env_seed = i + seed * num_envs
            env_id = binding.env_init(
                obs_slice, 
                act_slice, 
                rew_slice, 
                term_slice, 
                trunc_slice,
                env_seed,
                width=MAP_WIDTH, 
                height=MAP_HEIGHT,
                num_snakes=ns, 
                num_food=NUM_FOOD,
                vision=VISION, 
                leave_corpse_on_death=leave_corpse_on_death[i],
                reward_food=reward_food, 
                reward_corpse=reward_corpse,
                reward_death=reward_death, 
                reward_step=reward_step,
                use_potential_shaping=use_potential_shaping,
                potential_shaping_coef=potential_shaping_coef,
                max_snake_length=self.max_snake_length,
                max_episode_steps=max_episode_steps,
                cell_size=self.cell_size
            )
            c_envs.append(env_id)
            offset += ns
        self.c_envs = binding.vectorize(*c_envs)
 
    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, self.cell_size)

    def render_ansi(self, env_id=0, stride=1):
        return binding.vec_render_ansi(self.c_envs, env_id, stride)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = SingleSnake()
    env.reset()
    tick = 0

    total_snakes = env.num_agents
    actions = np.random.randint(0, 4, (atn_cache, total_snakes))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', total_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
