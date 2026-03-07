"""Minimal single-snake environment backed by a C extension."""

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.single_snake_v2 import binding


class SingleSnakeV2(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        width=5,
        height=5,
        cell_size=None,
        max_snake_length=None,
        num_food=1,
        reward_food=1.0,
        reward_step=-0.01,
        reward_death=-1.0,
        use_potential_shaping=False,
        shape_on_eat=False,
        potential_shaping_coef=0.0,
        max_episode_steps=400,
        report_interval=128,
        render_mode='human',
        buf=None,
        seed=0,
    ):
        if num_envs is None:
            num_envs = 1
        num_envs = int(num_envs)
        if num_envs < 1:
            raise pufferlib.APIUsageError('num_envs must be >= 1')

        width = int(width)
        height = int(height)
        num_food = int(num_food)
        if width < 2 or height < 2:
            raise pufferlib.APIUsageError('width and height must be >= 2')
        if num_food < 1:
            raise pufferlib.APIUsageError('num_food must be >= 1')
        if num_food != 1:
            raise pufferlib.APIUsageError('single_snake_v2 requires num_food == 1')
        max_area = width * height
        if max_snake_length is None:
            max_snake_length = max_area
        max_snake_length = int(max_snake_length)
        max_snake_length = max(1, min(max_snake_length, max_area))
        if cell_size is None:
            cell_size = int(np.ceil(1280 / max(width, height)))
        cell_size = int(cell_size)

        self.report_interval = int(report_interval)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=3, shape=(height, width), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.tick = 0
        self.board_width = width
        self.board_height = height

        super().__init__(buf)
        c_envs = []
        offset = 0
        for i in range(num_envs):
            obs_slice = self.observations[offset:offset+1]
            act_slice = self.actions[offset:offset+1]
            rew_slice = self.rewards[offset:offset+1]
            term_slice = self.terminals[offset:offset+1]
            trunc_slice = self.truncations[offset:offset+1]
            env_seed = i + seed * num_envs

            env_id = binding.env_init(
                obs_slice,
                act_slice,
                rew_slice,
                term_slice,
                trunc_slice,
                env_seed,
                width=width,
                height=height,
                num_food=num_food,
                reward_food=reward_food,
                reward_step=reward_step,
                reward_death=reward_death,
                use_potential_shaping=use_potential_shaping,
                shape_on_eat=shape_on_eat,
                potential_shaping_coef=potential_shaping_coef,
                max_episode_steps=max_episode_steps,
                max_snake_length=max_snake_length,
                cell_size=cell_size,
            )
            c_envs.append(env_id)
            offset += 1

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
        if self.report_interval > 0 and self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 1)

    def render_ansi(self, env_id=0, stride=1):
        return binding.vec_render_ansi(self.c_envs, env_id, stride)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    env = SingleSnakeV2(num_envs=512)
    env.reset()
    tick = 0
    total_agents = env.num_agents
    actions = np.random.randint(0, 4, (atn_cache, total_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: {total_agents * tick / (time.time() - start):.2f}')


if __name__ == '__main__':
    test_performance()
