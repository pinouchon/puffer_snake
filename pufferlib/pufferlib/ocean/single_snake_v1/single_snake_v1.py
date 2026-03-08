"""Simplified single-snake environment backed by a C extension."""

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.single_snake_v1 import binding


class SingleSnakeV1(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        width=8,
        height=8,
        reward_food=1.0,
        reward_step=-0.003,
        reward_death=-1.0,
        max_episode_steps=400,
        report_interval=256,
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
        if width < 2 or height < 2:
            raise pufferlib.APIUsageError('width and height must be >= 2')

        self.report_interval = int(report_interval)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=3, shape=(height, width), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = num_envs
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
                reward_food=reward_food,
                reward_step=reward_step,
                reward_death=reward_death,
                max_episode_steps=max_episode_steps,
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

    def render_ansi(self, env_id=0, stride=1):
        return binding.vec_render_ansi(self.c_envs, env_id, stride)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    env = SingleSnakeV1(num_envs=512)
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
