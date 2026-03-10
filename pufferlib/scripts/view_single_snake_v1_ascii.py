#!/usr/bin/env python3
import argparse
import glob
import math
import os
import select
import shutil
import sys
import termios
import time
import tty

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pufferlib.pytorch
from pufferlib import pufferl

ENV_NAME = "puffer_single_snake_v1"


class TerminalSession:
    def __init__(self):
        self._entered = False
        self._prev_lines = 0
        self._stdin_fd = None
        self._stdin_attrs = None

    def enter(self):
        if self._entered:
            return

        if sys.stdin.isatty():
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)

        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        self._entered = True

    def draw(self, text):
        lines = text.splitlines()
        out = ["\x1b[H"]
        for line in lines:
            out.append("\x1b[2K")
            out.append(line)
            out.append("\n")

        extra = self._prev_lines - len(lines)
        for _ in range(max(0, extra)):
            out.append("\x1b[2K\n")

        self._prev_lines = len(lines)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def poll_key(self):
        if self._stdin_fd is None:
            return None
        ready, _, _ = select.select([self._stdin_fd], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1)

    def wait_key(self):
        if self._stdin_fd is None:
            try:
                return input()
            except EOFError:
                return ""

        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], None)
            if ready:
                return sys.stdin.read(1)

    def leave(self):
        if not self._entered:
            return

        if self._stdin_fd is not None and self._stdin_attrs is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_attrs)

        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        self._entered = False


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Render single_snake_v1 policy as ASCII at fixed FPS")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to checkpoint (.pt). Defaults to latest experiments/**/model_*.pt")
    parser.add_argument("--fps", type=float, default=10.0,
        help="Locked render and env step rate")
    parser.add_argument("--device", type=str, default=default_device(),
        help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--policy-name", type=str, default=None,
        help="Override policy class name, e.g. SingleSnakeV1Policy")
    parser.add_argument("--rnn-name", type=str, default=None,
        help="Override recurrent wrapper class, e.g. SingleSnakeV1LSTM")
    parser.add_argument("--policy-cnn-channels", type=int, default=None,
        help="Override policy.cnn_channels")
    parser.add_argument("--policy-hidden-size", type=int, default=None,
        help="Override policy.hidden_size")
    parser.add_argument("--env-width", type=int, default=None,
        help="Override board width")
    parser.add_argument("--env-height", type=int, default=None,
        help="Override board height")
    parser.add_argument("--env-max-episode-steps", type=int, default=None,
        help="Override max episode steps")
    return parser.parse_args()


def find_latest_checkpoint(experiments_dir="experiments"):
    pattern = os.path.join(experiments_dir, "**", "model_*.pt")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise FileNotFoundError(
            f"No checkpoint found under {experiments_dir}. "
            "Expected files like experiments/**/model_*.pt"
        )
    return max(paths, key=os.path.getmtime)


def choose_stride(driver_env, term_size):
    cols, rows = term_size.columns, term_size.lines
    max_rows = max(8, rows - 7)
    max_cols = max(20, cols)
    board_w = max(1, int(driver_env.board_width))
    board_h = max(1, int(driver_env.board_height))
    stride_w = max(1, math.ceil(board_w / max_cols))
    stride_h = max(1, math.ceil(board_h / max_rows))
    return max(stride_w, stride_h)


def add_ascii_border(frame):
    lines = frame.splitlines()
    if not lines:
        return "+--+\n|  |\n+--+"

    width = max(len(line) for line in lines)
    top_bottom = "+" + ("-" * width) + "+"
    bordered = [top_bottom]
    for line in lines:
        bordered.append("|" + line.ljust(width) + "|")
    bordered.append(top_bottom)
    return "\n".join(bordered)


def snake_length_from_obs(obs):
    board = obs[0]
    return int(np.count_nonzero((board == 2) | (board == 3)))


def load_runtime(device, checkpoint):
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        args = pufferl.load_config(ENV_NAME)
    finally:
        sys.argv = saved_argv

    args["train"]["device"] = device
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["env"]["num_envs"] = 1
    args["load_model_path"] = checkpoint

    cli_args = parse_args()
    if cli_args.policy_name is not None:
        args["policy_name"] = cli_args.policy_name
    if cli_args.rnn_name is not None:
        args["rnn_name"] = cli_args.rnn_name
    if cli_args.policy_cnn_channels is not None:
        args["policy"]["cnn_channels"] = cli_args.policy_cnn_channels
    if cli_args.policy_hidden_size is not None:
        args["policy"]["hidden_size"] = cli_args.policy_hidden_size
    if cli_args.env_width is not None:
        args["env"]["width"] = cli_args.env_width
    if cli_args.env_height is not None:
        args["env"]["height"] = cli_args.env_height
    if cli_args.env_max_episode_steps is not None:
        args["env"]["max_episode_steps"] = cli_args.env_max_episode_steps

    vecenv = pufferl.load_env(ENV_NAME, args)
    policy = pufferl.load_policy(args, vecenv, ENV_NAME)
    policy.eval()
    return args, vecenv, policy


def init_recurrent_state(policy, device):
    hidden_size = getattr(policy, "hidden_size", None)
    if hidden_size is None:
        return {}

    return {
        "lstm_h": torch.zeros(1, hidden_size, device=device),
        "lstm_c": torch.zeros(1, hidden_size, device=device),
    }


def format_screen(checkpoint, args, obs, episode_idx, episode_step, total_step,
        episode_return, last_reward, last_loop_s, message):
    frame = add_ascii_border(obs["frame"])
    checkpoint_label = os.path.basename(checkpoint)
    actual_fps = 0.0 if last_loop_s <= 0 else 1.0 / last_loop_s
    header = [
        f"{ENV_NAME} ascii viewer | checkpoint: {checkpoint_label}",
        (
            f"episode: {episode_idx} | episode step: {episode_step} | total step: {total_step} | "
            f"snake length: {obs['snake_length']}"
        ),
        (
            f"episode return: {episode_return:.3f} | last reward: {last_reward:.3f} | "
            f"fps target: {args.fps:.1f} | fps actual: {actual_fps:.2f} | device: {args.device}"
        ),
    ]
    if message:
        header.append(message)

    return "\n".join(header) + "\n" + frame


def snapshot(driver_env, obs, stride):
    return {
        "frame": driver_env.render_ansi(0, stride),
        "snake_length": snake_length_from_obs(obs),
    }


def main():
    args = parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint("experiments")
    _, vecenv, policy = load_runtime(args.device, checkpoint)
    driver = vecenv.driver_env
    period = 1.0 / max(0.1, args.fps)
    recurrent_state = init_recurrent_state(policy, args.device)

    obs, _ = vecenv.reset()
    episode_idx = 1
    episode_step = 0
    total_step = 0
    episode_return = 0.0
    last_reward = 0.0
    last_loop_s = 0.0

    screen = TerminalSession()
    screen.enter()
    try:
        term_size = shutil.get_terminal_size((120, 40))
        stride = choose_stride(driver, term_size)

        while True:
            loop_start = time.perf_counter()

            key = screen.poll_key()
            if key in ("q", "Q"):
                return

            new_term_size = shutil.get_terminal_size((120, 40))
            if new_term_size != term_size:
                term_size = new_term_size
                stride = choose_stride(driver, term_size)

            pre_step_state = snapshot(driver, obs, stride)
            screen.draw(format_screen(
                checkpoint, args, pre_step_state, episode_idx, episode_step, total_step,
                episode_return, last_reward, last_loop_s, "q: quit"))

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=args.device)
                logits, _ = policy.forward_eval(obs_t, recurrent_state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            obs, rewards, terminals, truncations, _ = vecenv.step(action)
            last_reward = float(rewards[0])
            episode_return += last_reward
            total_step += 1
            episode_step += 1

            elapsed = time.perf_counter() - loop_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            last_loop_s = time.perf_counter() - loop_start

            done = bool(terminals[0] or truncations[0])
            if not done:
                continue

            ended_message = "episode ended on next action | showing board just before death | press any key to continue | q: quit"
            screen.draw(format_screen(
                checkpoint, args, pre_step_state, episode_idx, episode_step, total_step,
                episode_return, last_reward, last_loop_s, ended_message))

            key = screen.wait_key()
            if key in ("q", "Q"):
                return

            episode_idx += 1
            episode_step = 0
            episode_return = 0.0
            last_reward = 0.0
            last_loop_s = 0.0
            recurrent_state = init_recurrent_state(policy, args.device)
    finally:
        screen.leave()
        vecenv.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
