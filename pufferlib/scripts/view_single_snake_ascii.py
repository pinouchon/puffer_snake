#!/usr/bin/env python3
import argparse
import glob
import math
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pufferlib.pytorch
from pufferlib import pufferl


class TerminalScreen:
    def __init__(self):
        self._entered = False
        self._prev_lines = 0

    def enter(self):
        if self._entered:
            return
        # Alternate screen + hide cursor
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

        # Clear leftover lines from previous frame
        extra = self._prev_lines - len(lines)
        for _ in range(max(0, extra)):
            out.append("\x1b[2K\n")

        self._prev_lines = len(lines)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def leave(self):
        if not self._entered:
            return
        # Show cursor + leave alternate screen
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        self._entered = False


def parse_args():
    parser = argparse.ArgumentParser(description="Render single_snake policy as ASCII at fixed SPS")
    parser.add_argument("--env-name", type=str, default="puffer_single_snake_v2",
        help="Environment config name, e.g. puffer_single_snake or puffer_single_snake_v2")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to checkpoint (.pt). Defaults to latest experiments/**/model_*.pt")
    parser.add_argument("--fps", type=float, default=10.0, help="Target steps per second")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--env-id", type=int, default=0, help="Internal env index for rendering")
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
    max_rows = max(8, rows - 4)
    max_cols = max(20, cols)
    board_w = max(1, int(getattr(driver_env, "board_width", 1)))
    board_h = max(1, int(getattr(driver_env, "board_height", 1)))
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


def load_runtime(device, checkpoint, env_name):
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = saved_argv

    args["train"]["device"] = device
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["env"]["render_mode"] = "ansi"
    args["load_model_path"] = checkpoint

    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    policy.eval()
    return args, vecenv, policy


def run_loop(args):
    checkpoint = args.checkpoint or find_latest_checkpoint("experiments")
    cfg, vecenv, policy = load_runtime(args.device, checkpoint, args.env_name)
    device = cfg["train"]["device"]
    driver = vecenv.driver_env

    if not hasattr(driver, "render_ansi"):
        raise RuntimeError("single_snake env does not expose render_ansi")

    obs, _ = vecenv.reset()
    num_agents = vecenv.observation_space.shape[0]
    recurrent = cfg["rnn_name"] is not None
    state = {}
    if recurrent:
        state["lstm_h"] = torch.zeros(num_agents, policy.hidden_size, device=device)
        state["lstm_c"] = torch.zeros(num_agents, policy.hidden_size, device=device)

    period = 1.0 / max(0.1, args.fps)
    screen = TerminalScreen()
    screen.enter()
    term_size = shutil.get_terminal_size((120, 40))
    stride = choose_stride(driver, term_size)
    total_step = 0
    episode_step = 0
    episode_idx = 1
    try:
        while True:
            start = time.perf_counter()
            new_term_size = shutil.get_terminal_size((120, 40))
            if new_term_size != term_size:
                term_size = new_term_size
                stride = choose_stride(driver, term_size)
            frame = driver.render_ansi(args.env_id, stride)
            frame = add_ascii_border(frame)
            now = time.strftime("%H:%M:%S")
            header = (
                f"{args.env_name} ascii viewer | checkpoint: {checkpoint}\n"
                f"episode: {episode_idx} | episode step: {episode_step} | total step: {total_step}\n"
                f"time: {now} | fps target: {args.fps:.1f} | stride: {stride} | "
                f"device: {device} | ctrl+c to exit\n"
            )
            screen.draw(header + frame)

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device)
                logits, _ = policy.forward_eval(obs_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            obs, _, terminals, truncations, _ = vecenv.step(action)
            total_step += 1
            episode_step += 1

            done = np.logical_or(terminals, truncations)
            if recurrent:
                if np.any(done):
                    done_t = torch.as_tensor(done, dtype=torch.bool, device=device)
                    state["lstm_h"][done_t] = 0
                    state["lstm_c"][done_t] = 0

            if np.any(done):
                episode_idx += 1
                episode_step = 0
                continue

            elapsed = time.perf_counter() - start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        screen.leave()
        vecenv.close()


def main():
    args = parse_args()
    while True:
        try:
            run_loop(args)
        except KeyboardInterrupt:
            print("\nExiting viewer.")
            return
        except Exception as exc:
            print(f"\nViewer crashed: {exc}")
            traceback.print_exc()
            print("Restarting in 2 seconds...")
            time.sleep(2)


if __name__ == "__main__":
    main()
