import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pufferlib.pufferl as pufferl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, passthrough = parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *passthrough]
        cfg = pufferl.load_config_file(args.config)
    finally:
        sys.argv = old_argv

    pufferl.sweep(args=cfg, env_name=cfg["env_name"])


if __name__ == "__main__":
    main()
