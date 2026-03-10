import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_stats(path: Path):
    summary = {
        "best_score": None,
        "final_score": None,
        "final_uptime": None,
        "t50": None,
        "t100": None,
        "t120": None,
        "t150": None,
        "t200": None,
    }
    if not path.exists():
        return summary

    best = float("-inf")
    last = None
    first_ts = None
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            last = row
            score = row.get("environment/score")
            ts = row.get("timestamp")
            if first_ts is None and ts is not None:
                first_ts = ts
            uptime = row.get("performance/uptime")
            if uptime is None and ts is not None and first_ts is not None:
                uptime = ts - first_ts
            if score is None or uptime is None:
                continue
            best = max(best, score)
            for threshold, key in [(50, "t50"), (100, "t100"), (120, "t120"), (150, "t150"), (200, "t200")]:
                if summary[key] is None and score > threshold:
                    summary[key] = uptime

    if last is not None:
        summary["final_score"] = last.get("environment/score")
        summary["final_uptime"] = last.get("performance/uptime")
    if best != float("-inf"):
        summary["best_score"] = best
    return summary


def launch_job(repo_root: Path, batch_dir: Path, job: dict, gpu: int):
    job_dir = batch_dir / job["name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    stats_path = job_dir / "periodic_stats.jsonl"
    stdout_path = job_dir / "stdout.log"

    cmd = [
        str(repo_root / ".venv/bin/puffer"),
        "train",
        "puffer_single_snake_v1",
        "--vec.backend", "PufferEnv",
        "--vec.num-envs", "1",
        "--env.width", str(job.get("width", 16)),
        "--env.height", str(job.get("height", 16)),
        "--env.reward-food", "1.0",
        "--env.reward-step", "-0.003",
        "--env.reward-death", "-1.0",
        "--train.stats-log-interval", str(job.get("stats_log_interval", 2_000_000)),
        "--train.stats-log-path", str(stats_path),
        "--tag", job["name"],
        *job["args"],
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout = stdout_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "name": job["name"],
        "gpu": gpu,
        "proc": proc,
        "stdout": stdout,
        "job_dir": job_dir,
        "stats_path": stats_path,
        "args": job["args"],
        "start_time": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--stats-log-interval", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    batch_dir = Path(args.batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    queue = []
    for job in manifest:
        if args.width is not None and "width" not in job:
            job["width"] = args.width
        if args.height is not None and "height" not in job:
            job["height"] = args.height
        if args.stats_log_interval is not None and "stats_log_interval" not in job:
            job["stats_log_interval"] = args.stats_log_interval
        queue.append(job)
    running = []
    completed = []

    while queue or running:
        while queue and len(running) < len(gpus):
            used_gpus = {job["gpu"] for job in running}
            gpu = next(g for g in gpus if g not in used_gpus)
            running.append(launch_job(repo_root, batch_dir, queue.pop(0), gpu))

        next_running = []
        for job in running:
            if job["proc"].poll() is None:
                next_running.append(job)
                continue

            job["stdout"].close()
            result = {
                "name": job["name"],
                "gpu": job["gpu"],
                "returncode": job["proc"].returncode,
                "wall_time_s": time.time() - job["start_time"],
                "job_dir": str(job["job_dir"]),
                "args": job["args"],
                **parse_stats(job["stats_path"]),
            }
            completed.append(result)
            print(
                f"completed {job['name']} gpu={job['gpu']} rc={job['proc'].returncode} "
                f"best={result['best_score']} t100={result['t100']} "
                f"t120={result['t120']} t150={result['t150']} t200={result['t200']}",
                flush=True,
            )

        running = next_running
        (batch_dir / "summary.json").write_text(json.dumps(completed, indent=2))
        time.sleep(5)

    ranked = sorted(
        completed,
        key=lambda r: (
            r["t120"] if r["t120"] is not None else 1e18,
            -(r["best_score"] or -1e9),
            r["t150"] if r["t150"] is not None else 1e18,
            r["t100"] if r["t100"] is not None else 1e18,
            r["t200"] if r["t200"] is not None else 1e18,
        ),
    )
    (batch_dir / "summary_ranked.json").write_text(json.dumps(ranked, indent=2))
    print(f"finished {len(completed)} runs", flush=True)


if __name__ == "__main__":
    main()
