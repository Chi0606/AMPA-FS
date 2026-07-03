"""
Real-time health monitor: takes N samples at fixed intervals, records
(PID, log size, checkpoint variants, parsed progress) each sample, and
flags whether progress is actually advancing.

Usage:
    python experiments/_monitor_once.py [samples] [interval_seconds]
Defaults: 7 samples, 45 s interval  (~5 min 15 s total).
"""
import os
import re
import sys
import time
import pickle
import subprocess
from datetime import datetime

LOG_DIR = "results"
SCRIPT_NAME = "run_ablation"
CHECKPOINT = os.path.join(LOG_DIR, "ablation_results_checkpoint.pkl")


def find_pid():
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=8,
            creationflags=0x08000000,
        )
        for line in r.stdout.splitlines():
            if SCRIPT_NAME in line:
                parts = line.rsplit(",", 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    return int(parts[1].strip())
    except Exception:
        return None
    return None


def latest_log():
    files = [f for f in os.listdir(LOG_DIR)
             if f.startswith("run_ablation_log") and f.endswith(".txt")]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
    return os.path.join(LOG_DIR, files[-1])


def parse_progress(path):
    """Return dict with last-seen variant/run/done/pct/eta, or None."""
    try:
        with open(path, "rb") as f:
            content = f.read()
    except Exception:
        return None
    text = content.decode("utf-8", errors="ignore")
    matches = re.findall(
        r"(BMPA|AMPA_[A-Z]+)\s+run\s+(\d+)/\d+\s+\[(\d+)/(\d+)\]\s+"
        r"([\d.]+)%\s+ETA:\s+([\d.]+)m",
        text,
    )
    if not matches:
        return None
    v, run, done, total, pct, eta = matches[-1]
    return {
        "variant": v, "run": int(run),
        "done": int(done), "total": int(total),
        "pct": float(pct), "eta": float(eta),
    }


def checkpoint_variants():
    if not os.path.exists(CHECKPOINT):
        return 0
    try:
        with open(CHECKPOINT, "rb") as f:
            d = pickle.load(f)
        n = 0
        for ds in d:
            for v in d[ds]:
                if len(d[ds][v].get("fitness", [])) >= 30:
                    n += 1
        return n
    except Exception:
        return -1


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 45

    total_sec = samples * interval
    print("=" * 78)
    print(f"  REAL-TIME MONITOR  ({samples} samples x {interval}s "
          f"= {total_sec//60}m {total_sec%60}s)")
    print("=" * 78)
    header = (f"{'Time':>8s} {'PID':>6s} {'LogKB':>7s} {'Vars':>5s} "
              f"{'Variant':>9s} {'Run':>4s}/30 {'Pct':>6s} "
              f"{'ETA(m)':>7s} {'Delta':>10s}")
    print(header)
    print("-" * 78)

    prev_done = None
    prev_variant = None
    prev_run = None
    advances = 0

    for i in range(samples):
        pid = find_pid()
        log = latest_log()
        size = os.path.getsize(log) if log else 0
        prog = parse_progress(log) if log else None
        ncv = checkpoint_variants()
        now = datetime.now().strftime("%H:%M:%S")

        if prog:
            delta = ""
            if prev_done is not None:
                d_done = prog["done"] - prev_done
                same_run = (prog["variant"] == prev_variant
                            and prog["run"] == prev_run)
                if d_done > 0:
                    delta = f"+{d_done} runs"
                    advances += 1
                elif not same_run:
                    delta = "new run"
                    advances += 1
                else:
                    delta = "no change"
            prev_done = prog["done"]
            prev_variant = prog["variant"]
            prev_run = prog["run"]

            pid_s = str(pid) if pid else "--"
            print(f"{now:>8s} {pid_s:>6s} {size/1024:>7.1f} {ncv:>5d} "
                  f"{prog['variant']:>9s} {prog['run']:>4d}/30 "
                  f"{prog['pct']:>5.1f}% {prog['eta']:>7.1f} "
                  f"{delta:>10s}")
        else:
            pid_s = str(pid) if pid else "--"
            print(f"{now:>8s} {pid_s:>6s} {size/1024:>7.1f} {ncv:>5d} "
                  f"(no progress match)")

        if i < samples - 1:
            time.sleep(interval)

    print("-" * 78)
    print(f"Summary: {advances} of {samples - 1} inter-sample intervals "
          f"showed forward progress.")
    if advances >= (samples - 1) * 0.7:
        print("VERDICT: experiment is HEALTHY and actively running.")
    elif advances >= 1:
        print("VERDICT: experiment is running but slow (may just be a "
              "long-running dataset).")
    else:
        print("VERDICT: NO forward progress observed. Check process / logs.")


if __name__ == "__main__":
    main()
