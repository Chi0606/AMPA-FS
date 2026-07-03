"""
Real-time monitor for sensitivity experiment: samples PID, checkpoint
cell-count, and mtime at fixed intervals. Verdict based on advancing
cells or updating mtime.
"""
import os, sys, time, pickle, subprocess
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL  = os.path.join(BASE, "results", "sensitivity_results_checkpoint.pkl")

sys.path.insert(0, BASE)
import config

TOTAL_CELLS = sum(len(v) for v in config.SENSITIVITY_PARAMS.values()) \
              * len(config.SENSITIVITY_DATASETS)
TOTAL_RUNS = TOTAL_CELLS * config.SENSITIVITY_RUNS


def find_pid():
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=8,
            creationflags=0x08000000,
        )
        for line in r.stdout.splitlines():
            if "run_sensitivity" in line:
                parts = line.rsplit(",", 1)
                if len(parts) == 2 and parts[1].strip().isdigit():
                    return int(parts[1].strip())
    except Exception:
        return None
    return None


def snapshot():
    """Return (pid, mtime, done_cells, done_runs, last_cell_str)."""
    pid = find_pid()
    if not os.path.exists(PKL):
        return (pid, None, 0, 0, "(no checkpoint)")
    mtime = os.path.getmtime(PKL)
    try:
        d = pickle.load(open(PKL, "rb"))
    except Exception:
        return (pid, mtime, 0, 0, "(read err)")
    done_cells = 0
    done_runs = 0
    last = "(none)"
    for pn in d:
        for ds in d[pn]:
            for v, r in d[pn][ds].items():
                n = len(r.get("fitness_all", []))
                done_runs += n
                if n >= config.SENSITIVITY_RUNS:
                    done_cells += 1
                last = f"{pn}={v}/{ds}"
    return (pid, mtime, done_cells, done_runs, last)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    dt = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("=" * 82)
    print(f"  SENSITIVITY REAL-TIME MONITOR  ({n} samples x {dt}s "
          f"= {n*dt//60}m {n*dt%60}s)")
    print(f"  Total scope: {TOTAL_CELLS} cells x {config.SENSITIVITY_RUNS} "
          f"runs = {TOTAL_RUNS} trials")
    print("=" * 82)
    hdr = f"{'Time':>8s} {'PID':>6s} {'CP age':>8s} {'Cells':>8s} {'Runs':>8s} {'Pct':>6s}  {'Last cell':30s} {'Delta':>10s}"
    print(hdr)
    print("-" * 82)

    prev_runs = None
    advances = 0

    for i in range(n):
        pid, mtime, dc, dr, last = snapshot()
        now = datetime.now().strftime("%H:%M:%S")
        pid_s = str(pid) if pid else "--"
        cp_age = "--" if mtime is None else f"{time.time() - mtime:.0f}s"
        pct = 100.0 * dr / TOTAL_RUNS
        delta = ""
        if prev_runs is not None:
            d_runs = dr - prev_runs
            if d_runs > 0:
                delta = f"+{d_runs} runs"
                advances += 1
            else:
                delta = "no change"
        prev_runs = dr
        print(f"{now:>8s} {pid_s:>6s} {cp_age:>8s} "
              f"{dc:>4d}/{TOTAL_CELLS} {dr:>4d}/{TOTAL_RUNS} "
              f"{pct:>5.1f}%  {last[:30]:30s} {delta:>10s}")
        if i < n - 1:
            time.sleep(dt)

    print("-" * 82)
    if advances >= (n - 1) * 0.7:
        print("VERDICT: experiment is HEALTHY and actively running.")
    elif advances >= 1:
        print("VERDICT: experiment is slow but running (cells can take ~90s each).")
    else:
        print("VERDICT: NO forward progress. Check process / logs.")


if __name__ == "__main__":
    main()
