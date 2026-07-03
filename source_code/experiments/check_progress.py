"""
Progress monitor for AMPA-FS experiments.

Supports three experiment types simultaneously:
  - Main experiment:  main_results_checkpoint.pkl + run_main_log*.txt
  - Ablation study:   ablation_results_checkpoint.pkl + run_ablation_log*.txt
  - Sensitivity:      sensitivity_results_checkpoint.pkl + run_sensitivity_log*.txt

Features:
  - Auto-detects which experiments have data
  - Shows per-experiment progress bar, ETA, and completion matrix
  - Parses latest log file to display *currently running* variant/algorithm
  - Detects whether the experiment process is still alive (via log mtime)
  - Color-coded output (ANSI on Windows 10+)

Usage:
  python experiments/check_progress.py                 # show all experiments
  python experiments/check_progress.py --main          # main experiment only
  python experiments/check_progress.py --ablation      # ablation only
  python experiments/check_progress.py --sensitivity   # sensitivity only
  python experiments/check_progress.py --watch         # refresh every 10 seconds
"""

import os
import re
import sys
import time
import pickle
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ANSI color helpers (Windows 10+ supports this natively)
try:
    # Enable ANSI escape sequences on Windows 10+
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BLUE = "\033[34m"


def colored(text, color):
    return f"{color}{text}{C_RESET}"


# Progress bar
def progress_bar(current, total, width=40):
    if total == 0:
        return "[" + "?" * width + "]"
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    pct = ratio * 100
    color = C_GREEN if pct >= 95 else C_YELLOW if pct >= 50 else C_CYAN
    return colored(f"[{bar}] {pct:5.1f}%", color)


# File helpers
def read_text_any_encoding(path):
    """Try common encodings (Windows PowerShell writes UTF-16 by default)."""
    for enc in ("utf-16", "utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                data = f.read()
            if data:
                return data
        except Exception:
            continue
    return ""


def pick_latest_log(prefix):
    """Return path of most recently modified file results/{prefix}*.txt, or None."""
    if not os.path.isdir(config.RESULTS_DIR):
        return None
    best_path, best_mtime = None, 0
    for fn in os.listdir(config.RESULTS_DIR):
        if fn.startswith(prefix) and fn.endswith(".txt"):
            p = os.path.join(config.RESULTS_DIR, fn)
            m = os.path.getmtime(p)
            if m > best_mtime:
                best_mtime = m
                best_path = p
    return best_path


def human_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def elapsed_since(mtime):
    return time.time() - mtime


def _find_experiment_processes(script_name):
    """
    Query OS for running Python processes executing `script_name`.
    Returns list of (pid, cpu_seconds) or empty list.

    On Windows, file stat/mtime/size/content are all heavily cached while a
    process writes (OS retains the snapshot for tens of seconds). So we cannot
    reliably detect "alive" from file-system signals. Process list is the
    authoritative source.
    """
    pids = []
    if os.name == "nt":
        try:
            import subprocess
            # Use WMIC to fetch CommandLine; filter for the script name.
            # WMIC is deprecated in Win11 but still works for this simple query.
            result = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=8,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
            for line in result.stdout.splitlines():
                if script_name in line:
                    # CSV format: Node,CommandLine,ProcessId
                    parts = line.rsplit(",", 1)
                    if len(parts) == 2 and parts[1].strip().isdigit():
                        pids.append(int(parts[1].strip()))
        except Exception:
            # Fallback: tasklist does not expose CommandLine, so we can only
            # tell whether ANY python.exe is running.
            try:
                import subprocess
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=0x08000000,
                )
                if "python.exe" in result.stdout:
                    pids.append(-1)  # unknown PID, but something running
            except Exception:
                pass
    else:
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", script_name],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        except Exception:
            pass
    return pids


def process_status(log_path, script_name="run_ablation"):
    """
    Judge whether the experiment process is still alive.
    Returns ('RUNNING'|'STALLED'|'IDLE', age_seconds, pids)

    Primary: OS process list (authoritative).
    Fallback: mtime (unreliable on Windows but better than nothing).
    """
    pids = _find_experiment_processes(script_name)
    if pids:
        age = 0.0
        if log_path and os.path.exists(log_path):
            age = elapsed_since(os.path.getmtime(log_path))
        return ("RUNNING", age, pids)

    if log_path is None or not os.path.exists(log_path):
        return ("IDLE", 0.0, [])

    age = elapsed_since(os.path.getmtime(log_path))
    # No process found AND log is old -> likely finished or crashed
    if age < 120:
        return ("RUNNING", age, [])  # just finished very recently
    return ("STALLED", age, [])


# Experiment 1: Ablation Study
def show_ablation_progress():
    print(colored("\n" + "=" * 72, C_BOLD))
    print(colored("  ABLATION STUDY  (8 variants x 10 datasets x 30 runs = 2400 tasks)",
                  C_BOLD + C_MAGENTA))
    print(colored("=" * 72, C_BOLD))

    ckpt = os.path.join(config.RESULTS_DIR, "ablation_results_checkpoint.pkl")
    variants = config.ABLATION_VARIANTS
    datasets = config.DATASETS
    num_runs = config.NUM_RUNS
    total_tasks = len(datasets) * len(variants) * num_runs

    results = {}
    if os.path.exists(ckpt):
        try:
            with open(ckpt, "rb") as f:
                results = pickle.load(f)
        except Exception as e:
            print(colored(f"  [ERROR] Failed to load checkpoint: {e}", C_RED))
            return

    # Build completion matrix
    done_tasks = 0
    full_variants = 0  # fully-completed (dataset, variant) combos
    matrix = {}
    for ds in datasets:
        matrix[ds] = {}
        for v in variants:
            if ds in results and v in results[ds]:
                n = len(results[ds][v].get("fitness", []))
                matrix[ds][v] = n
                done_tasks += n
                if n >= num_runs:
                    full_variants += 1
            else:
                matrix[ds][v] = 0

    # Overall progress bar
    print(f"\n  {progress_bar(done_tasks, total_tasks)} "
          f"  {done_tasks}/{total_tasks} runs  "
          f"({full_variants}/{len(datasets)*len(variants)} variants complete)")

    # Find latest log for status
    log_path = pick_latest_log("run_ablation_log")
    status, age, pids = process_status(log_path, script_name="run_ablation")
    status_color = {"RUNNING": C_GREEN, "STALLED": C_RED, "IDLE": C_DIM}[status]
    log_name = os.path.basename(log_path) if log_path else "(no log)"
    pid_str = f", PID={pids[0]}" if pids and pids[0] > 0 else ""
    print(f"  Status: {colored(status, status_color)}{pid_str}  "
          f"(log mtime: {human_time(age)} ago, file: {log_name})")

    # Parse ETA from log
    if log_path:
        content = read_text_any_encoding(log_path)
        eta_matches = re.findall(r"ETA:\s*([\d.]+)m", content)
        pct_matches = re.findall(r"(\d+\.\d+)%", content)
        cur_var_matches = re.findall(
            r"(BMPA|AMPA_[A-Z]+)\s+run\s+(\d+)/(\d+)", content)
        if eta_matches and pct_matches:
            eta_min = float(eta_matches[-1])
            pct = float(pct_matches[-1])
            eta_done_at = datetime.now().timestamp() + eta_min * 60
            eta_str = datetime.fromtimestamp(eta_done_at).strftime("%H:%M")
            print(f"  Log-reported progress: {colored(f'{pct:.1f}%', C_CYAN)}  "
                  f"ETA: {human_time(eta_min * 60)}  "
                  f"(finish around {colored(eta_str, C_YELLOW)})")
        if cur_var_matches:
            var, cur, tot = cur_var_matches[-1]
            print(f"  Currently running: {colored(var, C_YELLOW)} "
                  f"run {cur}/{tot}")

    # Per-dataset matrix (8 variants x dataset)
    print(f"\n  Per-Dataset Status  ({num_runs} runs per cell)")
    header = "  Dataset       "
    for v in variants:
        header += f" {v:>8s}"
    print(colored(header, C_BOLD))
    print("  " + "-" * (14 + 9 * len(variants)))

    for ds in datasets:
        row = f"  {ds:14s}"
        for v in variants:
            n = matrix[ds][v]
            if n >= num_runs:
                cell = colored(f"{'OK':>8s}", C_GREEN)
            elif n > 0:
                cell = colored(f"{n:>2d}/{num_runs}", C_YELLOW)
            else:
                cell = colored(f"{'.':>8s}", C_DIM)
            row += f" {cell}"
        print(row)

    # Show preliminary results for datasets with all variants done
    import numpy as np
    complete_ds = [ds for ds in datasets
                   if all(matrix[ds][v] >= num_runs for v in variants)]
    if complete_ds:
        print(colored("\n  Preliminary Ablation Results "
                      f"({len(complete_ds)} datasets complete):",
                      C_BOLD))
        print(f"  {'Dataset':14s}  {'Variant':10s}  "
              f"{'Fitness':>10s}  {'Accuracy':>10s}  {'Selected':>10s}")
        print("  " + "-" * 60)
        for ds in complete_ds:
            for v in variants:
                r = results[ds][v]
                fit = np.mean(r["fitness"])
                acc = np.mean(r["accuracy"])
                nsel = np.mean(r["n_selected"])
                marker = colored(" <--", C_YELLOW) if v == "AMPA_FS" else ""
                print(f"  {ds:14s}  {v:10s}  "
                      f"{fit:>10.4f}  {acc:>10.4f}  {nsel:>10.1f}{marker}")

    if status == "STALLED":
        print(colored(f"\n  WARNING: no run_ablation process found AND log has "
                      f"not updated for {human_time(age)}.", C_RED))
        print(colored(f"  Restart with: python experiments/run_ablation.py",
                      C_DIM))
    elif status == "RUNNING" and age > 600:
        print(colored(f"\n  Note: process is ALIVE (PID found) but log mtime is "
                      f"{human_time(age)} old. This is normal on Windows - the "
                      f"OS caches file metadata while the writer holds it.",
                      C_DIM))


# Experiment 2: Main Comparison Experiment
def show_main_progress():
    print(colored("\n" + "=" * 72, C_BOLD))
    print(colored("  MAIN COMPARISON  (10 algorithms x 10 datasets x 30 runs = 3000 tasks)",
                  C_BOLD + C_BLUE))
    print(colored("=" * 72, C_BOLD))

    import numpy as np

    # Load checkpoint
    ckpt = os.path.join(config.RESULTS_DIR, "main_results_checkpoint.pkl")
    final = os.path.join(config.RESULTS_DIR, "main_results.pkl")
    results = {}
    src_file = None
    for path, label in ((final, "final"), (ckpt, "checkpoint")):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                src_file = label
                break
            except Exception:
                continue

    if not results:
        print(colored("  [INFO] No main experiment data yet.", C_DIM))
        return

    algorithms = config.ALGORITHMS
    datasets = config.DATASETS
    num_runs = config.NUM_RUNS
    total = len(algorithms) * len(datasets) * num_runs

    done = 0
    matrix = {}
    for ds in datasets:
        matrix[ds] = {}
        for alg in algorithms:
            if ds in results and alg in results[ds]:
                n = len(results[ds][alg].get("final_rf_acc", []))
                matrix[ds][alg] = n
                done += n
            else:
                matrix[ds][alg] = 0

    full_ds = sum(1 for ds in datasets
                  if all(matrix[ds][a] >= num_runs for a in algorithms))

    print(f"\n  {progress_bar(done, total)}  "
          f"{done}/{total} runs  ({full_ds}/{len(datasets)} datasets complete)  "
          f"[{src_file}]")

    log_path = pick_latest_log("run_main_log")
    status, age, pids = process_status(log_path, script_name="run_main")
    status_color = {"RUNNING": C_GREEN, "STALLED": C_RED, "IDLE": C_DIM}[status]
    log_name = os.path.basename(log_path) if log_path else "(no log)"
    pid_str = f", PID={pids[0]}" if pids and pids[0] > 0 else ""
    print(f"  Status: {colored(status, status_color)}{pid_str}  "
          f"(log mtime: {human_time(age)} ago, file: {log_name})")

    # Preliminary accuracy matrix
    if any(matrix[ds][a] >= num_runs for ds in datasets for a in algorithms):
        print(f"\n  RF Accuracy by Dataset and Algorithm:")
        print(f"  {'Dataset':14s}", end="")
        for a in algorithms:
            print(f" {a:>9s}", end="")
        print()
        print("  " + "-" * (14 + 10 * len(algorithms)))
        for ds in datasets:
            row = f"  {ds:14s}"
            for alg in algorithms:
                if matrix[ds][alg] >= num_runs:
                    acc = np.mean(results[ds][alg]["final_rf_acc"])
                    cell = f"{acc:>9.4f}"
                    if alg == "AMPA_FS":
                        cell = colored(cell, C_YELLOW)
                elif matrix[ds][alg] > 0:
                    cell = colored(f"{matrix[ds][alg]:>4d}/{num_runs}", C_CYAN)
                else:
                    cell = colored(f"{'.':>9s}", C_DIM)
                row += f" {cell}"
            print(row)


# Experiment 3: Parameter Sensitivity
def show_sensitivity_progress():
    print(colored("\n" + "=" * 72, C_BOLD))
    print(colored("  PARAMETER SENSITIVITY", C_BOLD + C_CYAN))
    print(colored("=" * 72, C_BOLD))

    ckpt = os.path.join(config.RESULTS_DIR, "sensitivity_results_checkpoint.pkl")
    final = os.path.join(config.RESULTS_DIR, "sensitivity_results.pkl")

    results = {}
    for path in (final, ckpt):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    results = pickle.load(f)
                break
            except Exception:
                continue

    if not results:
        print(colored("  [INFO] No sensitivity data yet (not started).", C_DIM))
        return

    datasets = config.SENSITIVITY_DATASETS
    params = config.SENSITIVITY_PARAMS
    num_runs = config.SENSITIVITY_RUNS

    total_configs = sum(len(vs) for vs in params.values()) * len(datasets)
    total_tasks = total_configs * num_runs

    done = 0
    for ds in datasets:
        for param in params:
            for val in params[param]:
                key = (ds, param, val)
                if key in results:
                    done += len(results[key].get("accuracy", []))
    print(f"\n  {progress_bar(done, total_tasks)}  {done}/{total_tasks} runs")


# Main
def main():
    parser = argparse.ArgumentParser(description="Experiment Progress Monitor")
    parser.add_argument("--main", action="store_true", help="Main experiment only")
    parser.add_argument("--ablation", action="store_true", help="Ablation only")
    parser.add_argument("--sensitivity", action="store_true", help="Sensitivity only")
    parser.add_argument("--watch", action="store_true",
                        help="Auto-refresh every 10 seconds")
    args = parser.parse_args()

    show_all = not (args.main or args.ablation or args.sensitivity)

    def render():
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        print(colored(f"AMPA-FS Experiment Progress  "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      C_BOLD + C_BLUE))
        if args.ablation or show_all:
            show_ablation_progress()
        if args.main or show_all:
            show_main_progress()
        if args.sensitivity or show_all:
            show_sensitivity_progress()
        print()

    if args.watch:
        try:
            while True:
                render()
                print(colored("  [auto-refresh every 10s; Ctrl-C to quit]", C_DIM))
                time.sleep(10)
        except KeyboardInterrupt:
            print(colored("\n  exiting watch mode.", C_DIM))
    else:
        render()


if __name__ == "__main__":
    main()
