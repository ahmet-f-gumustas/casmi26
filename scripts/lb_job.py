"""Leaderboard jobs for cron: submit a kernel version and log public scores.

``submit``: submits a version of the submission kernel unless a submission with the same message
exists already (so a rerun never spends a second slot), retries while the network is down, and
confirms the submission in the submissions list. ``log``: appends every completed submission not
yet in ``runs/lb_log.md`` with its public score and the difference to analog only (0.285).

    python scripts/lb_job.py submit 9 "recall probe 2a: ..."
    python scripts/lb_job.py log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
KERNEL = "red6161/casmi26-submission"
ANALOG_ONLY = 0.285
RUNS = REPO / "runs"
LOG = RUNS / "lb_log.md"
HEADER = ("# Leaderboard log\n\n| ref | submitted (UTC) | description | public | Δ vs analog only 0.285 |\n"
          "|---|---|---|---|---|\n")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def submissions() -> list[dict]:
    out = subprocess.run([KAGGLE, "competitions", "submissions", "-c", COMPETITION, "--format", "json"],
                         capture_output=True, text=True, timeout=120, check=True).stdout
    return json.loads(out)


def submit(version: int, message: str, tries: int, wait: int) -> int:
    for attempt in range(1, tries + 1):
        try:
            if any(s["description"] == message for s in submissions()):
                print(f"{now()} v{version}: already submitted ({message!r})", flush=True)
                return 0
            r = subprocess.run([KAGGLE, "competitions", "submit", "-c", COMPETITION, "-k", KERNEL, "-v", str(version),
                                "-f", "submission.csv", "-q", "-m", message], capture_output=True, text=True,
                               timeout=300)
            time.sleep(10)
            if r.returncode == 0 and any(s["description"] == message for s in submissions()):
                print(f"{now()} v{version}: submitted", flush=True)
                return 0
            print(f"{now()} v{version}: attempt {attempt} failed (rc {r.returncode}): {r.stderr.strip()[-300:]}",
                  flush=True)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as e:
            print(f"{now()} v{version}: attempt {attempt} error: {e}", flush=True)
        time.sleep(wait)
    print(f"{now()} v{version}: NOT submitted after {tries} attempts", flush=True)
    return 1


def log() -> int:
    RUNS.mkdir(exist_ok=True)
    text = LOG.read_text() if LOG.exists() else HEADER
    for s in sorted(submissions(), key=lambda s: s["date"]):
        ref = str(s["ref"])
        if s["status"].endswith("COMPLETE") and s["publicScore"] and f"| {ref} |" not in text:
            score = float(s["publicScore"])
            text += (f"| {ref} | {s['date'][:16].replace('T', ' ')} | {s['description']} | {score:.3f} | "
                     f"{score - ANALOG_ONLY:+.3f} |\n")
            print(f"{now()} logged {ref}: {score:.3f}", flush=True)
    LOG.write_text(text)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("version", type=int)
    s.add_argument("message")
    s.add_argument("--tries", type=int, default=20)
    s.add_argument("--wait", type=int, default=90, help="seconds between attempts")
    sub.add_parser("log")
    args = ap.parse_args()
    sys.exit(submit(args.version, args.message, args.tries, args.wait) if args.cmd == "submit" else log())


if __name__ == "__main__":
    main()
