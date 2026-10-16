"""Leaderboard jobs for cron: an ordered submission queue and a score log.

``submit``: submits a version of the submission kernel unless a submission with the same message
exists already (so a rerun never spends a second slot), retries while the network is down, and
confirms the submission in the submissions list. ``add`` appends a version to ``runs/lb_queue.json``;
``queue`` (run from cron) submits the first queued entry not on the leaderboard yet, once per run and
only while fewer than ``--per-day`` submissions were made that UTC day. ``log`` appends every completed
submission not yet in ``runs/lb_log.md`` with its public score and the difference to analog only (0.285).

    python scripts/lb_job.py add 12 "combiner without FP: ..."
    python scripts/lb_job.py queue
    python scripts/lb_job.py list
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
QUEUE = RUNS / "lb_queue.json"
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


def load_queue() -> list[dict]:
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else []


def add(version: int, message: str) -> int:
    q = load_queue()
    if any(e["message"] == message for e in q):
        print(f"{now()} v{version}: already queued", flush=True)
        return 0
    q.append({"version": version, "message": message, "added": now()})
    RUNS.mkdir(exist_ok=True)
    QUEUE.write_text(json.dumps(q, indent=1) + "\n")
    print(f"{now()} v{version}: queued at position {len(q)}", flush=True)
    return 0


def queue(per_day: int) -> int:
    subs = submissions()
    done = {s["description"] for s in subs}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used = sum(s["date"].startswith(today) for s in subs)
    pending = [e for e in load_queue() if e["message"] not in done]
    if not pending or used >= per_day:
        return 0
    e = pending[0]
    return submit(e["version"], e["message"], tries=20, wait=90)


def show() -> int:
    done = {s["description"]: s for s in submissions()}
    for i, e in enumerate(load_queue(), 1):
        s = done.get(e["message"])
        state = f"submitted {s['date'][:16]} → {s['publicScore'] or s['status'].split('.')[-1]}" if s else "pending"
        print(f"{i}. v{e['version']} [{state}] {e['message']}")
    return 0


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
    a = sub.add_parser("add")
    a.add_argument("version", type=int)
    a.add_argument("message")
    qp = sub.add_parser("queue")
    qp.add_argument("--per-day", type=int, default=2)
    sub.add_parser("list")
    sub.add_parser("log")
    args = ap.parse_args()
    if args.cmd == "submit":
        sys.exit(submit(args.version, args.message, args.tries, args.wait))
    if args.cmd == "add":
        sys.exit(add(args.version, args.message))
    if args.cmd == "queue":
        sys.exit(queue(args.per_day))
    sys.exit(show() if args.cmd == "list" else log())


if __name__ == "__main__":
    main()
