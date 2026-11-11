"""Run the SZZ bug detector over repositories. Works on a laptop, Colab or GitHub Actions.

Repositories are ranked by number of target PRs (1 = most). With --shard/--num-shards,
each machine takes every Nth repository of the chosen range, so big repositories are
spread evenly across machines. Finished repositories are skipped on a rerun (resume).
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import rq2_szz as s

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, default=1, help="first repository rank (1 = most PRs)")
ap.add_argument("--last", type=int, default=100, help="last repository rank")
ap.add_argument("--repos", default="", help="comma-separated repository names (overrides ranks; for tests)")
ap.add_argument("--workers", type=int, default=3, help="repositories processed at the same time")
ap.add_argument("--inner-workers", type=int, default=3, help="threads per repository")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--targets", default="data/rq2_targets.parquet")
ap.add_argument("--out", default="results")
ap.add_argument("--skip-file", default="skip.txt", help="optional file listing repositories to skip")
a = ap.parse_args()

OUT = Path(a.out) / "szz"
LOG = Path(a.out) / f"szz_log_shard{a.shard}.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
if s.CLONES.exists():
    shutil.rmtree(s.CLONES, onexc=s._force_remove)      # leftovers from an interrupted run
s.CLONES.mkdir(parents=True, exist_ok=True)

targets = pd.read_parquet(a.targets)
rank = targets.groupby("full_name").size().sort_values(ascending=False, kind="stable")
if a.repos.strip():
    chosen = [r.strip() for r in a.repos.split(",") if r.strip()]
else:
    chosen = list(rank.index[a.first - 1:a.last])
mine = [r for i, r in enumerate(chosen) if i % a.num_shards == a.shard]

skip = set()
if Path(a.skip_file).exists():
    skip = {l.strip() for l in open(a.skip_file, encoding="utf-8") if l.strip()}
todo = [r for r in mine if r not in skip and not (OUT / f"{s.safe_name(r)}.csv").exists()]
print(f"Shard {a.shard}/{a.num_shards}: {len(mine)} repositories, {len(mine) - len(todo)} skipped or done, "
      f"{len(todo)} to process ({a.workers} at a time, {a.inner_workers} threads each).", flush=True)

start, done = time.time(), 0
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    jobs = {pool.submit(s.process_repo, r, targets[targets["full_name"] == r], 10, a.inner_workers): r
            for r in todo}
    for job in as_completed(jobs):
        r = jobs[job]
        try:
            rows, log = job.result()
        except Exception as e:
            rows, log = None, {"repo": r, "status": "error", "error": str(e)[:200]}
        if rows is not None:
            pd.DataFrame(rows).to_csv(OUT / f"{s.safe_name(r)}.csv", index=False)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        done += 1
        mins = (time.time() - start) / 60
        print(f"[{done}/{len(todo)}] {log['status']:12s} {r} ({log.get('seconds', '-')}s) | "
              f"{mins:.0f} min elapsed, about {mins / done * (len(todo) - done):.0f} min left", flush=True)
print("Finished this batch.", flush=True)
