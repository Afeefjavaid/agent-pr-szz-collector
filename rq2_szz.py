"""SZZ-based post-merge bug detection for AI agent pull requests (v7.1).

v7.1: if the recent-history (shallow) clone fails twice, the last attempt clones the full history.

v7 uses exactly the same per-PR logic as v5. The only change is speed:
the pull requests of one repository are analyzed in parallel threads.
Results do not depend on the order of work, so output is identical to v5.
"""
import json
import os
import re
import shutil
import stat
import subprocess
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLONES = Path(os.environ.get("SZZ_CLONES", "clones"))

SRC_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".cs", ".c", ".h", ".cc",
           ".cpp", ".hpp", ".rb", ".php", ".swift", ".scala", ".m", ".dart", ".vue", ".svelte", ".lua", ".sh"}
NOISE_FILE = re.compile(r"(^|/)(package(-lock)?\.json|yarn\.lock|pnpm-lock\.yaml|.*\.lock|go\.(sum|mod)|CHANGELOG[^/]*)$", re.I)
TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|specs?|testing)/|(_test|\.test|\.spec|_spec)\.\w+$|(^|/)test_[^/]+$|Tests?\.\w+$")
BOT_AUTHOR = re.compile(r"dependabot|renovate|github-actions|greenkeeper|snyk|pre-commit-ci|mergify|allcontributors", re.I)
FIX = re.compile(r"\b(fix(e[sd])?|bug|hotfix|regression|broken|crash)\b", re.I)
DEPS = re.compile(r"\b(deps?|dependenc\w*|bump\w*|upgrade\w*|lockfile)\b", re.I)
DOCFIX = re.compile(r"\b(typos?|spelling|grammar|docs?|documentation|comments?|readme|lint|format(ting)?|style|whitespace)\b", re.I)
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+")
SHA_LINE = re.compile(r"^([0-9a-f]{40}) \d+ \d+")


def safe_name(full_name):
    return full_name.replace("/", "__")


def size_gb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9


def _force_remove(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def git(args, cwd, timeout=900):
    try:
        r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if r.returncode == 0 else None


def src_only(files):
    return {f for f in files if Path(f).suffix.lower() in SRC_EXT and not NOISE_FILE.search(f)}


def load_history(repo_dir):
    """Non-merge commits on the default branch: (sha, unix time, author, message, files)."""
    out = git(["log", "--no-merges", "--no-renames", "--name-only",
               "--format=%x1e%H%x1f%ct%x1f%an <%ae>%x1f%B%x1f", "HEAD"], repo_dir, timeout=1800) or ""
    commits = []
    for rec in out.split("\x1e")[1:]:
        p = rec.split("\x1f")
        if len(p) < 5:
            continue
        files = {l.strip() for l in p[4].splitlines() if l.strip()}
        commits.append((p[0], int(p[1]), p[2], p[3], files))
    commits.sort(key=lambda c: c[1])
    return commits


def pr_files(repo_dir, sha):
    """Files changed by the PR's merge (or squash/rebase) commit."""
    line = git(["rev-list", "--parents", "-n", "1", sha], repo_dir)
    if not line:
        return None
    p = line.split()
    if len(p) >= 3:
        out = git(["diff", "--name-only", "--no-renames", p[1], sha], repo_dir)
    else:
        out = git(["show", "--name-only", "--no-renames", "--format=", sha], repo_dir)
    return {l.strip() for l in out.splitlines() if l.strip()} if out is not None else None


def pr_commit_set(repo_dir, merge_sha):
    """Commits that belong to the PR: the squash commit, or the branch commits of a merge commit."""
    line = git(["rev-list", "--parents", "-n", "1", merge_sha], repo_dir)
    if not line:
        return set()
    p = line.split()
    if len(p) >= 3:
        out = git(["rev-list", f"{p[1]}..{p[2]}"], repo_dir) or ""
        return set(out.split()) | {merge_sha}
    return {merge_sha}


def removed_ranges(repo_dir, fix_sha, path):
    """Line ranges (in the parent version) that the fix commit changed or deleted in this file."""
    out = git(["diff", "-U0", "--no-renames", f"{fix_sha}^", fix_sha, "--", path], repo_dir) or ""
    ranges = []
    for line in out.splitlines():
        m = HUNK.match(line)
        if m:
            start, count = int(m.group(1)), int(m.group(2) or 1)
            if count > 0:
                ranges.append((start, count))
    return ranges


def blamed_commits(repo_dir, fix_sha, path, cache):
    """Commits that last wrote the lines this fix changed (the SZZ step), cached per (fix, file)."""
    key = (fix_sha, path)
    if key not in cache:
        shas, ranges = set(), removed_ranges(repo_dir, fix_sha, path)
        if ranges:
            args = ["blame", "--porcelain", "-w"]
            for s, c in ranges[:50]:
                args += ["-L", f"{s},+{c}"]
            out = git(args + [f"{fix_sha}^", "--", path], repo_dir) or ""
            shas = {m.group(1) for m in map(SHA_LINE.match, out.splitlines()) if m}
        cache[key] = shas
    return cache[key]


def clone_repo(full_name, d, tries=3, timeout=3600):
    url, err = f"https://github.com/{full_name}.git", "unknown"
    for attempt in range(tries):
        try:
            shallow = ["--shallow-since=2024-10-01"] if attempt < tries - 1 else []   # last try: full history
            r = subprocess.run(["git", "clone", "--bare", "--quiet"] + shallow + [url, str(d)],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return None
            err = r.stderr[:200]
        except subprocess.TimeoutExpired:
            err = "clone timeout"
        if d.exists():
            shutil.rmtree(d, onexc=_force_remove)
        time.sleep(10 * (attempt + 1))
    return err


def analyze_pr(d, pr, commits, times, cache, max_hits=5):
    """Same per-PR logic as v5: up to `max_hits` candidate SZZ fixes, plus follow-up and revert flags."""
    files = pr_files(d, pr.merge_sha)
    src = src_only(files) if files else set()
    src_nt = {f for f in src if not TEST_FILE.search(f)}
    mine = pr_commit_set(d, pr.merge_sha) if src else set()
    hits = []
    row = {"pr_id": pr.pr_id, "repo": pr.full_name, "files_found": files is not None,
           "n_src": len(src), "n_src_nontest": len(src_nt),
           "followup_180": 0, "followup_fix_180": 0, "revert_180": 0}
    if src:
        t0 = int(pr.merged_at.timestamp())
        lo, hi = bisect_right(times, t0), bisect_right(times, t0 + 180 * 86400)
        ref = re.compile(rf"#{pr.number}\b")
        for sha, ct, author, msg, cf in commits[lo:hi]:
            if sha in mine or BOT_AUTHOR.search(author):
                continue
            subject = msg.splitlines()[0] if msg else ""
            days = (ct - t0) / 86400
            if pr.merge_sha in msg or (subject.lower().startswith("revert") and ref.search(msg)):
                row["revert_180"] = 1
            if ref.search(msg):
                row["followup_180"] = 1
                if FIX.search(subject):
                    row["followup_fix_180"] = 1
            if len(hits) >= max_hits:
                continue
            if not (FIX.search(subject) and not DEPS.search(subject) and not DOCFIX.search(subject)):
                continue
            for path in sorted(cf & src):
                if blamed_commits(d, sha, path, cache) & mine:
                    hits.append({"sha": sha, "days": round(days, 1), "subject": subject[:150],
                                 "file": path, "nontest": path in src_nt})
                    break
    first = hits[0] if hits else None
    first_nt = next((h for h in hits if h["nontest"]), None)
    for w in (30, 90, 180):
        row[f"szz_{w}"] = int(bool(first) and first["days"] <= w)
        row[f"szz_nt_{w}"] = int(bool(first_nt) and first_nt["days"] <= w)
    row["szz_days"] = first["days"] if first else None
    row["szz_fix_subject"] = first["subject"] if first else None
    row["szz_file"] = first["file"] if first else None
    row["n_hits"] = len(hits)
    row["szz_hits"] = json.dumps(hits)
    return row


def process_repo(full_name, prs, max_hits=5, inner_workers=3):
    """Clone one repository, analyze all its PRs (in parallel threads), delete the clone.

    Returns (rows, log); rows is None if the clone failed.
    """
    d = CLONES / (safe_name(full_name) + ".git")
    if d.exists():
        shutil.rmtree(d, onexc=_force_remove)
    t = time.time()
    err = clone_repo(full_name, d)
    if err:
        return None, {"repo": full_name, "status": "clone_failed", "error": err}
    try:
        clone_gb = size_gb(d)
        commits = load_history(d)
        times = [c[1] for c in commits]
        cache = {}
        pr_list = list(prs.itertuples())
        with ThreadPoolExecutor(max_workers=inner_workers) as ex:
            rows = list(ex.map(lambda pr: analyze_pr(d, pr, commits, times, cache, max_hits), pr_list))
        log = {"repo": full_name, "status": "ok", "n_prs": len(prs), "clone_gb": round(clone_gb, 2),
               "commits": len(commits), "blame_checks": len(cache),
               "seconds": round(time.time() - t), "version": "v7"}
        return rows, log
    finally:
        if d.exists():
            shutil.rmtree(d, onexc=_force_remove)
