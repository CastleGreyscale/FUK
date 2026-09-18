#!/usr/bin/env python3
"""
NAND refresh — rewrite model weights that have gone cold.

Flash cells leak charge. Data written once and never touched again drifts far
enough that the controller starts paying for ECC retries on every read, and a
file that arrived at 3.2GB/s comes back months later at 500MB/s. Nothing is
corrupt and SMART stays clean; reads just get slow. Model weights are the worst
case for this — written once, read forever, never modified.

The fix is to make the controller write the data again. Copying a file and
renaming the copy over the original does it: new cells, fresh charge, full read
speed restored. Verified on this machine 2026-07-11 — one 10GB shard went from
~500MB/s back to 3.2GB/s, and a 507GB sweep took 43 minutes.

So this module does two things:

  scan      Measure the cold read speed of every large file under models_root
            by dropping its page cache and timing a sample read. Reads only.

  refresh   The same measurement, but files below the threshold get rewritten
            and re-measured, so the report says what each one gained.

Both are driven from Utilities -> Models, and both run standalone for headless
or cron use:

    python fuk/utils/nand_refresh.py scan
    python fuk/utils/nand_refresh.py refresh --threshold 1200

Safety, because this rewrites the model library:

  * The copy is written to a temp file beside the original, fsynced, size-checked,
    and then os.replace'd — an atomic rename. A crash at any point leaves either
    the old file or the new one, never a truncated one. Safe to run while the
    server is up: a pipeline mid-load holds the old inode open and finishes from
    it untouched.
  * Every rewrite checks free space first and refuses without room for the file
    plus headroom. On btrfs both copies exist for an instant, and a reflinked or
    snapshotted file is unshared by this, which costs real space.
  * Content is never altered. Size is verified before the rename, mode and mtime
    are preserved, and btrfs checksums every block it writes, so a bad copy
    surfaces as a read error rather than silent corruption.
  * Symlinks are resolved and the real file rewritten once, however many links
    point at it. Hardlinked files are skipped: replacing one would break the link.

Cost: a sweep rewrites the whole library, so plan on ~800GB of writes. That is
under a tenth of a percent of a 2TB drive's endurance rating — the wear is
irrelevant, the hour it takes is not.

POSIX only (posix_fadvise, statvfs). Meaningless on spinning rust or a network
mount, which `describe_root()` reports so the UI can say so instead of guessing.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Only large files are worth this. Aging shows up in sustained sequential reads;
# a 5MB config that takes 40ms instead of 8ms is not what makes a load slow, and
# walking thousands of them would cost more than it saves.
DEFAULT_MIN_SIZE_MB = 64

# Sample read per file. 64MB is long enough to leave seek and syscall overhead
# behind and short enough that measuring 156 files costs ~10GB of reads.
DEFAULT_SAMPLE_MB = 64

# Below this, call a file aged. Fresh NVMe here reads 1.7-3.2GB/s and aged data
# lands at 500-600MB/s, so the gap is wide and the exact line is not delicate.
DEFAULT_THRESHOLD_MBPS = 1200

# Free space demanded before a rewrite: the file itself, plus room to spare so a
# sweep can never be the thing that fills the disk.
FREE_HEADROOM_BYTES = 8 * 1024 ** 3

_CHUNK = 8 * 1024 ** 2          # copy/read block
_PROGRESS_EVERY = 256 * 1024 ** 2   # bytes between job progress updates

_TMP_SUFFIX = ".fukrefresh.tmp"

# Alongside perf_history.json — same kind of data, same place to look for it.
_STATE_PATH = Path(__file__).resolve().parent.parent / "ui" / "data" / "nand_refresh.json"
_MAX_TRACKED = 4000     # per-file refresh records kept in the state file

MB = 1024 ** 2


# ---------------------------------------------------------------------------
# Roots and filesystem facts
# ---------------------------------------------------------------------------

def models_root(config_dir: Optional[Path] = None) -> Path:
    """models_root from defaults.json, the same one the downloader writes into."""
    cfg = Path(config_dir) if config_dir else Path(__file__).resolve().parent.parent / "config"
    with open(cfg / "defaults.json") as f:
        return Path(json.load(f).get("models_root", "./models")).expanduser()


# Filesystems with no local flash behind them. Aging is not a thing that
# happens here, and rewriting 800GB over the network would be a long mistake.
_NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "sshfs", "fuse.sshfs", "9p", "ceph"}


def _mount_of(path: Path) -> Optional[dict]:
    """The mountinfo entry for the filesystem holding `path`.

    Read from /proc rather than derived from st_dev, because btrfs (and every
    other filesystem on an anonymous block device) reports a made-up device
    number that has no /sys/dev/block entry at all. mountinfo carries the real
    source device and the filesystem type.
    """
    target = os.path.realpath(path)
    best = None
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                # <id> <parent> <maj:min> <root> <mountpoint> <opts>... - <fstype> <source> ...
                left, _, right = line.partition(" - ")
                lf, rf = left.split(), right.split()
                if len(lf) < 5 or len(rf) < 2:
                    continue
                mountpoint = lf[4].replace("\\040", " ")
                if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
                    # Longest prefix wins — /home and /home/brad/ai both match a
                    # path under the latter, and only the deeper one is its mount.
                    # On a tie the later line wins: that is the mount stacked on
                    # top, which is how an autofs trigger hides the real NFS mount.
                    if best is None or len(mountpoint) >= len(best["mountpoint"]):
                        best = {"mountpoint": mountpoint, "fstype": rf[0], "source": rf[1]}
    except OSError:
        return None
    return best


def _rotational(source: str) -> Optional[bool]:
    """True for spinning disks, False for flash, None when it cannot be told."""
    if not source.startswith("/dev/"):
        return None
    base = Path("/sys/class/block") / Path(source).name
    # A partition has no queue/ of its own; the parent disk carries it.
    for candidate in (base / "queue" / "rotational", base / ".." / "queue" / "rotational"):
        try:
            return candidate.read_text().strip() == "1"
        except OSError:
            continue
    return None


def describe_root(root: Path) -> dict:
    """What the UI needs to decide whether any of this applies here."""
    info = {
        "root": str(root),
        "exists": root.is_dir(),
        "free_bytes": 0,
        "total_bytes": 0,
        "rotational": None,
        "fstype": None,
        "device": None,
        "supported": False,
        "reason": None,
    }
    if not info["exists"]:
        info["reason"] = f"{root} does not exist"
        return info

    try:
        vfs = os.statvfs(root)
        info["free_bytes"] = vfs.f_bavail * vfs.f_frsize
        info["total_bytes"] = vfs.f_blocks * vfs.f_frsize
    except OSError as e:
        info["reason"] = f"statvfs failed: {e}"
        return info

    if not hasattr(os, "posix_fadvise"):
        info["reason"] = "posix_fadvise unavailable — cold reads cannot be measured"
        return info

    mount = _mount_of(root)
    if mount:
        info["fstype"] = mount["fstype"]
        info["device"] = mount["source"]
        info["rotational"] = _rotational(mount["source"])

        if mount["fstype"] in _NETWORK_FS:
            info["reason"] = (f"models are on a {mount['fstype']} mount — "
                              f"there is no local flash to refresh")
            return info

    if info["rotational"]:
        info["reason"] = "models live on a rotational disk — NAND aging does not apply"
        return info

    info["supported"] = True
    return info


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def candidates(root: Path, min_size_bytes: int) -> List[dict]:
    """Large regular files under `root`, one entry per real file.

    Symlinks resolve to their target: the LoRA tree is links into the model
    cache, and the blob behind twelve camera-move links should be rewritten once,
    not twelve times. Dedup is by (device, inode) rather than path, which catches
    that and any other aliasing for free.
    """
    seen: set = set()
    out: List[dict] = []
    skipped: List[dict] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(_TMP_SUFFIX):
                continue
            path = Path(dirpath) / name
            try:
                st = os.stat(path)       # follows symlinks, which is the point
            except OSError:
                continue                 # dangling link or vanished mid-walk
            if not os.path.isfile(path) or st.st_size < min_size_bytes:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)

            real = os.path.realpath(path)
            if st.st_nlink > 1:
                # Rewriting via rename would silently break the other links.
                skipped.append({"path": real, "size": st.st_size, "reason": "hardlinked"})
                continue

            out.append({"path": real, "size": st.st_size, "mtime": st.st_mtime})

    out.sort(key=lambda f: f["size"], reverse=True)
    for f in skipped:
        f["skipped"] = True
    return out + skipped


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _drop_cache(fd: int, offset: int = 0, length: int = 0) -> None:
    """Evict this file's clean pages. Best effort — never worth raising over."""
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass


def measure_mbps(path: str, sample_bytes: int = DEFAULT_SAMPLE_MB * MB) -> Optional[float]:
    """Cold read speed in MB/s, or None if the file could not be read.

    The page cache is dropped before timing and again afterwards, so the number
    reflects the drive rather than RAM, and running this does not evict whatever
    the rest of the machine had cached. Measures logical bytes: on a compressed
    filesystem a compressible file will read faster than the device can go, which
    for incompressible fp16 weights is not a distinction that arises.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    n = min(sample_bytes, size)
    if n <= 0:
        return None

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        _drop_cache(fd, 0, n)
        read = 0
        t0 = time.perf_counter()
        while read < n:
            block = os.read(fd, min(_CHUNK, n - read))
            if not block:
                break
            read += len(block)
        dt = time.perf_counter() - t0
        _drop_cache(fd, 0, read)
    except OSError:
        return None
    finally:
        os.close(fd)

    if dt <= 0 or read == 0:
        return None
    return (read / MB) / dt


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

class RefreshError(RuntimeError):
    """A rewrite that did not happen. `fatal` means the rest of the sweep is
    doomed too — running out of disk is not a per-file problem, and grinding
    through a hundred more failures to prove it helps no one."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def rewrite(path: str, progress: Optional[Callable[[int], None]] = None) -> int:
    """Rewrite one file in place through a temp copy and an atomic rename.

    Returns the bytes written. Raises RefreshError with a reason rather than
    leaving anything half-done: the original is only replaced once the copy is
    complete, fsynced, and verified to be the same size.
    """
    src = Path(path)
    try:
        st = os.stat(src)
    except OSError as e:
        raise RefreshError(f"stat failed: {e}") from e

    if st.st_nlink > 1:
        raise RefreshError("hardlinked — a rename would break the other links")

    try:
        vfs = os.statvfs(src.parent)
        free = vfs.f_bavail * vfs.f_frsize
    except OSError as e:
        raise RefreshError(f"statvfs failed: {e}") from e
    if free < st.st_size + FREE_HEADROOM_BYTES:
        raise RefreshError(
            f"not enough free space — needs {(st.st_size + FREE_HEADROOM_BYTES) / 1024**3:.1f}GB, "
            f"{free / 1024**3:.1f}GB free",
            fatal=True,
        )

    tmp = src.with_name(f".{src.name}{_TMP_SUFFIX}")
    written = 0
    since_report = 0
    try:
        with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
            while True:
                block = fsrc.read(_CHUNK)
                if not block:
                    break
                fdst.write(block)
                written += len(block)
                since_report += len(block)
                if progress and since_report >= _PROGRESS_EVERY:
                    progress(since_report)
                    since_report = 0
                # Neither copy needs to stay cached; a full sweep would otherwise
                # push everything else on the machine out of RAM. Drop just the
                # block we read — re-advising the whole range each time would
                # walk more pages with every chunk.
                _drop_cache(fsrc.fileno(), written - len(block), len(block))
            fdst.flush()
            os.fsync(fdst.fileno())

        if written != st.st_size:
            raise RefreshError(f"short copy: wrote {written} of {st.st_size} bytes")

        os.chmod(tmp, st.st_mode)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, src)

        # Persist the rename itself, so a power cut cannot leave the directory
        # pointing at a file that was never durably linked.
        dfd = os.open(src.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

        # The data we just wrote is sitting clean in cache; drop it so the
        # follow-up measurement reads the drive and not RAM.
        fd = os.open(src, os.O_RDONLY)
        try:
            _drop_cache(fd, 0, 0)
        finally:
            os.close(fd)
    except RefreshError:
        _unlink_quietly(tmp)
        raise
    except OSError as e:
        _unlink_quietly(tmp)
        if e.errno == errno.ENOSPC:
            raise RefreshError("ran out of disk space mid-copy", fatal=True) from e
        raise RefreshError(f"{type(e).__name__}: {e}") from e

    if progress and since_report:
        progress(since_report)
    return written


def _unlink_quietly(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def clean_temp_files(root: Path) -> List[str]:
    """Remove temp copies left behind by a killed run. Originals are untouched."""
    removed = []
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            if name.endswith(_TMP_SUFFIX):
                p = Path(dirpath) / name
                _unlink_quietly(p)
                removed.append(str(p))
    return removed


# ---------------------------------------------------------------------------
# Persistent state
#
# What was measured, what was rewritten, when. Small enough to rewrite whole,
# written atomically, and never allowed to take a sweep down with it.
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()


def _load_state() -> dict:
    try:
        with open(_STATE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        tmp.replace(_STATE_PATH)
    except Exception:
        pass


def history() -> dict:
    """Last scan, last refresh, and per-file refresh records."""
    with _state_lock:
        state = _load_state()
    return {
        "last_scan": state.get("last_scan"),
        "last_refresh": state.get("last_refresh"),
        "files_tracked": len(state.get("files", {})),
    }


def _record(kind: str, summary: dict, files: List[dict]) -> None:
    with _state_lock:
        state = _load_state()
        state[f"last_{kind}"] = summary
        if kind == "refresh":
            tracked = state.get("files", {})
            for f in files:
                if f.get("rewritten"):
                    tracked[f["path"]] = {
                        "t": summary["finished_at"],
                        "before": f.get("before_mbps"),
                        "after": f.get("after_mbps"),
                    }
            # Oldest records fall off — this is a log, not an inventory.
            if len(tracked) > _MAX_TRACKED:
                ordered = sorted(tracked.items(), key=lambda kv: kv[1].get("t") or "")
                tracked = dict(ordered[-_MAX_TRACKED:])
            state["files"] = tracked
        _save_state(state)


# ---------------------------------------------------------------------------
# Jobs
#
# A sweep is tens of minutes of disk work, so it runs on a thread and the UI
# polls, exactly as model downloads do. One at a time: two sweeps competing for
# the same drive would measure each other rather than the flash.
# ---------------------------------------------------------------------------

_job: Optional[dict] = None
_job_lock = threading.Lock()
_cancel = threading.Event()


def get_job() -> Optional[dict]:
    with _job_lock:
        return dict(_job) if _job else None


def _update(**fields) -> None:
    with _job_lock:
        if _job is not None:
            _job.update(fields)


def cancel() -> bool:
    """Ask a running sweep to stop after the current file. Never mid-rewrite."""
    with _job_lock:
        running = _job is not None and _job["status"] == "running"
    if running:
        _cancel.set()
    return running


def start(mode: str = "scan", *, root: Optional[Path] = None,
          threshold_mbps: float = DEFAULT_THRESHOLD_MBPS,
          min_size_mb: int = DEFAULT_MIN_SIZE_MB,
          sample_mb: int = DEFAULT_SAMPLE_MB,
          rewrite_all: bool = False,
          log=None) -> dict:
    """Kick off a scan or refresh on a background thread. Returns the job."""
    global _job

    if mode not in ("scan", "refresh"):
        raise ValueError(f"unknown mode '{mode}'")

    root = Path(root) if root else models_root()
    info = describe_root(root)
    if not info["supported"]:
        raise RefreshError(info["reason"] or "unsupported filesystem")

    with _job_lock:
        if _job is not None and _job["status"] == "running":
            raise RefreshError("a sweep is already running")
        _cancel.clear()
        _job = {
            "id": uuid.uuid4().hex[:12],
            "mode": mode,
            "root": str(root),
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "threshold_mbps": threshold_mbps,
            "rewrite_all": bool(rewrite_all),
            "total": 0,
            "completed": 0,
            "current": None,
            "aged": 0,
            "rewritten": 0,
            "bytes_total": 0,
            "bytes_done": 0,
            "skipped": [],
            "failed": [],
            "files": [],
            "error": None,
        }
        job = dict(_job)

    threading.Thread(
        target=_run,
        args=(mode, root, threshold_mbps, min_size_mb * MB, sample_mb * MB, rewrite_all, log),
        name=f"nand-{mode}",
        daemon=True,
    ).start()
    return job


def _run(mode, root, threshold, min_size, sample_bytes, rewrite_all, log):
    def _log(level, msg):
        if log:
            getattr(log, level)("NandRefresh", msg)
        else:
            print(f"[nand-refresh] {msg}", flush=True)

    try:
        stale = clean_temp_files(root)
        if stale:
            _log("warning", f"cleared {len(stale)} leftover temp file(s) from an interrupted run")

        found = candidates(root, min_size)
        files = [f for f in found if not f.get("skipped")]
        skipped = [f for f in found if f.get("skipped")]
        _update(total=len(files), skipped=skipped,
                bytes_total=sum(f["size"] for f in files))
        _log("info", f"{mode}: {len(files)} file(s), "
                     f"{sum(f['size'] for f in files) / 1024**3:.1f}GB under {root}")

        results, aged, rewritten, done_bytes = [], 0, 0, 0

        for i, f in enumerate(files):
            if _cancel.is_set():
                break
            short = os.path.relpath(f["path"], root)
            _update(completed=i, current=short)

            before = measure_mbps(f["path"], sample_bytes)
            row = {"path": f["path"], "rel": short, "size": f["size"],
                   "before_mbps": round(before, 1) if before else None,
                   "after_mbps": None, "rewritten": False}

            is_aged = before is not None and before < threshold
            if is_aged:
                aged += 1

            if mode == "refresh" and (rewrite_all or is_aged):
                def _tick(n):
                    nonlocal done_bytes
                    done_bytes += n
                    _update(bytes_done=done_bytes)
                try:
                    rewrite(f["path"], progress=_tick)
                    row["rewritten"] = True
                    rewritten += 1
                    after = measure_mbps(f["path"], sample_bytes)
                    row["after_mbps"] = round(after, 1) if after else None
                except RefreshError as e:
                    row["error"] = str(e)
                    with _job_lock:
                        if _job is not None:
                            _job["failed"].append({"path": short, "error": str(e)})
                    _log("warning", f"{short}: {e}")
                    if e.fatal:
                        _update(error=str(e))
                        break
            else:
                # Not rewritten, but its bytes still count toward the bar so the
                # progress reflects the sweep rather than only the slow files.
                done_bytes += f["size"]
                _update(bytes_done=done_bytes)

            results.append(row)
            _update(completed=i + 1, aged=aged, rewritten=rewritten)

        cancelled = _cancel.is_set()
        # A fatal error (out of space) broke the loop early rather than raising,
        # so the sweep finished with a report but is not a success.
        aborted = (get_job() or {}).get("error")
        measured = [r["before_mbps"] for r in results if r["before_mbps"]]
        gains = [r for r in results if r["rewritten"] and r["after_mbps"] and r["before_mbps"]]

        summary = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "root": str(root),
            "threshold_mbps": threshold,
            "files_measured": len(results),
            "files_aged": aged,
            "files_rewritten": rewritten,
            "bytes_rewritten": sum(r["size"] for r in results if r["rewritten"]),
            "aged_bytes": sum(r["size"] for r in results
                              if r["before_mbps"] and r["before_mbps"] < threshold),
            "slowest_mbps": round(min(measured), 1) if measured else None,
            "median_mbps": round(sorted(measured)[len(measured) // 2], 1) if measured else None,
            "fastest_mbps": round(max(measured), 1) if measured else None,
            "mean_gain": round(
                sum(r["after_mbps"] / r["before_mbps"] for r in gains) / len(gains), 2
            ) if gains else None,
            "elapsed_s": round(time.time() - (get_job() or {}).get("started_at", time.time()), 1),
            "cancelled": cancelled,
            "aborted": aborted,
        }

        # Slowest first: the report is a worklist, and the worst offenders are
        # the only rows anyone reads.
        results.sort(key=lambda r: (r["before_mbps"] is None, r["before_mbps"] or 0))

        _record(mode, summary, results)
        state = "cancelled" if cancelled else "failed" if aborted else "completed"
        _update(status=state, finished_at=time.time(), current=None,
                files=results, summary=summary)

        _log("warning" if aborted else "info",
             f"{mode} {state}: {len(results)} measured, "
             f"{aged} aged"
             + (f", {rewritten} rewritten ({summary['bytes_rewritten'] / 1024**3:.0f}GB)"
                if rewritten else "")
             + (f", {summary['mean_gain']}x mean speedup" if summary["mean_gain"] else ""))

    except Exception as e:
        _update(status="failed", finished_at=time.time(), current=None,
                error=f"{type(e).__name__}: {e}")
        _log("error", f"{mode} failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f}GB"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nand_refresh",
        description="Measure and restore cold read speed on the model library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Flash loses charge over time; rewriting a file restores full read speed.\n"
               "  scan     measure only, changes nothing\n"
               "  refresh  rewrite anything below the threshold",
    )
    parser.add_argument("mode", choices=["scan", "refresh"])
    parser.add_argument("--root", type=Path, default=None,
                        help="defaults to models_root from defaults.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_MBPS,
                        metavar="MBPS", help=f"aged below this (default {DEFAULT_THRESHOLD_MBPS})")
    parser.add_argument("--min-size-mb", type=int, default=DEFAULT_MIN_SIZE_MB)
    parser.add_argument("--sample-mb", type=int, default=DEFAULT_SAMPLE_MB)
    parser.add_argument("--all", action="store_true",
                        help="refresh: rewrite every file, not just the slow ones")
    parser.add_argument("--limit", type=int, default=15, help="rows to print (default 15)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    root = args.root or models_root()
    info = describe_root(root)
    if not info["supported"]:
        print(f"error: {info['reason']}", file=sys.stderr)
        return 2

    print(f"{args.mode}: {root}  ({_fmt_gb(info['free_bytes'])} free)")
    try:
        start(args.mode, root=root, threshold_mbps=args.threshold,
              min_size_mb=args.min_size_mb, sample_mb=args.sample_mb,
              rewrite_all=args.all)
    except RefreshError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # The worker logs to stdout too, so the redrawing counter is for terminals
    # only — in a cron log it would just be a wall of carriage returns.
    live = sys.stdout.isatty()
    last = -1
    while True:
        job = get_job()
        if job is None or job["status"] != "running":
            break
        if live and job["completed"] != last:
            last = job["completed"]
            print(f"\r  {job['completed']}/{job['total']}  {job['current'] or ''}"[:110].ljust(110),
                  end="", flush=True)
        time.sleep(0.5)
    if live:
        print()

    job = get_job() or {}
    if job.get("status") == "failed" and not job.get("summary"):
        print(f"error: {job.get('error')}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"summary": job.get("summary"), "files": job.get("files", [])}, indent=2))
        return 0

    s = job.get("summary") or {}
    for row in (job.get("files") or [])[:args.limit]:
        mark = "*" if row["rewritten"] else " "
        before = f"{row['before_mbps']:>7.1f}" if row["before_mbps"] else "      ?"
        after = f" -> {row['after_mbps']:>7.1f}" if row["after_mbps"] else ""
        print(f" {mark} {before}{after} MB/s  {_fmt_gb(row['size']):>8}  {row['rel']}")

    print(f"\n{s.get('files_measured', 0)} measured · {s.get('files_aged', 0)} aged "
          f"({_fmt_gb(s.get('aged_bytes', 0))}) · median {s.get('median_mbps')} MB/s "
          f"· slowest {s.get('slowest_mbps')} MB/s")
    if s.get("files_rewritten"):
        print(f"{s['files_rewritten']} rewritten, {_fmt_gb(s['bytes_rewritten'])} in "
              f"{s.get('elapsed_s', 0) / 60:.0f}min"
              + (f", {s['mean_gain']}x mean speedup" if s.get("mean_gain") else ""))
    elif args.mode == "scan" and s.get("files_aged"):
        print(f"run: {sys.argv[0]} refresh --threshold {args.threshold:g}")
    if s.get("cancelled"):
        print("cancelled — files already rewritten stay rewritten")
    if s.get("aborted"):
        print(f"stopped early: {s['aborted']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
