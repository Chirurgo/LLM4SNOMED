"""Resource measurement for the benchmark: latency stats, and the database container's
own CPU/memory/disk (via the docker CLI), which is what differs between backends --
the Python client process is the same harness for all of them."""
import json
import re
import statistics
import subprocess
import threading
import time
from typing import Dict, List, Optional

_UNITS = {"b": 1, "kb": 1e3, "kib": 1024, "mb": 1e6, "mib": 1024 ** 2,
          "gb": 1e9, "gib": 1024 ** 3, "tb": 1e12, "tib": 1024 ** 4}

# docker compose service -> data directory inside its container
DATA_DIRS = {
    "neo4j": "/data",
    "postgres": "/var/lib/postgresql/data",
    "qdrant": "/qdrant/storage",
    "weaviate": "/var/lib/weaviate",
    "meilisearch": "/meili_data",
}


def _to_bytes(s: str) -> float:
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", s)
    if not m:
        return float("nan")
    return float(m.group(1)) * _UNITS.get(m.group(2).lower(), float("nan"))


def _run(args, timeout_s: float = 120):
    """docker CLI call that can't hang the caller forever (returns None on timeout)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None


def container_id(service: str) -> Optional[str]:
    out = _run(["docker", "compose", "ps", "-q", service])
    if out is None:
        return None
    cid = out.stdout.strip().splitlines()
    return cid[0] if cid else None


def container_stats(cid: str) -> Optional[dict]:
    out = _run(["docker", "stats", "--no-stream", "--format", "{{json .}}", cid], timeout_s=60)
    if out is None or out.returncode != 0 or not out.stdout.strip():
        return None
    s = json.loads(out.stdout.strip().splitlines()[0])
    return {
        "mem_bytes": _to_bytes(s["MemUsage"].split("/")[0]),
        "cpu_pct": float(s["CPUPerc"].rstrip("%") or 0),
    }


def container_disk_bytes(service: str) -> Optional[float]:
    cid = container_id(service)
    path = DATA_DIRS.get(service)
    if not cid or not path:
        return None
    out = _run(["docker", "exec", cid, "du", "-sk", path], timeout_s=600)
    if out is None or out.returncode != 0:
        return None
    return float(out.stdout.split()[0]) * 1024


class ContainerSampler:
    """Samples a container's memory/CPU in a background thread while a config runs.
    `docker stats --no-stream` itself takes ~1-2 s, which bounds the sampling rate --
    fine for multi-second sweeps, but peak memory of very short bursts may be missed."""

    def __init__(self, service: str):
        self.service = service
        self._cid = container_id(service)
        self._samples: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if self._cid:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            s = container_stats(self._cid)
            if s:
                self._samples.append(s)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self._samples:
            return {"db_mem_peak_mb": None, "db_mem_mean_mb": None, "db_cpu_mean_pct": None, "db_samples": 0}
        mem = [s["mem_bytes"] for s in self._samples]
        cpu = [s["cpu_pct"] for s in self._samples]
        return {
            "db_mem_peak_mb": max(mem) / 1e6,
            "db_mem_mean_mb": statistics.fmean(mem) / 1e6,
            "db_cpu_mean_pct": statistics.fmean(cpu),
            "db_cpu_peak_pct": max(cpu),
            "db_samples": len(self._samples),
        }


def latency_summary(latencies_s: List[float], prefix: str) -> Dict[str, Optional[float]]:
    if not latencies_s:
        return {f"{prefix}_n": 0, f"{prefix}_mean_ms": None, f"{prefix}_p50_ms": None, f"{prefix}_p95_ms": None}
    ms = sorted(x * 1000 for x in latencies_s)
    p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    return {
        f"{prefix}_n": len(ms),
        f"{prefix}_mean_ms": statistics.fmean(ms),
        f"{prefix}_p50_ms": statistics.median(ms),
        f"{prefix}_p95_ms": p95,
    }


class Stopwatch:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


class HostSampler:
    """Samples this Python process (plus children) for CPU and RSS in a background
    thread -- the client side of a load: embedding, Parquet decoding, request building.
    cpu_pct is psutil's per-process figure, so 100 = one fully busy core."""

    def __init__(self, interval_s: float = 1.0):
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process()
        self.interval_s = interval_s
        self._rss: List[float] = []
        self._cpu: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _procs(self):
        try:
            return [self._proc] + self._proc.children(recursive=True)
        except self._psutil.Error:
            return [self._proc]

    def _run(self):
        for p in self._procs():
            try:
                p.cpu_percent(None)  # prime
            except self._psutil.Error:
                pass
        while not self._stop.wait(self.interval_s):
            rss = cpu = 0.0
            for p in self._procs():
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except self._psutil.Error:
                    pass
            self._rss.append(rss)
            self._cpu.append(cpu)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self._rss:
            return {"host_rss_peak_mb": None, "host_rss_mean_mb": None, "host_cpu_mean_pct": None, "host_samples": 0}
        return {
            "host_rss_peak_mb": max(self._rss) / 1e6,
            "host_rss_mean_mb": statistics.fmean(self._rss) / 1e6,
            "host_cpu_mean_pct": statistics.fmean(self._cpu),
            "host_cpu_peak_pct": max(self._cpu),
            "host_samples": len(self._rss),
        }
