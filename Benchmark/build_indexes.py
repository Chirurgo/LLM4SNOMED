"""Full-scale index build with time + resource monitoring, one step at a time.

Steps (default order): corpus -> neo4j -> postgres -> qdrant -> weaviate -> meilisearch

  corpus  : RF2 -> denormalized descriptions -> SapBERT dense + Qdrant/bm25 sparse vectors
            -> DATA/corpus_cache/<...>.parquet. The only embedding pass; host-sampled.
  <db>    : wipe + (re)create schema + bulk load from that Parquet + build indexes.

For every DB step only that DB's container is running (the others are stopped), so the
container CPU/memory samples and the host numbers are attributable to it alone. This also
keeps the run inside Docker Desktop's 8 GB VM. Per step it records wall time per phase,
DB-container CPU/memory (sampled during the step), this process's CPU/RSS, the DB's
on-disk size before and after, idle memory once loaded, and row counts.

Results: <out>/indexing/<step>.json (+ indexing.md summary). A step whose JSON exists is
skipped unless --force, so the run is resumable after a crash.

    python -m Benchmark.build_indexes [--steps corpus neo4j ...] [--out Benchmark/results/full]
"""
import argparse
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from dotenv import load_dotenv

from Benchmark.resource_profile import (ContainerSampler, HostSampler, container_disk_bytes, container_id,
                                        container_stats)

DB_SERVICES = ("neo4j", "postgres", "qdrant", "weaviate", "meilisearch")
STEPS = ("corpus",) + DB_SERVICES
DEFAULT_PARQUET = "DATA/corpus_cache/snomed_20250301_sapbert_bm25.parquet"


def _sh(*args, check=True, timeout_s: float = 180, retries: int = 4) -> str:
    """Run a docker CLI command with a timeout, retrying transient engine failures
    (Docker Desktop returns HTTP 500 / hangs while its VM is pausing or resuming)."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            out = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout_s}s"
        else:
            if out.returncode == 0 or not check:
                return out.stdout.strip()
            last = out.stderr.strip()
        print(f"  [docker] {' '.join(args)} failed (attempt {attempt}/{retries}): {last}", flush=True)
        if attempt < retries:
            time.sleep(15 * attempt)
            _wait_docker(timeout_s=300)
    raise RuntimeError(f"{' '.join(args)} failed after {retries} attempts: {last}")


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True,
                              text=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _wait_docker(timeout_s: float = 300) -> None:
    start = time.perf_counter()
    while not _docker_ok():
        if time.perf_counter() - start > timeout_s:
            raise RuntimeError(f"Docker engine not responding after {timeout_s}s -- restart Docker Desktop")
        time.sleep(5)


def _resource_saver_enabled() -> Optional[bool]:
    """Docker Desktop's Resource Saver pauses the VM once no container is running. The corpus
    step stops every container for ~2 h, and on 2026-11-13 the VM got stuck 'pausing',
    killing the next step. None = setting not found (not Docker Desktop / unknown layout)."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    for name in ("settings-store.json", "settings.json"):
        p = Path(appdata) / "Docker" / name
        if p.exists():
            try:
                return bool(json.loads(p.read_text(encoding="utf-8")).get("UseResourceSaver", False))
            except (OSError, ValueError):
                return None
    return None


def _drop_vm_caches() -> None:
    """Best effort: make Docker Desktop's WSL VM drop the page cache the previous DB's
    load left behind, so vmmemWSL hands the memory back to Windows (on a 16 GB host the
    leftover cache starved the loader process). Only discards clean cache; no data loss."""
    if platform.system() != "Windows":
        return
    try:
        subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _isolate(service: str) -> None:
    others = [s for s in DB_SERVICES if s != service]
    _sh("docker", "compose", "stop", *others)
    _drop_vm_caches()
    _sh("docker", "compose", "up", "-d", service, timeout_s=600)


def _wait_ready(service: str, timeout_s: float = 300) -> None:
    from Benchmark.wait_for_services import CHECKS, _tcp_ok
    check = CHECKS.get(service) or (lambda: _neo4j_ok())
    start = time.perf_counter()
    while not check():
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(f"{service} not ready after {timeout_s}s")
        time.sleep(2)
    if service == "postgres":  # port opens before initdb finishes on first start
        time.sleep(5)


def _neo4j_ok() -> bool:
    from neo4j import GraphDatabase
    try:
        with GraphDatabase.driver(os.environ["NEO4J_HOST"],
                                  auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])) as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


def _image(service: str) -> Optional[str]:
    cid = container_id(service)
    return _sh("docker", "inspect", "--format", "{{.Config.Image}}", cid, check=False) if cid else None


class Phases:
    """Times named sub-phases of a step."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    def run(self, name: str, fn: Callable, *a, **kw):
        print(f"[{name}] start", flush=True)
        t = time.perf_counter()
        res = fn(*a, **kw)
        self.timings[f"{name}_s"] = round(time.perf_counter() - t, 2)
        print(f"[{name}] done in {self.timings[f'{name}_s']:.1f}s", flush=True)
        return res


# --- steps -----------------------------------------------------------------------------

def step_corpus(args, ph: Phases) -> dict:
    from fastembed import SparseTextEmbedding
    from Database.build_corpus import build_and_cache_corpus
    from Database.embed_models import AsyncSapBERTEmbedder
    from Src.stores.qdrant_store import SPARSE_MODEL_ID
    import pyarrow.parquet as pq
    import torch

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dense = AsyncSapBERTEmbedder(device=device, batch_size=args.embed_batch or (512 if device == "cuda" else 64))
    sparse = SparseTextEmbedding(SPARSE_MODEL_ID)
    ph.run("embed", build_and_cache_corpus, os.environ["SNOMED_RELEASE_PATH"], args.parquet,
           dense, sparse_model=sparse, batch_size=10_000, force=args.force_corpus)
    meta = pq.ParquetFile(args.parquet).metadata
    return {"rows": meta.num_rows, "parquet_mb": Path(args.parquet).stat().st_size / 1e6,
            "embedding_model": getattr(dense, "model_id", type(dense).__name__),
            "sparse_model": SPARSE_MODEL_ID, "device": device, "torch_threads": torch.get_num_threads()}


def step_neo4j(args, ph: Phases) -> dict:
    from Database.loaders import neo4j_loader as L
    url, user, pw = os.environ["NEO4J_HOST"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
    ph.run("wipe", L.wipe, url, user, pw)
    ph.run("schema", L.create_schema, url, user, pw)
    sub = ph.run("concepts_and_relationships", L.load_concepts_and_relationships,
                 url, user, pw, os.environ["SNOMED_RELEASE_PATH"])
    ph.timings.update({k: round(v, 2) for k, v in sub.items()})
    ph.run("insert_descriptions", L.load_descriptions, url, user, pw, args.parquet)
    ph.run("vector_index", L.build_vector_index, url, user, pw)
    c = L.counts(url, user, pw)
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(args.parquet).metadata.num_rows
    # Descriptions collapse slightly (md5(text) MERGE), HAS_NAME should not.
    if not (c["concepts"] and c["is_a"] and c["descriptions"] >= 0.99 * rows and c["has_name"] >= 0.99 * rows):
        raise RuntimeError(f"neo4j load looks incomplete: {c} (parquet rows {rows})")
    return {"counts": c,
            "memory_config": {"heap": "2G", "pagecache": "3G"},
            "note": "embeddings reused from corpus parquet (not recomputed); vector index built after insert"}


def step_postgres(args, ph: Phases) -> dict:
    from Benchmark.backends import POSTGRES_DSN
    from Database.loaders import postgres_loader as L
    from Src.stores.postgres_store import connect
    conn = connect(POSTGRES_DSN)
    try:
        ph.run("schema", L.create_schema, conn, drop_existing=True)
        ph.run("insert", L.load_corpus, conn, args.parquet, build_indexes=False, batch_size=5000)
        ph.run("indexes", L.create_indexes, conn)
        n = conn.execute(f"SELECT count(*) FROM {L.TABLE_NAME}").fetchone()[0]
        sizes = conn.execute(
            "SELECT indexrelname, pg_relation_size(indexrelid) FROM pg_stat_user_indexes "
            "WHERE relname = %s", (L.TABLE_NAME,)).fetchall()
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    finally:
        conn.close()
    return {"counts": {"rows": n}, "index_sizes_mb": {k: v / 1e6 for k, v in sizes},
            "pgvector": version, "index_build_settings": L.INDEX_BUILD_SETTINGS}


def step_qdrant(args, ph: Phases) -> dict:
    from Benchmark.backends import QDRANT_URL, _wait_qdrant_indexed
    from Database.loaders import qdrant_loader as L
    client = L.connect(QDRANT_URL)
    try:
        ph.run("schema", L.create_collection, client, drop_existing=True)
        ph.run("insert", L.load_corpus, client, args.parquet)
        ph.run("wait_indexed", _wait_qdrant_indexed, client, L.COLLECTION_NAME, 4 * 3600)
        info = client.get_collection(L.COLLECTION_NAME)
    finally:
        client.close()
    return {"counts": {"points": info.points_count, "indexed_vectors": info.indexed_vectors_count,
                       "segments": info.segments_count}}


def step_weaviate(args, ph: Phases) -> dict:
    from Benchmark.backends import WEAVIATE_GRPC_PORT, WEAVIATE_HOST, WEAVIATE_HTTP_PORT
    from Database.loaders import weaviate_loader as L
    client = L.connect(WEAVIATE_HOST, WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_PORT)
    try:
        ph.run("schema", L.create_collection, client, drop_existing=True)
        ph.run("insert", L.load_corpus, client, args.parquet, batch_size=1000)
        n = client.collections.get(L.COLLECTION_NAME).aggregate.over_all(total_count=True).total_count
    finally:
        client.close()
    return {"counts": {"objects": n}, "note": "HNSW is built synchronously during insert"}


def step_meilisearch(args, ph: Phases) -> dict:
    from Benchmark.backends import MEILI_API_KEY, MEILI_URL
    from Database.loaders import meilisearch_loader as L
    client = L.connect(MEILI_URL, MEILI_API_KEY)
    resume = getattr(args, "resume_meili", False)
    if not resume:
        ph.run("schema", L.create_index, client, drop_existing=True)
    ph.run("insert_and_index", L.load_corpus, client, args.parquet, batch_size=5000, max_in_flight=20,
           resume=resume)
    stats = client.index(L.INDEX_NAME).get_stats()
    return {"counts": {"documents": stats.number_of_documents},
            "note": "insert_and_index includes Meilisearch's async indexing (all tasks awaited)"}


STEP_FNS = {"corpus": step_corpus, "neo4j": step_neo4j, "postgres": step_postgres,
            "qdrant": step_qdrant, "weaviate": step_weaviate, "meilisearch": step_meilisearch}


def run_step(step: str, args) -> dict:
    ph = Phases()
    rec = {"step": step, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": {
        "platform": platform.platform(), "cpu_count": os.cpu_count()}}
    if getattr(args, "no_monitor", False):
        # Plain build: no container/host sampling, disk or idle readings.
        if step == "corpus":
            _sh("docker", "compose", "stop", *DB_SERVICES, check=False)
        else:
            _wait_docker()
            _isolate(step)
            _wait_ready(step)
        stop = threading.Event()
        if args.drop_caches_every and step != "corpus":
            threading.Thread(target=lambda: [_drop_vm_caches() for _ in iter(
                lambda: stop.wait(args.drop_caches_every), True)], daemon=True).start()
        t = time.perf_counter()
        try:
            rec.update(STEP_FNS[step](args, ph))
        finally:
            stop.set()
        rec["total_s"] = round(time.perf_counter() - t, 2)
        rec["monitored"] = False
    elif step == "corpus":
        _sh("docker", "compose", "stop", *DB_SERVICES, check=False)  # nothing competes for CPU
        with HostSampler() as host:
            t = time.perf_counter()
            rec.update(STEP_FNS[step](args, ph))
            rec["total_s"] = round(time.perf_counter() - t, 2)
        rec.update(host.summary())
    else:
        _wait_docker()
        _isolate(step)
        _wait_ready(step)
        rec["image"] = _image(step)
        disk_before = container_disk_bytes(step)
        with ContainerSampler(step) as db, HostSampler() as host:
            t = time.perf_counter()
            rec.update(STEP_FNS[step](args, ph))
            rec["total_s"] = round(time.perf_counter() - t, 2)
        rec.update(db.summary())
        rec.update(host.summary())
        time.sleep(15)  # let background work settle before the "idle" reading
        idle = container_stats(container_id(step))
        rec["db_mem_idle_after_load_mb"] = idle["mem_bytes"] / 1e6 if idle else None
        disk_after = container_disk_bytes(step)
        rec["db_disk_before_mb"] = disk_before / 1e6 if disk_before is not None else None
        rec["db_disk_after_mb"] = disk_after / 1e6 if disk_after is not None else None
    rec["phases"] = ph.timings
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def write_summary(out: Path) -> None:
    rows = []
    for step in STEPS:
        p = out / "indexing" / f"{step}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    f = lambda v, d=1: "-" if v is None else f"{v:,.{d}f}"
    lines = ["| step | total (min) | phases (s) | DB CPU mean / peak % | DB mem peak / idle MB | "
             "host CPU mean % | host RSS peak MB | disk after MB | counts |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        phases = ", ".join(f"{k[:-2]} {v:,.0f}" for k, v in r.get("phases", {}).items())
        lines.append(
            f"| {r['step']} | {f(r['total_s'] / 60)} | {phases} | "
            f"{f(r.get('db_cpu_mean_pct'))} / {f(r.get('db_cpu_peak_pct'))} | "
            f"{f(r.get('db_mem_peak_mb'), 0)} / {f(r.get('db_mem_idle_after_load_mb'), 0)} | "
            f"{f(r.get('host_cpu_mean_pct'))} | {f(r.get('host_rss_peak_mb'), 0)} | "
            f"{f(r.get('db_disk_after_mb'), 0)} | {r.get('counts', r.get('rows', ''))} |")
    (out / "indexing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--out", default="Benchmark/results/full")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--force", action="store_true", help="re-run steps that already have results")
    ap.add_argument("--allow-resource-saver", action="store_true",
                    help="run even though Docker Desktop's Resource Saver is enabled")
    ap.add_argument("--no-monitor", action="store_true",
                    help="skip resource sampling (just build the indexes)")
    ap.add_argument("--resume-meili", action="store_true",
                    help="continue an interrupted Meilisearch load instead of recreating the index")
    ap.add_argument("--drop-caches-every", type=float, default=0, metavar="SECONDS",
                    help="also drop the Docker VM page cache periodically during each DB step")
    ap.add_argument("--force-corpus", action="store_true", help="re-embed even if the corpus cache is valid")
    ap.add_argument("--device", default="auto", help="SapBERT device for the corpus step (auto = cuda if available)")
    ap.add_argument("--embed-batch", type=int, default=0, help="SapBERT batch size (default 512 on GPU, 64 on CPU)")
    args = ap.parse_args(argv)
    load_dotenv(".env", override=False)
    out = Path(args.out)
    (out / "indexing").mkdir(parents=True, exist_ok=True)

    if _resource_saver_enabled() and not args.allow_resource_saver:
        raise SystemExit("Docker Desktop Resource Saver is ON. Turn it off (Settings > Resources > "
                         "Advanced > Resource Saver) or pass --allow-resource-saver.")
    _wait_docker(timeout_s=60)

    for step in args.steps:
        res_path = out / "indexing" / f"{step}.json"
        if res_path.exists() and not args.force:
            print(f"=== {step}: already done ({res_path}), skipping", flush=True)
            continue
        if step != "corpus" and not Path(args.parquet).exists():
            raise FileNotFoundError(f"{args.parquet} missing -- run the corpus step first")
        print(f"=== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
        rec = run_step(step, args)
        res_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        write_summary(out)
        print(f"=== {step}: done in {rec['total_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
