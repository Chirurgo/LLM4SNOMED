"""End-to-end experiment runner: corpus -> indexes -> retrieval sweep -> ranking study ->
Laya -> LLM selection -> SNO-PI (small + whole-SNOMED) -> REPORT.md.

Every stage calls the existing module as a subprocess (same CLIs as before), logs to
<results>/logs/<stage>.log, and records its status in <results>/pipeline_state.json.
A finished stage is skipped on rerun (and the modules skip finished sub-steps), so after a
crash just run the same command again. See EXPERIMENT.md for the full walkthrough.

  python -m Benchmark.run_all                       # everything (Laya / LLM only if configured)
  python -m Benchmark.run_all --stages snopi report  # selected stages
  python -m Benchmark.run_all --from rerank         # this stage and all after it
  python -m Benchmark.run_all --dry-run             # print the commands only

Stages: check corpus index queries sweep latency sweep_report freeze rerank laya llm snopi report
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGES = ["check", "corpus", "index", "queries", "sweep", "latency", "sweep_report", "freeze",
          "rerank", "laya", "llm", "snopi", "report"]
ALL_BACKENDS = ["neo4j", "postgres", "qdrant", "weaviate", "meilisearch"]
FREEZE_BACKENDS = ["neo4j", "qdrant", "meilisearch"]   # ranking-study backends (2026-11-19 cut)
CROSS_ENCODERS = ["bge-large", "bge-v2-m3", "jina-v2-ml", "gte-ml", "qwen3-0.6b"]
GOLD = "Minimal_Dataset.json"
TERMS = "Benchmark/data/canonical_terms_minimal.tsv"
LLM_BASE = "qdrant_hybrid_k50_x2_w0.5_d50_rrf60__k20__none"

# SNO-PI profiles: (prep args, train args, eval args). `small` replicates the CPU v2 run
# (4,000 queries, 1 epoch) on the GPU; `full` trains on the whole SNOMED CT clinical corpus.
SNOPI_PROFILES = {
    "small": (["--profile", "small", "--n-train", "4000", "--n-dev", "400", "--query-mode", "mixed",
               "--children", "--data", "_small"],
              ["--data", "_small", "--epochs", "1", "--bs", "8", "--out", "DATA/snopi/model-gpu-small"],
              ["--data", "_small", "--model", "DATA/snopi/model-gpu-small", "--name", "snopi-small",
               "--k", "10", "20"]),
    "full": (["--profile", "full", "--n-dev", "2000", "--queries-per-concept", "2", "--query-mode", "mixed",
              "--children", "--negation", "--product", "--data", "_full"],
             ["--data", "_full", "--epochs", "3", "--bs", "32", "--lr", "2e-5", "--none", "--p-none", "0.15",
              "--log-every", "200", "--resume", "--out", "DATA/snopi/model-gpu-full"],
             ["--data", "_full", "--model", "DATA/snopi/model-gpu-full", "--name", "snopi-full",
              "--k", "10", "20", "50"]),
}


class Runner:
    def __init__(self, args):
        self.a = args
        self.root = Path(args.results)
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "pipeline_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() \
            else {"stages": {}}
        self.env = {**os.environ, "BENCH_RESULTS": str(self.root), "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1"}
        self.py = sys.executable
        self.log = None
        if args.device == "auto":   # resolved once: harness/freeze/laya need a concrete device
            try:
                import torch
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                args.device = "cpu"

    # -- plumbing -------------------------------------------------------------------------
    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def sh(self, *cmd, python=None, check=True):
        """Run `python -m ...` (or any command) with output to console + stage log."""
        cmd = [str(c) for c in cmd]
        if cmd[0] == "-m":
            cmd = [python or self.py] + cmd
        line = " ".join(cmd)
        print(f"$ {line}", flush=True)
        if self.log:
            self.log.write(f"$ {line}\n"); self.log.flush()
        if self.a.dry_run:
            return 0
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env,
                             stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
        for out in p.stdout:
            sys.stdout.write(out)
            if self.log:
                self.log.write(out)
        rc = p.wait()
        if self.log:
            self.log.flush()
        if rc and check:
            raise RuntimeError(f"command failed (exit {rc}): {line}")
        return rc

    def db_up(self, service):
        """Only `service` running (fits Docker Desktop's 8 GB VM), or all with --no-isolate."""
        if self.a.dry_run:
            print(f"$ [docker] isolate {service}")
            return
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
        from Benchmark.build_indexes import _isolate, _wait_docker, _wait_ready, _sh
        _wait_docker()
        if self.a.no_isolate:
            _sh("docker", "compose", "up", "-d", service, timeout_s=600)
        else:
            _isolate(service)
        _wait_ready(service, timeout_s=900)

    def db_stop_all(self):
        if not self.a.dry_run:
            from Benchmark.build_indexes import DB_SERVICES, _sh
            _sh("docker", "compose", "stop", *DB_SERVICES, check=False, timeout_s=600)

    def run(self, stages):
        for st in stages:
            rec = self.state["stages"].get(st, {})
            if rec.get("status") == "done" and st not in self.a.force:
                print(f"=== {st}: done earlier, skipping (--force {st} to redo)", flush=True)
                continue
            print(f"\n=== {st}: start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            t0 = time.perf_counter()
            self.log = open(self.logs / f"{st}.log", "a", encoding="utf-8")
            self.log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {st}\n")
            try:
                status = getattr(self, f"st_{st}")() or "done"
            except Exception as e:
                self.state["stages"][st] = {"status": "failed", "error": str(e)[:500],
                                            "seconds": time.perf_counter() - t0,
                                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                self.save()
                print(f"=== {st}: FAILED: {e}\nlog: {self.logs / (st + '.log')}\n"
                      f"Fix the cause and rerun the same command; finished work is kept.", flush=True)
                raise SystemExit(1)
            finally:
                self.log.close()
                self.log = None
            if not self.a.dry_run:
                self.state["stages"][st] = {"status": status, "seconds": time.perf_counter() - t0,
                                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                self.save()
            print(f"=== {st}: {status} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    # -- stages ---------------------------------------------------------------------------
    def st_check(self):
        from dotenv import dotenv_values
        info = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
                "os": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
        problems = []
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
                info["bf16"] = torch.cuda.is_bf16_supported()
            elif self.a.device == "cuda" or not self.a.allow_cpu:
                problems.append("no CUDA GPU visible to torch (install the CUDA build of torch, see EXPERIMENT.md)")
        except ImportError:
            problems.append("torch not installed")
        try:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        except ImportError:
            pass
        try:
            info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                text=True).stdout.strip()
            info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                                    text=True).stdout.strip())
        except OSError:
            pass
        env = dotenv_values(".env")
        for k in ("NEO4J_HOST", "NEO4J_USER", "NEO4J_PASSWORD", "SNOMED_RELEASE_PATH"):
            if not env.get(k) and not os.environ.get(k):
                problems.append(f".env is missing {k}")
        rf2 = env.get("SNOMED_RELEASE_PATH") or os.environ.get("SNOMED_RELEASE_PATH")
        if rf2 and not Path(rf2).exists():
            problems.append(f"SNOMED_RELEASE_PATH does not exist: {rf2}")
        for f in (GOLD, TERMS):
            if not Path(f).exists():
                problems.append(f"missing {f}")
        if shutil.which("docker") is None:
            problems.append("docker CLI not found")
        free = shutil.disk_usage(".").free / 2**30
        info["disk_free_gb"] = round(free, 1)
        if free < 60:
            problems.append(f"only {free:.0f} GB free disk (need ~60 GB: corpus 6, DB volumes ~25, models ~10)")
        if self.a.laya_python and not Path(self.a.laya_python).exists():
            problems.append(f"--laya-python not found: {self.a.laya_python}")
        info["problems"] = problems
        info["options"] = {k: v for k, v in vars(self.a).items() if k != "dry_run"}
        (self.root / "env.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        for k, v in info.items():
            if k not in ("problems", "options"):
                print(f"  {k}: {v}")
        if problems:
            print("PROBLEMS:\n  " + "\n  ".join(problems))
            if not self.a.ignore_check:
                raise RuntimeError("environment check failed (fix, or pass --ignore-check)")

    def _build_indexes(self, steps):
        cmd = ["-m", "Benchmark.build_indexes", "--steps", *steps, "--out", self.root / "full",
               "--device", self.a.device]
        if not self.a.monitor:
            cmd.append("--no-monitor")
        if self.a.allow_resource_saver:
            cmd.append("--allow-resource-saver")
        self.sh(*cmd)

    def st_corpus(self):
        self._build_indexes(["corpus"])

    def st_index(self):
        self._build_indexes(self.a.backends)

    def st_queries(self):
        self.sh("-m", "Benchmark.gold_queries", "--gold", GOLD, "--terms", TERMS,
                "--out", self.root / "gold" / "extractions_gold.json")

    def _gold_args(self):
        return ["--gold", GOLD, "--extractions", self.root / "gold" / "extractions_gold.json",
                "--out", self.root / "gold", "--device", self.a.device]

    def st_sweep(self):
        for b in self.a.backends:
            self.db_up(b)
            self.sh("-m", "Benchmark.harness", *self._gold_args(), "--backends", b)

    def st_latency(self):
        for b in self.a.backends:
            self.db_up(b)
            self.sh("-m", "Benchmark.latency_warm", *self._gold_args(), "--backends", b)

    def st_sweep_report(self):
        cmd = ["-m", "Benchmark.report", "--out", self.root / "gold", "--gold", GOLD,
               "--extractions", self.root / "gold" / "extractions_gold.json"]
        if "neo4j" in self.a.backends:
            self.db_up("neo4j")
            cmd.append("--neo4j")
        self.sh(*cmd)

    def st_freeze(self):
        bs = [b for b in FREEZE_BACKENDS if b in self.a.backends]
        for b in bs:
            self.db_up(b)
            self.sh("-m", "Benchmark.freeze_candidates", "--backends", b,
                    "--extractions", self.root / "gold" / "extractions_gold.json",
                    "--out", self.root / "rank" / "frozen", "--device", self.a.device)
        self.db_stop_all()   # the rest is offline; give the RAM back

    def st_rerank(self):
        self.sh("-m", "Benchmark.rank_study", "rerank", "--rankers", "none", "--k", "10", "20", "50")
        for m in self.a.rerankers:   # one model per process: frees GPU memory between models
            self.sh("-m", "Benchmark.rank_study", "rerank", "--rankers", m, "--queries", "en", "ctx",
                    "--k", "20", "50", "--device", self.a.device)

    def st_laya(self):
        if not self.a.laya_python:
            print("  skipped: pass --laya-python <python of the laya env> (see EXPERIMENT.md)")
            return "skipped"
        for m in self.a.laya_models:
            self.sh("-m", "Benchmark.laya_study", "run", "--model", m, "--device", self.a.device,
                    python=self.a.laya_python)
            self.sh("-m", "Benchmark.laya_study", "validate", "--model", m, "--device", self.a.device,
                    python=self.a.laya_python)
        self.sh("-m", "Benchmark.rank_study", "score")

    def st_llm(self):
        if not self.a.llm:
            print("  skipped: pass --llm openai:<model> | gemini:<model> | lmstudio:<model>")
            return "skipped"
        # GOLD is the public Minimal_Dataset. Never point a hosted LLM at the restricted dataset.
        for spec in self.a.llm:
            self.sh("-m", "Benchmark.rank_study", "llm-select", "--base", self.a.llm_base, "--llm", spec)

    def st_snopi(self):
        for prof in self.a.snopi_profiles:
            prep, train, ev = SNOPI_PROFILES[prof]
            data = prep[prep.index("--data") + 1]
            if not Path(f"DATA/snopi/train{data}.jsonl").exists() or prof in self.a.force_snopi_prep:
                self.sh("-m", "Benchmark.snopi_study", "prep", *prep, "--device", self.a.device)
            model = train[train.index("--out") + 1]
            meta = Path(model) / "train_meta.json"
            done = meta.exists() and json.loads(meta.read_text()).get("epochs") == int(train[train.index("--epochs") + 1])
            if not done:
                self.sh("-m", "Benchmark.snopi_study", "train", *train, "--device", self.a.device)
            self.sh("-m", "Benchmark.snopi_study", "eval", *ev, "--device", self.a.device)

    def st_report(self):
        self.sh("-m", "Benchmark.final_report", "--root", self.root)
        print(f"\nREPORT: {self.root / 'REPORT.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="Benchmark/results_gpu", help="results root (kept apart from the CPU run)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, help="run only these (in pipeline order)")
    ap.add_argument("--from", dest="from_", choices=STAGES, help="run this stage and every later one")
    ap.add_argument("--force", nargs="+", default=[], choices=STAGES, help="redo stages marked done")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--backends", nargs="+", default=ALL_BACKENDS, choices=ALL_BACKENDS)
    ap.add_argument("--rerankers", nargs="+", default=CROSS_ENCODERS)
    ap.add_argument("--laya-python", help="python.exe / bin/python of the separate laya env")
    ap.add_argument("--laya-models", nargs="+", default=["multilingual", "vega"], choices=["multilingual", "vega"])
    ap.add_argument("--llm", nargs="+", help="LLM selection on public data, e.g. openai:gpt-4.1 gemini:gemini-2.5-flash")
    ap.add_argument("--llm-base", default=LLM_BASE, help="run whose top-5 the LLM chooses from")
    ap.add_argument("--snopi-profiles", nargs="+", default=["small", "full"], choices=sorted(SNOPI_PROFILES))
    ap.add_argument("--force-snopi-prep", nargs="*", default=[], choices=sorted(SNOPI_PROFILES))
    ap.add_argument("--monitor", action="store_true", help="sample DB/host resources while indexing")
    ap.add_argument("--no-isolate", action="store_true", help="don't stop other DB containers (>= 32 GB RAM)")
    ap.add_argument("--allow-resource-saver", action="store_true")
    ap.add_argument("--ignore-check", action="store_true", help="continue despite environment problems")
    ap.add_argument("--allow-cpu", action="store_true", help="don't treat a missing GPU as a problem")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not Path("Benchmark").is_dir():
        raise SystemExit("run from the repository root")
    stages = STAGES
    if args.from_:
        stages = STAGES[STAGES.index(args.from_):]
    if args.stages:
        stages = [s for s in STAGES if s in args.stages]
    if "check" not in stages and not args.dry_run and not (Path(args.results) / "env.json").exists():
        stages = ["check"] + stages
    Runner(args).run(stages)


if __name__ == "__main__":
    main()
