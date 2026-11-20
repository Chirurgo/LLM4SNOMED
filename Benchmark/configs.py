"""Benchmark configurations and the sweep grid.

Sweeps are one-factor-at-a-time (OFAT) around a baseline rather than a full Cartesian
product: each axis is varied alone, holding everything else at the baseline. A full
grid over these axes would be thousands of runs per backend; OFAT keeps it to a few
dozen while still showing each knob's quality/speed/resource curve. Use
`--full-grid` in the harness for a Cartesian product over a hand-picked subset.

Edit SWEEP (and BASELINE) below to change what is explored.
"""
import itertools
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

from Src.stores.base import SearchParams

# Rerankers already in the local HF cache, ordered fastest -> slowest on CPU.
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",   # ~22M params, general-domain
    "medcpt": "ncbi/MedCPT-Cross-Encoder",              # ~110M, biomedical
    "bge-large": "BAAI/bge-reranker-large",             # ~560M, paper baseline
}


@dataclass(frozen=True)
class RunConfig:
    backend: str
    mode: str = "dense"                      # "dense" | "hybrid"
    k_candidate: int = 20                    # paper baseline (02_main.ipynb)
    expand_factor: float = 2.0               # domain-filter over-fetch (Neo4j baseline value)
    min_hits: int = 3
    params: SearchParams = field(default_factory=SearchParams)
    reranker: Optional[str] = "bge-large"    # key into RERANKERS, or None = retrieval order
    selection: str = "top1"                  # "top1" | "llm"
    axis: str = "baseline"                   # which OFAT axis produced this config

    @property
    def relevant_params(self) -> frozenset:
        return relevant_params(self.backend, self.mode)

    @property
    def config_id(self) -> str:
        p, rel = self.params, self.relevant_params
        parts = [self.backend, self.mode, f"k{self.k_candidate}", f"x{self.expand_factor:g}"]
        for name, tag in (("hybrid_weight", "w"), ("arm_depth_min", "d"), ("rrf_k", "rrf")):
            if name in rel:
                parts.append(f"{tag}{getattr(p, name):g}")
        if "fusion" in rel and p.fusion:
            parts.append(p.fusion)
        if "ef_search" in rel and p.ef_search is not None:
            parts.append(f"ef{p.ef_search}")
        parts += [f"rr-{self.reranker or 'none'}", self.selection]
        return "_".join(parts)

    def as_dict(self) -> dict:
        """Params the backend/mode does not use are reported as None, not as their
        (ignored) default, so result tables never suggest a knob was in effect."""
        d = asdict(self)
        d.pop("params")
        rel = self.relevant_params
        d.update({f"p_{k}": (v if k in rel else None) for k, v in self.params.as_dict().items()})
        d["config_id"] = self.config_id
        return d


HYBRID_ONLY_PARAMS = frozenset({"hybrid_weight", "arm_depth_min", "rrf_k", "fusion"})


def relevant_params(backend: str, mode: str) -> frozenset:
    supported = _supported_params(backend)
    return supported if mode == "hybrid" else supported - HYBRID_ONLY_PARAMS


# --- the sweep -----------------------------------------------------------------------
# axis name -> (applies to which mode(s), list of values, how to apply a value)
SWEEP: Dict[str, dict] = {
    # Retrieval depth: candidate recall ceiling vs. cross-encoder cost (linear in k).
    "k_candidate":   {"modes": ("dense", "hybrid"), "values": [5, 10, 20, 50, 100]},
    # Domain-filter over-fetch: more survivors after filtering vs. bigger fetch.
    "expand_factor": {"modes": ("dense",), "values": [1.0, 2.0, 4.0]},
    # HNSW beam width: the classic ANN speed/recall knob (no effect on tiny corpora,
    # which are brute-forced by every backend).
    "ef_search":     {"modes": ("dense",), "values": [16, 64, 128, 256, 512], "param": True},
    # Dense share of hybrid fusion: 0 = keyword only, 1 = dense only.
    "hybrid_weight": {"modes": ("hybrid",), "values": [0.0, 0.25, 0.5, 0.75], "param": True},
    # Per-arm candidate depth before fusion (Postgres, Qdrant).
    "arm_depth_min": {"modes": ("hybrid",), "values": [20, 50, 100], "param": True},
    # RRF constant: small k = top ranks dominate, large k = flatter (Postgres, Qdrant).
    "rrf_k":         {"modes": ("hybrid",), "values": [10, 60, 120], "param": True},
    # Fusion algorithm (backend-specific values; filtered per backend below).
    "fusion":        {"modes": ("hybrid",), "values": ["rrf", "dbsf", "ranked", "relative_score"], "param": True},
    # Reranker size/domain: the dominant per-entity CPU cost.
    "reranker":      {"modes": ("dense", "hybrid"), "values": [None, "minilm", "medcpt", "bge-large"]},
}

FUSIONS_BY_BACKEND = {"qdrant": {"rrf", "dbsf"}, "weaviate": {"ranked", "relative_score"}}
# What fusion=None already means in each store -- sweeping it again would duplicate the baseline.
DEFAULT_FUSION = {"qdrant": "rrf", "weaviate": "ranked"}


def _supported_params(backend: str) -> frozenset:
    from Benchmark.backends import STORE_CLASSES
    return STORE_CLASSES[backend].supported_params


def _supports_hybrid(backend: str) -> bool:
    return backend != "neo4j"


def build_sweep(backends: List[str], axes: Optional[List[str]] = None, llm: bool = False) -> List[RunConfig]:
    """OFAT sweep per backend around a dense and (where supported) a hybrid baseline.
    Axes a backend cannot honour are skipped rather than run as silent no-ops."""
    axes = axes or list(SWEEP)
    configs: Dict[str, RunConfig] = {}
    for backend in backends:
        supported = _supported_params(backend)
        modes = ["dense"] + (["hybrid"] if _supports_hybrid(backend) else [])
        for mode in modes:
            base = RunConfig(backend=backend, mode=mode)
            configs.setdefault(base.config_id, base)
            for axis in axes:
                spec = SWEEP[axis]
                if mode not in spec["modes"]:
                    continue
                if spec.get("param") and axis not in supported:
                    continue
                for value in spec["values"]:
                    if axis == "fusion" and (value not in FUSIONS_BY_BACKEND.get(backend, set())
                                             or value == DEFAULT_FUSION.get(backend)):
                        continue
                    if spec.get("param"):
                        cfg = replace(base, params=replace(base.params, **{axis: value}), axis=axis)
                    else:
                        cfg = replace(base, **{axis: value}, axis=axis)
                    configs.setdefault(cfg.config_id, cfg)
            if llm:
                cfg = replace(base, selection="llm", axis="selection")
                configs.setdefault(cfg.config_id, cfg)
    return list(configs.values())


def build_full_grid(backends: List[str], grid: Dict[str, list], llm: bool = False) -> List[RunConfig]:
    """Cartesian product over `grid` (axis -> values), e.g.
    {"mode": ["dense", "hybrid"], "k_candidate": [10, 50], "reranker": ["minilm", "bge-large"]}."""
    configs: Dict[str, RunConfig] = {}
    keys = list(grid)
    for backend in backends:
        supported = _supported_params(backend)
        for combo in itertools.product(*(grid[k] for k in keys)):
            kw = dict(zip(keys, combo))
            if kw.get("mode") == "hybrid" and not _supports_hybrid(backend):
                continue
            param_kw = {k: v for k, v in kw.items() if k in SearchParams.__dataclass_fields__}
            if any(k not in supported for k in param_kw):
                continue
            run_kw = {k: v for k, v in kw.items() if k not in param_kw}
            cfg = RunConfig(backend=backend, params=SearchParams(**param_kw), axis="grid", **run_kw)
            configs.setdefault(cfg.config_id, cfg)
            if llm:
                cfg = replace(cfg, selection="llm")
                configs.setdefault(cfg.config_id, cfg)
    return list(configs.values())
