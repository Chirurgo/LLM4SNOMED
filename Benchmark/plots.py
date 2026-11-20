"""Trade-off plots for Benchmark.report (static PNGs, light surface).

1. pareto_overview.png  -- every backend's Pareto front (link accuracy vs. estimated
   cost per entity) on one log-x axis; non-front configs as recessive grey dots.
2. pareto_by_backend.png -- small multiples, one panel per backend, front points
   labelled with what makes each config different from the baseline.

Colors: categorical slots 1-5 of the dataviz reference palette, fixed per backend
(never by rank), validated with validate_palette.js (light, all checks pass; the
sub-3:1 contrast slots are relieved by direct labels + results.md as the table view).
Marker shape is a secondary encoding for retrieval mode: circle = dense, triangle = hybrid.
"""
from pathlib import Path
from typing import List

BACKEND_COLORS = {  # fixed slot per entity
    "postgres": "#2a78d6",
    "qdrant": "#eb6834",
    "weaviate": "#1baf7a",
    "meilisearch": "#eda100",
    "neo4j": "#e87ba4",
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODE_MARKER = {"dense": "o", "hybrid": "^"}
QUALITY, COST = "link_acc", "est_ms_entity"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.tick_params(which="both", colors=MUTED, labelsize=7)
    ax.set_xlabel("Estimated cost per entity (ms, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("Linking accuracy (exact-span entities)", color=INK_2, fontsize=9)


def _diff_label(r: dict) -> str:
    """Short label naming what this config changes vs. its mode's baseline."""
    axis = r.get("axis") or "baseline"
    if axis == "baseline":
        return f"{r['mode']} baseline"
    key = {"k_candidate": "k_candidate", "expand_factor": "expand_factor", "reranker": "reranker",
           "ef_search": "p_ef_search", "hybrid_weight": "p_hybrid_weight", "arm_depth_min": "p_arm_depth_min",
           "rrf_k": "p_rrf_k", "fusion": "p_fusion"}.get(axis)
    val = r.get(key) if key else None
    if axis == "reranker" and val in (None, ""):
        val = "none"
    short = {"k_candidate": "k", "expand_factor": "expand", "hybrid_weight": "w", "arm_depth_min": "depth",
             "ef_search": "ef", "rrf_k": "rrf_k"}.get(axis, axis)
    return f"{r['mode'][0]}: {short}={val}"


def _front(rows: List[dict]) -> List[dict]:
    return sorted((r for r in rows if r.get("pareto")), key=lambda r: r[COST])


def _plot_front(ax, rows, color, lw=2.0):
    import matplotlib.pyplot as plt  # noqa: F401
    front = _front(rows)
    if not front:
        return front
    xs = [r[COST] for r in front]
    ys = [r[QUALITY] for r in front]
    ax.plot(xs, ys, color=color, linewidth=lw, drawstyle="steps-post", zorder=2)
    for r in front:
        ax.scatter(r[COST], r[QUALITY], s=64, marker=MODE_MARKER.get(r["mode"], "o"), color=color,
                   edgecolors=SURFACE, linewidths=2, zorder=3)
    return front


def plot_tradeoffs(rows: List[dict], out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r.get(QUALITY) is not None and r.get(COST) is not None and r[COST] > 0]
    if not rows:
        return []
    backends = [b for b in BACKEND_COLORS if any(r["backend"] == b for r in rows)]
    written = []

    # 1) overview
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.scatter([r[COST] for r in rows if not r.get("pareto")], [r[QUALITY] for r in rows if not r.get("pareto")],
               s=14, color=AXIS, zorder=1, linewidths=0)
    for i, b in enumerate(backends):
        front = _plot_front(ax, [r for r in rows if r["backend"] == b], BACKEND_COLORS[b])
        if front:  # direct label beside each front's best point, staggered so close fronts don't collide
            best = max(front, key=lambda r: (r[QUALITY], -r[COST]))
            ax.annotate(b, (best[COST], best[QUALITY]), xytext=(10, -4 - 11 * i), textcoords="offset points",
                        fontsize=8, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6, shrinkA=0, shrinkB=4))
    handles = [Line2D([], [], color=BACKEND_COLORS[b], linewidth=2, label=b) for b in backends]
    handles += [Line2D([], [], color=MUTED, marker=m, linestyle="", markersize=7, label=mode)
                for mode, m in MODE_MARKER.items()]
    handles.append(Line2D([], [], color=AXIS, marker="o", linestyle="", markersize=4, label="off-front config"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    ax.set_title("Pareto fronts: linking accuracy vs. cost per entity", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    p = out_dir / "pareto_overview.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(p)

    # 2) small multiples, shared axes so panels compare directly
    n = len(backends)
    cols = min(2, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(6 * cols, 4.2 * nrows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, b in zip(axes.flat, backends):
        _style(ax)
        rs = [r for r in rows if r["backend"] == b]
        for mode, marker in MODE_MARKER.items():
            off = [r for r in rs if r["mode"] == mode and not r.get("pareto")]
            ax.scatter([r[COST] for r in off], [r[QUALITY] for r in off], s=22, marker=marker,
                       color=AXIS, linewidths=0, zorder=1)
        front = _plot_front(ax, rs, BACKEND_COLORS[b])
        for i, r in enumerate(front):  # selective labels: front points only, alternating offsets
            ax.annotate(_diff_label(r), (r[COST], r[QUALITY]), xytext=(5, -12 if i % 2 else 6),
                        textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_title(b, color=INK, fontsize=10, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-backend trade-offs (labelled points are on the Pareto front)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = out_dir / "pareto_by_backend.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(p)
    return written
