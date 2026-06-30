from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_points(path: Path) -> dict[str, list[dict[str, str]]]:
    panels: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            panels.setdefault(row["panel"], []).append(row)
    return panels


def rows_to_arrays(rows: list[dict[str, str]], kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = [row for row in rows if row["kind"] == kind]
    xy = np.array([[float(row["x"]), float(row["y"])] for row in keep], dtype=np.float32)
    labels = np.array([int(row["class"]) for row in keep], dtype=np.int64)
    indices = np.array([int(row["index"]) for row in keep], dtype=np.int64)
    return xy, labels, indices


def set_clean_limits(ax: plt.Axes, xy: np.ndarray, extra: np.ndarray | None = None) -> None:
    pts = xy if extra is None else np.vstack([xy, extra])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.07 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def class_colormap(num_classes: int, name: str) -> ListedColormap:
    palettes = {
        "photo": [
            "#2F6DAE",
            "#E6852A",
            "#3B9E58",
            "#8E62B8",
            "#C45A4A",
            "#7C6A5C",
            "#52B7C8",
            "#C9B63B",
            "#7F7F7F",
            "#D675A5",
        ],
        "muted": [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#B279A2",
            "#E45756",
            "#72B7B2",
            "#9D755D",
            "#EECA3B",
            "#BAB0AC",
            "#FF9DA6",
        ],
    }
    colors = palettes.get(name, palettes["muted"])
    if num_classes > len(colors):
        base = plt.get_cmap("tab20", num_classes)
        return ListedColormap([base(i) for i in range(num_classes)])
    return ListedColormap(colors[:num_classes])


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw paper-ready baseline-vs-GRAPPLE t-SNE from exported points.")
    parser.add_argument(
        "--points",
        type=Path,
        default=Path(
            "runs/aaai_figures/embedding_comparison_amazon_computers/"
            "amazon-computers_graphsage_vs_grapple_tsne_seed0_points.csv"
        ),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("runs/grapple_paper_figures"))
    parser.add_argument("--stem", default="fig3_graphsage_vs_grapple_tsne")
    parser.add_argument("--baseline_panel", default="graphsage")
    parser.add_argument("--baseline_title", default="(a) GraphSAGE + Linear")
    parser.add_argument("--grapple_title", default="(b) GRAPPLE")
    parser.add_argument("--split_prototype_markers", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--palette", default="muted", choices=["muted", "photo"])
    parser.add_argument("--compact_prototype_legend", action="store_true")
    args = parser.parse_args()

    panels = read_points(args.points)
    if args.baseline_panel not in panels or "grapple" not in panels:
        raise SystemExit(f"Expected panels {args.baseline_panel!r} and 'grapple' in {args.points}.")

    base_xy, base_y, base_idx = rows_to_arrays(panels[args.baseline_panel], "node")
    gr_xy, gr_y, gr_idx = rows_to_arrays(panels["grapple"], "node")
    proto_xy, proto_y, _ = rows_to_arrays(panels["grapple"], "prototype")
    if len(base_idx) != len(gr_idx) or not np.array_equal(base_idx, gr_idx):
        raise SystemExit("GraphSAGE and GRAPPLE panels do not use the same sampled node indices.")
    if not np.array_equal(base_y, gr_y):
        raise SystemExit("GraphSAGE and GRAPPLE panels do not use the same sampled node labels.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    num_classes = int(max(base_y.max(), proto_y.max()) + 1)
    cmap = class_colormap(num_classes, args.palette)

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.85), constrained_layout=True)
    panel_specs = [
        (axes[0], base_xy, None, args.baseline_title),
        (axes[1], gr_xy, proto_xy, args.grapple_title),
    ]

    for ax, node_xy, maybe_proto_xy, title in panel_specs:
        ax.scatter(
            node_xy[:, 0],
            node_xy[:, 1],
            c=base_y,
            cmap=cmap,
            s=5.2,
            alpha=0.52,
            linewidths=0,
            rasterized=True,
        )
        if maybe_proto_xy is not None:
            if args.split_prototype_markers:
                for offset, marker, size in [(0, "*", 105), (1, "D", 44)]:
                    mask = np.arange(len(maybe_proto_xy)) % 2 == offset
                    ax.scatter(
                        maybe_proto_xy[mask, 0],
                        maybe_proto_xy[mask, 1],
                        c=proto_y[mask],
                        cmap=cmap,
                        s=size,
                        marker=marker,
                        edgecolors="black",
                        linewidths=0.85,
                        zorder=5,
                    )
            else:
                ax.scatter(
                    maybe_proto_xy[:, 0],
                    maybe_proto_xy[:, 1],
                    c=proto_y,
                    cmap=cmap,
                    s=145,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.85,
                    zorder=5,
                )
        ax.set_title(title, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#222222")
        set_clean_limits(ax, node_xy, maybe_proto_xy)

    node_handle = plt.Line2D([], [], color="#7A7A7A", marker="o", linestyle="None", markersize=3.3, label="Node")
    if args.split_prototype_markers:
        p1_label = "P1" if args.compact_prototype_legend else "Prototype 1"
        p2_label = "P2" if args.compact_prototype_legend else "Prototype 2"
        proto_handles = [
            plt.Line2D(
                [],
                [],
                color="black",
                marker="*",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6.7,
                label=p1_label,
            ),
            plt.Line2D(
                [],
                [],
                color="black",
                marker="D",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=4.4,
                label=p2_label,
            ),
        ]
    else:
        proto_handles = [
            plt.Line2D(
                [],
                [],
                color="black",
                marker="*",
                linestyle="None",
                markerfacecolor="white",
        markeredgecolor="black",
                markersize=6.7,
                label="Prototype",
            )
        ]
    axes[1].legend(
        handles=[node_handle, *proto_handles],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.35,
        handletextpad=0.45,
        fontsize=6.3 if args.compact_prototype_legend else 7.0,
    )

    fig.savefig(args.out_dir / f"{args.stem}.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / f"{args.stem}.png", dpi=600, bbox_inches="tight")
    dataset = args.dataset
    if dataset is None:
        name = args.points.name
        dataset = name.split("_")[0] if "_" in name else "unknown"

    metadata = {
        "dataset": dataset,
        "source_points": str(args.points),
        "output_pdf": str(args.out_dir / f"{args.stem}.pdf"),
        "output_png": str(args.out_dir / f"{args.stem}.png"),
        "baseline_panel": f"{args.baseline_title} node embeddings exported by experiments/plot_embedding_comparison_with_prototypes.py",
        "grapple_panel": "GRAPPLE v_clipped node tangent vectors with rho * prototype_directions, jointly t-SNE-reduced with all prototypes in the GRAPPLE panel.",
        "sampled_nodes": int(len(base_idx)),
        "sampled_nodes_per_class": {str(cls): int((base_y == cls).sum()) for cls in sorted(set(base_y.tolist()))},
        "num_classes": num_classes,
        "num_prototypes": int(len(proto_y)),
        "same_sampled_nodes_both_panels": True,
        "nodes_used": "class-balanced random split test-node sample from the source export",
        "tsne_settings": "Source export used sklearn TSNE with random_state=0, init='pca', learning_rate='auto', and perplexity=30.",
        "prototype_marker_encoding": (
            "Two learned prototypes per class are shown with separate marker shapes: star for prototype index 1 and diamond for prototype index 2."
            if args.split_prototype_markers
            else "All learned prototypes are shown as stars."
        ),
        "palette": args.palette,
        "compact_prototype_legend": bool(args.compact_prototype_legend),
        "seed": 0,
        "note": "This script redraws an existing real export; it does not fabricate or alter coordinates.",
    }
    (args.out_dir / f"{args.stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
