"""
Plot score distributions and transition heatmaps from lexichash-trace JSON output.

Usage: cargo r -r -- -k 21 --repeat 1000 | python plot.py [--out-dir DIR] [--format png pdf svg ...] [--plots best second transition drift inverse]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import approx.lexichash as lh
import approx.minhash as mh

ALL_PLOTS = [
    "best",
    "second",
    "transition",
    "drift",
    "inverse",
    "converge",
    "converge-inverse",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        help="save plots in this directory instead of showing them",
    )
    parser.add_argument(
        "-f",
        "--format",
        nargs="+",
        default=["pdf"],
        help="file format(s) to save plots as, e.g. --format pdf svg (default: pdf)",
    )
    parser.add_argument(
        "-p",
        "--plots",
        nargs="+",
        choices=ALL_PLOTS,
        default=ALL_PLOTS,
        help="which plot(s) to generate: best, second, transition, drift, inverse, converge, converge-inverse (default: all)",
    )
    return parser.parse_args()


def fmt_rate(rate):
    return f"{rate * 100:g}%"


def forward_prediction(data, rho):
    """Theoretical mean drift score at substitution rate(s) `rho`, dispatching on algorithm."""
    if data["algorithm"] == "minhash":
        return mh.score(data["k"], rho)
    return lh.score(data["len"], data["k"], rho)


def inverse_prediction(data, score):
    """Recovered mutation rate from empirical mean score(s), dispatching on algorithm."""
    if data["algorithm"] == "minhash":
        return mh.inverse(data["k"], score)
    return lh.inverse(data["len"], data["k"], score)


def block_group_means(blocks, min_groups=10):
    """Regroup disjoint `blocks` (each `{repeats, sum}`) into windows of `q`
    consecutive blocks, for increasing `q`, stopping once fewer than
    `min_groups` independent windows remain. Blocks are i.i.d., so this
    yields, for each sketch size, several independent samples of the mean
    at that size instead of just one point on a single noisy trajectory.

    Yields (sketch_size, group_means) pairs, where sketch_size is the
    average number of repeats per window and group_means holds one
    empirical mean per independent window.
    """
    counts = np.array([b["repeats"] for b in blocks])
    sums = np.array([b["sum"] for b in blocks])
    num_blocks = len(counts)
    max_q = num_blocks // min_groups
    for q in range(1, max_q + 1):
        num_groups = num_blocks // q
        c = counts[: num_groups * q].reshape(num_groups, q).sum(axis=1)
        s = sums[: num_groups * q].reshape(num_groups, q).sum(axis=1)
        yield c.mean(), s / c


def plot_gap_vs_sketch_size(
    xs, gap_mean, gap_std, color, ylabel, title, fit_reference=True
):
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    lo = np.clip(gap_mean - gap_std, 0, None)
    ax.fill_between(xs, lo, gap_mean + gap_std, color=color, alpha=0.15)
    ax.plot(xs, gap_mean, color=color, marker="o", markersize=3, label="mean gap")

    inv_sqrt_xs = 1.0 / np.sqrt(xs)
    if fit_reference:
        # C/sqrt(n) reference, with C fit by least squares (equivalent to
        # minimizing RMSE against gap_mean, since C enters linearly)
        C = np.sum(gap_mean * inv_sqrt_xs) / np.sum(inv_sqrt_xs**2)
    else:
        C = 1.0
    ref = C * inv_sqrt_xs
    ax.plot(
        xs,
        ref,
        "--",
        color="grey",
        linewidth=1,
        label=rf"${C:.2g}/\sqrt{{n}}$ reference",
    )

    ax.set_xlabel("number of sketched $k$-mers")
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    ax.legend()
    return fig


def plot_best(data, lo, hi):
    hists = data["score_histograms"]
    fig, axes = plt.subplots(1, len(hists), figsize=(4 * len(hists), 3.5), sharey=True)
    for ax, h in zip(axes, hists):
        counts = np.array(h["counts"])
        probs = counts / counts.sum()
        ax.bar(np.arange(len(probs)), probs)
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_xticks(np.arange(lo, hi + 1))
        ax.set_title(f"mutation rate = {fmt_rate(h['rate'])}")
        ax.set_xlabel("score")
    axes[0].set_ylabel("probability")
    fig.suptitle(
        f"Best score distribution ({data['algorithm']}, $k$={data['k']}, len={data['len']}, repeat={data['repeat']})"
    )
    fig.tight_layout()
    return fig


def plot_second(data, lo, hi):
    sb_hists = data["second_best_histograms"]
    sb_data = []
    for h in sb_hists:
        counts = np.array(h["counts"], dtype=float)
        card_sum = np.array(h["cardinality_sum"], dtype=float)
        probs = counts / counts.sum()
        avg_card = np.divide(
            card_sum, counts, out=np.zeros_like(card_sum), where=counts != 0
        )
        sb_data.append((probs, avg_card))
    max_prob = max(probs.max() for probs, _ in sb_data)
    max_card = max(avg_card.max() for _, avg_card in sb_data)

    fig, axes = plt.subplots(
        2,
        len(sb_hists),
        figsize=(4 * len(sb_hists), 5),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
        squeeze=False,
        layout="constrained",
    )
    for col, (h, (probs, avg_card)) in enumerate(zip(sb_hists, sb_data)):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.bar(np.arange(len(probs)), probs, color="tab:blue")
        ax_top.set_ylim(0, max_prob * 1.05)
        ax_top.set_title(f"mutation rate = {fmt_rate(h['rate'])}")
        ax_top.tick_params(labelbottom=False)

        ax_bot.bar(np.arange(len(avg_card)), avg_card, color="tab:red")
        ax_bot.set_ylim(
            max_card * 1.05, 0
        )  # inverted: 0 touches ax_top, grows downward
        ax_bot.set_xlim(lo - 0.5, hi + 0.5)
        ax_bot.set_xticks(np.arange(lo, hi + 1))
        ax_bot.set_xlabel("second-best score")

        if col > 0:
            ax_top.set_yticklabels([])
            ax_bot.set_yticklabels([])
    axes[0, 0].set_ylabel("probability", color="tab:blue")
    axes[1, 0].set_ylabel("avg. tied $k$-mers", color="tab:red")
    fig.suptitle(
        f"Second-best score distribution ({data['algorithm']}, $k$={data['k']}, len={data['len']}, repeat={data['repeat']})"
    )
    return fig


def plot_transition(data, lo, hi):
    bands = data["band_transitions"]
    prob_matrices = []
    for b in bands:
        bsize = b["matrix"]["size"]
        counts = np.array(b["matrix"]["data"], dtype=float).reshape(bsize, bsize)
        # normalize by the whole matrix (joint probability), not per-row
        total = counts.sum()
        probs = counts / total if total > 0 else counts
        prob_matrices.append(probs)

    # shared log color scale across bands, so they stay comparable; 0 is
    # masked out (rendered white) since it has no place on a log scale
    nonzero = np.concatenate([p[p > 0] for p in prob_matrices])
    vmin = nonzero.min()
    vmax = nonzero.max()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, len(bands), figsize=(4.5 * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]
    for ax, b, probs in zip(axes, bands, prob_matrices):
        masked = np.ma.masked_equal(probs, 0)
        im = ax.imshow(
            masked, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_ylim(lo - 0.5, hi + 0.5)
        ax.set_xticks(np.arange(lo, hi + 1))
        ax.set_yticks(np.arange(lo, hi + 1))
        ax.set_xticks(np.arange(lo - 0.5, hi + 1), minor=True)
        ax.set_yticks(np.arange(lo - 0.5, hi + 1), minor=True)
        ax.grid(which="minor", color="lightgrey", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f"{fmt_rate(b['from_rate'])} → {fmt_rate(b['to_rate'])}")
        ax.set_xlabel("score after")
        ax.set_ylabel("score before")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Score transition probabilities by mutation-rate band ({data['algorithm']}, log scale)"
    )
    fig.tight_layout()
    return fig


def percentile_from_counts(counts, q):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(counts)
    idx = np.searchsorted(cum, q / 100 * total, side="left")
    return min(idx, len(counts) - 1)


def mean_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    return np.dot(counts, np.arange(len(counts))) / total


def plot_drift(data):
    drift = data["original_drift"]
    rates = np.array([p["mutations"] for p in drift]) / data["len"] * 100

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    bold = 2.25

    if data["algorithm"] == "minhash":
        p_match = np.array([c["counts"][1] / sum(c["counts"]) for c in drift])
        std = np.sqrt(p_match * (1 - p_match))
        empirical_handle = ax.plot(
            rates,
            p_match,
            color="tab:red",
            marker="o",
            markersize=3,
            label="empirical mean",
        )[0]
        lo_band, hi_band = np.clip(p_match - std, 0, 1), np.clip(p_match + std, 0, 1)
        ax.fill_between(rates, lo_band, hi_band, color="tab:red", alpha=0.15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(original best hash still wins)")
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        theoretical = forward_prediction(data, rates / 100)
        theoretical_handle = ax.plot(
            rates,
            theoretical,
            "--",
            color="tab:orange",
            linewidth=bold,
            label="theoretical model",
        )[0]

        theoretical_old = mh.old_score(data["k"], rates / 100)
        theoretical_old_handle = ax.plot(
            rates,
            theoretical_old,
            ":",
            color="tab:green",
            linewidth=bold,
            label="theoretical model (old approx.)",
        )[0]
        legend_handles = [empirical_handle, theoretical_handle, theoretical_old_handle]
    else:
        # LexicHash's drift score is a skewed/bimodal continuous score, so
        # the envelope comes from exact percentiles of the per-point score
        # histogram rather than +/- std
        percentiles = list(range(10, 100, 10))
        cmap = plt.get_cmap("cool")
        colors = cmap(np.linspace(0, 1, len(percentiles)))

        vals_list = [
            np.array([percentile_from_counts(p["counts"], q) for p in drift])
            for q in percentiles
        ]
        for lo_vals, hi_vals, color in zip(vals_list, vals_list[1:], colors):
            ax.fill_between(rates, lo_vals, hi_vals, color=color, alpha=0.15, zorder=0)

        percentile_handles = []
        for q, color, vals in zip(percentiles, colors, vals_list):
            linewidth = bold if q == 50 else 1.5
            label = "median" if q == 50 else f"{q}th"
            percentile_handles.append(
                ax.plot(
                    rates,
                    vals,
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=linewidth,
                    label=label,
                    zorder=5,
                )[0]
            )

        empirical = np.array([mean_from_counts(p["counts"]) for p in drift])
        empirical_handle = ax.plot(
            rates,
            empirical,
            color="black",
            marker="o",
            markersize=3,
            linewidth=bold,
            label="empirical mean",
            zorder=10,
        )[0]

        theoretical = forward_prediction(data, rates / 100)
        theoretical_handle = ax.plot(
            rates,
            theoretical,
            "--",
            color="tab:orange",
            linewidth=bold,
            label="theoretical model",
            zorder=20,
        )[0]

        ax.set_ylim(0, data["k"] + 1)
        ax.set_ylabel("shared prefix length with original best $k$-mer")
        ax.yaxis.set_major_locator(MultipleLocator(3))

        header = Line2D([], [], linestyle="none", label="percentile")
        legend_handles = [
            empirical_handle,
            theoretical_handle,
            header,
        ] + percentile_handles

    ax.set_xlabel("mutation rate")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    fig.suptitle(
        f"Best $k$-mer drift from original ({data['algorithm']}, $k$={data['k']}, len={data['len']}, repeat={data['repeat']})"
    )
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig


def plot_inverse(data):
    drift = data["original_drift"]
    n = data["len"]
    rates = np.array([p["mutations"] for p in drift]) / n * 100

    if data["algorithm"] == "minhash":
        scores = np.array([c["counts"][1] / sum(c["counts"]) for c in drift])
    else:
        scores = np.array([mean_from_counts(p["counts"]) for p in drift])
    recovered = inverse_prediction(data, scores) * 100
    label = "inverse(empirical score)"

    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    lim = rates.max()
    ax.plot(
        [0, lim], [0, lim], "--", color="grey", linewidth=1, label="identity", zorder=0
    )
    ax.plot(
        rates,
        recovered,
        color="tab:orange",
        marker="o",
        markersize=3,
        label=label,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true mutation rate")
    ax.set_ylabel("recovered mutation rate")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    ax.set_aspect("equal")
    fig.suptitle(
        f"Mutation rate recovery from empirical score ({data['algorithm']}, $k$={data['k']}, len={data['len']}, repeat={data['repeat']})"
    )
    ax.legend()
    return fig


def plot_converge(data):
    conv = data["convergence"]
    pred = forward_prediction(data, conv["rate"])
    # only minhash has an old, biased approximation to compare against
    # show_old = data["algorithm"] == "minhash"
    show_old = False
    pred_old = mh.old_score(data["k"], conv["rate"]) if show_old else None

    xs, gap_mean, gap_std, old_gap_mean = [], [], [], []
    for x, means in block_group_means(conv["blocks"]):
        gaps = np.abs(means - pred) / pred
        xs.append(x)
        gap_mean.append(gaps.mean())
        gap_std.append(gaps.std())  # population std; few groups at large sizes
        if show_old:
            old_gaps = np.abs(means - pred_old) / pred_old
            old_gap_mean.append(old_gaps.mean())

    fig = plot_gap_vs_sketch_size(
        np.array(xs),
        np.array(gap_mean),
        np.array(gap_std),
        color="tab:blue",
        ylabel="relative gap of score: |empirical - theoretical| / theoretical",
        title=(
            f"Relative gap between empirical and theoretical score "
            f"({data['algorithm']}, $k$={data['k']}, len={data['len']}, rate={fmt_rate(conv['rate'])})"
        ),
    )

    if show_old:
        ax = fig.axes[0]
        ax.plot(
            xs,
            old_gap_mean,
            color="tab:green",
            marker="o",
            markersize=3,
            label="mean gap (old approx.)",
        )
        ax.legend()

    return fig


def plot_converge_inverse(data):
    conv = data["convergence"]
    true_rate = conv["rate"]

    xs, gap_mean, gap_std = [], [], []
    for x, means in block_group_means(conv["blocks"]):
        try:
            recovered = inverse_prediction(data, means)
        except ValueError:
            # empirical mean out of the inverse model's valid range at this
            # sketch size (only happens for lexichash, and only when noisy);
            # skip rather than crash
            continue
        gaps = np.abs(recovered - true_rate) / true_rate
        gaps = gaps[np.isfinite(gaps)]
        if gaps.size == 0:
            continue
        xs.append(x)
        gap_mean.append(gaps.mean())
        gap_std.append(gaps.std())

    return plot_gap_vs_sketch_size(
        np.array(xs),
        np.array(gap_mean),
        np.array(gap_std),
        color="tab:orange",
        ylabel="relative gap of mutation rate: |recovered - truth| / truth",
        title=(
            f"Relative gap between recovered and true mutation rate "
            f"({data['algorithm']}, $k$={data['k']}, len={data['len']}, rate={fmt_rate(conv['rate'])})"
        ),
    )


def main():
    args = parse_args()
    data = json.load(sys.stdin)
    figs = {}

    # scores concentrate around log4(len)
    # crop figures there instead of showing the mostly-empty full 0..k range
    size = data["k"] + 1
    center = np.ceil(np.log(data["len"]) / np.log(4))
    lo = max(0, int(center - 5))
    hi = min(size - 1, int(center + 5))

    if "best" in args.plots:
        figs["score_distribution"] = plot_best(data, lo, hi)

    if "second" in args.plots:
        figs["second_best_score"] = plot_second(data, lo, hi)

    if "transition" in args.plots:
        figs["transitions"] = plot_transition(data, lo, hi)

    if "drift" in args.plots:
        figs["drift"] = plot_drift(data)

    if "inverse" in args.plots:
        figs["inverse"] = plot_inverse(data)

    if "converge" in args.plots:
        figs["convergence"] = plot_converge(data)

    if "converge-inverse" in args.plots:
        figs["convergence_inverse"] = plot_converge_inverse(data)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, f in figs.items():
            for fmt in args.format:
                f.savefig(args.out_dir / f"{name}.{fmt}", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
