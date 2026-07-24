"""Plot score distributions and transition heatmaps from lexichash-trace JSON output.

Usage: cargo r -r -- -k 21 --repeat 1000 --sketch both | python plot.py [--out-dir DIR] [--format png pdf svg ...] [--plots best second transition drift inverse gap-size inverse-gap-size gap-rate inverse-gap-rate]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

import approx.lexichash as lh
import approx.minhash as mh

ALL_PLOTS = [
    "best",
    "second",
    "transition",
    "drift",
    "inverse",
    "gap-size",
    "inverse-gap-size",
    "gap-rate",
    "inverse-gap-rate",
]

DEFAULT_PLOTS = [
    # "best",
    # "second",
    # "transition",
    "drift",
    "inverse",
    # "gap-size",
    "inverse-gap-size",
    # "gap-rate",
    "inverse-gap-rate",
]

# one color per sketch algorithm; tab:orange/grey stay free for
# theoretical/identity lines, tab:green for drift's alt/old comparison
SKETCH_COLOR = {"lexichash": "tab:blue", "minhash": "tab:red"}

# lexichash's fast approximate inverse (--compare-approx)
APPROX_COLOR = "tab:purple"

# a primary curve draws above its own alt/old/approx comparison variants
PRIMARY_ZORDER = 3
COMPARISON_ZORDER = 2

# drift plots stop here regardless of --max-mutation-rate: past this,
# minhash's score has mostly collapsed to its floor
DRIFT_MAX_RATE = 14.0


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
        default=DEFAULT_PLOTS,
        help=(
            "which plot(s) to generate: best, second, transition, drift, inverse, "
            "gap-size, inverse-gap-size, gap-rate, inverse-gap-rate (default: all)"
        ),
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help=(
            "also show lexichash's alt model / minhash's old model on plots "
            "that support it, for comparison against the primary model"
        ),
    )
    parser.add_argument(
        "--compare-approx",
        action="store_true",
        help=(
            "also show lexichash's fast closed-form approximate inverse "
            "(inverse_approx) on inverse/inverse-gap-size/inverse-gap-rate, "
            "for comparison against the exact bisection-based inverse"
        ),
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "generate one figure per algorithm instead of combining both "
            "into one, for input with more than one algorithm (drift always "
            "does this already, regardless of this flag)"
        ),
    )
    return parser.parse_args()


def fmt_rate(rate):
    return f"{rate * 100:g}%"


def fmt_len(length):
    """Format a sequence length in scientific notation for figure titles,
    e.g. 30000 -> "3 \\cdot 10^{4}", 100000 -> "10^{5}"."""
    exp = int(np.floor(np.log10(length)))
    coeff = length / 10**exp
    if np.isclose(coeff, round(coeff)):
        coeff = int(round(coeff))
    if coeff == 1:
        return f"10^{{{exp}}}"
    return rf"{coeff:g} \cdot 10^{{{exp}}}"


def sketch_names(data_list):
    return " & ".join(d["algorithm"] for d in data_list)


def forward_prediction(data, rho):
    """Theoretical mean drift score at substitution rate(s) `rho`, dispatching on algorithm."""
    if data["algorithm"] == "minhash":
        return mh.score(data["k"], rho)
    return lh.score(data["len"], data["k"], rho)


def inverse_prediction(data, score):
    """Recovered mutation rate from empirical mean score(s), dispatching on algorithm."""
    if data["algorithm"] == "minhash":
        return mh.fast_inverse(data["k"], score)
    return lh.inverse(data["len"], data["k"], score)
    # return lh.inverse_approx(data["len"], data["k"], score)


def masked_predict(predict, data, scores):
    """Call `predict(data, s)` per-element, returning `nan` where the model's
    valid range is exceeded instead of raising for the whole array -- one
    out-of-range point would otherwise blank out an entire sweep."""
    scores = np.asarray(scores, dtype=float)
    result = np.full(scores.shape, np.nan)
    for i, s in enumerate(scores):
        try:
            result[i] = predict(data, np.array([s]))[0]
        except ValueError:
            pass
    return result


def relative_gap(values, reference):
    """abs(values - reference) / reference, filtered to finite entries."""
    gaps = np.abs(np.asarray(values, dtype=float) - reference) / reference
    return gaps[np.isfinite(gaps)]


def try_relative_gap(predict, data, means, reference):
    """`relative_gap(predict(data, means), reference)`, or empty if
    `predict` raises (`means` outside that model's valid range)."""
    try:
        return relative_gap(predict(data, means), reference)
    except ValueError:
        return np.array([])


def alt_prediction(data, rho):
    """Theoretical mean drift score under lexichash's alternative
    discrete-substitution model (see `lh.alt_score`); lexichash only."""
    return lh.alt_score(data["len"], data["k"], rho)


def alt_inverse_prediction(data, score):
    """Recovered mutation rate under lexichash's alternative
    discrete-substitution model (see `lh.alt_inverse`); lexichash only."""
    return lh.alt_inverse(data["len"], data["k"], score)


def approx_inverse_prediction(data, score):
    """Recovered mutation rate under lexichash's fast closed-form
    approximate inverse (see `lh.inverse_approx`); lexichash only, valid
    only for rho up to ~0.1 (much narrower than the exact `lh.inverse`)."""
    return lh.inverse_approx(data["len"], data["k"], score)


def old_prediction(data, rho):
    """Theoretical mean drift score under minhash's old exponential-approx
    model (see `mh.old_score`); minhash only."""
    return mh.old_score(data["k"], rho)


def old_inverse_prediction(data, score):
    """Recovered mutation rate under minhash's old exponential-approx
    model (see `mh.old_inverse`); minhash only."""
    return mh.old_inverse(data["k"], score)


def forward_extras(data, compare_models):
    """`(predict, label, linestyle, color)` for each `--compare-models`
    forward (score) comparison curve applicable to `data`'s algorithm."""
    if not compare_models:
        return []
    if data["algorithm"] == "minhash":
        return [(old_prediction, "old approx.", ":", SKETCH_COLOR["minhash"])]
    return [(alt_prediction, "alt", ":", SKETCH_COLOR["lexichash"])]


def inverse_extras(data, compare_models, compare_approx):
    """`(predict, label, linestyle, color)` for each `--compare-models`/
    `--compare-approx` inverse comparison curve applicable to `data`'s
    algorithm."""
    extras = []
    if compare_models:
        if data["algorithm"] == "minhash":
            extras.append(
                (old_inverse_prediction, "old approx.", ":", SKETCH_COLOR["minhash"])
            )
        else:
            extras.append(
                (alt_inverse_prediction, "alt", ":", SKETCH_COLOR["lexichash"])
            )
    if compare_approx and data["algorithm"] != "minhash":
        extras.append((approx_inverse_prediction, "approx.", "-.", APPROX_COLOR))
    return extras


def block_group_means(blocks, min_groups=10):
    """Regroup disjoint i.i.d. `blocks` into windows of `q` consecutive
    blocks, down to `min_groups` windows. Yields (sketch_size, group_means):
    several independent mean samples per sketch size, not just one."""
    counts = np.array([b["repeats"] for b in blocks])
    sums = np.array([b["sum"] for b in blocks])
    num_blocks = len(counts)
    max_q = num_blocks // min_groups
    for q in range(1, max_q + 1):
        num_groups = num_blocks // q
        c = counts[: num_groups * q].reshape(num_groups, q).sum(axis=1)
        s = sums[: num_groups * q].reshape(num_groups, q).sum(axis=1)
        yield c.mean(), s / c


def add_gap_series(
    ax,
    xs,
    gap_mean,
    gap_std,
    color,
    label,
    fit_reference=False,
    alpha=0.8,
    zorder=PRIMARY_ZORDER,
):
    """Draw one mean-gap-vs-x curve (+ std band) on `ax`. If `fit_reference`,
    also fit (least squares) and draw a C/sqrt(x) reference line."""
    # on a log-scale axis, a hard 0 floor makes fill_between drop those
    # vertices entirely, leaving gaps in the band; a small positive floor
    # keeps it continuous without visibly changing the linear-scale case
    floor = 1e-4 if ax.get_yscale() == "log" else 0.0
    lo = np.clip(gap_mean - gap_std, floor, None)
    ax.fill_between(xs, lo, gap_mean + gap_std, color=color, alpha=0.15, zorder=zorder)
    ax.plot(
        xs,
        gap_mean,
        color=color,
        alpha=alpha,
        marker="o",
        markersize=3,
        label=label,
        zorder=zorder,
    )

    if fit_reference:
        inv_sqrt_xs = 1.0 / np.sqrt(xs)
        C = np.sum(gap_mean * inv_sqrt_xs) / np.sum(inv_sqrt_xs**2)
        ref = C * inv_sqrt_xs
        ax.plot(
            xs,
            ref,
            "--",
            color=color,
            linewidth=1,
            alpha=0.7,
            label=rf"{label}: ${C:.2g}/\sqrt{{n}}$ reference",
            zorder=zorder,
        )


def finish_size_axes(ax):
    ax.set_xlabel("number of sketched $k$-mers")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:g}%")


def bound_gap_axis(ax, mean_arrays, margin=1.15, log_margin=3.0):
    """Set y-limits from the plotted means (with a margin), instead of
    autoscaling to the fill_between std bands, which can be much wider.
    Log axes get a multiplicative margin on both ends; linear only on top."""
    values = np.concatenate([a[a > 0] for a in mean_arrays if len(a)])
    if values.size == 0:
        return
    if ax.get_yscale() == "log":
        ax.set_ylim(values.min() / log_margin, values.max() * log_margin)
    else:
        ax.set_ylim(0, values.max() * margin)


def rate_tick_locator():
    """Tick locator that adapts to the swept range instead of a fixed step,
    which looks fine at 10% but overlaps unreadably at 40%."""
    return MaxNLocator(nbins=8, steps=[1, 2, 5, 10])


def finish_rate_axes(ax):
    ax.set_xlabel("mutation rate")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:g}%")
    ax.xaxis.set_major_locator(rate_tick_locator())
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")


def last_stable_rate(rates, gap_mean, threshold=0.2):
    """Largest rate before `gap_mean` permanently exceeds `threshold` (once
    collapsed to its floor it stays there, but noise can cross `threshold`
    in brief earlier spikes, so look for the last sustained breach)."""
    if len(gap_mean) == 0 or gap_mean[-1] < threshold:
        # never permanently breaches the threshold; keep the full range
        return rates[-1] if len(rates) else 0.0
    below = np.flatnonzero(gap_mean < threshold)
    return rates[below[-1]] if below.size else 0.0


def rate_axes_for_range(max_rate, stable_cutoff):
    """Figure/axes for a gap-vs-rate plot: one linear axis if the range stays
    within `stable_cutoff`, otherwise two panels (zoomed linear + full-range
    log), since the gap can blow up by orders of magnitude past it."""
    if max_rate <= stable_cutoff * 1.2:
        fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
        return fig, [ax]

    fig, (stable_ax, full_ax) = plt.subplots(
        1, 2, figsize=(13, 4.5), layout="constrained"
    )
    stable_ax.set_title(f"stable regime (≤{stable_cutoff:.3g}%)")
    full_ax.set_title("full range (log scale)")
    full_ax.set_yscale("log")
    return fig, [stable_ax, full_ax]


def draw_oor_markers(axes, oor_markers):
    """Draw each axis's collected `(x, y, color)` out-of-range markers (see
    `add_gap_rate_series`) as vertical dotted lines up to that axis's top,
    plus a single legend entry on `axes[0]` if any were drawn."""
    for ax, markers in zip(axes, oor_markers):
        top = ax.get_ylim()[1]
        for x, y, color in markers:
            ax.plot([x, x], [y, top], ":", color=color, linewidth=1.5, zorder=1)
    if any(oor_markers):
        axes[0].plot(
            [],
            [],
            ":",
            color="grey",
            linewidth=1.5,
            label="out of range beyond this point",
        )


def add_gap_rate_series(
    axes, xs, gap_mean, gap_std, color, label, stable_cutoff, max_rate, extras=None
):
    """Plot one gap-vs-rate series plus `extras` (`(xs, gap_mean, label,
    linestyle, color)` tuples) onto `axes` (two-panel: stable subset only
    on panel 0). Returns `(panel_means, oor_markers)` for y-limits/markers."""
    two_panel = len(axes) == 2
    panel_means = [[] for _ in axes]
    oor_markers = []

    def axes_for_x(x):
        if not two_panel:
            return [0]
        return [0, 1] if x <= stable_cutoff else [1]

    def plot_one(
        ax_idx,
        ax,
        xs_i,
        mean_i,
        std_i,
        linestyle,
        curve_label,
        curve_color,
        alpha,
        zorder,
        filter_to_stable,
    ):
        if filter_to_stable:
            m = xs_i <= stable_cutoff
            xs_i, mean_i = xs_i[m], mean_i[m]
            std_i = std_i[m] if std_i is not None else None
        if ax.get_yscale() == "log":
            # an exact-zero mean (e.g. the rate=0 point) can't be placed on
            # a log axis and otherwise draws a near-vertical line down to it
            m = mean_i > 0
            xs_i, mean_i = xs_i[m], mean_i[m]
            std_i = std_i[m] if std_i is not None else None
        if std_i is not None:
            add_gap_series(
                ax,
                xs_i,
                mean_i,
                std_i,
                curve_color,
                curve_label,
                alpha=alpha,
                zorder=zorder,
            )
        else:
            ax.plot(
                xs_i,
                mean_i,
                linestyle,
                color=curve_color,
                alpha=alpha,
                marker="o",
                markersize=3,
                label=curve_label,
                zorder=zorder,
            )
        panel_means[ax_idx].append(mean_i)

    plot_one(
        0,
        axes[0],
        xs,
        gap_mean,
        gap_std,
        "-",
        label,
        color,
        0.8,
        PRIMARY_ZORDER,
        filter_to_stable=two_panel,
    )
    if two_panel:
        plot_one(
            1,
            axes[1],
            xs,
            gap_mean,
            gap_std,
            "-",
            label,
            color,
            0.8,
            PRIMARY_ZORDER,
            filter_to_stable=False,
        )
    if xs.size and xs[-1] < max_rate - 1e-9:
        oor_markers += [
            (ax_idx, xs[-1], gap_mean[-1], color) for ax_idx in axes_for_x(xs[-1])
        ]

    for extra_xs, extra_gap_mean, extra_label, linestyle, extra_color in extras or []:
        full_label = f"{label} ({extra_label})"
        plot_one(
            0,
            axes[0],
            extra_xs,
            extra_gap_mean,
            None,
            linestyle,
            full_label,
            extra_color,
            0.8,
            COMPARISON_ZORDER,
            filter_to_stable=two_panel,
        )
        if two_panel:
            plot_one(
                1,
                axes[1],
                extra_xs,
                extra_gap_mean,
                None,
                linestyle,
                full_label,
                extra_color,
                0.8,
                COMPARISON_ZORDER,
                filter_to_stable=False,
            )
        if extra_xs.size and extra_xs[-1] < max_rate - 1e-9:
            oor_markers += [
                (ax_idx, extra_xs[-1], extra_gap_mean[-1], extra_color)
                for ax_idx in axes_for_x(extra_xs[-1])
            ]

    return panel_means, oor_markers


def add_row_label(ax, text, x=-0.35, y=0.5):
    """Bold, rotated label to the left of `ax`, identifying a whole row (or
    row-group) of a multi-algorithm stacked figure."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )


def plot_best(data_list, lo, hi):
    nrows = len(data_list)
    ncols = len(data_list[0]["score_histograms"])
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, data in enumerate(data_list):
        for col, h in enumerate(data["score_histograms"]):
            ax = axes[row, col]
            counts = np.array(h["counts"])
            probs = counts / counts.sum()
            ax.bar(np.arange(len(probs)), probs)
            ax.set_xlim(lo - 0.5, hi + 0.5)
            ax.set_xticks(np.arange(lo, hi + 1))
            if row == 0:
                ax.set_title(f"mutation rate = {fmt_rate(h['rate'])}")
            if row == nrows - 1:
                ax.set_xlabel("score")
        axes[row, 0].set_ylabel("probability")
        if nrows > 1:
            add_row_label(axes[row, 0], data["algorithm"])
    fig.suptitle(
        f"Best score distribution ({sketch_names(data_list)}, $k$={data_list[0]['k']}, "
        f"len=${fmt_len(data_list[0]['len'])}$, repeat={data_list[0]['repeat']})"
    )
    return fig


def plot_second(data_list, lo, hi):
    n_algo = len(data_list)
    ncols = len(data_list[0]["second_best_histograms"])

    per_algo = []
    for data in data_list:
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
        per_algo.append((data, sb_hists, sb_data))

    max_prob = max(probs.max() for _, _, sb_data in per_algo for probs, _ in sb_data)
    max_card = max(
        avg_card.max() for _, _, sb_data in per_algo for _, avg_card in sb_data
    )

    fig, axes = plt.subplots(
        2 * n_algo,
        ncols,
        figsize=(4 * ncols, 2.5 * 2 * n_algo),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1] * n_algo, "hspace": 0.05},
        squeeze=False,
        layout="constrained",
    )
    for algo_idx, (data, sb_hists, sb_data) in enumerate(per_algo):
        top_row, bot_row = 2 * algo_idx, 2 * algo_idx + 1
        for col, (h, (probs, avg_card)) in enumerate(zip(sb_hists, sb_data)):
            ax_top, ax_bot = axes[top_row, col], axes[bot_row, col]
            ax_top.bar(np.arange(len(probs)), probs, color="tab:blue")
            ax_top.set_ylim(0, max_prob * 1.05)
            if algo_idx == 0:
                ax_top.set_title(f"mutation rate = {fmt_rate(h['rate'])}")
            ax_top.tick_params(labelbottom=False)

            ax_bot.bar(np.arange(len(avg_card)), avg_card, color="tab:red")
            ax_bot.set_ylim(
                max_card * 1.05, 0
            )  # inverted: 0 touches ax_top, grows downward
            ax_bot.set_xlim(lo - 0.5, hi + 0.5)
            ax_bot.set_xticks(np.arange(lo, hi + 1))
            if algo_idx == n_algo - 1:
                ax_bot.set_xlabel("second-best score")

            if col > 0:
                ax_top.set_yticklabels([])
                ax_bot.set_yticklabels([])
        axes[top_row, 0].set_ylabel("probability", color="tab:blue")
        axes[bot_row, 0].set_ylabel("avg. tied $k$-mers", color="tab:red")
        if n_algo > 1:
            add_row_label(axes[top_row, 0], data["algorithm"], x=-0.55, y=0.0)
    fig.suptitle(
        f"Second-best score distribution ({sketch_names(data_list)}, $k$={data_list[0]['k']}, "
        f"len=${fmt_len(data_list[0]['len'])}$, repeat={data_list[0]['repeat']})"
    )
    return fig


def plot_transition(data_list, lo, hi):
    n_algo = len(data_list)
    ncols = len(data_list[0]["band_transitions"])

    prob_matrices_per_algo = []
    for data in data_list:
        prob_matrices = []
        for b in data["band_transitions"]:
            bsize = b["matrix"]["size"]
            counts = np.array(b["matrix"]["data"], dtype=float).reshape(bsize, bsize)
            # normalize by the whole matrix (joint probability), not per-row
            total = counts.sum()
            probs = counts / total if total > 0 else counts
            prob_matrices.append(probs)
        prob_matrices_per_algo.append(prob_matrices)

    # shared log color scale across bands and algorithms, so they stay
    # comparable; 0 is masked out (rendered white) since it has no place on
    # a log scale
    nonzero = np.concatenate(
        [p[p > 0] for prob_matrices in prob_matrices_per_algo for p in prob_matrices]
    )
    vmin = nonzero.min()
    vmax = nonzero.max()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(
        n_algo,
        ncols,
        figsize=(4.5 * ncols, 4 * n_algo),
        squeeze=False,
        layout="constrained",
    )
    for row, (data, prob_matrices) in enumerate(zip(data_list, prob_matrices_per_algo)):
        for col, (b, probs) in enumerate(zip(data["band_transitions"], prob_matrices)):
            ax = axes[row, col]
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
            title = f"{fmt_rate(b['from_rate'])} → {fmt_rate(b['to_rate'])}"
            ax.set_title(f"{data['algorithm']}: {title}" if n_algo > 1 else title)
            ax.set_xlabel("score after")
            ax.set_ylabel("score before")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Score transition probabilities by mutation-rate band ({sketch_names(data_list)}, log scale)"
    )
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


def aggregate_counts(point):
    """Sum an `original_drift` point's per-group histograms into the single
    histogram aggregated over all repeats."""
    return np.sum(point["group_counts"], axis=0)


def group_means_at_rate(point, algorithm):
    """Per-group empirical mean score at one `original_drift` point, one
    value per disjoint repeat group (`RATE_GROUPS` of them)."""
    counts = np.array(point["group_counts"], dtype=float)
    if algorithm == "minhash":
        return counts[:, 1] / counts.sum(axis=1)
    idx = np.arange(counts.shape[1])
    return (counts * idx).sum(axis=1) / counts.sum(axis=1)


def repeats_per_rate_group(data):
    """Number of repeats in each of the disjoint groups `group_means_at_rate`
    draws one sample from, i.e. `repeat // RATE_GROUPS`."""
    num_groups = len(data["original_drift"][0]["group_counts"])
    return data["repeat"] // num_groups


def plot_drift(data, compare_models=False):
    drift = data["original_drift"]
    rates = np.array([p["mutations"] for p in drift]) / data["len"] * 100
    keep = rates <= DRIFT_MAX_RATE
    drift = [p for p, k in zip(drift, keep) if k]
    rates = rates[keep]
    agg = [aggregate_counts(p) for p in drift]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    bold = 2.25

    if data["algorithm"] == "minhash":
        p_match = np.array([a[1] / a.sum() for a in agg])
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
        legend_handles = [empirical_handle, theoretical_handle]

        if compare_models:
            theoretical_old = mh.old_score(data["k"], rates / 100)
            theoretical_old_handle = ax.plot(
                rates,
                theoretical_old,
                ":",
                color="tab:green",
                linewidth=bold,
                label="theoretical model (old approx.)",
            )[0]
            legend_handles.append(theoretical_old_handle)
    else:
        # LexicHash's drift score is a skewed/bimodal continuous score, so
        # the envelope comes from exact percentiles of the per-point score
        # histogram rather than +/- std
        percentiles = list(range(10, 100, 10))
        cmap = plt.get_cmap("cool")
        colors = cmap(np.linspace(0, 1, len(percentiles)))

        vals_list = [
            np.array([percentile_from_counts(a, q) for a in agg]) for q in percentiles
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

        empirical = np.array([mean_from_counts(a) for a in agg])
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

        legend_handles = [empirical_handle, theoretical_handle]

        if compare_models:
            theoretical_alt = alt_prediction(data, rates / 100)
            theoretical_alt_handle = ax.plot(
                rates,
                theoretical_alt,
                ":",
                color="tab:green",
                linewidth=bold,
                label="theoretical model (alt)",
                zorder=20,
            )[0]
            legend_handles.append(theoretical_alt_handle)

        ax.set_ylim(0, data["k"] + 1)
        ax.set_ylabel("shared prefix length with original best $k$-mer")
        ax.yaxis.set_major_locator(MultipleLocator(3))

        header = Line2D([], [], linestyle="none", label="percentile")
        legend_handles += [header] + percentile_handles

    ax.set_xlabel("mutation rate")
    ax.xaxis.set_major_locator(rate_tick_locator())
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    fig.suptitle(
        f"Best $k$-mer drift from original ({data['algorithm']}, $k$={data['k']}, len=${fmt_len(data['len'])}$, repeat={data['repeat']})"
    )
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig


def plot_inverse(data_list, compare_models=False, compare_approx=False):
    def track_max(current, values):
        # comparison models can hit +-inf/nan at the extremes (e.g. log(0)
        # when the empirical match probability is exactly 0 at high
        # mutation rates); ignore those when sizing the axes
        finite = values[np.isfinite(values)]
        return max(current, np.max(finite)) if finite.size else current

    fig, ax = plt.subplots(figsize=(9.5, 6.5), layout="constrained")
    lim = 0.0
    ymax = 0.0
    # (x, y_start, color) for each curve that runs out of its model's
    # invertible range before the swept rate ends, so a vertical marker can
    # be drawn up to the axis top once the final y-limit is known
    oor_markers = []

    for data in data_list:
        drift = data["original_drift"]
        rates = np.array([p["mutations"] for p in drift]) / data["len"] * 100
        lim = max(lim, rates.max())
        agg = [aggregate_counts(p) for p in drift]
        color = SKETCH_COLOR[data["algorithm"]]

        if data["algorithm"] == "minhash":
            scores = np.array([a[1] / a.sum() for a in agg])
        else:
            scores = np.array([mean_from_counts(a) for a in agg])

        curves = [(inverse_prediction, "-", color, PRIMARY_ZORDER, "")]
        curves += [
            (predict, linestyle, xcolor, COMPARISON_ZORDER, f" ({label})")
            for predict, label, linestyle, xcolor in inverse_extras(
                data, compare_models, compare_approx
            )
        ]

        for predict, linestyle, curve_color, zorder, label_suffix in curves:
            recovered = masked_predict(predict, data, scores) * 100
            valid = np.flatnonzero(np.isfinite(recovered))
            if valid.size == 0:
                continue
            ymax = track_max(ymax, recovered)
            if valid[-1] < len(recovered) - 1:
                oor_markers.append(
                    (rates[valid[-1]], recovered[valid[-1]], curve_color)
                )
            ax.plot(
                rates,
                recovered,
                linestyle,
                color=curve_color,
                alpha=0.8,
                marker="o",
                markersize=3,
                label=f"{data['algorithm']}: inverse(empirical score){label_suffix}",
                zorder=zorder,
            )

    ax.plot(
        [0, lim], [0, lim], "--", color="grey", linewidth=1, label="identity", zorder=0
    )
    ax.set_xlim(0, lim)
    # extend the y-axis (not just up to `lim`) so points that overshoot the
    # true-rate range, e.g. a noisy or biased recovery, stay visible instead
    # of getting clipped off the top
    ax.set_ylim(0, max(lim, ymax) * 1.02)
    top = ax.get_ylim()[1]
    for x, y0, color in oor_markers:
        ax.plot([x, x], [y0, top], ":", color=color, linewidth=1.5, zorder=1)
    if oor_markers:
        ax.plot(
            [],
            [],
            ":",
            color="grey",
            linewidth=1.5,
            label="out of range beyond this point",
        )
    ax.set_xlabel("true mutation rate")
    ax.set_ylabel("recovered mutation rate")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    # only force 1:1 (clean 45° identity line) when overshoot is modest;
    # a big one (e.g. minhash saturating) would squeeze the plot into a
    # tall, mostly-empty sliver
    if ymax <= lim * 1.3:
        ax.set_aspect("equal")
    fig.suptitle(
        f"Mutation rate recovery from empirical score ({sketch_names(data_list)}, "
        f"$k$={data_list[0]['k']}, len=${fmt_len(data_list[0]['len'])}$, repeat={data_list[0]['repeat']})"
    )
    # "identity" first so the reference line reads before the data series
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != "identity")
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order], loc="lower right"
    )
    return fig


def plot_gap_size(data_list, compare_models=False):
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    all_means = []

    for data in data_list:
        conv = data["convergence"]
        rate = conv["rate"]
        pred = forward_prediction(data, rate)
        color = SKETCH_COLOR[data["algorithm"]]
        extras = forward_extras(data, compare_models)
        extra_preds = [predict(data, rate) for predict, *_ in extras]

        xs, gap_mean, gap_std = [], [], []
        extra_means = [[] for _ in extras]
        for x, means in block_group_means(conv["blocks"]):
            xs.append(x)
            gaps = relative_gap(means, pred)
            gap_mean.append(gaps.mean())
            gap_std.append(gaps.std())  # population std; few groups at large sizes
            for acc, extra_pred in zip(extra_means, extra_preds):
                acc.append(relative_gap(means, extra_pred).mean())

        add_gap_series(
            ax,
            np.array(xs),
            np.array(gap_mean),
            np.array(gap_std),
            color,
            data["algorithm"],
            fit_reference=True,
        )
        all_means.append(np.array(gap_mean))
        for (_, label, linestyle, xcolor), means in zip(extras, extra_means):
            ax.plot(
                xs,
                means,
                linestyle,
                color=xcolor,
                alpha=0.8,
                marker="o",
                markersize=3,
                label=f"{data['algorithm']} ({label})",
                zorder=COMPARISON_ZORDER,
            )
            all_means.append(np.array(means))

    finish_size_axes(ax)
    bound_gap_axis(ax, all_means)
    ax.set_ylabel("relative gap of score: |empirical - theoretical| / theoretical")
    fig.suptitle(
        f"Relative gap between empirical and theoretical score "
        f"({sketch_names(data_list)}, $k$={data_list[0]['k']}, len=${fmt_len(data_list[0]['len'])}$, "
        f"rate={fmt_rate(data_list[0]['convergence']['rate'])})",
        fontsize=11,
    )
    ax.legend()
    return fig


def plot_inverse_gap_size(data_list, compare_models=False, compare_approx=False):
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    all_means = []

    for data in data_list:
        conv = data["convergence"]
        true_rate = conv["rate"]
        color = SKETCH_COLOR[data["algorithm"]]
        extras = inverse_extras(data, compare_models, compare_approx)

        xs, gap_mean, gap_std = [], [], []
        extra_xs = [[] for _ in extras]
        extra_means = [[] for _ in extras]
        for x, means in block_group_means(conv["blocks"]):
            if x < 64:
                continue

            gaps = try_relative_gap(inverse_prediction, data, means, true_rate)
            if gaps.size:
                xs.append(x)
                gap_mean.append(gaps.mean())
                gap_std.append(gaps.std())

            for exs, ems, (predict, *_) in zip(extra_xs, extra_means, extras):
                g = try_relative_gap(predict, data, means, true_rate)
                if g.size:
                    exs.append(x)
                    ems.append(g.mean())

        add_gap_series(
            ax,
            np.array(xs),
            np.array(gap_mean),
            np.array(gap_std),
            color,
            data["algorithm"],
            fit_reference=True,
        )
        all_means.append(np.array(gap_mean))
        for (_, label, linestyle, xcolor), exs, ems in zip(
            extras, extra_xs, extra_means
        ):
            if not exs:
                continue
            ax.plot(
                exs,
                ems,
                linestyle,
                color=xcolor,
                alpha=0.8,
                marker="o",
                markersize=3,
                label=f"{data['algorithm']} ({label})",
                zorder=COMPARISON_ZORDER,
            )
            all_means.append(np.array(ems))

    finish_size_axes(ax)
    bound_gap_axis(ax, all_means)
    ax.set_ylabel("relative gap of mutation rate: |recovered - truth| / truth")
    fig.suptitle(
        f"Relative gap between recovered and true mutation rate "
        f"({sketch_names(data_list)}, $k$={data_list[0]['k']}, len=${fmt_len(data_list[0]['len'])}$, "
        f"rate={fmt_rate(data_list[0]['convergence']['rate'])})",
        fontsize=11,
    )
    ax.legend()
    return fig


def plot_gap_rate(data_list, compare_models=False):
    # compute each algorithm's own gap-vs-rate series up front, so the
    # stable/full-range cutoff (below) can be detected directly from the
    # exact series being plotted, instead of from some other proxy metric
    series = []
    for data in data_list:
        drift = data["original_drift"]
        rates = np.array([p["mutations"] for p in drift]) / data["len"] * 100
        color = SKETCH_COLOR[data["algorithm"]]
        extras_spec = forward_extras(data, compare_models)

        # RATE_GROUPS i.i.d. repeat groups per rate checkpoint give
        # RATE_GROUPS independent gap samples, the same batch-means trick
        # as plot_gap_size but sweeping rate instead of sketch size
        gap_mean, gap_std = [], []
        extra_means = [[] for _ in extras_spec]
        for point, rate in zip(drift, rates):
            means = group_means_at_rate(point, data["algorithm"])
            pred = forward_prediction(data, rate / 100)
            gaps = relative_gap(means, pred)
            gap_mean.append(gaps.mean())
            gap_std.append(gaps.std())
            for acc, (predict, *_) in zip(extra_means, extras_spec):
                acc.append(relative_gap(means, predict(data, rate / 100)).mean())

        extras = [
            (rates, np.array(means), label, linestyle, xcolor)
            for means, (_, label, linestyle, xcolor) in zip(extra_means, extras_spec)
        ]
        series.append(
            (data, rates, np.array(gap_mean), np.array(gap_std), color, extras)
        )

    max_rate = max(rates.max() for _, rates, _, _, _, _ in series)
    stable_cutoff = min(
        last_stable_rate(rates, gap_mean) for _, rates, gap_mean, _, _, _ in series
    )
    fig, axes = rate_axes_for_range(max_rate, stable_cutoff)
    panel_means = [[] for _ in axes]
    oor_markers = [[] for _ in axes]

    for data, rates, gap_mean, gap_std, color, extras in series:
        means_per_axis, oor = add_gap_rate_series(
            axes,
            rates,
            gap_mean,
            gap_std,
            color,
            data["algorithm"],
            stable_cutoff,
            max_rate,
            extras,
        )
        for i, means in enumerate(means_per_axis):
            panel_means[i] += means
        for ax_idx, x, y, marker_color in oor:
            oor_markers[ax_idx].append((x, y, marker_color))

    for ax in axes:
        finish_rate_axes(ax)
        ax.set_ylabel("relative gap of score: |empirical - theoretical| / theoretical")
    for ax, means in zip(axes, panel_means):
        bound_gap_axis(ax, means)
    draw_oor_markers(axes, oor_markers)
    fig.suptitle(
        f"Relative gap between empirical and theoretical score "
        f"({sketch_names(data_list)}, $k$={data_list[0]['k']}, len=${fmt_len(data_list[0]['len'])}$, "
        f"repeat/group={repeats_per_rate_group(data_list[0])})",
        fontsize=11,
    )
    axes[0].legend()
    return fig


def plot_inverse_gap_rate(data_list, compare_models=False, compare_approx=False):
    # compute each algorithm's own gap-vs-rate series up front, so the
    # stable/full-range cutoff (below) can be detected directly from the
    # exact series being plotted, instead of from some other proxy metric
    series = []
    for data in data_list:
        drift = data["original_drift"]
        rates = np.array([p["mutations"] for p in drift]) / data["len"] * 100
        color = SKETCH_COLOR[data["algorithm"]]
        extras_spec = inverse_extras(data, compare_models, compare_approx)

        xs, gap_mean, gap_std = [], [], []
        extra_xs = [[] for _ in extras_spec]
        extra_means = [[] for _ in extras_spec]
        for point, rate in zip(drift, rates):
            if rate == 0:
                continue  # true rate is 0 here, relative gap is undefined
            means = group_means_at_rate(point, data["algorithm"])
            true_rate = rate / 100

            gaps = try_relative_gap(inverse_prediction, data, means, true_rate)
            if gaps.size:
                xs.append(rate)
                gap_mean.append(gaps.mean())
                gap_std.append(gaps.std())

            for exs, ems, (predict, *_) in zip(extra_xs, extra_means, extras_spec):
                g = try_relative_gap(predict, data, means, true_rate)
                if g.size:
                    exs.append(rate)
                    ems.append(g.mean())

        extras = [
            (np.array(exs), np.array(ems), label, linestyle, xcolor)
            for exs, ems, (_, label, linestyle, xcolor) in zip(
                extra_xs, extra_means, extras_spec
            )
            if exs
        ]
        series.append(
            (data, np.array(xs), np.array(gap_mean), np.array(gap_std), color, extras)
        )

    max_rate = max(
        (
            np.array([p["mutations"] for p in data["original_drift"]])
            / data["len"]
            * 100
        ).max()
        for data in data_list
    )
    stable_cutoff = min(
        last_stable_rate(xs, gap_mean) for _, xs, gap_mean, _, _, _ in series
    )
    fig, axes = rate_axes_for_range(max_rate, stable_cutoff)
    panel_means = [[] for _ in axes]
    oor_markers = [[] for _ in axes]

    for data, xs, gap_mean, gap_std, color, extras in series:
        means_per_axis, oor = add_gap_rate_series(
            axes,
            xs,
            gap_mean,
            gap_std,
            color,
            data["algorithm"],
            stable_cutoff,
            max_rate,
            extras,
        )
        for i, means in enumerate(means_per_axis):
            panel_means[i] += means
        for ax_idx, x, y, marker_color in oor:
            oor_markers[ax_idx].append((x, y, marker_color))

    for ax in axes:
        finish_rate_axes(ax)
        ax.set_ylabel("relative gap of mutation rate: |recovered - truth| / truth")
    for ax, means in zip(axes, panel_means):
        bound_gap_axis(ax, means)
    draw_oor_markers(axes, oor_markers)
    fig.suptitle(
        f"Relative gap between recovered and true mutation rate "
        f"({sketch_names(data_list)}, $k$={data_list[0]['k']}, len=${fmt_len(data_list[0]['len'])}$, "
        f"repeat/group={repeats_per_rate_group(data_list[0])})",
        fontsize=11,
    )
    axes[0].legend()
    return fig


def main():
    args = parse_args()
    data_list = json.load(sys.stdin)
    figs = {}

    # scores concentrate around log4(len)
    # crop figures there instead of showing the mostly-empty full 0..k range
    k, length = data_list[0]["k"], data_list[0]["len"]
    size = k + 1
    center = np.ceil(np.log(length) / np.log(4))
    lo = max(0, int(center - 5))
    hi = min(size - 1, int(center + 5))

    def add_figs(base_name, plot_fn, *fn_args):
        """Call `plot_fn(data_list, *fn_args)`, or once per algorithm with
        `--separate` (drift already always does this, regardless of the
        flag, so it doesn't go through here)."""
        if args.separate and len(data_list) > 1:
            for data in data_list:
                figs[f"{base_name}_{data['algorithm']}"] = plot_fn([data], *fn_args)
        else:
            figs[base_name] = plot_fn(data_list, *fn_args)

    if "best" in args.plots:
        add_figs("score_distribution", plot_best, lo, hi)

    if "second" in args.plots:
        add_figs("second_best_score", plot_second, lo, hi)

    if "transition" in args.plots:
        add_figs("transitions", plot_transition, lo, hi)

    if "drift" in args.plots:
        # kept as separate figures, one per algorithm
        for data in data_list:
            name = "drift" if len(data_list) == 1 else f"drift_{data['algorithm']}"
            figs[name] = plot_drift(data, args.compare_models)

    if "inverse" in args.plots:
        add_figs("inverse", plot_inverse, args.compare_models, args.compare_approx)

    if "gap-size" in args.plots:
        add_figs("gap_size", plot_gap_size, args.compare_models)

    if "inverse-gap-size" in args.plots:
        add_figs(
            "inverse_gap_size",
            plot_inverse_gap_size,
            args.compare_models,
            args.compare_approx,
        )

    if "gap-rate" in args.plots:
        add_figs("gap_rate", plot_gap_rate, args.compare_models)

    if "inverse-gap-rate" in args.plots:
        add_figs(
            "inverse_gap_rate",
            plot_inverse_gap_rate,
            args.compare_models,
            args.compare_approx,
        )

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, f in figs.items():
            for fmt in args.format:
                f.savefig(args.out_dir / f"{name}.{fmt}", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
