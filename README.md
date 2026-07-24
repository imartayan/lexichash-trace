# lexichash-trace

Simulate random substitutions in random DNA sequences and track how LexicHash/MinHash sketches behave as mutation rate increases:
which *k*-mer gets selected, how often the runner-up changes, and how similar the selected *k*-mer stays to the original one.

Each run repeats the simulation many times in parallel and aggregates the results into JSON, `plot.py` then turns that JSON into figures.

## Build

```sh
RUSTFLAGS="-C target-cpu=native" cargo b -r
cp target/release/lexichash-trace .
```

## Run the simulation

```sh
./lexichash-trace [OPTIONS] > out.json
```

| Flag | Meaning | Default |
|---|---|---|
| `-l <LEN>` | sequence length | 100 Kbp |
| `-k <K>` | *k*-mer size (<= 32) | 21 |
| `-r, --repeat <N>` | number of independent repeats | 1000 |
| `-t, --threads <N>` | size of the thread pool | all cores |
| `-s, --sketch <lexichash\|minhash\|both>` | which sketch algorithm(s) to simulate | `lexichash` |
| `--converge-rate <RATE>` | mutation rate at which to track convergence of the empirical mean drift score to its prediction | 0.05 |
| `--max-mutation-rate <RATE>` | mutation rate up to which the drift/gap-rate plots sweep | 0.1 |

Each repeat mutates the sequence one random base at a time, up to
`--max-mutation-rate` of its length, running all repeats in parallel. Along
the way it samples the current best score at the 0%, 0.1%, 1%, 10%
mutation-rate checkpoints (fixed regardless of `--max-mutation-rate`, used
for the transition/best/second-best plots), and at 50 evenly spaced points
across the whole `--max-mutation-rate` range (used for the drift/gap-rate
plots), then aggregates everything across repeats into the JSON written to
stdout.

The output is always a JSON array, with one entry per simulated algorithm
(one entry for `--sketch lexichash`/`minhash`, two for `--sketch both`).
With `both`, LexicHash and MinHash are simulated independently (separate
random sequences), not on shared mutation trajectories.

## Plot

Requires `matplotlib` and `numpy`.

```sh
python3 plot.py < out.json
```
or directly
```sh
./lexichash-trace | python3 plot.py
```

It reads the JSON array from stdin and generates different plots:
- best-score distribution (`best`)
- second-best score distribution (`second`)
- score transitions (`transition`)
- drift of the selected *k*-mer from the original as mutations increase (`drift`)
- mutation rate recovered from the drift signal vs the true rate (`inverse`)
- gap between the empirical mean drift score and its theoretical prediction as sketch size grows, at the fixed `--converge-rate` (`gap-size`)
- gap between the mutation rate recovered from that empirical score and the true rate, as sketch size grows (`inverse-gap-size`)
- gap between the empirical mean drift score and its theoretical prediction across the whole mutation-rate range (`gap-rate`)
- gap between the mutation rate recovered from that empirical score and the true rate, across the whole mutation-rate range (`inverse-gap-rate`)

The `gap-size`/`inverse-gap-size` plots use batch means over disjoint groups
of repeats at the fixed rate to get several independent samples of the gap
at each sketch size. The `gap-rate`/`inverse-gap-rate` plots use the same
idea but with a fixed group count (10) at every mutation-rate checkpoint
instead, sweeping rate rather than sketch size.

When the input contains both algorithms (`--sketch both`), `best`/`second`/
`transition` stack LexicHash on top of MinHash in one figure (each row
labeled); `drift` stays as two separate figures (`drift_lexichash`,
`drift_minhash`); `inverse` and the `gap-*`/`inverse-gap-*` plots overlay
both algorithms on shared axes, colored blue (LexicHash) and red (MinHash),
including their fit-reference lines. Pass `--separate` to always get one
figure per algorithm instead (matching `drift`'s behavior), regardless of
which plots are selected.

`gap-rate`/`inverse-gap-rate` auto-detect where each algorithm's
theoretical score starts approaching its floor (the point past which a
relative gap tends to blow up, since it divides by an ever-smaller
denominator) and switch to a two-panel view once the swept range extends
past it: a zoomed linear view of the stable regime, and a log-scale view of
the full range.

| Flag | Meaning | Default |
|---|---|---|
| `-p, --plots <best second transition drift inverse gap-size inverse-gap-size gap-rate inverse-gap-rate>` | which plot(s) to generate | all |
| `-o, --out-dir <DIR>` | save plots here instead of showing them | show interactively |
| `-f, --format <pdf svg png>` | output format(s), only used with `-o` | `pdf` |
| `--compare-models` | also show lexichash's alt model / minhash's old model, for comparison | off |
| `--compare-approx` | also show lexichash's fast closed-form approximate inverse on `inverse`/`inverse-gap-size`/`inverse-gap-rate`, for comparison | off |
| `--separate` | one figure per algorithm instead of combining both, when input has both | off |

## Complete example

Run both LexicHash & MinHash with *k*=31 and 10K sketched *k*-mers, make comparative plots and save them as pdf & svg to `plots` folder:
```sh
./lexichash-trace --sketch both -k 31 --repeat 10000 | python3 plot.py -o plots -f pdf svg
```

Run MinHash and save only the `inverse` plot as png in `plots`:
```sh
./lexichash-trace --sketch minhash -k 31 --repeat 10000 | python3 plot.py -o plots -f png -p inverse
```
