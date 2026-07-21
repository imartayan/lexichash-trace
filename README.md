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
| `--minhash` | use MinHash instead of LexicHash | off (LexicHash) |

Each repeat mutates the sequence one random base at a time, up to 10% of its
length, running all repeats in parallel. Along the way it samples the current
best score at the 0%, 0.1%, 1%, 10% mutation-rate checkpoints (and 50 evenly
spaced points across the whole range for the drift plot), then aggregates
everything across repeats into the JSON written to stdout.

## Plot

Requires `matplotlib` and `numpy`.

```sh
python3 plot.py < out.json
```
or directly
```sh
./lexichash-trace | python3 plot.py
```

It reads the JSON from stdin and generates five plots:
- best-score distribution (`best`)
- second-best score distribution (`second`)
- score transitions (`transition`)
- drift of the selected *k*-mer from the original as mutations increase (`drift`)
- mutation rate recovered from the drift signal vs the true rate (`inverse`)

| Flag | Meaning | Default |
|---|---|---|
| `-p, --plots <best second transition drift inverse>` | which plot(s) to generate | all five |
| `-o, --out-dir <DIR>` | save plots here instead of showing them | show interactively |
| `-f, --format <pdf svg png>` | output format(s), only used with `-o` | `pdf` |

## Example

Run MinHash and save two plots as svg in the `out` directory:
```sh
./lexichash-trace --minhash -k 25 --repeat 500 | python3 plot.py -o out -f svg -p best drift
```
