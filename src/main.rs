use clap::Parser;
use packed_seq::{PackedSeqVec, SeqVec};
use rayon::prelude::*;
use serde::Serialize;

use lexichash_trace::KT;
use lexichash_trace::matrix::SquareMatrix;
use lexichash_trace::sim::{LexicHashSim, MinHashSim, Sim};

const RATES: [f64; 4] = [0.0, 0.001, 0.01, 0.1];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Length of the sequence
    #[arg(short, default_value_t = 100_000)]
    len: usize,
    /// K-mer size
    #[arg(short, default_value_t = 21)]
    k: usize,
    /// Number of repetitions
    #[arg(short, long, default_value_t = 1000)]
    repeat: usize,
    /// Number of threads [default: all]
    #[arg(short, long)]
    threads: Option<usize>,
    /// Use MinHash instead of LexicHash
    #[arg(long)]
    minhash: bool,
    /// Mutation rate at which to track convergence of the empirical mean
    /// drift score to its prediction, as repeats accumulate
    #[arg(short, long, default_value_t = 0.05)]
    converge_rate: f64,
    /// Mutation rate up to which original_drift/gap-rate plots sweep;
    /// score_histograms/band_transitions always stop at RATES.last() (10%)
    /// regardless of this value
    #[arg(short, long, default_value_t = 0.1)]
    max_mutation_rate: f64,
}

#[derive(Serialize)]
struct ScoreHistogram {
    rate: f64,
    counts: Vec<usize>,
}

#[derive(Serialize)]
struct BandTransitions {
    from_rate: f64,
    to_rate: f64,
    matrix: SquareMatrix<usize>,
}

#[derive(Serialize)]
struct SecondBestHistogram {
    rate: f64,
    counts: Vec<usize>,
    cardinality_sum: Vec<usize>,
}

/// Histograms of the drift score at this mutation-rate checkpoint, one per
/// disjoint group of repeats (`RATE_GROUPS` equal-sized groups by repeat
/// index). Since repeats are i.i.d., each group's histogram is an
/// independent sample of the same distribution, letting plot.py compute a
/// mean and std of the empirical score at this rate instead of only the
/// single aggregate (which is just these histograms summed).
#[derive(Serialize)]
struct OriginalDriftPoint {
    mutations: usize,
    group_counts: Vec<Vec<usize>>,
}

/// Sum and count of the fixed-rate drift score over one of 50 disjoint,
/// equal-sized (up to rounding) blocks of repeats, in whatever order the
/// parallel reduce combined them. Since repeats are i.i.d., these blocks can
/// be freely regrouped (e.g. into `q`-block windows) to get multiple
/// independent samples of the mean at sample size `q * repeats_per_block`,
/// rather than a single noisy running mean.
#[derive(Serialize)]
struct ConvergenceBlock {
    repeats: usize,
    sum: usize,
}

#[derive(Serialize)]
struct Convergence {
    rate: f64,
    mutations: usize,
    blocks: Vec<ConvergenceBlock>,
}

#[derive(Serialize)]
struct Output {
    algorithm: String,
    k: usize,
    len: usize,
    repeat: usize,
    score_histograms: Vec<ScoreHistogram>,
    band_transitions: Vec<BandTransitions>,
    second_best_histograms: Vec<SecondBestHistogram>,
    original_drift: Vec<OriginalDriftPoint>,
    convergence: Convergence,
}

fn main() {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let output = if args.minhash {
        run::<MinHashSim>(&args, "minhash")
    } else {
        run::<LexicHashSim>(&args, "lexichash")
    };

    serde_json::to_writer(std::io::stdout().lock(), &output).unwrap();
}

fn add_assign(dst: &mut [usize], src: &[usize]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

// number of disjoint repeat groups tracked at every mutation-rate
// checkpoint, for the gap-vs-rate plots (mirrors CONVERGENCE_BLOCKS's role
// for the gap-vs-sketch-size plots, but fixed rather than swept)
const RATE_GROUPS: usize = 10;

fn run<S: Sim + Send>(args: &Args, algorithm: &str) -> Output {
    let num_states = args.k + 1;
    let drift_buckets = S::drift_buckets(args.k);
    let checkpoints: Vec<usize> = RATES
        .iter()
        .map(|r| (r * args.len as f64).round() as usize)
        .collect();
    // last RATES checkpoint: score_histograms/band_transitions stop here
    let total_mutations = *checkpoints.last().unwrap();
    // 50 evenly spaced sample points across the (possibly longer) drift-tracking range
    let drift_max_mutations = (args.max_mutation_rate * args.len as f64).round() as usize;
    let sample_points: Vec<usize> = (0..=50).map(|p| p * drift_max_mutations / 50).collect();
    // nearest of those same 50 sample points to the requested convergence rate
    let converge_index =
        ((args.converge_rate / args.max_mutation_rate * 50.0).round() as usize).clamp(0, 50);

    let (
        score_histograms,
        band_matrices,
        second_best_hist,
        second_best_card_sum,
        og_score_hist,
        converge_scores,
    ) = (0..args.repeat)
        .into_par_iter()
        .map_init(
            || {
                let sim = unsafe { S::new_uninit() };
                let seq = PackedSeqVec::random(args.len);
                (sim, seq)
            },
            |(sim, seq), repeat_idx| {
                let mask = rand::random::<KT>();
                sim.reset(args.k, mask, seq.as_slice());

                let mut score_hist = vec![vec![0usize; num_states]; RATES.len()];
                let mut band_matrices =
                    vec![SquareMatrix::<usize>::new(num_states); RATES.len() - 1];
                let mut second_best_hist = vec![vec![0usize; num_states]; RATES.len()];
                let mut second_best_card_sum = vec![vec![0usize; num_states]; RATES.len()];
                let mut og_score_hist =
                    vec![vec![vec![0usize; drift_buckets]; RATE_GROUPS]; sample_points.len()];
                let group = repeat_idx * RATE_GROUPS / args.repeat;
                let mut next_sample = 0;
                let mut converge_score = 0usize;

                let record_second_best = |sim: &S, hist: &mut [usize], card_sum: &mut [usize]| {
                    let (score, cardinality) = sim.heap().second_bests();
                    hist[score as usize] += 1;
                    card_sum[score as usize] += cardinality;
                };

                let og = sim.heap().best_item();
                let mut i = sim.heap().best_score() as usize;
                score_hist[0][i] += 1;
                record_second_best(sim, &mut second_best_hist[0], &mut second_best_card_sum[0]);
                // sample point 0 (no mutations yet) never occurs inside the
                // loop below, since steps there start at 1: record it here,
                // trivially at the "identical to original" bucket
                if sample_points.first() == Some(&0) {
                    let drift = sim.drift_score(og);
                    og_score_hist[0][group][drift] += 1;
                    if converge_index == 0 {
                        converge_score = drift;
                    }
                    next_sample = 1;
                }
                // score_histograms/band_transitions only cover steps up to
                // total_mutations (RATES.last()); og_score_hist/convergence
                // keep going past that, up to drift_max_mutations
                let mut band = 0usize;
                for step in 1..=drift_max_mutations {
                    sim.mutate();
                    let j = sim.heap().best_score() as usize;
                    if step <= total_mutations {
                        while step > checkpoints[band + 1] {
                            band += 1;
                        }
                        band_matrices[band][(i, j)] += 1;
                        score_hist[band + 1][j] += 1;
                        record_second_best(
                            sim,
                            &mut second_best_hist[band + 1],
                            &mut second_best_card_sum[band + 1],
                        );
                    }
                    if next_sample < sample_points.len() && step == sample_points[next_sample] {
                        let drift = sim.drift_score(og);
                        og_score_hist[next_sample][group][drift] += 1;
                        if next_sample == converge_index {
                            converge_score = drift;
                        }
                        next_sample += 1;
                    }
                    i = j;
                }
                (
                    score_hist,
                    band_matrices,
                    second_best_hist,
                    second_best_card_sum,
                    og_score_hist,
                    vec![converge_score],
                )
            },
        )
        .reduce(
            || {
                (
                    vec![vec![0usize; num_states]; RATES.len()],
                    vec![SquareMatrix::<usize>::new(num_states); RATES.len() - 1],
                    vec![vec![0usize; num_states]; RATES.len()],
                    vec![vec![0usize; num_states]; RATES.len()],
                    vec![vec![vec![0usize; drift_buckets]; RATE_GROUPS]; sample_points.len()],
                    Vec::new(),
                )
            },
            |mut a, b| {
                for (ha, hb) in a.0.iter_mut().zip(b.0.iter()) {
                    add_assign(ha, hb);
                }
                for (ma, mb) in a.1.iter_mut().zip(b.1.iter()) {
                    add_assign(ma.as_slice_mut(), mb.as_slice());
                }
                for (ha, hb) in a.2.iter_mut().zip(b.2.iter()) {
                    add_assign(ha, hb);
                }
                for (ha, hb) in a.3.iter_mut().zip(b.3.iter()) {
                    add_assign(ha, hb);
                }
                for (pa, pb) in a.4.iter_mut().zip(b.4.iter()) {
                    for (ha, hb) in pa.iter_mut().zip(pb.iter()) {
                        add_assign(ha, hb);
                    }
                }
                a.5.extend(b.5);
                a
            },
        );

    let second_best_histograms: Vec<SecondBestHistogram> = RATES
        .iter()
        .zip(second_best_hist)
        .zip(second_best_card_sum)
        .map(|((&rate, counts), cardinality_sum)| SecondBestHistogram {
            rate,
            counts,
            cardinality_sum,
        })
        .collect();

    let original_drift: Vec<OriginalDriftPoint> = sample_points
        .iter()
        .zip(og_score_hist)
        .map(|(&mutations, group_counts)| OriginalDriftPoint {
            mutations,
            group_counts,
        })
        .collect();

    // cumulative sum over repeats, in whatever order the parallel reduce
    // combined them; since repeats are i.i.d. that order is as good as any
    // other for splitting into blocks below
    let mut cum_sum = Vec::with_capacity(converge_scores.len() + 1);
    cum_sum.push(0usize);
    let mut cumulative = 0usize;
    for &s in &converge_scores {
        cumulative += s;
        cum_sum.push(cumulative);
    }
    // split into disjoint blocks, same evenly spaced idea as `sample_points`
    // but along the repeat-count axis instead of the mutation-count axis;
    // plot.py regroups these into windows of >=10 blocks for each plotted
    // point, so CONVERGENCE_BLOCKS needs to be 10x the number of points it
    // wants along that axis
    const CONVERGENCE_BLOCKS: usize = 500;
    let edges: Vec<usize> = (0..=CONVERGENCE_BLOCKS)
        .map(|p| p * args.repeat / CONVERGENCE_BLOCKS)
        .collect();
    let convergence = Convergence {
        rate: sample_points[converge_index] as f64 / args.len as f64,
        mutations: sample_points[converge_index],
        blocks: edges
            .windows(2)
            .map(|w| ConvergenceBlock {
                repeats: w[1] - w[0],
                sum: cum_sum[w[1]] - cum_sum[w[0]],
            })
            .collect(),
    };

    Output {
        algorithm: algorithm.to_string(),
        k: args.k,
        len: args.len,
        repeat: args.repeat,
        score_histograms: RATES
            .iter()
            .zip(score_histograms)
            .map(|(&rate, counts)| ScoreHistogram { rate, counts })
            .collect(),
        band_transitions: RATES
            .windows(2)
            .zip(band_matrices)
            .map(|(w, matrix)| BandTransitions {
                from_rate: w[0],
                to_rate: w[1],
                matrix,
            })
            .collect(),
        second_best_histograms,
        original_drift,
        convergence,
    }
}
