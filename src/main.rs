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

#[derive(Serialize)]
struct OriginalDriftPoint {
    mutations: usize,
    counts: Vec<usize>,
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

fn run<S: Sim + Send>(args: &Args, algorithm: &str) -> Output {
    let num_states = args.k + 1;
    let drift_buckets = S::drift_buckets(args.k);
    let checkpoints: Vec<usize> = RATES
        .iter()
        .map(|r| (r * args.len as f64).round() as usize)
        .collect();
    // 50 evenly spaced sample points across the whole mutation range
    let total_mutations = *checkpoints.last().unwrap();
    let sample_points: Vec<usize> = (0..=50).map(|p| p * total_mutations / 50).collect();

    let (score_histograms, band_matrices, second_best_hist, second_best_card_sum, og_score_hist) =
        (0..args.repeat)
            .into_par_iter()
            .map_init(
                || {
                    let sim = unsafe { S::new_uninit() };
                    let seq = PackedSeqVec::random(args.len);
                    (sim, seq)
                },
                |(sim, seq), _| {
                    let mask = rand::random::<KT>() << (KT::BITS as usize - 2 * args.k);
                    sim.reset(args.k, mask, seq.as_slice());

                    let mut score_hist = vec![vec![0usize; num_states]; RATES.len()];
                    let mut band_matrices =
                        vec![SquareMatrix::<usize>::new(num_states); RATES.len() - 1];
                    let mut second_best_hist = vec![vec![0usize; num_states]; RATES.len()];
                    let mut second_best_card_sum = vec![vec![0usize; num_states]; RATES.len()];
                    let mut og_score_hist = vec![vec![0usize; drift_buckets]; sample_points.len()];
                    let mut next_sample = 0;

                    let record_second_best =
                        |sim: &S, hist: &mut [usize], card_sum: &mut [usize]| {
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
                        og_score_hist[0][sim.drift_score(og)] += 1;
                        next_sample = 1;
                    }
                    for band in 0..RATES.len() - 1 {
                        for step in checkpoints[band] + 1..=checkpoints[band + 1] {
                            sim.mutate();
                            let j = sim.heap().best_score() as usize;
                            band_matrices[band][(i, j)] += 1;
                            score_hist[band + 1][j] += 1;
                            record_second_best(
                                sim,
                                &mut second_best_hist[band + 1],
                                &mut second_best_card_sum[band + 1],
                            );
                            if next_sample < sample_points.len()
                                && step == sample_points[next_sample]
                            {
                                og_score_hist[next_sample][sim.drift_score(og)] += 1;
                                next_sample += 1;
                            }
                            i = j;
                        }
                    }
                    (
                        score_hist,
                        band_matrices,
                        second_best_hist,
                        second_best_card_sum,
                        og_score_hist,
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
                        vec![vec![0usize; drift_buckets]; sample_points.len()],
                    )
                },
                |mut a, b| {
                    for (ha, hb) in a.0.iter_mut().zip(b.0.iter()) {
                        for (ca, cb) in ha.iter_mut().zip(hb.iter()) {
                            *ca += cb;
                        }
                    }
                    for (ma, mb) in a.1.iter_mut().zip(b.1.iter()) {
                        for (va, vb) in ma.as_slice_mut().iter_mut().zip(mb.as_slice().iter()) {
                            *va += vb;
                        }
                    }
                    for (ha, hb) in a.2.iter_mut().zip(b.2.iter()) {
                        for (ca, cb) in ha.iter_mut().zip(hb.iter()) {
                            *ca += cb;
                        }
                    }
                    for (ha, hb) in a.3.iter_mut().zip(b.3.iter()) {
                        for (ca, cb) in ha.iter_mut().zip(hb.iter()) {
                            *ca += cb;
                        }
                    }
                    for (ha, hb) in a.4.iter_mut().zip(b.4.iter()) {
                        for (ca, cb) in ha.iter_mut().zip(hb.iter()) {
                            *ca += cb;
                        }
                    }
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
        .map(|(&mutations, counts)| OriginalDriftPoint { mutations, counts })
        .collect();

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
    }
}
