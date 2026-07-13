use super::Sim;
use crate::KT;
use crate::heap::LexicHeap;
use packed_seq::{PackedSeq, Seq, SeqVec};
use rand::{RngExt, rngs::SmallRng};
use rustc_hash::FxHasher;
use std::hash::Hasher;

#[derive(Debug, Clone)]
pub struct MinHashSim {
    k: usize,
    seq_len: usize,
    seed: KT,
    /// 2-bit-packed bases, same layout as `packed_seq`: base `i` lives in bits
    /// `[2*(i%4), 2*(i%4)+2)` of byte `i/4`, padded with 16 trailing zero bytes
    /// so `get_kmer` can always safely read a full `u128` at any valid offset.
    bases: Vec<u8>,
    /// reused across `mutate_at` calls to avoid a per-mutation allocation
    scratch: Vec<KT>,
    heap: LexicHeap,
    rng: SmallRng,
}

impl Sim for MinHashSim {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn new_uninit() -> Self {
        let rng = rand::make_rng::<SmallRng>();
        Self {
            k: Default::default(),
            seq_len: Default::default(),
            seed: Default::default(),
            bases: Default::default(),
            scratch: Default::default(),
            heap: Default::default(),
            rng,
        }
    }

    fn reset(&mut self, k: usize, seed: KT, seq: PackedSeq) {
        assert!(k > 0);
        assert!(k <= 32);
        assert!(k <= seq.len());
        self.k = k;
        self.seq_len = seq.len();
        self.seed = seed;
        self.bases = seq.to_vec().into_raw();
        self.bases.resize(self.bases.len() + 16, 0);
        let num_kmers = seq.len() - (k - 1);
        self.heap.clear();
        self.heap.reserve(num_kmers);
        for i in 0..num_kmers {
            let kmer = get_kmer(&self.bases, i, k);
            self.heap.push(hash(kmer, seed));
        }
    }

    #[inline(always)]
    fn k(&self) -> usize {
        self.k
    }

    #[inline(always)]
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    #[inline(always)]
    fn heap(&self) -> &LexicHeap {
        &self.heap
    }

    #[inline(always)]
    fn mutate(&mut self) {
        let pos = self.rng.random_range(0..self.seq_len());
        self.mutate_at(pos);
    }

    #[inline(always)]
    fn mutate_at(&mut self, pos: usize) {
        let start = pos.saturating_sub(self.k - 1);
        let stop = (pos + 1).min(self.seq_len - (self.k - 1));
        let delta = self.rng.random_range(0u8..4);
        xor_base(&mut self.bases, pos, delta);
        self.scratch.clear();
        for i in start..stop {
            let kmer = get_kmer(&self.bases, i, self.k);
            self.scratch.push(hash(kmer, self.seed));
        }
        self.heap.update_range_with(start..stop, &self.scratch);
    }

    #[inline(always)]
    fn drift_score(&self, original: KT) -> usize {
        (self.heap().best_item() == original) as usize
    }

    #[inline(always)]
    fn drift_buckets(_k: usize) -> usize {
        2
    }
}

#[inline(always)]
fn get_kmer(bases: &[u8], index: usize, k: usize) -> u64 {
    let byte_idx = index / 4;
    let bit_offset = 2 * (index % 4);
    let chunk = u128::from_le_bytes(bases[byte_idx..byte_idx + 16].try_into().unwrap());
    let mask = (1u128 << (2 * k)) - 1;
    ((chunk >> bit_offset) & mask) as u64
}

#[inline(always)]
fn xor_base(bases: &mut [u8], index: usize, delta: u8) {
    bases[index / 4] ^= delta << (2 * (index % 4));
}

#[inline(always)]
fn hash(x: KT, seed: KT) -> KT {
    let mut hasher = FxHasher::default();
    hasher.write_u64(x ^ seed);
    hasher.finish()
}
