use super::Sim;
use super::packed_bases::{get_kmer, xor_base};
use crate::KT;
use crate::heap::LexicHeap;
use packed_seq::{PackedSeq, Seq, SeqVec};
use rand::{RngExt, rngs::SmallRng, seq::SliceRandom};

#[derive(Debug, Clone)]
pub struct LexicHashSim {
    k: usize,
    seq_len: usize,
    mask: KT,
    /// `KT::BITS - 2*k`: left-shift that aligns a k-mer's packed bits to the
    /// top of `KT`, so prefix comparisons order by shared-prefix length
    shift: usize,
    /// 2-bit-packed bases, same layout as `packed_seq`: base `i` lives in bits
    /// `[2*(i%4), 2*(i%4)+2)` of byte `i/4`, padded with 16 trailing zero bytes
    /// so `get_kmer` can always safely read a full `u128` at any valid offset.
    bases: Vec<u8>,
    heap: LexicHeap,
    mut_pos: Vec<u32>,
    rng: SmallRng,
}

impl Sim for LexicHashSim {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn new_uninit() -> Self {
        let rng = rand::make_rng::<SmallRng>();
        Self {
            k: Default::default(),
            seq_len: Default::default(),
            mask: Default::default(),
            shift: Default::default(),
            bases: Default::default(),
            heap: Default::default(),
            mut_pos: Default::default(),
            rng,
        }
    }

    fn reset(&mut self, k: usize, mask: KT, seq: PackedSeq) {
        assert!(k > 0);
        assert!(k <= 32);
        assert!(k <= seq.len());
        self.k = k;
        self.seq_len = seq.len();
        self.shift = KT::BITS as usize - 2 * k;
        self.mask = mask | ((1 << self.shift) - 1);
        self.bases = seq.to_vec().into_raw();
        self.bases.resize(self.bases.len() + 16, 0);
        self.mut_pos.clear();
        self.mut_pos.extend(0..seq.len() as u32);
        self.mut_pos.shuffle(&mut self.rng);
        let num_kmers = seq.len() - (k - 1);
        self.heap.clear();
        self.heap.choose_best_threshold(k, seq.len(), 10_000);
        let (bases, shift, mask) = (&self.bases, self.shift, self.mask);
        self.heap
            .init_with_fn(num_kmers, |i| (get_kmer(bases, i, k) << shift) ^ mask);
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
    fn heap_mut(&mut self) -> &mut LexicHeap {
        &mut self.heap
    }

    #[inline(always)]
    fn mutate(&mut self) {
        let pos = self.mut_pos.pop().unwrap_or(0) as usize;
        self.mutate_at(pos);
    }

    #[inline(always)]
    fn mutate_at(&mut self, pos: usize) {
        let start = pos.saturating_sub(self.k - 1);
        let stop = (pos + 1).min(self.seq_len - (self.k - 1));
        let xor = self.rng.random_range(1..4);
        xor_base(&mut self.bases, pos, xor);
        let (bases, k, shift, mask) = (&self.bases, self.k, self.shift, self.mask);
        self.heap
            .update_range_with_fn(start..stop, |i| (get_kmer(bases, i, k) << shift) ^ mask);
    }

    #[inline(always)]
    fn drift_score(&self, original: KT) -> usize {
        let xor = original ^ self.heap().best_item();
        (LexicHeap::prefix_score(xor) as usize).min(self.k)
    }

    #[inline(always)]
    fn drift_buckets(k: usize) -> usize {
        k + 1
    }
}
