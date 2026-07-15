use super::Sim;
use crate::KT;
use crate::heap::LexicHeap;
use packed_seq::{PackedSeq, Seq};
use rand::{RngExt, rngs::SmallRng};

#[derive(Debug, Clone)]
pub struct LexicHashSim {
    k: usize,
    seq_len: usize,
    heap: LexicHeap,
    rng: SmallRng,
}

impl Sim for LexicHashSim {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn new_uninit() -> Self {
        let rng = rand::make_rng::<SmallRng>();
        Self {
            k: Default::default(),
            seq_len: Default::default(),
            heap: Default::default(),
            rng,
        }
    }

    fn reset(&mut self, k: usize, mask: KT, seq: PackedSeq) {
        assert!(k > 0);
        assert!(k <= 32);
        assert!(k <= seq.len());
        self.k = k;
        self.seq_len = seq.len();
        let num_kmers = seq.len() - (k - 1);
        self.heap.clear();
        self.heap.reserve(num_kmers);
        let shift = KT::BITS as usize - 2 * k;
        let mask = mask | ((1 << shift) - 1);
        for i in 0..num_kmers {
            let kmer = seq.slice(i..(i + k)).as_u64() << shift;
            self.heap.push(kmer ^ mask);
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
        let delta = (self.k - 1).saturating_sub(pos);
        let first_xor = self.rng.random_range(0..4) << (KT::BITS as usize - 2 * (delta + 1));
        self.heap.update_range(start..stop, first_xor);
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
