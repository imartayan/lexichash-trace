use crate::KT;
use crate::heap::LexicHeap;
use packed_seq::PackedSeq;

mod lexichash;
mod minhash;
mod packed_bases;

pub use lexichash::LexicHashSim;
pub use minhash::MinHashSim;

pub trait Sim: Sized {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn new_uninit() -> Self;
    fn reset(&mut self, k: usize, seed: KT, seq: PackedSeq);
    fn k(&self) -> usize;
    fn seq_len(&self) -> usize;
    fn heap(&self) -> &LexicHeap;
    fn heap_mut(&mut self) -> &mut LexicHeap;
    fn mutate(&mut self);
    fn mutate_at(&mut self, pos: usize);
    /// Bucket index for "current best vs `original`", used for the drift-from-original plot.
    fn drift_score(&self, original: KT) -> usize;
    /// Number of buckets `drift_score` can return, for a given `k`.
    fn drift_buckets(k: usize) -> usize;

    fn new(k: usize, seed: KT, seq: PackedSeq) -> Self {
        let mut res = unsafe { Self::new_uninit() };
        res.reset(k, seed, seq);
        res
    }
}
