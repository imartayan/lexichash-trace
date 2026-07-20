use crate::KT;
use branches::unlikely;
use core::ops::Range;
use rustc_hash::FxHashMap;

/// Offset-within-a-level type.
type OT = u16;

#[derive(Debug, Clone)]
pub struct LexicHeap {
    /// total number of items ever set (tracked or not)
    count: usize,
    /// scores below threshold are not stored
    threshold: u8,
    /// `score(x) >= threshold` iff `x <= untracked_cutoff`
    untracked_cutoff: KT,
    /// (score, item, index)
    best: (u8, KT, u32),
    /// index -> (score, offset in level), absent iff score < threshold
    coord: FxHashMap<u32, (u8, OT)>,
    /// one level per prefix score,
    /// only populated for score >= threshold,
    /// offset -> (item, index)
    levels: [Vec<(KT, u32)>; 33],
}

impl Default for LexicHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl LexicHeap {
    #[inline(always)]
    pub fn new() -> Self {
        Self::new_with_threshold(0)
    }

    #[inline(always)]
    pub fn new_with_threshold(threshold: u8) -> Self {
        Self {
            count: 0,
            best: (0, KT::MAX, 0),
            coord: FxHashMap::default(),
            levels: core::array::from_fn(|_| Vec::new()),
            threshold,
            untracked_cutoff: Self::untracked_cutoff_for(threshold),
        }
    }

    #[inline(always)]
    pub fn new_with_best_threshold(k: usize, seq_len: usize, repeat: usize) -> Self {
        Self::new_with_threshold(Self::best_threshold(k, seq_len, repeat))
    }

    #[inline(always)]
    pub fn set_threshold(&mut self, threshold: u8) {
        self.threshold = threshold;
        self.untracked_cutoff = Self::untracked_cutoff_for(threshold);
    }

    #[inline(always)]
    fn untracked_cutoff_for(threshold: u8) -> KT {
        KT::MAX.checked_shr(2 * threshold as u32).unwrap_or(0)
    }

    #[inline(always)]
    pub fn choose_best_threshold(&mut self, k: usize, seq_len: usize, repeat: usize) -> u8 {
        let best_threshold = Self::best_threshold(k, seq_len, repeat);
        self.set_threshold(best_threshold);
        best_threshold
    }

    #[inline(always)]
    fn best_threshold(k: usize, seq_len: usize, repeat: usize) -> u8 {
        const SAFETY_MARGIN: usize = 10_000;
        let num_kmers = seq_len - (k - 1);
        let num_mut = k * num_kmers * repeat;
        let target_ln_prob = -((num_mut * SAFETY_MARGIN) as f64).ln();
        let target_ratio = target_ln_prob / (num_kmers as f64);
        let threshold_f = -0.5 * (-target_ratio.exp_m1()).log2();
        threshold_f.floor().max(0.) as u8
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.count = 0;
        self.best = (0, KT::MAX, 0);
        self.coord.clear();
        self.levels.iter_mut().for_each(|level| level.clear());
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn reserve(&mut self, additional: usize) {
        let mut n = additional;
        let mut i = 0;
        let mut tracked_estimate = 0;
        while n > 0 {
            let m = n >> 2;
            if i >= self.threshold as usize {
                let level_capacity = (n - m).wrapping_mul(11).div_ceil(10);
                self.levels[i].reserve(level_capacity);
                tracked_estimate += level_capacity;
            }
            n = m;
            i += 1;
        }
        // reserve at 2x the expected tracked count
        self.coord.reserve(tracked_estimate * 2);
    }

    #[inline(always)]
    pub fn prefix_score(x: KT) -> u8 {
        x.leading_zeros() as u8 / 2
    }

    #[inline(always)]
    pub fn best_score(&self) -> u8 {
        self.best.0
    }

    #[inline(always)]
    pub fn best_item(&self) -> KT {
        self.best.1
    }

    #[inline(always)]
    pub fn best_index(&self) -> u32 {
        self.best.2
    }

    #[inline(always)]
    pub fn second_bests(&self) -> (u8, usize) {
        let mut i = self.best_score();
        let mut level = self.levels.get(i as usize).unwrap();
        if level.len() > 1 {
            (i, level.len() - 1)
        } else {
            i -= 1;
            level = self.levels.get(i as usize).unwrap();
            while level.is_empty() {
                i -= 1;
                level = self.levels.get(i as usize).unwrap();
            }
            (i, level.len())
        }
    }

    #[inline(always)]
    fn _pop(&mut self, index: usize) {
        if let Some((i, offset)) = self.coord.remove(&(index as u32)) {
            let level = &mut self.levels[i as usize];
            level.swap_remove(offset as usize);
            if let Some(&(_, swap_index)) = level.get(offset as usize) {
                self.coord.insert(swap_index, (i, offset));
            }
        }
        // we delay the rescan of best values
    }

    #[inline(always)]
    fn _set(&mut self, index: usize, x: KT) {
        if unlikely(x <= self.untracked_cutoff) {
            self._set_tracked(index, x);
        }
    }

    /// Tracked-item half of `_set`, for callers that already know `x <= untracked_cutoff`.
    #[inline(always)]
    fn _set_tracked(&mut self, index: usize, x: KT) {
        let i = Self::prefix_score(x);
        let level = &mut self.levels[i as usize];
        let offset = OT::try_from(level.len()).expect("level exceeds OT capacity");
        level.push((x, index as u32));
        self.coord.insert(index as u32, (i, offset));
        if unlikely(x < self.best_item()) {
            self.best = (i, x, index as u32);
        }
    }

    #[inline(always)]
    pub fn push(&mut self, x: KT) {
        let index = self.count;
        self.count += 1;
        self._set(index, x);
    }

    /// Bulk-populates the heap with `count` items computed via `f`.
    pub fn init_with_fn(&mut self, count: usize, mut f: impl FnMut(usize) -> KT) {
        self.reserve(count);
        self.count = count;
        for index in 0..count {
            let x = f(index);
            if unlikely(x <= self.untracked_cutoff) {
                self._set_tracked(index, x);
            }
        }
    }

    /// Updates `range`, computing each new value via `f`.
    #[inline(always)]
    pub fn update_range_with_fn(&mut self, range: Range<usize>, mut f: impl FnMut(usize) -> KT) {
        let prev_best_index = self.best_index();
        for index in range.clone() {
            self._pop(index);
            let x_new = f(index);
            self._set(index, x_new);
        }
        self._rescan_best_if_stale(range, prev_best_index);
    }

    #[inline(always)]
    fn _rescan_best_if_stale(&mut self, range: Range<usize>, prev_best_index: u32) {
        // rescan best if we altered it
        if range.contains(&(prev_best_index as usize)) && prev_best_index == self.best_index() {
            let mut i = self.best_score();
            let mut level = self.levels.get(i as usize).unwrap();
            while level.is_empty() {
                i -= 1;
                level = self.levels.get(i as usize).unwrap();
            }
            let min = level.iter().min().unwrap();
            self.best = (i, min.0, min.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    /// Drives a threshold=0 heap and an auto-thresholded one through the
    /// same updates and checks their observable state always matches.
    #[test]
    fn threshold_matches_baseline() {
        let mut rng = SmallRng::seed_from_u64(42);
        let n = 5_000;
        let k = 21;

        let mut baseline = LexicHeap::new_with_threshold(0);
        let mut optimized = LexicHeap::new();
        let best_threshold = optimized.choose_best_threshold(k, n, 1);
        dbg!(best_threshold);
        baseline.reserve(n);

        let initial: Vec<KT> = (0..n).map(|_| rng.random()).collect();
        for &x in &initial {
            baseline.push(x);
        }
        optimized.init_with_fn(initial.len(), |i| initial[i]);
        assert_heaps_match(&baseline, &optimized);

        for _ in 0..2_000 {
            let pos = rng.random_range(0..n);
            let start = pos.saturating_sub(k - 1);
            let stop = (pos + 1).min(n - (k - 1));
            let values: Vec<KT> = (start..stop).map(|_| rng.random()).collect();
            baseline.update_range_with_fn(start..stop, |i| values[i - start]);
            optimized.update_range_with_fn(start..stop, |i| values[i - start]);
            assert_heaps_match(&baseline, &optimized);
        }
    }

    fn assert_heaps_match(baseline: &LexicHeap, optimized: &LexicHeap) {
        assert_eq!(baseline.best_score(), optimized.best_score());
        assert_eq!(baseline.best_item(), optimized.best_item());
        assert_eq!(baseline.best_index(), optimized.best_index());
        assert_eq!(baseline.second_bests(), optimized.second_bests());
    }
}
