use crate::KT;
use core::ops::Range;

#[derive(Debug, Clone)]
pub struct LexicHeap {
    /// (score, item, index)
    best: (u8, KT, u32),
    /// index -> (score, offset in level)
    coord: Vec<(u8, u32)>,
    /// one level per prefix score,
    /// offset -> (item, index)
    levels: [Vec<(KT, u32)>; 32],
}

impl Default for LexicHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl LexicHeap {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            best: (0, KT::MAX, 0),
            coord: Default::default(),
            levels: Default::default(),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.best = (0, KT::MAX, 0);
        self.coord.clear();
        self.levels.iter_mut().for_each(|level| level.clear());
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.coord.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reserve(&mut self, additional: usize) {
        self.coord.reserve(additional);
        let mut n = additional;
        let mut i = 0;
        while n > 0 {
            let m = n >> 2;
            self.levels[i].reserve((n - m).wrapping_mul(11).div_ceil(10));
            n = m;
            i += 1;
        }
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
    fn _pop(&mut self, index: usize) -> KT {
        let (i, j) = self.coord[index];
        let level = self.levels.get_mut(i as usize).unwrap();
        let (x, _) = level.swap_remove(j as usize);
        if let Some(&(_, swap_index)) = level.get(j as usize) {
            self.coord.swap(index, swap_index as usize);
        }
        // we delay the rescan of best values
        x
    }

    #[inline(always)]
    fn _set(&mut self, index: usize, x: KT) {
        let i = Self::prefix_score(x);
        let level = self.levels.get_mut(i as usize).unwrap();
        self.coord[index] = (i, level.len() as u32);
        level.push((x, index as u32));
        if x < self.best_item() {
            self.best = (i, x, index as u32)
        }
    }

    #[inline(always)]
    pub fn push(&mut self, x: KT) {
        let index = self.len();
        self.coord.push((0, 0));
        self._set(index, x);
    }

    #[inline(always)]
    pub fn update_range(&mut self, range: Range<usize>, first_xor: KT) {
        let prev_best_index = self.best_index();
        for (delta, index) in range.clone().enumerate() {
            let x = self._pop(index);
            let x_new = x ^ (first_xor >> (2 * delta));
            self._set(index, x_new);
        }
        self._rescan_best_if_stale(range, prev_best_index);
    }

    #[inline(always)]
    pub fn update_range_with(&mut self, range: Range<usize>, values: &[KT]) {
        debug_assert_eq!(range.len(), values.len());
        let prev_best_index = self.best_index();
        for (index, &x_new) in range.clone().zip(values) {
            self._pop(index);
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
