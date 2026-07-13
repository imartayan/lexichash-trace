use crate::matrix::SquareMatrix;

#[derive(Debug, Clone)]
pub struct TransitionTrace {
    num_states: usize,
    total_count: usize,
    count: SquareMatrix<usize>,
}

impl TransitionTrace {
    #[inline(always)]
    pub fn new(num_states: usize) -> Self {
        assert!(num_states > 0);
        Self::from_count_matrix(0, SquareMatrix::new(num_states))
    }

    #[inline(always)]
    pub fn from_count_matrix(total_count: usize, count: SquareMatrix<usize>) -> Self {
        Self {
            num_states: count.size(),
            total_count,
            count,
        }
    }

    #[inline(always)]
    pub fn into_count_matrix(self) -> SquareMatrix<usize> {
        self.count
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.count.as_slice_mut().fill(0);
        self.total_count = 0;
    }

    #[inline(always)]
    pub fn count_matrix(&self) -> &SquareMatrix<usize> {
        &self.count
    }

    #[inline(always)]
    pub fn increment(&mut self, i: usize, j: usize) {
        self.count[(i, j)] += 1;
        self.total_count += 1;
    }

    #[inline(always)]
    pub fn count(&self, i: usize, j: usize) -> usize {
        self.count[(i, j)]
    }

    #[inline(always)]
    pub fn count_outgoing(&self, i: usize) -> usize {
        self.count.row(i).sum()
    }

    #[inline(always)]
    pub fn count_incoming(&self, j: usize) -> usize {
        self.count.column(j).sum()
    }

    #[inline(always)]
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    #[inline(always)]
    pub fn transition_matrix(&self) -> SquareMatrix<f64> {
        let mut matrix = SquareMatrix::new(self.num_states);
        self.transition_matrix_in(&mut matrix);
        matrix
    }

    #[inline(always)]
    pub fn transition_matrix_in(&self, matrix: &mut SquareMatrix<f64>) {
        assert_eq!(matrix.size(), self.num_states);
        for v in matrix.as_slice_mut() {
            *v = f64::NAN;
        }
        let mut i = 0;
        let mut outgoing = self.count_outgoing(i) as f64;
        let mut j = if outgoing == 0. { self.num_states } else { 0 };
        loop {
            while j == self.num_states {
                i += 1;
                if i == self.num_states {
                    return;
                }
                outgoing = self.count_outgoing(i) as f64;
                j = if outgoing == 0. { self.num_states } else { 0 };
            }
            matrix[(i, j)] = self.count[(i, j)] as f64 / outgoing;
            j += 1;
        }
    }
}
