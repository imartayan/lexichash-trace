use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, serde::Serialize)]
pub struct SquareMatrix<T> {
    size: usize,
    data: Vec<T>,
}

impl<T> SquareMatrix<T> {
    pub fn new(size: usize) -> Self
    where
        T: Default,
    {
        Self {
            size,
            data: (0..size * size).map(|_| T::default()).collect(),
        }
    }

    #[inline(always)]
    pub fn from_vec(size: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), size * size);
        Self { size, data }
    }

    #[inline(always)]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline(always)]
    pub fn row(&self, i: usize) -> std::slice::Iter<'_, T> {
        self.data[i * self.size..(i + 1) * self.size].iter()
    }

    #[inline(always)]
    pub fn column(&self, j: usize) -> impl Iterator<Item = &T> + '_ {
        self.data.iter().skip(j).step_by(self.size)
    }
}

impl<T: Display> Display for SquareMatrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.size {
            if i > 0 {
                writeln!(f)?;
            }
            for j in 0..self.size {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", self[(i, j)])?;
            }
        }
        Ok(())
    }
}

impl<T> std::ops::Index<(usize, usize)> for SquareMatrix<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.size + j]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for SquareMatrix<T> {
    #[inline(always)]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.size + j]
    }
}
