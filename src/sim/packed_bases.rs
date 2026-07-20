/// Extract the k-mer starting at `index` from `bases`.
///
/// `bases` must use the same 2-bit-packed layout as `packed_seq`: base `i`
/// lives in bits `[2*(i%4), 2*(i%4)+2)` of byte `i/4`, padded with 16
/// trailing zero bytes so this can always safely read a full `u128` at any
/// valid offset.
#[inline(always)]
pub(super) fn get_kmer(bases: &[u8], index: usize, k: usize) -> u64 {
    let byte_idx = index / 4;
    let bit_offset = 2 * (index % 4);
    let chunk = u128::from_le_bytes(bases[byte_idx..byte_idx + 16].try_into().unwrap());
    let mask = (1u128 << (2 * k)) - 1;
    ((chunk >> bit_offset) & mask) as u64
}

/// Apply a substitution to the base at `index` (see `get_kmer` for the packed layout).
#[inline(always)]
pub(super) fn xor_base(bases: &mut [u8], index: usize, xor: u8) {
    bases[index / 4] ^= xor << (2 * (index % 4));
}

#[cfg(test)]
mod tests {
    use super::*;
    use packed_seq::{PackedSeqVec, Seq, SeqVec};

    /// `get_kmer` reads directly from a raw packed byte buffer instead of
    /// going through `packed_seq`'s `Seq::slice(..).as_u64()`; both must
    /// agree bit-for-bit, since LexicHash relies on exact prefix comparisons.
    #[test]
    fn get_kmer_matches_packed_seq_as_u64() {
        let len = 500;
        let seq = PackedSeqVec::random(len);
        let mut bases = seq.as_slice().to_vec().into_raw();
        bases.resize(bases.len() + 16, 0);

        for k in [1, 5, 21, 32] {
            for i in 0..=(len - k) {
                let expected = seq.as_slice().slice(i..i + k).as_u64();
                assert_eq!(get_kmer(&bases, i, k), expected, "i={i} k={k}");
            }
        }
    }
}
