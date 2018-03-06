//! Misc numerical stuff and other handy functions.

use num;

/// fcopysign:
///
/// If z1 and z2 have the same sign, then return z1
/// Else return z1 with negated sign.
pub fn copysign<T>(f1: T, f2: T) -> T
where
    T: num::Float,
{
    // This probably even works as intended for NaN's, since
    // is_sign_positive() and such just look at the sign bit.
    if f1.is_sign_positive() && f2.is_sign_positive()
        || f1.is_sign_negative() && f2.is_sign_negative()
    {
        f1
    } else {
        f1.neg()
    }
}

/// Truncates the float into the given integer type.
pub fn truncate_to_int<From, To>(f1: From) -> To
where
    From: num::Float,
    To: num::NumCast,
{
    // TODO: Needs more correctness checking!  Inf, NaN, etc.
    // Also not sure NumCast is the right thing to use here but it seems to work.
    To::from(f1.trunc()).unwrap()
}

/// Rounds the float into the given integer type.
pub fn round_to_int<From, To>(f1: From) -> To
where
    From: num::Float,
    To: num::NumCast,
{
    // TODO: Needs more correctness checking!  Inf, NaN, etc.
    // Also needs to verify that the rounding behavior is correct.
    // Also not sure NumCast is the right thing to use here but it seems to work.
    To::from(f1.round()).unwrap()
}

/// A convenience function for doing a bitwise conversion of a u32 to an f32.
/// We don't just use `f32::from_bits()` so that we can insert extra checking
/// to filter out signaling NaN's, which are disallowed by wasm.
pub fn u32_to_f32(i: u32) -> f32 {
    // TODO: Insert extra checking to filter out signaling NaN's, which are
    // disallowed by wasm.  :-P
    // BUGGO: from_bits is technically incorrect because a signaling NaN
    // *may* slip through from_bits(), and WebAssembly currently
    // does not support signaling NaN's.
    // See https://webassembly.github.io/spec/core/exec/numerics.html#floating-point-operations
    f32::from_bits(i)
}

/// A convenience function for doing a bitwise conversion of a u64 to an f64
/// We don't just use `f64::from_bits()` so that we can insert extra checking
/// to filter out signaling NaN's, which are disallowed by wasm.
pub fn u64_to_f64(i: u64) -> f64 {
    // TODO: Insert extra checking to filter out signaling NaN's, which are
    // disallowed by wasm.  :-P
    f64::from_bits(i)
}

/// Returns whether or not the float is a signaling NaN.
/// A signaling NaN has a format like:
/// `s111 1111 1nxx xxxx xxxx xxxx xxxx xxxx`
/// where the `x`'s represent a non-zero number (zero
/// would be infinity) and `n` is 0.
/// The sign bit `s` may be anything.
///
/// On some old-fashioned platforms (PA-RISC, some MIPS)
/// a signaling NaN is marked by `n=1`, but the 2008 revision of
/// IEEE754 defines it to be `n=0`.
pub fn f32_is_signaling_nan(f: f32) -> bool {
    let uf: u32 = f.to_bits();
    let signal_bit = 0b0000_0000_0100_0000_0000_0000_0000_0000;
    let signal_bit = 1 << 22;
    // signaling nan := is NAN and signal bit is clear
    let signal_bit_clear = (uf & signal_bit) == 0;
    f32::is_nan(f) && signal_bit_clear
}

/// Same as `f32_is_signaling_nan()` for `f64`'s.
/// The signaling-nan-bit is bit 51 instead of bit 22
pub fn f64_is_signaling_nan(f: f64) -> bool {
    let uf: u64 = f.to_bits();
    let signal_bit = 1 << 51;
    // signaling nan := is NAN and signal bit is clear
    let signal_bit_clear = (uf & signal_bit) == 0;
    f64::is_nan(f) && signal_bit_clear
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_is_signaling_nan() {
        use std::f32;
        assert!(!f32_is_signaling_nan(3.0));
        assert!(!f32_is_signaling_nan(f32::NAN));
        assert!(!f32_is_signaling_nan(f32::INFINITY));
        assert!(!f32_is_signaling_nan(f32::NEG_INFINITY));
        assert!(!f32_is_signaling_nan(f32::MAX));
        assert!(!f32_is_signaling_nan(f32::MIN));

        // Create a signaling NaN by taking f32::NAN and setting
        // bit 22 to 0
        // Then in case that was the only bit set in the mantissa,
        // set bit 0 too
        let mask: u32 = !(1 << 22);
        let uf = f32::NAN.to_bits();
        let signaling_nan = f32::from_bits(uf & mask | 1);
        //println!();
        //println!("{:b}", uf);
        //println!("{:b}", mask);
        //println!("{:b}", uf & mask);
        assert!(f32_is_signaling_nan(signaling_nan));
    }

    #[test]
    fn test_f64_is_signaling_nan() {
        use std::f64;
        assert!(!f64_is_signaling_nan(3.0));
        assert!(!f64_is_signaling_nan(f64::NAN));
        assert!(!f64_is_signaling_nan(f64::INFINITY));
        assert!(!f64_is_signaling_nan(f64::NEG_INFINITY));
        assert!(!f64_is_signaling_nan(f64::MAX));
        assert!(!f64_is_signaling_nan(f64::MIN));

        // Create a signaling NaN by taking f64::NAN and setting
        // bit 51 to 0
        // Then in case that was the only bit set in the mantissa,
        // set bit 0 too
        let mask: u64 = !(1 << 51);
        let uf = f64::NAN.to_bits();
        let signaling_nan = f64::from_bits(uf & mask | 1);
        //println!();
        //println!("{:b}", uf);
        //println!("{:b}", mask);
        //println!("{:b}", uf & mask | 1);
        assert!(f64_is_signaling_nan(signaling_nan));
    }

}
