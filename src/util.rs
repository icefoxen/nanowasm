//! Misc numerical and handiness functions. functions.

use num;
use parity_wasm::elements;

/// fcopysign:
///
/// If z1 and z2 have the same sign, then return z1
/// Else return z1 with negated sign.
pub fn copysign<T>(f1: T, f2: T) -> T
    where T: num::Float {
    // This probably even works as intended for NaN's, since
    // is_sign_positive() and such just look at the sign bit.
    if f1.is_sign_positive() && f2.is_sign_positive()
        || f1.is_sign_negative() && f2.is_sign_negative() {
            f1
        } else {
            f1.neg()
        }
}

/// Truncates the float into the given integer type.
pub fn truncate_to_int<From, To>(f1: From) -> To
    where From: num::Float,
          To: num::NumCast {
    // TODO: Needs more correctness checking!  Inf, NaN, etc.
    // Also not sure NumCast is the right thing to use here but it seems to work.
    To::from(f1.trunc()).unwrap()
}

/// Rounds the float into the given integer type.
pub fn round_to_int<From, To>(f1: From) -> To
    where From: num::Float,
          To: num::NumCast {
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




/// Takes a slice of `Local`'s (local variable *specifications*),
/// and creates a vec of their types.
///
/// Slightly trickier than just a map+collect.
pub(crate) fn types_from_locals(locals: &[elements::Local]) -> Vec<elements::ValueType> {
    // This looks like it should just be a map and collect but actually isn't,
    // 'cause we need to iterate the inner loop.  We could make it a map but it's
    // trickier and not worth the bother.
    let num_local_slots = locals.iter().map(|x| x.count() as usize).sum();
    let mut v = Vec::with_capacity(num_local_slots);
    for local in locals {
        for _i in 0..local.count() {
            let t = local.value_type();
            v.push(t);
        }
    }
    v
}



