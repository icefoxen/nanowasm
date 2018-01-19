//! A small embeddable and extendable WebAssembly interpreter.
//! Uses the `parity-wasm` crate for loading wasm binaries.

extern crate byteorder;
extern crate num;
extern crate parity_wasm;


#[cfg(test)]
mod tests;

pub mod types;
pub mod util;
pub mod loader;
pub mod interpreter;

