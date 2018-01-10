# nanowasm

A small standalone WebAssembly interpreter in Rust

# Goals

This is a hobby project, not intended to be a professional-grade tool.  But if it gets to that point, great!  I think a small,
lightweight interpreter would be really useful for things like embedding, running unit tests on generated wasm code, as an
extension system, etc.

The road map is, more or less in order:

 * Implement all of wasm32 1.0 correctly, passing standard tests and fuzzing
 * Make it *easy* to run as a standalone interpreter, or embedded in other programs
 * Make it *easy* to add custom modules written in Rust
 * Nice debugging tools?
 * Make it reasonably fast?
 * JIT???
 * Load custom modules written in Rust or C as DLL's???

# Building programs

Fetch and build [wabt](https://github.com/WebAssembly/wabt), which contains useful low-level tools like assemblers.

```
sudo apt install clang cmake
git clone --recursive https://github.com/WebAssembly/wabt.git
cd wabt
make -j$(nproc)
```

The assembler is `wat2wasm`, so use that.

```
cd test_programs
wat2wasm inc.wast
```

This should create a `inc.wasm` program which is what you can actually load and run:

```
cargo run -- test_programs/inc.wasm
```

# Similar projects

 * parity-wasm: A crate for serializing/deserializing wasm code.  Also includes its own interpreter, but I wanted to write my own and I find theirs hard to extend
 * [WebAssembly reference interpreter](https://github.com/WebAssembly/spec/tree/master/interpreter)
 * [WebAssembly Binary Toolkit](https://github.com/WebAssembly/wabt)

# Licence

Apache/MIT
