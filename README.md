# nanowasm

A small standalone WebAssembly interpreter in Rust.  Currently not really usable, yet.

# Goals

This is a hobby project, not intended to be a professional-grade tool.  But if it gets to that point, great!  I think a small, lightweight interpreter would be really useful for things like embedding, running unit tests on generated wasm code, as an extension system, etc.

The road map is, more or less in order:

 * Implement all of wasm32 1.0 core correctly, passing standard tests and fuzzing
 * Make it *easy* to run as a standalone interpreter, or embedded in other programs
 * Make it *easy* to extend with custom modules written in Rust
 * Make it *easy* to run Rust code built with `no_std` for the wasm32-unknown-unknown target, and provide a basic console API
 * Make it possible to set hard execution limits on memory and CPU consumed (somehow) and easily sandbox it to forbid access to random resources (like files)
 * Make it reasonably fast?
 * Nice debugging tools?
 * JIT?  Either using LLVM, Cranelift, or writing my own; I kind of want to write my own for the experience.
 * Load custom modules from DLL's written in Rust or C???

# Non-goals

 * Don't intend to run correctly on big-endian platforms, since where byte layout matters to wasm (in memory's) wasm specifies little-endian.  Since I kinda want to make it a JIT then it will be platform-specific anyway.
 * Don't intend to never use `unsafe`; if we can make a *significant* performance win with unsafe code, we should.  Properly-validated WebAssembly code should be safe itself.  Naturally, not using unsafe would be best.


# Building programs

It's useful to fetch and build [wabt](https://github.com/WebAssembly/wabt), which contains useful low-level tools like
an assembler.


```
sudo apt install clang cmake
git clone --recursive https://github.com/icefoxen/nanowasm
# Or if you've already checked out nanowasm, cd into it and run:
# git submodule update --init --recursive
cd nanowasm/spec/wabt
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

# Safety

This should aim for the goal that if a wasm module is invalid, it will return a `Result::Err` on trying to load or
validate it, or construct a program from it.  Then actually executing the code should not need to return a `Result`
because it has already been verified correct; things that are incorrect are bugs in the interpreter or verifier and should
result in a panic (should we use `assert!` or `debug_assert!` here?  Depends on the situation; `assert!` can
sometimes lead to the compiler making smarter code, for example eliding bounds checks after one).

One place where this is *not* possible to statically verify is if you set resource bounds and the program exceeds
them.  Need to think about this.  There may be other places; the wasm test suite should (hopefully) cover these.

# Similar projects

 * parity-wasm: A crate for serializing/deserializing wasm code.  
 * [wasmi](https://github.com/pepyakin/wasmi): A webassembly interpreter; used to be part of parity-wasm but got
   split off into its own thing.  Quite good, but I wanted to write my own
 * <https://github.com/CraneStation/wasmtime>: A wasm standalone JIT using Cranelift
 * [WebAssembly reference interpreter](https://github.com/WebAssembly/spec/tree/master/interpreter)
 * [WebAssembly Binary Toolkit](https://github.com/WebAssembly/wabt)

# Licence

Apache/MIT
