# nanowasm
A small standalone WebAssembly interpreter in Rust

# Building programs

Fetch and build [wabt](https://github.com/WebAssembly/wabt), which contains useful low-level tools like assemblers.

The assembler is `wat2wasm`, so use that.

```
cd test_programs
wat2wasm inc.wast
```

This should create a `inc.wasm` program which is what you can actually load and run.
