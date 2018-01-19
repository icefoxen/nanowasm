#!/usr/bin/env python3

import shutil
import subprocess

def main():
    nanowasm_binary_debug = "../target/debug/nanowasm"
    nanowasm_binary_release = "../target/release/nanowasm"

    # TODO: For now we only test the debug build.
    # Options for the future include testing whichever was
    # built more recently, or testing both, or building both
    # and testing both...
    nanowasm_binary = nanowasm_binary_debug

    # Instead of trying to fiddle with wabt's find-exe.py or such
    # we just copy nanowasm over wabt's wasm-interp binary.
    wasm_interp = "wabt/bin/wasm-interp"
    shutil.copyfile(nanowasm_binary, wasm_interp)

    # run wabt interpreter tests
    wabt_test_runner = "wabt/test/run-tests.py"
    subprocess.run([wabt_test_runner, "interp"])

if __name__ == '__main__':
    main()
