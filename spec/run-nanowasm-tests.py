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
    # TODO Lots of the tests are not reeeeeaally valid for nanowasm,
    # so we should list a subset of them.
    target_tests = ["start"]
    for test in target_tests:
        test_name = "spec/"
        wabt_test_runner = "wabt/test/run-tests.py"
        subprocess.run([wabt_test_runner, test_name, "-v", "-p"])

if __name__ == '__main__':
    main()
