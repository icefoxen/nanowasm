#!/usr/bin/env python3

import os
import shutil
import subprocess


def main():
    wast2json_binary = "wabt/bin/wast2json"
    target_tests = ["start.wast"]
    outdir = "OUT/"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    # Build nanowasm, so we only do it once and fail if it doesn't work.
    cargo_command_line = ["cargo", "build"]
    # TODO: Make sure this completes successfully
    subprocess.run(cargo_command_line)

    for test in target_tests:
        test_name = "wasm_testsuite/" + test
        print("Generating", test_name)
        # TODO: Make sure this completes successfully
        outfile = outdir + test + ".json"
        subprocess.run([wast2json_binary, test_name, "-o", outfile])
    
        # Run nanowasm on test results.
        cargo_command_line = ["cargo", "run", "--bin", "nanowasm", "--", "--load-test", outfile ]
        subprocess.run(cargo_command_line)

if __name__ == '__main__':
    main()
