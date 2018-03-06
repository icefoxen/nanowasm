This directory contains code to run nanowasm against the standard test suite, using (hijacking) tools from the `wabt` binary toolkit.

If you didn't check out the git repo with all submodules (`git clone
--recursive ...`), we can get them with this incantation:

```
git submodule update --init --recursive
```

Build wabt:

```
cd spec/wabt
make -j$(nproc)
cd ..
```

Run tests (from the `nanowasm/spec` directory)

```
python3 run-nanowasm-tests.py
```
