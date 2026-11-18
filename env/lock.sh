#!/bin/sh
# Recompile the frozen environment (plan 3.6) from env/requirements.in.
# Run from the repository root after changing a dependency; commit the result.
set -eu
uv pip compile env/requirements.in \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
    --generate-hashes --emit-index-url --no-emit-package dsp-props \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    -o env/requirements-py312-linux.txt
