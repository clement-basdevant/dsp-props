# Frozen environment for dsp-props (plan 3.6): every number of the paper and of
# benchmarks/results.md can be recomputed in it.
#   docker build -t dsp-props .
#   docker run --rm dsp-props python -m pytest -q
#   docker run --rm dsp-props python benchmarks/make_results.py --check
# Base image pinned by digest (python 3.12.16, Debian bookworm, linux/amd64 and arm64).
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /dsp-props

# Dependencies first, every one pinned with its hash (env/requirements-py312-linux.txt, from env/lock.sh)
COPY env/requirements-py312-linux.txt env/
RUN python -m pip install --require-hashes --no-deps -r env/requirements-py312-linux.txt

COPY . .
RUN python -m pip install --no-deps --no-build-isolation . \
    && python -m pip check
