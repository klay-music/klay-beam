# ────────────────────────────────────────────────────────────────────────
# Stage 1 – klay-data builder: pull LFS assets + build the wheel
# ────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS klay-data-builder

# 1. OS packages needed to build and to run git-lfs
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git git-lfs build-essential && \
    git lfs install --skip-repo && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy the klay-data source tree into the builder
#    (the directory path below assumes your Cloud Build context already
#    contains the sub-module checkout; adjust if different)
COPY klay_beam/submodules/klay-data ./submodules/klay-data

# 3. Build the wheel _inside_ the image; no outside index needed
RUN pip install --no-cache-dir build && \
    python -m build --wheel ./submodules/klay-data -o /klay-data-wheel


# ────────────────────────────────────────────────────────────────────────
# Stage 2 – final image: install the wheel + run the app
# ────────────────────────────────────────────────────────────────────────
FROM klaymusic/klay-beam:0.13.7-py3.10-beam2.53.0-torch2.1-cuda12.1

WORKDIR /klay/app

# Install git
RUN apt-get update && apt-get install -y git

# 2) install klay_data FROM THAT WHEEL ONLY  ─────────────────────────────
COPY --from=klay-data-builder /klay-data-wheel /tmp/klay-data-wheel
RUN pip install --no-cache-dir --no-index --find-links=/tmp/klay-data-wheel klay_data \
 && rm -rf /tmp/klay-data-wheel

# These paths are relatived to the repository root
COPY klay_beam/submodules/klay-mtrpp ./submodules/klay-mtrpp

# Install the klay-mtrpp package
RUN pip install --no-cache-dir --no-index ./submodules/klay-mtrpp

# Set git to use token for all GitHub URLs
# RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Download model
RUN python3 -c "from klay_data.extractors import MTRPPExtractor; MTRPPExtractor()"
