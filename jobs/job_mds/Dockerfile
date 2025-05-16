# ────────────────────────────────────────────────────────────────────────
# Stage 2 – final image: install the wheel + run the app
# ────────────────────────────────────────────────────────────────────────
FROM klaymusic/klay-beam:0.14.5-py3.10-beam2.64.0-torch2.1

WORKDIR /klay/app

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
    python -m build --wheel ./submodules/klay-data -o /tmp/klay-data-wheel

# 2) install klay_data FROM THAT WHEEL ONLY  ─────────────────────────────
RUN pip install --no-cache-dir --find-links=/tmp/klay-data-wheel klay_data
