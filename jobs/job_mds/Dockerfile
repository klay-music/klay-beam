# ────────────────────────────────────────────────────────────────────────
# Stage 2 – final image: install the wheel + run the app
# ────────────────────────────────────────────────────────────────────────
FROM klaymusic/klay-beam:0.14.5-py3.10-beam2.64.0-torch2.5

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
RUN python3 -m pip install --no-cache-dir build pipdeptree && \
    python3 -m build --wheel ./submodules/klay-data -o ./klay-data-wheel

# 2) install klay_data FROM THAT WHEEL ONLY  ─────────────────────────────
RUN python3 -m pip install --no-cache-dir --find-links=./klay-data-wheel klay_data

RUN python3 -m pipdeptree
