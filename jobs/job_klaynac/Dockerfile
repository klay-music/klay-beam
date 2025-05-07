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
COPY klay_beam/submodules/klay-data ./submodules/klay-data

# 3. Build the wheel _inside_ the image; no outside index needed
RUN pip install --no-cache-dir build && \
    python -m build --wheel ./submodules/klay-data -o /klay-data-wheel


# ────────────────────────────────────────────────────────────────────────
# Stage 2 – final image: install the wheel + run the app
# ────────────────────────────────────────────────────────────────────────
FROM klaymusic/klay-beam:0.14.0-py3.10-beam2.61.0-torch2.1-cuda12.1

WORKDIR /klay/app

COPY jobs/job_klaynac ./job_klaynac

COPY klay_beam/submodules/klay-codecs/submodules/klay-nac/submodules/klay-data/submodules/klay-audiotools ./submodules/klay-audiotools
COPY klay_beam/submodules/klay-codecs/submodules/klay-nac/submodules/klay-config ./submodules/klay-config
COPY klay_beam/submodules/klay-codecs/submodules/klay-nac ./submodules/klay-nac
COPY klay_beam/submodules/klay-codecs ./submodules/klay-codecs

RUN python3 -m pip install \
  './submodules/klay-audiotools' \
  './submodules/klay-config' \
  './submodules/klay-nac' \
  './submodules/klay-codecs'


ENTRYPOINT ["/opt/apache/beam/boot"]
