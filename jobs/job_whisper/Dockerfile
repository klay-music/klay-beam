FROM klaymusic/klay-beam:0.14.1-py3.10-beam2.64.0-torch2.3-cuda12.1

WORKDIR /klay/app

# 2. Copy the klay-data source tree into the builder
#    (the directory path below assumes your Cloud Build context already
#    contains the sub-module checkout; adjust if different)
COPY klay_beam/submodules/klay-data ./submodules/klay-data

# 3. Build the wheel _inside_ the image; no outside index needed
RUN pip install build && \
    python -m build --wheel ./submodules/klay-data -o ./klay-data-wheel


# 4) install klay_data FROM THAT WHEEL ONLY  ─────────────────────────────
RUN pip install --find-links=./klay-data-wheel klay_data[whisper]

# 5) Copy the rest of the app code into the image
COPY klay_beam/submodules/klay-audiotools ./submodules/klay-audiotools
COPY jobs/job_whisper ./job_whisper

# 6) Install klay-data with whisperx. ctranslate2 is needed to make CUDA available.
RUN pip install './submodules/klay-audiotools' 'ctranslate2==4.4.0'

# This is only allowed for torch>=2.3
# ENV TORCH_DISABLE_GLOBAL_LIBS=1

# Smoke test: check that torch is installed and CUDA is available
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
