FROM klaymusic/klay-beam:0.14.6-py3.10-beam2.64.0-torch2.5-cuda12.1

WORKDIR /klay/app

RUN apt-get update && apt-get install git -y

# ---------- cuDNN runtime for CUDA 12.1 ---------------------------
RUN python3 -m pip install --no-cache-dir nvidia-cudnn-cu12==8.9.6.50
ENV LD_LIBRARY_PATH=/env/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ENV TORCH_DISABLE_GLOBAL_LIBS=1

COPY klay_beam/submodules/klay-data ./submodules/klay-data
RUN python3 -m pip install build && \
    python3 -m build --wheel ./submodules/klay-data -o ./klay-data-wheel
RUN python3 -m pip install --find-links=./klay-data-wheel klay_data[whisper]

COPY klay_beam/submodules/klay-audiotools ./submodules/klay-audiotools
COPY jobs/job_whisper ./job_whisper
RUN python3 -m pip install ./submodules/klay-audiotools ctranslate2==4.4.0

RUN python3 -c "import torch"
