FROM klaymusic/klay-beam:0.14.6-py3.10-beam2.64.0-torch2.5-cuda12.1

WORKDIR /klay/app

RUN apt-get update && apt-get install git -y

COPY klay_beam/submodules/klay-data ./submodules/klay-data
RUN python3 -m pip install build && \
    python3 -m build --wheel ./submodules/klay-data -o ./klay-data-wheel
RUN python3 -m pip install --find-links=./klay-data-wheel klay_data[whisper]

COPY klay_beam/submodules/klay-audiotools ./submodules/klay-audiotools
COPY jobs/job_whisper ./job_whisper
RUN python3 -m pip install ./submodules/klay-audiotools ctranslate2==4.4.0

RUN python3 -c "import torch"
RUN python3 -c "from klay_data.extractors import WhisperExtractor; WhisperExtractor()"
