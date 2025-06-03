FROM klaymusic/klay-beam:0.14.6-py3.10-beam2.64.0-torch2.5

WORKDIR /klay/app

COPY klay_beam/submodules/klay-data ./submodules/klay-data
RUN python3 -m pip install build && \
    python3 -m build --wheel ./submodules/klay-data -o ./klay-data-wheel
run python3 -m pip install --find-links=./klay-data-wheel 'klay_data[torch]'

COPY klay_beam/submodules/klay-audiotools ./submodules/klay-audiotools
COPY jobs/job_stats ./job_stats
RUN python3 -m pip install ./submodules/klay-audiotools ctranslate2==4.4.0

RUN python3 -c "import torch"
