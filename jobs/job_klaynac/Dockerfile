FROM klaymusic/klay-beam:0.14.6-py3.10-beam2.64.0-torch2.5-cuda12.1

WORKDIR /klay/app

COPY klay_beam/submodules/klay-audiotools ./submodules/klay-audiotools
COPY klay_beam/submodules/klay-config ./submodules/klay-config
COPY klay_beam/submodules/klay-nac ./submodules/klay-nac
COPY klay_beam/submodules/klay-codecs ./submodules/klay-codecs
COPY jobs/job_klaynac ./job_klaynac

RUN python3 -m pip install \
  './submodules/klay-audiotools' \
  './submodules/klay-config' \
  './submodules/klay-nac' \
  './submodules/klay-codecs'
