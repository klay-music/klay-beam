# Example usage:
# docker build -t us-docker.pkg.dev/klay-home/klay-docker/klay-beam-discogs-effnet .

FROM us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.12.1-py3.9-beam2.51.0

WORKDIR /klay/app

ADD https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_multi_embeddings-effnet-bs64-1.pb ./models/

# Install submodules and klay_beam
COPY . ./job_discogs_effnet

# If you change submodules below, you also may want to also change the local
# conda environment.
RUN python3 -m pip install './job_discogs_effnet'
RUN pip check
