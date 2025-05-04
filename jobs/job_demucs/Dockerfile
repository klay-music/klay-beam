# Example usage:
# docker build -t us-docker.pkg.dev/klay-home/klay-docker/klay-beam-demucs .

FROM klaymusic/klay-beam:0.13.7-py3.10-beam2.53.0-torch2.1-cuda12.1

WORKDIR /klay/app
COPY . ./job_demucs

# demucs needs git
RUN apt-get update && \
    apt-get install -y git

RUN python3 -m pip install './job_demucs'
RUN pip check

# Download the demucs models to "/root/.cache/torch/hub/checkpoints"
RUN python3 -c "import demucs.pretrained as p; p.get_model('hdemucs_mmi'); print('Models downloaded')"
RUN python3 -c "import torch; print(f'cuda is available: {torch.cuda.is_available()}')"

ENTRYPOINT ["/opt/apache/beam/boot"]
