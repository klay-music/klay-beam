# Example usage:
# docker build -t us-docker.pkg.dev/klay-home/klay-docker/klay-beam-demucs .

FROM us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.12.1-py3.9-beam2.51.0-torch2.0

WORKDIR /klay/app
COPY . ./job_demucs
RUN python3 -m pip install './job_demucs'
RUN pip check

# Download the demucs models to "/root/.cache/torch/hub/checkpoints"
RUN python3 -c "import demucs.pretrained as p; p.get_model('htdemucs_ft'); print('Models downloaded')"
