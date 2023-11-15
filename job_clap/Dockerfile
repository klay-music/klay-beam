# Example usage:
# docker build -t us-docker.pkg.dev/klay-home/klay-docker/klay-beam-clap .

FROM us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.12.2-py3.9-beam2.51.0-torch2.0-torchvision0.15

WORKDIR /klay/app
COPY . ./job_clap
RUN python3 -m pip install './job_clap'
RUN pip check

# Download the model
RUN python3 -c "import job_clap.transforms as t; t.download_model(); print('Models downloaded')"
