# klay-beam

Klay Beam is a toolkit for running Apache Beam jobs that process audio data. It
can be used for parallelizing audio transformations and feature extraction. The
basic workflow:

1. Write an audio transformation or feature-extraction pipeline
1. Run and test the pipeline locally
1. Launch a job with the [GCP Dataflow](https://cloud.google.com/dataflow)
   runner, scaling up execution to hundreds or thousands of concurrent
   processes.

This repository bundles:

- The `klay_beam` python package with basic utilities for reading, transforming,
  and writing audio in beam pipelines
- Docker image build processes for images that can be used in pipelines executed
  on the Dataflow Beam Runner on GCP. Core images include support for a python
  versions 3.9, 3.10, 3.11, and PyTorch (optionally with CUDA support). (See:
  [hub.docker.com/r/klaymusic/klay-beam/](https://hub.docker.com/r/klaymusic/klay-beam/tags))
- Examples of simple pipeline launch scripts for running jobs locally and on GCP
  Dataflow.

See the [python package readme in `klay_beam`](./klay_beam/README.md) for more
information.

# Run an example job

```bash
pip install klay_beam

python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road/mp3/'

python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3'

python -m klay_beam.run_example \
    --runner DataflowRunner \
    --max_num_workers=128 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image 'us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-docker-py3.9-beam2.51-torch2.0' \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/example-job/ \
    --project klay-training \
    --source_audio_suffix .mp3 \
    --source_audio_path 'gs://klay-datasets-001/mtg-jamendo/00' \
    --machine_type n1-standard-16 \
    --job_name 'read-audio-example'
```