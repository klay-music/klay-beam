# klay-beam

Base repo for running Apache Beam jobs locally or on GCP via Dataflow. This is
how we run massively parallel jobs with hundreds-of-thousands of input audio
files, for example:

- Source separation
- Feature extraction
- Cropping
- Resampling

Most Beam jobs will require three ingredients which must be engineered for
compatibility:

1. A pipeline script to define and launch the job
2. A specialized Docker image that includes Beam SDK and any dependencies required
   by the pipeline (Dataflow worker nodes will run instances of this image)
3. A local python environment that also includes the Beam SDK and job
   dependencies (The job will be launched from this environment. Also used to
   run the job when running locally with `--runner=Direct`)

This repo includes helpers and examples for creating compatible scripts, Docker
images, and local environments (1, 2, and 3 respectively).


# Run an example job

```bash
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
    --temp_location gs://klay-dataflow-test-000/tmp/convert-audio/ \
    --project klay-training \
    --source_audio_suffix .mp3 \
    --source_audio_path 'gs://klay-datasets-001/mtg-jamendo/00' \
    --machine_type n1-standard-16 \
    --job_name 'read-audio-example'
```