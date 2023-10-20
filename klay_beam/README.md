# Klay Beam

Helpers for running massively parallel Apache Beam jobs on audio data.

Processing large batches of audio data can be very time consuming. It is often
helpful to spin up many instances to apply transformations or extract features
from a large audio dataset. This package bundles a collection of utility methods
and examples designed to ease the process of massively parallel audio jobs with
GCP Dataflow and Apache Beam.

The core transformations include:

- File manipulations (for local filesystem and cloud storage)
  - Audio File writing and reading
  - Feature writing (and reading?)
  - file name mutations: Moving, and extension mutation
- SkipCompleted
- Audio data resampling
- Audio channel manipulation

## Example Job

The example job uses the following ingredients:
- `bin/run_job_example.py` pipeline script
- `klay-beam:0.2.0` docker container
- `environment/py310-torch.local.yml` local environment

To run the example job:
1. Talk to Charles or Max for GCP IAP permissions
2. Activate a `klay-beam` conda environment locally, (for example
   `environment/py310-torch.local.yml`)
3. Run `bin/run_job_example.py` (see example below for arguments)

```bash
# Run Locally in the klay-beam conda environment. Running locally allows you to
# use --input and --output paths on your local filesystem OR in object storage.
python bin/run_job_example.py \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/00500*.mp3' \
    --output 'test_audio/job_output/{}.wav' \
    --runner Direct
```

```bash
# Run remotely via GCP Dataflow. Should be executed in the `klay-beam` conda
# environment to ensure Beam SDK, python, and dependency parity between the
# local environment and Worker environments.
python bin/run_job_example.py \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --disk_size_gb 50 \
    --experiments=use_runner_v2 \
    --sdk_container_image us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.2.0 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/ \
    --project klay-beam-tests \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/**' \
    --output 'gs://klay-dataflow-test-000/results/outputs/17/{}.wav' \
    --job_name 'klay-audio-test-017'
```

Notes:

- When running remotely you can use the `--setup_file` option to upload a local
  package to the workers. For example `--setup_file=./klay_beam/setup.py` would
  cause `klay_beam` to be bundled as an `sdist` (when you execute
  `bin/run_job_example.py`) and installed on the worker nodes replacing any existing
  installation of `klay_beam` that may be in the docker container. Any missing
  pip dependencies specified in `pyproject.toml` will also be installed at
  runtime.
- When running on Dataflow, view the job execution details and logs at
  https://console.cloud.google.com/dataflow/jobs?project=klay-beam-tests
- options for `--autoscaling_algorithm` are `THROUGHPUT_BASED` and `NONE`

# Development
## Quick Start

Create `conda` environment. Environments labeled `local` are likely to work on
linux albeit without cuda support:

```sh
conda env create -f environments/py310-torch.local.yml
```

To create or update an environment:

```sh
conda env update -f environment/py310-torch.local.yml
```

## Docker Container

This docker will be run on all workers. When running a Beam job on GCP Dataflow
with the `--setup_file` option missing dependencies will be installed using pip.
However, to save time, large dependencies (or non-pip dependencies such as
ffmpeg 4) should be included in the docker container.

### Docker Build Steps

These steps build the Docker container and push to our GCP docker registry.

1. `cd ..` return to parent dir
1. `./make-conda-lock-file.sh klay_beam/environment/py310-torch.linux-64.yml` to generate a new
   `klay_beam/environment/py310-torch.linux-64.lock`. **IMPORTANT: Only run this step
   when you are changing the conda dependencies in the
   `klay_beam/environment/py310-torch.linux-64.yml` file.**
2. Run `docker build -f Dockerfile.klay-beam -t klay-beam:latest .`
3. Edit `tag-klay-beam-py310.sh` to update the version, for example `0.2.0-rc.2`
4. Configure docker to authorize it to write to the artifact registry: `gcloud auth configure-docker us-docker.pkg.dev` (only needs to be done once)
5. Run `tag-klay-beam-py310.sh` to tag and push to GCP.

To test the container interactively:
`docker run --rm -it --entrypoint /bin/sh klay-beam:latest`

## Code Quality
### Testing
We use `pytest` for testing, there's no coverage target at the moment but
essential functions and custom logic should definitely be tested. To run the
tests:
```sh
make tests
```

### Code Style
We use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

### Static Typing
We check static types using `mypy`.
```sh
make type-check
```

## Design Patterns

In Apache Beam, a **Pipeline** is a Directed Acyclic Graph.
- Each node in the graph is a data processing operation called a **PTransform**
or "Parallel Transform".
- **PTransforms** accept oen or more **PCollections** as input, and output one
  or more **PCollections**
-

```python
with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
    audio, failed, durations = (
        p
        # MatchFiles produces a PCollection of FileMetadata objects
        | beam_io.MatchFiles(match_pattern)
        # Prevent "fusion"
        | "Reshuffle" >> beam.Reshuffle()
        # ReadMatches produces a PCollection of ReadableFile objects
        | beam_io.ReadMatches()
        | "Load Audio"
        >> beam.ParDo(LoadWithTorchaudio()).with_outputs(
            "failed", "duration_seconds", main="audio"
        )
    )
    (
        durations
        | "SumLengths" >> beam.CombineGlobally(sum)
        | "LogDuration"
        >> beam.Map(
            lambda x: logging.info(
                "Total duration of loaded audio: "
                f"~= {x:.3f} seconds "
                f"~= {x / 60:.3f} minutes "
                f"~= {x / 60 / 60:.3f} hours"
            )
        )
    )

    (
        failed
        | "Log Failed" >> beam.Map(lambda x: logging.warning(x))
        | "Count" >> beam.combiners.Count.Globally()
        | "Log Failed Count"
        >> beam.Map(lambda x: logging.warning(f"Failed to decode {x} files"))
    )
```


### MatchFiles

The `MatchFiles` Transforms returns a PCollection of
`apache_beam.io.filesystem.FileMetadata` instances, which have the following
properties ([code](https://beam.apache.org/releases/pydoc/2.50.0/_modules/apache_beam/io/filesystem.html#FileMetadata)):

```
path: str
size_in_bytes: int
last_updated_in_seconds: float
```

Note that in GCP Cloud Storage, the `last_update_in_seconds` property reflects
[AutoClass](https://cloud.google.com/storage/docs/autoclass) changes.


### Preventing Fusion

Transforms such as `MatchFiles` output PCollections with MANY elements relative
to the number of input elements. This called a "fan-out" transform. Large
fan-out transforms should pre proceeded by a Reshuffle when running on GCP
Dataflow. See [Preventing
Fusion](https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion)
in the Dataflow docs.


### ReadMatches

The `ReadMatches` Transform returns a PCollection of
`apache_beam.io.fileio.ReadableFile` instances ([code](https://beam.apache.org/releases/pydoc/2.24.0/_modules/apache_beam/io/fileio.html#ReadableFile)), which have a `.metadata` property and 3 additional methods:

```python
metadata: apache_beam.io.filesystem.FileMetadata
open(self, mime_type='text/plain', compression_type=None) -> io.BufferedReader # (for gs:// paths)
read(self, mime_type='application/octet-stream') -> FileLike
read_utf8(self)
```

<!--
Each filesystem has its own .open method. see details of each here:
>>> from apache_beam.io.filesystems import FileSystem
>>> FileSystem.__subclasses__()
[
    <class 'apache_beam.io.hadoopfilesystem.HadoopFileSystem'>
    <class 'apache_beam.io.localfilesystem.LocalFileSystem'>
    <class 'apache_beam.io.gcp.gcsfilesystem.GCSFileSystem'>
    <class 'apache_beam.io.aws.s3filesystem.S3FileSystem'>
    <class 'apache_beam.io.azure.blobstoragefilesystem.BlobStorageFileSystem'>
]-->


### LoadWithPytorch

LoadWithTorchaudio is a custom `beam.DoFn`, turned into a PTransform via the
`beam.ParDo` helper. See the source for implementation details. Generally,
custom functions have a few requirements that help them work well in on
distributed runners. They are:

- The function should be thread-compatible
- The function should be serializable
- Recommended: the function be idempotent

For details about these requirements, see the Apache Beam documentation:
https://beam.apache.org/documentation/programming-guide/#requirements-for-writing-user-code-for-beam-transforms
