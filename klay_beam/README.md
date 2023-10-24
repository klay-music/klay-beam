# Klay Beam

Helpers for running massively parallel Apache Beam jobs on audio data.

**NOTE:** This is Beta. Documentation is incomplete. Expect breaking changes
prior to v1.0.

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
- `klay_beam.run_example` pipeline script
- A specialized docker image created by `Makefile`
- `environment/py310-torch.local.yml` local environment

Notes:


To run the example job:
1. Ensure you have GCP permissions
2. Activate a `klay-beam` conda environment locally, (for example
   `environment/py310-torch.local.yml`)
3. Invoke `python -m klay_beam.run_example` as per examples below

```bash
# Running locally allows you to use --source_audio_path values paths on your
# local filesystem OR in gs://object-storage.
python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path '/local/path/to/mp3s/'
```

```bash
# Run remotely via GCP Dataflow. Should be executed in the `klay-beam` conda
# environment to ensure Beam SDK, python, and dependency parity between the
# local environment and Worker environments.

KLAY_BEAM_CONTAINER=us-docker.pkg.dev/<your-gcp-project>/<your-docker-artifact-registry>/<your-docker-image>:<tag>
SERVICE_ACCOUNT_EMAIL=<your-service-account>@<your-gcp-project>.iam.gserviceaccount.com
TEMP_GS_URL=gs://<your-gs-bucket>/<your-writable-dir/>
AUDIO_URL='gs://<your-audio-bucket>/audio/'

python -m klay_beam.run_example \
    --runner DataflowRunner \
    --max_num_workers=128 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email ${SERVICE_ACCOUNT_EMAIL} \
    --experiments=use_runner_v2 \
    --sdk_container_image ${KLAY_BEAM_CONTAINER} \
    --sdk_location=container \
    --temp_location ${TEMP_GS_URL} \
    --project klay-training \
    --source_audio_suffix .mp3 \
    --source_audio_path ${AUDIO_URL} \
    --machine_type n1-standard-8 \
    --job_name 'example-job-000'
```

Notes:

- When running on Dataflow you can use the `--setup_file` option to upload a
  local package to the workers. For example, when running with
  `--runner DataflowRunner`, `--setup_file=./your_job/setup.py` would cause
  `your_job` to be bundled as an `sdist` and installed on the worker nodes
  replacing any existing installation of `your_job` that may be in the docker
  container. Any missing pip dependencies specified in `your_job/pyproject.toml`
  will also be installed at runtime.
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

See an example of building a Compatible docker image in `Makefile`.

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
- **PTransforms** accept one or more **PCollections** as input, and output one
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
