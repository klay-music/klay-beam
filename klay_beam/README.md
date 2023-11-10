# Klay Beam

Helpers for running massively parallel Apache Beam jobs on audio data.

**NOTE:** This is Beta. Documentation is incomplete. Expect breaking changes
prior to v1.0.

Processing large batches of audio data can be very time consuming. It is often
helpful to use many VM instances to apply transformations or extract features
from a large audio dataset. This package bundles a collection of utilities and
examples designed to ease the process of massively parallel audio jobs with
GCP Dataflow and Apache Beam.

The core transformations include:

- File manipulations (for local filesystem and cloud storage)
  - Audio File writing and reading
  - Feature writing (and reading?)
  - file name mutations: Moving, and extension mutation
- SkipCompleted
- Audio data resampling
- Audio channel manipulation

You can use `klay-beam` to write and launch your own custom jobs that build on
top of these primitives. It is setup to support a wide variety of custom
dependencies and environments, including pinned versions of Python, Pytorch,
CUDA (for GPU support), and more.

## Running Locally

Typically you will want to write and test a job on local machine, before testing
and executing on a massive dataset. For example:

```bash
# Create the environment and install klay_beam
conda env create -f environment/py3.10-torch2.0.yml`
conda activate klay-beam-py3.10-torch2.0
pip install -e .
```

Then launch the example job:

```bash
# Then launch the job. Running locally allows you to use --source_audio_path
#  values paths on your local filesystem OR in gs://object-storage. To use gs://
# directories, you must be authenticated with GCP
python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path '/local/path/to/mp3s/'
```

## Running on GCP via Dataflow

If your audio files are in cloud storage you can process them in using GCP
Dataflow, which allows for massive parallel execution. This requires additional
setup, including:

1. Activate Dataflow API
1. Create GCP service account
1. Create GCP Cloud Storage bucket
1. Setup GCP permissions for launching and executing jobs

Finally, you need a specialized docker container that bundles `apache_beam`,
`klay_beam`, and any additional dependencies. See `Makefile` for examples.

### Setup GCP

To get started, setup a GCP project by following this steps below, which are
adapted from the [Dataflow Quickstart Guide][dataflow-quickstart].

[dataflow-quickstart]: https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python

```bash
# manually set the following examples
GCP_PROJECT_ID=your-gcp-project  # ID of the GCP project that will run jobs
USER_EMAIL=you@example.com       # The email associated with your GCP account
DATAFLOW_BUCKET_NAME=your-bucket # Temp data storage bucket for beam workers
GCP_SA_NAME=beam-worker          # GCP service account name used by beam workers

# Compute the full email of the service account used by beam workers
GCP_SA_EMAIL=${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com
# Compute a valid tempo storage path job workers. This is just a proposal. You
# can put this anywhere, as long the Beam workers are able to write temporary
# files to this path during job execution.
TEMP_GS_URL=gs://${DATAFLOW_BUCKET_NAME}/tmp

# Create and activate a GCP project. Skip if you want to reuse an existing one.
gcloud init
gcloud projects create ${GCP_PROJECT_ID}
gcloud config set project ${GCP_PROJECT_ID}
# Make sure that billing is enabled for your project
gcloud beta billing projects describe ${GCP_PROJECT_ID}
gcloud services enable dataflow compute_component logging storage_component storage_api bigquery pubsub datastore.googleapis.com cloudresourcemanager.googleapis.com
gcloud auth application-default login

# Dataflow jobs need to write temporary data to cloud storage during job
# execution. Create a bucket using the gsutil mb (make bucket) command. See
# `gsutil help mb` for details.
gsutil mb --autoclass -l US -b on gs://${DATAFLOW_BUCKET_NAME}

# Create a service account which will be used by the worker nodes
gcloud iam service-accounts create $GCP_SA_NAME --description="Service account used by Apache Beam workers" --display-name="Beam Worker"

# Give the service account access it needs
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} --member="serviceAccount:${GCP_SA_EMAIL}" --role=roles/dataflow.admin
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} --member="serviceAccount:${GCP_SA_EMAIL}" --role=roles/dataflow.worker
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} --member="serviceAccount:${GCP_SA_EMAIL}" --role=roles/storage.objectAdmin
# Note that the last command above will give the service account (and any users
# who can impersonate the service account) full access to ALL buckets in the
# project. If this is undesirable, you can use the Cloud Storage section of
# console.cloud.google.com to give the service account access to ONLY specific
# buckets. To do this, navigate to a bucket, and click the "permissions" button.
#
# If you choose bucket level permissions, you must also grant:
# - read+list access to buckets where source data is saved
# - write access to buckets where result data will be persisted

# To allow users to impersonate the service account, run the following command
# which grants a user the `roles/iam.serviceAccountUser` (AKA "Service Account
#  User") role, but only for a specific service account:
gcloud iam service-accounts add-iam-policy-binding ${GCP_SA_EMAIL} \
    --member="user:${USER_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# Alternatively, if you want to grant the user access to impersonate ALL service
# accounts, use this command instead:
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
    --member="user:${USER_EMAIL}" \
    --role=roles/iam.serviceAccountUser
```


### Launch GCP Dataflow Job

```bash
# Run remotely via GCP Dataflow. Should be executed in the `klay-beam` conda
# environment to ensure Beam SDK, python, and dependency parity between the
# local environment and Worker environments.

# You will need the following configuration values from the setup (above)
GCP_PROJECT_ID=<your-gcp-project>
GCP_SA_EMAIL=<your-service-account>@<your-gcp-project>.iam.gserviceaccount.com
TEMP_GS_URL=gs://<your-gs-bucket>/<your-writable-dir/>

# Additionally, you need a custom Beam container, and an gs:// url that contains
# the audio files you want to read. You must ensure that the service account
# has read access to these audio files.
KLAY_BEAM_CONTAINER=us-docker.pkg.dev/<your-gcp-project>/<your-docker-artifact-registry>/<your-docker-image>:<tag>
AUDIO_URL='gs://<your-audio-bucket>/audio/'

python -m klay_beam.run_example \
    --runner DataflowRunner \
    --max_num_workers=128 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email ${GCP_SA_EMAIL} \
    --experiments=use_runner_v2 \
    --sdk_container_image ${KLAY_BEAM_CONTAINER} \
    --sdk_location=container \
    --temp_location ${TEMP_GS_URL} \
    --project ${GCP_PROJECT_ID} \
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

### Custom Docker Images on Dataflow

If you are storing your docker images in a private repo use the IAM section of
console.cloud.google.com and grant the "Artifact Registry Reader" role to your
beam worker service account.

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


### LoadWithTorchaudio

LoadWithTorchaudio is a custom `beam.DoFn`, turned into a PTransform via the
`beam.ParDo` helper. See the source for implementation details. Generally,
custom functions have a few requirements that help them work well in on
distributed runners. They are:

- The function should be thread-compatible
- The function should be serializable
- Recommended: the function be idempotent

For details about these requirements, see the Apache Beam documentation:
https://beam.apache.org/documentation/programming-guide/#requirements-for-writing-user-code-for-beam-transforms
