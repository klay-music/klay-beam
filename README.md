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

# Development
## Versioning
### Job Versioning
The version of any job is updated using the following steps:
```
1. First edit the version in the files its defined
  - `.env`
  - `src/<pkg>/__init__.py::__version__`

2. git add && git commit -m '<job_name>: <version>'
3. git tag -a job-essentia-<version> -m '<job_name>: <version>'
4. git push --tag
```

