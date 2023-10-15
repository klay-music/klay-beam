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
