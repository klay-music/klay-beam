import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    ResampleAudio,
    write_file,
    numpy_to_wav,
)



"""

Job for extracting EnCodec and Chroma features:

1. Recursively search a path for `.<something>.wav` files (`--source_audio_path`) where `<something>` is one of `source`, `bass`, `vocals`, `drums`, or `other`.
1. For each audio file, extract the Chroma and Encoded numpy features, write them to 
1. Save results to (`--target_audio_path`) preserving the directory structure.

To run, activate a suitable python environment such as
``../environments/osx-64-job-random-trim.yml`.

```
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_extract_features.py \
    --source_audio_path '/absolute/path/to/source.wav/files/' \
    --runner Direct

# Run remote job with autoscaling
python bin/run_job_extract_features.py \
    --runner DataflowRunner \
    --machine_type n1-standard-2 \
    --max_num_workers=256 \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.5.0-demucs \
    --sdk_location=container \
    --setup_file ./klay_beam/setup.py \
    --temp_location gs://klay-dataflow-test-000/tmp/extract_features/ \
    --project klay-training \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --target_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --experiments=no_use_multiple_sdk_containers \
    --number_of_worker_harness_threads=1 \
    --job_name 'extract_features-001'

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE

# Extra options to consider

Reduce the maximum number of threads that run DoFn instances. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#reduce-threads
    --number_of_worker_harness_threads

Create one Apache Beam SDK process per worker. Prevents the shared objects and
data from being replicated multiple times for each Apache Beam SDK process. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#one-sdk
    --experiments=no_use_multiple_sdk_containers
```

"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_audio_path",
        dest="input",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.

        To get only some wav files, try:
        '/Users/alice/datasets/fma_large/005/'

        To run on the full dataset use:
        'gs://klay-datasets/mtg_jamendo_autotagging/audios/'
        """,
    )

    parser.add_argument(
        "--target_audio_path",
        dest="output",
        required=True,
        help="""
        Specify the output directory.

        For example:
        'gs://klay-dataflow-test-000/results/outputs/1/'
        """,
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    def to_wav(element):
        key, numpy_audio, sr = element
        return key, numpy_to_wav(numpy_audio, sr)

    # Pattern to recursively find mp3s inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.source.wav")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()

            # SkipCompleted goes here

            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
            | "Resample: 44.1k"
            >> beam.ParDo(
                ResampleAudio(
                    target_sr=44_100,
                    source_sr_hint=48_000,
                )
            )
            | "SourceSeparate"
            >> beam.ParDo(
                SeparateSources(
                    source_dir=known_args.input,
                    target_dir=known_args.output,
                    model_name="htdemucs_ft",
                )
            )
            | "Resample: 48K"
            >> beam.ParDo(
                ResampleAudio(
                    target_sr=48_000,
                    source_sr_hint=44_100,
                    output_numpy=True,
                )
            )
            | "CreateWavFile" >> beam.Map(to_wav)
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
