import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    convert_audio,
    LoadWithTorchaudio,
    SkipCompleted,
    write_file,
    numpy_to_file,
)

from job_clap.transforms import ExtractCLAP


"""
Job for extracting EnCodec features:

1. Recursively search a path for `.wav` files
1. For each audio file, extract CLAP embeddings
1. Write the results to an .npy file adjacent to the source audio file

To run, activate a suitable python environment such as
`../environments/osx-64-clap.yml`.

```
# CD into the root klay_beam dir to the launch script:
python bin/run_job_extract_clap.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --audio_suffix .wav

# Run remote job with autoscaling
python bin/run_job_extract_clap.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers 100 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract-clap/ \
    --setup_file ./job_clap/setup.py \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.10.3-clap \
    --source_audio_path \
        'gs://klay-dataflow-test-001/mtg-jamendo-90s-crop/00' \
    --job_name 'extract-clap-004'
    --number_of_worker_harness_threads 1 \
    --experiments no_use_multiple_sdk_containers
    --audio_suffix .wav

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
        "--audio-suffix",
        default=".wav",
    )
    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Pattern to recursively find audio files inside source_audio_path
    match_pattern = os.path.join(known_args.input, f"**{known_args.audio_suffix}")
    extract_fn = ExtractCLAP(device="cpu")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        audio_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=known_args.audio_suffix,
                    new_suffix=extract_fn.suffix,
                    check_timestamp=True,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        (
            audio_files
            | "Convert to Mono and Resample to 48k"
            >> beam.Map(
                lambda x: (
                    x[0],
                    convert_audio(x[1], x[2], 48_000, 1),
                    48_000,
                )
            )
            | "ExtractCLAP" >> beam.ParDo(extract_fn)
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
