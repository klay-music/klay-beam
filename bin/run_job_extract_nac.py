import argparse
import os.path
import logging
from typing import Union

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    ResampleAudio,
    SkipCompleted,
    write_file,
    numpy_to_file,
)

from job_nac.transforms import ExtractDAC, ExtractEncodec


"""
Job for extracting EnCodec features:

1. Recursively search a path for `.wav` files
1. For each audio file, extract EnCodec tokens
1. Write the results to an .npy file adjacent to the source audio file

To run, activate a suitable python environment such as
`../environments/osx-64-nac.yml`.

```
# CD into the root klay_beam dir to the launch script:
python bin/run_job_extract_nac.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_name dac \
    --nac_input_sr 44100 \
    --audio_suffix .wav \

python bin/run_job_extract_nac.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_name encodec \
    --nac_input_sr 48000 \
    --audio_suffix .wav \

# Run remote job with autoscaling
python bin/run_job_extract_nac.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers 1000 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract-ecdc-48k/ \
    --setup_file ./job_nac/setup.py \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.10.0-nac \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --nac_name encodec \
    --nac_input_sr 48000 \
    --audio_suffix .wav \
    --job_name 'extract-ecdc-002'


    --number_of_worker_harness_threads 1 \
    --experiments no_use_multiple_sdk_containers

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
        "--nac_name",
        required=True,
        choices=["dac", "encodec"],
        help="""
        Which neural audio codec should we use? Options are ['dac' or 'encodec']
        """,
    )

    parser.add_argument(
        "--nac_input_sr",
        required=True,
        type=int,
        choices=[16000, 24000, 44100, 48000],
        help="""
        Which audio sample rate should we extract from?
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff"],
        help="""
        Which audio file extension to search for when scanning input dir?
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

    # Pattern to recursively find audio files inside source_audio_path
    match_pattern = os.path.join(known_args.input, f"**{known_args.audio_suffix}")

    # instantiate NAC extractor here so we can use computed variables
    extract_fn: Union[ExtractDAC, ExtractEncodec]
    if known_args.nac_name == "dac":
        extract_fn = ExtractDAC(known_args.nac_input_sr)
    elif known_args.nac_name == "encodec":
        extract_fn = ExtractEncodec(known_args.nac_input_sr)

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

        npy, ecdc = (
            audio_files
            | f"Resample: {extract_fn.sample_rate}Hz"
            >> beam.ParDo(
                ResampleAudio(
                    source_sr_hint=48_000,
                    target_sr=extract_fn.sample_rate,
                )
            )
            | "ExtractNAC" >> beam.ParDo(extract_fn).with_outputs("ecdc", main="npy")
        )

        (
            npy
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )

        (ecdc | "PersistEcdcFile" >> beam.Map(write_file))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
