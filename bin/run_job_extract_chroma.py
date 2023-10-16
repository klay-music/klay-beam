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
    ExtractChromaFeatures,
    SkipCompleted,
    write_file,
    numpy_to_file,
)


"""
Job for extracting EnCodec and Chroma features:

1. Recursively search a path for `.<something>.wav` files
   (`--source_audio_path`) where `<something>` is one of `source`, `bass`,
   `vocals`, `drums`, or `other` as specified by `--stem`. If no stem is given,
    all .wav files will be extracted.
1. For each audio file, extract features
1. Write the results to an .npy file adjacent to the source audio file

To run, activate a suitable python environment such as one of
- `klay_beam/environment/py310-torch.local.yml`
- `klay_beam/environment/py310-torch.linux-64.yml`

```
# CD into the root klay_beam dir to the launch script:
python bin/run_job_extract_chroma.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'

# Run remote job with autoscaling
python bin/run_job_extract_chroma.py \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers=550 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --experiments=use_runner_v2 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract_chroma/ \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.8.1-py310 \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --job_name 'extract-chroma-006'

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
        "--stem",
        dest="stem",
        required=False,
        default=None,
        choices=["source", "bass", "drums", "other", "vocals"],
        help="The stem to extract",
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Pattern to recursively find mp3s inside source_audio_path

    match_pattern = os.path.join(known_args.input, "**.wav")
    if known_args.stem is not None:
        match_pattern = os.path.join(known_args.input, f"**.{known_args.stem}.wav")

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
                    old_suffix=".wav",
                    # CAUTION! This if we change the chroma parameters, we need to change this too
                    new_suffix=".chroma_50hz.npy",
                    check_timestamp=True,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        chroma_audio_sr = 16_000

        (
            audio_files
            | "Resample: 16k"
            >> beam.ParDo(
                ResampleAudio(
                    source_sr_hint=48_000,
                    target_sr=chroma_audio_sr,
                )
            )
            | "ExtractChroma"
            >> beam.ParDo(
                ExtractChromaFeatures(
                    # CAUTION! This if we change the chroma parameters, we need
                    # to also update the SkipCompleted transform.
                    audio_sr=chroma_audio_sr,
                    n_chroma=12,
                    n_fft=2048,
                    win_length=1280,
                    hop_length=320,
                    norm=1,
                )
            )
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
