import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    write_file,
    numpy_to_wav,
    SkipCompleted,
    ResampleAudio
)

from klay_beam.path import move, remove_suffix


"""
Example usage:

python bin/run_job_convert_audio.py \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road/mp3/' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/wav/' 

python bin/run_job_convert_audio.py \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/jamendo_copy' 


python bin/run_job_convert_audio.py \
    --runner DataflowRunner \
    --max_num_workers=128 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.10.0-py310 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/convert-audio/ \
    --project klay-training \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo/' \
    --target_audio_path \
        'gs://klay-datasets-001/mtg-jamendo/' \
    --machine_type n1-standard-8 \
    --job_name 'convert-jamendo-full-to-source-wav-48k-004'

    # Possible values for --source_audio_path
        'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \
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
        'gs://klay-datasets/mtg-jamendo/'
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

    parser.add_argument(
        "--source_audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff"],
        help="""
        Which format are candidate audio files saved with?
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

    # Pattern to recursively find mp3s inside source_audio_path
    match_pattern = os.path.join(known_args.input, f"**{known_args.source_audio_suffix}")
    NEW_SAMPLE_RATE = 48_000

    def rename_file(element):
        original_filename, audio_tensor, sr = element
        moved_filename = move(original_filename, known_args.input, known_args.output)
        moved_no_suffix = remove_suffix(moved_filename, known_args.source_audio_suffix)
        new_filename = f"{moved_no_suffix}.source.wav" # Caution: must resemble SkipCompleted
        logging.info(f"Renaming: {original_filename} -> {new_filename}")
        return (new_filename, audio_tensor, sr)

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | "Reshuffle" >> beam.Reshuffle()

            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=known_args.source_audio_suffix,
                    new_suffix=".source.wav", # Caution: must resemble rename_file
                    source_dir=known_args.input,
                    target_dir=known_args.output,
                )
            )

            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "Load Audio" >> beam.ParDo(LoadWithTorchaudio())
            | f"Resample: {NEW_SAMPLE_RATE}Hz"
            >> beam.ParDo(
                ResampleAudio(
                    source_sr_hint=44_100,
                    target_sr=NEW_SAMPLE_RATE,
                    output_numpy=True,
                )
            )

            | "Rename File" >> beam.Map(rename_file)
            | "Convert to Wav" >> beam.Map(lambda x: (x[0], numpy_to_wav(x[1], x[2])))
            | "Write Audio" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
