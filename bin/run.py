import argparse
import pathlib

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import logging

# import klay_beam.audio
from klay_beam.transforms import (
    LoadWithTorchaudio,
    write_file,
    numpy_to_wav,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        dest="input",
        required=True,
        help="""
        Specify the input file pattern. This can be a local path or a GCS path,
        and may use the * or ** wildcard.

        To get only some wav files, try:
        '/Users/alice/datasets/fma_large/005/00591*'

        To find all files in directory and subdirectories, use **:
        'gs://klay-datasets/char-lossless-50gb/The Beatles/**'

        This indirectly uses apache_beam.io.filesystems.FileSystems.match:
        https://beam.apache.org/releases/pydoc/2.48.0/apache_beam.io.filesystems.html#apache_beam.io.filesystems.FileSystems.match
        """,
    )

    parser.add_argument(
        "--output",
        dest="output",
        required=True,
        help="""
        Specify the output file format, using {} as a filename placeholder.

        For example:
        'gs://klay-dataflow-test-000/results/outputs/1/{}.wav'
        """,
    )
    return parser.parse_known_args(None)


def run():
    logging.basicConfig(level=logging.INFO)
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # An example from the Apache Beam documentation uses the save_main_session
    # option. They describe the motivation for this option:
    #
    # > because one or more DoFn's in this workflow rely on global context
    # > (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        readable_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(known_args.input)
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
        )

        audio_elements = readable_files | "Load audio with pytorch" >> beam.ParDo(
            LoadWithTorchaudio()
        )

        # Convert audio tensors to in-memory files. Persist resulting files.
        (
            audio_elements
            | "Creating (filename, tensor, sr) tuples"
            >> beam.Map(
                lambda x: (
                    known_args.output.format(pathlib.Path(x[0]).name),
                    x[2],
                    x[3],
                )
            )
            | "Convert to (filename, BytesIO) tuples"
            >> beam.Map(
                lambda x: (
                    x[0],
                    numpy_to_wav(x[1].numpy(), x[2]),
                )
            )
            | "Write audio files" >> beam.Map(write_file)
        )

        # Log every processed filename to a local file (this is unhelpful when
        # running remotely via Dataflow)
        (
            audio_elements
            | "Get writable text" >> beam.Map(lambda x: "{}\t({})".format(x[0], x[1]))
            | "Log to local file"
            >> beam.io.WriteToText(
                "out.txt", append_trailing_newlines=True  # hard coded for now
            )
        )

        p.run().wait_until_finish()


if __name__ == "__main__":
    run()
