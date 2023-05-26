import argparse
import pathlib

import apache_beam as beam
import apache_beam.io.fileio
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

# import klay_beam.audio
from klay_beam.transforms import (
    LoadWithTorchaudio,
    write_file,
    numpy_to_mp3,
    numpy_to_ogg,
    numpy_to_wav,
)


input_1 = "/Users/charles/projects/fma_large/005/00591*"
input_2 = "gs://klay-datasets/char-lossless-50gb/The Beatles/**"
output_1 = "/Users/charles/projects/klay/python/klay-beam/output/{}.mp3"
output_2 = "/Users/charles/projects/klay/python/klay-beam/output/ogg/{}.ogg"
output_3 = "/Users/charles/projects/klay/python/klay-beam/output/wav/{}.wav"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", dest="input", default=input_1, help="Input files (glob) to process."
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=output_3,
        help="Output format.",
    )
    return parser.parse_known_args(None)


def run():
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
            | beam.io.fileio.MatchFiles(known_args.input)
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam.io.fileio.ReadMatches()
        )

        audio_elements = readable_files | "Load audio with pytorch" >> beam.ParDo(
            LoadWithTorchaudio()
        )

        # Write each file to an mp3. Note the ungodly lambda function, which
        # interleaves the audio channels, and creates a (filename, mp3_data)
        # tuple.
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
            | "Convert to (filename, mp3_blob) tuples"
            >> beam.Map(
                lambda x: (
                    x[0],
                    numpy_to_wav(x[1].numpy(), x[2], bit_depth=24),
                )
            )
            | "Write mp3 files" >> beam.Map(write_file)
        )

        # Log every processed filename to a local file
        (
            audio_elements
            | "Get writable text" >> beam.Map(lambda x: "{}\t({})".format(x[0], x[1]))
            | "Log to local file"
            >> beam.io.WriteToText(
                "out.txt", append_trailing_newlines=True  # hard coded for now
            )
        )


if __name__ == "__main__":
    run()
