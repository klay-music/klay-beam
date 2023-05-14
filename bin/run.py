import argparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import LoadWithTorchaudio


input_1 = "/Users/charles/Downloads/fma_large/005/0051*"
input_2 = "gs://klay-datasets/char-lossless-50gb/The Beatles/**"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", dest="input", default=input_1, help="Input files (glob) to process."
    )
    parser.add_argument(
        "--output",
        dest="output",
        default="out.txt",
        help="Output file to log results to.",
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
        audio_elements = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam.io.fileio.MatchFiles(known_args.input)
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam.io.fileio.ReadMatches()
            | "Load audio with pytorch" >> beam.ParDo(LoadWithTorchaudio())
        )

        # Log every processed filename to a local file
        (
            audio_elements
            | "Get writable text" >> beam.Map(lambda x: "{}\t({})".format(x[0], x[1]))
            | "Log to local file" >> beam.io.WriteToText(
                known_args.output,
                append_trailing_newlines=True
            )
        )

if __name__ == "__main__":
    run()
