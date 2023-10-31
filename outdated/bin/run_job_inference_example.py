import io
import math
import numpy as np
import os.path
import argparse
import pathlib
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor

import torch
import torch.hub
import encodec
import encodec.modules
import encodec.quantization

from klay_beam.transforms import (
    LoadWithTorchaudio,
    write_file,
)

from klay_data.transform import convert_audio


"""
This is an example use of the apache beam RunInference interface. As of August
16, 2023, I found it easier to just instantiate models in a DoFn's setup method
than to use the RunInference interface. However, I am leaving this code here
because we may want to use RunInference in the future, and it's interface is not
well documented. RunInference does support batched which would be non-trivial
to setup manually.
"""


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


def format_output_path(pattern: str, source_filename: str):
    """
    Given a pattern like 'gs://data/outputs/01/{}.wav'
    and a source_filename like 'gs://klay-datasets/jamendo/00/okay.what.mp3'
    return 'gs://data/outputs/01/okay.wav'
    """
    path = pathlib.Path(source_filename)
    id = str(path.parent / path.name.split(".")[0])
    return pattern.format(pathlib.Path(id).name)


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
            | "Creating (output_filename, tensor, sr) tuples"
            >> beam.Map(
                lambda x: (
                    format_output_path(known_args.output, x[0]),
                    x[1],
                    x[2],
                )
            )
            | "Convert to (filename, mono_numpy) tuples"
            >> beam.Map(
                lambda x: (
                    x[0],
                    convert_audio(x[1], x[2], 24_000, 1),
                )
            )
            | "Extract tokens" >> RunInference(model_handler=get_model_handler())
            | "Handle keyed result" >> beam.Map(handle_keyed_result)
            | "Write files" >> beam.Map(write_file)
        )

        # Log every processed filename to a local file (this is unhelpful when
        # running remotely via Dataflow)
        (
            audio_elements
            | "Get writable text" >> beam.Map(lambda x: x[0])
            | "Log to local file"
            >> beam.io.WriteToText(
                "out.txt", append_trailing_newlines=True  # hard coded for now
            )
        )


def get_encodec_model_params():
    """Get parameters suitable for EncodecModel(**params)

    Usually, you would just call EncodecModel.encodec_model_24khz(). However the
    apache beam RunInference interface requires that the model be constructed
    with a **params argument. This params returned from this function were
    compiled by reverse engineering the process of calling

    ```
    model = encodec.EncodecModel.e
    model.set_target_bandwidth(24.0).
    ```

    Alternatively, we could write a wrapper function, but that necessitates
    saving custom model parameters to disk because the Beam API ALSO requires
    that we pass in an path to parameters that will be automatically loaded via
    model.load_state_dict(torch.load(params_path)). Ultimately I decided that
    just passing in the params dict was simpler and cleaner than maintaining a
    custom wrapper AND a custom parameters file.
    """
    target_bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
    # bandwidth = 24.0
    sample_rate = 24_000
    channels = 1
    causal: bool = True
    model_norm: str = "weight_norm"
    name = "encodec_24khz"
    encoder = encodec.modules.SEANetEncoder(
        channels=channels, norm=model_norm, causal=causal
    )
    decoder = encodec.modules.SEANetDecoder(
        channels=channels, norm=model_norm, causal=causal
    )
    n_q = int(
        1000
        * target_bandwidths[-1]
        // (math.ceil(sample_rate / encoder.hop_length) * 10)
    )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "quantizer": encodec.quantization.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        ),
        "target_bandwidths": target_bandwidths,
        # "bandwidth": bandwidth,
        "sample_rate": sample_rate,
        "channels": channels,
        "normalize": False,
        "segment": None,
        "name": name,
    }


def get_encodec_state_dict_url():
    """Apache Beam requires that we pass in a path to the model parameters.
    Usually we would just call EncodecModel.encodec_model_24khz() which
    automatically downloads the parameters to a local cache.
    """

    # The path below is hard coded in the encodec package, but it is not exposed
    # by the API. As a result, there is a risk that a future version of the
    # package changes this path and breaks this code. If this assertion fails,
    # we must inspect the source code of the encodec package ane ensure the path
    # below is correct.
    assert encodec.__version__ == "0.1.1"
    return "https://dl.fbaipublicfiles.com/encodec/v0/" + "encodec_24khz-d7cc33bc.th"


def get_encodec_state_dict_local_path():
    return os.path.join(torch.hub.get_dir(), "checkpoints", "encodec_24khz-d7cc33bc.th")


# References
# https://cloud.google.com/dataflow/docs/notebooks/run_inference_generative_ai
# https://beam.apache.org/documentation/transforms/python/elementwise/runinference-pytorch/


def handle_keyed_result(keyed_result):
    """Convert a keyed result to a numpy array"""
    key, result = keyed_result
    in_memory_file = io.BytesIO()
    np.save(in_memory_file, result.inference.numpy())
    in_memory_file.seek(0)
    return (key, in_memory_file)


def create_encodec_model():
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(24.0)
    model.eval()
    return model


def get_model_handler():
    return KeyedModelHandler(
        PytorchModelHandlerTensor(
            min_batch_size=1,
            max_batch_size=2,
            model_class=create_encodec_model,
            model_params={},
            state_dict_path=get_encodec_state_dict_local_path(),
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
