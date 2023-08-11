# %%
import io
import math
import numpy as np
import os.path
import torch.hub
import encodec
import encodec.modules
import encodec.quantization


# %%
import torch
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor


# %%
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
