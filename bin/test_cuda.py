import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import logging
import torch

"""
To run locally
python bin/test_cuda.py \
    --runner Direct

Remotely
us-docker.pkg.dev/klay-home/klay-docker/klay-beam-cuda:latest
"""
def log_message(element):
    is_available = torch.cuda.is_available()
    logging.info('Checking CUDA availability!')
    logging.info('CUDA availability: %s', is_available)
    logging.info('CUDA device count: %s', torch.cuda.device_count())

    if is_available:
        logging.info('CUDA current device: %s', torch.cuda.current_device())

    try:
        # Create a dummy tensor
        tensor = torch.tensor([1.0])
        # Try to send the tensor to the GPU
        tensor_gpu = tensor.cuda()
        logging.info('Successfully created a tensor and moved it to GPU.')
    except Exception as e:
        logging.error('Error during tensor creation or GPU transfer: %s', str(e))
    return element


def parse_args():
    parser = argparse.ArgumentParser()
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

    with beam.Pipeline(options=pipeline_options) as p:
        _ = (p
             | 'CreateSingleton' >> beam.Create([None])
             | 'LogMessage' >> beam.Map(log_message))

        p.run().wait_until_finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
