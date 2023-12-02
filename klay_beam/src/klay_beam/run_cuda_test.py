import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import logging
import torch
import sys
import os
import os.path


"""
Check if CUDA is available and log debug information.

Creating a docker image with CUDA support can be tricky. See the klay_beam
GitHub actions for an example of how to build such an image.

When running on Dataflow, the `--sdk_container_image` needs pytorch to be
installed. Unlike most klay_beam jobs, the image does not actually need the
klay_beam package installed because this file does not actually import klay_beam
(klay_beam should still be installed in the local launch environment).

Try using a local development environment such as:

conda env create -f environment/py3.10-torch1.11-cuda11.3.yml
conda env create -f environment/py3.10-torch1.11.yml
conda activate <your-env-name>

pip install apache_beam==2.51.0 klay_beam==0.12.2

# You can test your local environment
python -m klay_beam.run_cuda_test --runner Direct

# To run on Dataflow, configure the following environment variables as per the
# klay_beam README:
GCP_PROJECT_ID=<your project id>
GCP_SA_EMAIL=<your service account email>
TEMP_GS_URL=<gs:// path to a temporary directory>

python -m klay_beam.run_cuda_test \
    --region us-east4 \
    --autoscaling_algorithm NONE \
    --runner DataflowRunner \
    --service_account_email ${GCP_SA_EMAIL} \
    --disk_size_gb 50 \
    --experiments use_runner_v2 \
    --sdk_container_image \
        klaymusic/klay-beam:0.12.2-py3.10-beam2.51.0-torch1.11-cuda11.3 \
    --sdk_location container \
    --temp_location ${TEMP_GS_URL} \
    --project ${GCP_PROJECT_ID} \
    --dataflow_service_options \
        "worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver" \
    --job_name 'cuda-test'
"""


def log_debug_info(element):
    logging.info("python location: %s", sys.executable)
    logging.info("python version: %s", sys.version)
    logging.info("python path: %s", ":".join(sys.path))
    is_available = torch.cuda.is_available()
    logging.info("Checking CUDA availability!")
    logging.info("CUDA availability: %s", is_available)
    logging.info("CUDA device count: %s", torch.cuda.device_count())
    logging.info("pytorch was built with CUDA version: %s", torch.version.cuda)

    if is_available:
        logging.info("CUDA current device: %s", torch.cuda.current_device())

    try:
        # Create a dummy tensor
        tensor = torch.tensor([1.0])
        # Try to send the tensor to the GPU
        _ = tensor.cuda()
        logging.info("Successfully created a tensor and moved it to GPU.")
    except Exception as e:
        logging.error("Error during tensor creation or GPU transfer: %s", str(e))
    return element


def log_directory_contents(dir_path):
    try:
        # Check if the directory exists
        logging.info(f"-----------scanning directory: {dir_path}")
        if os.path.exists(dir_path):
            contents = os.listdir(dir_path)
            for fn in contents:
                logging.info(os.path.join(dir_path, fn))
    except Exception as e:
        # Catch any other errors (like permission errors)
        logging.info(f"Error while accessing {dir_path}: {e}")


def list_some_directories(element):
    log_directory_contents("/usr/local/nvidia")
    log_directory_contents("/usr/local/nvidia/lib64")
    log_directory_contents("/usr/local/cuda/lib64")
    log_directory_contents("/opt/apache/beam/")
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
        (
            p
            | "CreateSingleton" >> beam.Create([None])
            | "CudaDebug" >> beam.Map(log_debug_info)
            | "ListDirectories" >> beam.Map(list_some_directories)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
