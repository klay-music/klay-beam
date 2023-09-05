import argparse
import os.path
import tarfile
import requests
import logging
import pathlib
from urllib.parse import urlparse

from google.cloud import storage
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

"""
This is a quick and dirty script for downloading the Jamendo dataset in parallel
and uploading it to GCS. If this ends up being useful, we should consider
generalizing, testing, and merging the functionality into `klay_beam`.

To run, activate a suitable python environment such as
``../environments/osx-64-klay-beam-py310.yml`.

```
# CD into the root klay_beam dir to the launch script:
python bin/run_job_get_jamendo.py --runner Direct

# Run remote job with autoscaling
python bin/run_job_get_jamendo.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers=101 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments=use_runner_v2 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/get-jamendo/ \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.6.0-py310 \
    --disk_size_gb 50 \
    --job_name 'get-jamendo-002'

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE
```

"""


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_known_args(None)


class GetPaths(beam.PTransform):
    def __init__(self):
        pass

    def expand(self, pcoll) -> beam.PCollection[str]:
        urls = [
            f"https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/raw_30s_audio-{i:02}.tar"
            for i in range(0, 100)
        ]
        return pcoll.pipeline | beam.Create(urls)


class HandleTar(beam.DoFn):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name

    def process(self, url: str):
        logging.info(f"Received: {url}")

        working_dir = pathlib.Path(os.path.expanduser("~/jamendo-tmp/"))
        ensure_dir_exists(working_dir)

        path = pathlib.Path(urlparse(url).path)
        local_tar_path = working_dir / path.name

        # check if tar already exists
        if os.path.exists(local_tar_path):
            logging.info(f"Tar already exists: {local_tar_path}")
        else:
            logging.info(f"Saving to: {local_tar_path}")
            download_tar(url, local_tar_path)

        local_extract_path = working_dir / "extract" / path.name
        logging.info(f"Extracting to: {local_extract_path}")
        ensure_dir_exists(local_extract_path)
        extract_tar(local_tar_path, local_extract_path)
        logging.info("Done extracting tar file")

        # we have to upload in this DoFn because we depend on the local filesystem
        for file_path in list_files_recursive(local_extract_path):
            relative_path = pathlib.Path(file_path).relative_to(local_extract_path)
            gcs_path = f"mtg-jamendo/{relative_path}"

            local_path = str(file_path)
            full_gcs_path = f"gs://{self.bucket_name}/{gcs_path}"
            logging.info(f"Uploading {local_path} to {full_gcs_path}")

            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logging.info(f"Uploaded: {full_gcs_path}")
            yield full_gcs_path


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            | GetPaths()
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | beam.ParDo(HandleTar(bucket_name="klay-datasets-001"))
            | beam.Map(lambda x: logging.info(f"Uploaded: {x}"))
        )


def download_tar(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    save_path = pathlib.Path(save_path)
    dl_name = pathlib.Path(save_path).with_suffix(".dl")

    with open(dl_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    os.rename(dl_name, save_path)


def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_path)


def ensure_dir_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def list_files_recursive(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
