import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from .torch_transforms import LoadWithTorchaudio


"""
# Example usage:

```bash
python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path \
        '/local/path/to/mp3s/' \

python -m klay_beam.run_example \
    --runner Direct \
    --source_audio_suffix .mp3 \
    --source_audio_path ${AUDIO_URL}


KLAY_BEAM_CONTAINER=us-docker.pkg.dev/<your-gcp-project>/<your-docker-artifact-registry>/<your-docker-image>:<tag>
SERVICE_ACCOUNT_EMAIL=dataset-dataflow-worker@klay-training.iam.gserviceaccount.com
TEMP_GS_URL=gs://<your-gs-bucket>/<your-writable-dir/>
AUDIO_URL='gs://<your-audio-bucket>/audio/'


python -m klay_beam.run_example \
    --runner DataflowRunner \
    --max_num_workers=128 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email ${SERVICE_ACCOUNT_EMAIL} \
    --experiments=use_runner_v2 \
    --sdk_container_image ${KLAY_BEAM_CONTAINER} \
    --sdk_location=container \
    --temp_location ${TEMP_GS_URL} \
    --project klay-training \
    --source_audio_suffix .mp3 \
    --source_audio_path ${AUDIO_URL} \
    --machine_type n1-standard-8 \
    --job_name 'example-job-000'
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
        'gs://klay-datasets/mtg-jamendo/'
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
    match_pattern = os.path.join(
        known_args.input, f"**{known_args.source_audio_suffix}"
    )

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        _, failed, durations = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | "Reshuffle" >> beam.Reshuffle()
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "Load Audio"
            >> beam.ParDo(LoadWithTorchaudio()).with_outputs(
                "failed", "duration_seconds", main="audio"
            )
        )

        # log duration of audio
        (
            durations
            | "SumLengths" >> beam.CombineGlobally(sum)
            | "LogDuration"
            >> beam.Map(
                lambda x: logging.info(
                    "Total duration of loaded audio: "
                    f"~= {x:.3f} seconds "
                    f"~= {x / 60:.3f} minutes "
                    f"~= {x / 60 / 60:.3f} hours"
                )
            )
        )

        # log count of number of audio files
        (
            durations
            | "Count" >> beam.combiners.Count.Globally()
            | "LogCount"
            >> beam.Map(lambda x: logging.info(f"Total number of audio files: {x}"))
        )

        # log number of audio files that failed to load
        (
            failed
            | "Log Failed" >> beam.Map(lambda x: logging.warning(x))
            | "Count Failed" >> beam.combiners.Count.Globally()
            | "Log Failed Count"
            >> beam.Map(lambda x: logging.warning(f"Failed to decode {x} files"))
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
