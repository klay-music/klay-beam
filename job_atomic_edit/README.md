# job_atomic_edit

Parsing Song stem files into (source, edit, target) triplets for editing model, using only *atomic* edits.

An edit is *atomic* if it only specifies direct composition of stems, such as:
1. Adding tracks
2. Removing tracks
3. Replacing tracks with other instruments
4. Extracting tracks

The general pipeline is as follows:
1. Recursively search a path for `.wav` files for each song
2. For each song (i.e. group of `.wav` files in DBVO), generate all the corresponding edit triplets
3. Write the results out as `[SONG-ID].[src|tgt].[EDIT-ID].wav`

Note that `run_job_extract_atomic` accepts the optional argument `--t`, which specifies the maximum
length for each output triplet in seconds (if not provided, outputs will include the entire song).

To run, activate the conda dev+launch environment: `environment/dev.yml`.

```bash
# Example invocation to run locally
python bin/run_job_extract_atomic.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_input_sr 44100 \
    --audio_suffix .wav \
    --t 10 \

python bin/run_job_extract_atomic.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_input_sr 48000 \
    --audio_suffix .wav \
    --t 10 \

# Run remote job with autoscaling
python bin/run_job_extract_atomic.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --max_num_workers 1000 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract-ecdc-48k/ \
    --setup_file ./setup.py \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --nac_input_sr 48000 \
    --audio_suffix .wav \
    --machine_type n1-standard-2 \
    --number_of_worker_harness_threads 2 \
    --job_name 'extract-ecdc-002' \
    --t 10 \


# Possible test values for --source_audio_path
    'klay-dataflow-test-000/test-audio/abbey_road/wav' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE
```

# Development
## Quick Start
Install dependencies (we highly recommend creating and activating a virtual
python environment first):

```sh
pip install [-e] '.[code-style, type-check, tests]'
```

## Dependencies
### conda
We use `pip` to handle python dependencies.  To create or update an environment:

```sh
pip install [-e] '.[code-style, type-check, tests]'
```

All dependencies are listed in the `pyproject.toml` file in the 'dependencies'
section.

## Code Quality
### Testing
We use `pytest` for testing, there's no coverage target at the moment but
essential functions and custom logic should definitely be tested. To run the
tests:
```sh
make tests
```

### Code Style
We use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

### Static Typing
We check static types using `mypy`.
```sh
make type-check
```
