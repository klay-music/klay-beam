.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

run-local:
	python bin/run_workflow.py \
    --runner Direct \
		--src_dir $(src_dir) \
		--audio_suffix $(audio_suffix) \
		--target_audio_suffix $(target_audio_suffix) \
		--target_sample_rate $(target_sample_rate) \
		--crop_duration $(crop_duration)

run-dataflow:
	python bin/run_workflow.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2,no_use_multiple_sdk_containers \
    --sdk_location container \
    --temp_location gs://klay-beam-scratch-storage/tmp/job_copy/ \
    --machine_type t2d-standard-1 \
    --setup_file ./setup.py \
		--number_of_worker_harness_threads 6 \
    --max_num_workers $(max_num_workers) \
    --job_name $(job_name) \
		--src_dir $(src_dir) \
		--audio_suffix $(audio_suffix) \
		--target_audio_suffix $(target_audio_suffix) \
		--target_sample_rate $(target_sample_rate)
