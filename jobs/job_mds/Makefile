.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src bin
	black src bin

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

run-local:
	python bin/run_workflow.py \
    --runner Direct \
		--src_dir $(src_dir) \
		--dest_dir $(dest_dir) \
		--audio_suffix $(audio_suffix) \
		--suffixes $(suffixes) \
		--num_files_per_shard $(num_files_per_shard) \
		--max_dataset_size $(max_dataset_size) \
		--min_shard_idx $(min_shard_idx) \

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
    --machine_type n2-standard-4 \
    --setup_file ./setup.py \
		--number_of_worker_harness_threads 32 \
    --max_num_workers $(max_num_workers) \
    --job_name $(job_name) \
		--src_dir $(src_dir) \
		--dest_dir $(dest_dir) \
		--audio_suffix $(audio_suffix) \
		--suffixes $(suffixes) \
		--num_files_per_shard $(num_files_per_shard) \
		--max_dataset_size $(max_dataset_size) \
		--min_shard_idx $(min_shard_idx) \
