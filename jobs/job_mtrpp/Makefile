.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 src tests bin
	black --check src tests bin

type-check:
	mypy src --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

docker:
	docker build -t ${DOCKER_IMAGE_NAME} .

docker-push:
	docker push ${DOCKER_IMAGE_NAME}

run-local:
	python bin/run_workflow.py \
		--runner Direct \
		--source_audio_path=$(source_audio_path) \
		--match_suffix=$(match_suffix) \
		--audio_suffix=$(audio_suffix) \
		--overwrite

run-dataflow:
	python bin/run_workflow.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2,no_use_multiple_sdk_containers \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/mtrpp/ \
    --machine_type g2-standard-4 \
    --dataflow_service_options "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    --disk_size_gb 50 \
    --setup_file ./setup.py \
    --job_name $(job_name) \
		--max_num_workers $(max_num_workers) \
		--source_audio_path=$(source_audio_path) \
		--match_suffix=$(match_suffix) \
		--audio_suffix=$(audio_suffix) \
		--number_of_worker_harness_threads=3
