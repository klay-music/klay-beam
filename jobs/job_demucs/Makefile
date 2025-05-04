.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

# If you change the image name, you should also change the DEFAULT_IMAGE in the
# launch script.
docker:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker-push:
	docker push $(DOCKER_IMAGE_NAME)

run-local:
	python bin/run_workflow.py \
		--runner Direct \
		--audio_suffix $(audio_suffix) \
		--match_pattern $(match_pattern) \

run-dataflow:
	python bin/run_workflow.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2,no_use_multiple_sdk_containers \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/demucs/ \
    --machine_type g2-standard-4 \
    --dataflow_service_options "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    --disk_size_gb 50 \
    --setup_file ./setup.py \
	  --number_of_worker_harness_threads=1 \
    --job_name $(job_name) \
	  --max_num_workers $(max_num_workers) \
	  --audio_suffix $(audio_suffix) \
	  --match_pattern $(match_pattern) \
	  --only_if_vocals \
