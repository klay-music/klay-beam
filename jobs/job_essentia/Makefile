.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

docker:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker-push:
	docker push $(DOCKER_IMAGE_NAME)

make run-dataflow:
	python bin/run_workflow.py \
	    --runner DataflowRunner \
	    --project klay-training \
	    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
	    --region us-central1 \
	    --autoscaling_algorithm THROUGHPUT_BASED \
	    --experiments use_runner_v2,no_use_multiple_sdk_containers \
	    --sdk_location container \
	    --temp_location gs://klay-beam-scratch-storage/tmp/extract-essentia/ \
	    --machine_type n1-standard-16 \
	    --number_of_worker_harness_threads=8 \
	    --setup_file ./setup.py \
	    --src_dir $(src_dir) \
	    --audio_suffix $(audio_suffix) \
	    --features $(features) \
	    --job_name $(job_name) \
	    --max_num_workers $(max_num_workers) \
