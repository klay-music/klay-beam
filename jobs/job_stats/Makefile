.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 src tests bin
	black --check src tests bin

type-check:
	mypy src --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

help:
	python bin/run_workflow.py --help

docker:
	mkdir -p submodules
	cp -r ../submodules/klay-data/ submodules/
	docker build -t ${DOCKER_IMAGE_NAME} .
	rm -r submodules/klay-data

docker-push:
	docker push ${DOCKER_IMAGE_NAME}

run-local:
	python bin/run_workflow.py \
		--runner Direct \
		--source_audio_path $(source_audio_path) \
		--match_suffix $(match_suffix)

run-dataflow:
	python bin/run_workflow.py \
		--runner DataflowRunner \
		--project klay-training \
		--service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
		--region us-central1 \
		--autoscaling_algorithm THROUGHPUT_BASED \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/job_stats/ \
    --machine_type t2d-standard-1 \
    --experiments=use_runner_v2,no_use_multiple_sdk_containers \
    --setup_file ./setup.py \
    --number_of_worker_harness_threads=4 \
    --job_name $(job_name) \
    --max_num_workers $(max_num_workers) \
    --source_audio_path $(source_audio_path) \
    --match_suffix $(match_suffix)
