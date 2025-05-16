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
	git submodule update --init --recursive
	rm -fr submodules; mkdir -p submodules
	cp -r ../submodules/klay-data ./submodules/klay-data
	docker buildx build --platform linux/amd64 -t $(DOCKER_IMAGE_NAME) .
	rm -fr submodules

docker-push:
	docker push $(DOCKER_IMAGE_NAME)

run-local:
	pythoe bin/run_workflow.py \
		--runner Direct \
		--match_pattern=$(match_pattern)


run-dataflow:
	python bin/run_workflow.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2,no_use_multiple_sdk_containers \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/byt5/ \
    --machine_type n1-standard-4 \
    --dataflow_service_options "worker_accelerator=type:nvidia-tesla-t4;count:1;install-nvidia-driver" \
    --disk_size_gb 50 \
    --setup_file ./setup.py \
	--number_of_worker_harness_threads=2 \
    --job_name $(job_name) \
	--max_num_workers $(max_num_workers) \
	--match_pattern $(match_pattern)
