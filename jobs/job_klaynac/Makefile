.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 bin src tests
	black --check bin src tests

type-check:
	mypy bin src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

# If you change the image name, you should also change the DEFAULT_IMAGE in the
# launch script.
docker:
	mkdir -p models/klaynac/$(KLAY_CODECS_VERSION) models/klaynacvae/$(KLAY_CODECS_VERSION)
	echo "Copying models from klay-codecs@$(KLAY_CODECS_VERSION)"
	cp submodules/klay-codecs/models/klaynac/* models/klaynac/$(KLAY_CODECS_VERSION)/
	cp submodules/klay-codecs/models/klaynacvae/* models/klaynacvae/$(KLAY_CODECS_VERSION)/
	docker buildx build --platform linux/amd64 -t $(DOCKER_IMAGE_NAME) .
	rm -fr models

docker-push:
	docker push $(DOCKER_IMAGE_NAME)

run-local:
	python bin/run_workflow.py \
		--runner Direct \
		--model_type=discrete \
		--audio_suffix=$(audio_suffix) \
		--source_audio_path=$(source_audio_path) \
		--match_suffix=$(match_suffix)

run-dataflow:
	python bin/run_workflow.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2,no_use_multiple_sdk_containers \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/klaynac/ \
    --machine_type g2-standard-4 \
		--dataflow_service_options "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    --disk_size_gb 50 \
    --setup_file ./setup.py \
    --job_name $(job_name) \
	--max_num_workers $(max_num_workers) \
	--source_audio_path $(source_audio_path) \
	--match_suffix=$(match_suffix) \
	--model_type=continuous \
	--audio_suffix=$(audio_suffix) \
	--number_of_worker_harness_threads=1
