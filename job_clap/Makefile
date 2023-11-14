.PHONY: code-style type-check pre-test tests docker docker-push

code-style:
	flake8 src tests bin
	black --check src tests bin

type-check:
	mypy src --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

# If you change the image name, you should also change the DEFAULT_IMAGE in the
# launch script.
DOCKER_IMAGE_NAME=us-docker.pkg.dev/klay-home/klay-docker/klay-beam-clap
docker:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker-push:
	docker push $(DOCKER_IMAGE_NAME)
