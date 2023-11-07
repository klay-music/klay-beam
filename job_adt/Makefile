.PHONY: code-style type-check pre-test tests

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
DOCKER_IMAGE_NAME=us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:0.4.2

docker:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker-push:
	docker push $(DOCKER_IMAGE_NAME)
