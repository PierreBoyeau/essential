IMAGE_NAME ?= docker.io/pierreboyeau/essential
GIT_COMMIT := $(shell git rev-parse --short HEAD)

.PHONY: all build push pull

all: build

build:
	@echo "Pulling latest image to use as cache"
	@docker pull "${IMAGE_NAME}:latest" || true
	@if [ -n "$$(docker images -q ${IMAGE_NAME}:latest 2>/dev/null)" ]; then \
		echo "--- Using cache from ${IMAGE_NAME}:latest"; \
		docker build \
			--cache-from "${IMAGE_NAME}" \
			-t "${IMAGE_NAME}:latest" \
			-t "${IMAGE_NAME}:${GIT_COMMIT}" \
			.; \
	else \
		echo "--- No cache image found, building from scratch"; \
		docker build \
			-t "${IMAGE_NAME}:latest" \
			-t "${IMAGE_NAME}:${GIT_COMMIT}" \
			.; \
	fi

push:
	@echo "Pushing image: ${IMAGE_NAME}:latest and ${IMAGE_NAME}:${GIT_COMMIT}"
	docker push "${IMAGE_NAME}:latest"
	docker push "${IMAGE_NAME}:${GIT_COMMIT}"

pull:
	@echo "Pulling image: ${IMAGE_NAME}:latest"
	docker pull "${IMAGE_NAME}:latest" || true
