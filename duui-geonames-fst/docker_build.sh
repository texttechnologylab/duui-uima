#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.3.4}"
REGISTRY="docker.texttechnologylab.org"
IMAGE_NAME="duui-geonames-fst"
PUSH="${PUSH:-true}"

build_image() {
  local variant="$1"
  local dockerfile="$2"
  local country_arg="${3:-}"

  echo "============================================================"
  echo "Building ${IMAGE_NAME}/${variant}:${VERSION}"
  echo "Dockerfile: ${dockerfile}"
  echo "============================================================"

  if [[ -n "${country_arg}" ]]; then
    docker build \
      -f "${dockerfile}" \
      -t "${IMAGE_NAME}/${variant}:${VERSION}" \
      --build-arg COUNTRY="${country_arg}" \
      .
  else
    docker build \
      -f "${dockerfile}" \
      -t "${IMAGE_NAME}/${variant}:${VERSION}" \
      .
  fi

  docker tag \
    "${IMAGE_NAME}/${variant}:${VERSION}" \
    "${IMAGE_NAME}/${variant}:latest"

  docker tag \
    "${IMAGE_NAME}/${variant}:${VERSION}" \
    "${REGISTRY}/${IMAGE_NAME}/${variant}:${VERSION}"

  docker tag \
    "${IMAGE_NAME}/${variant}:${VERSION}" \
    "${REGISTRY}/${IMAGE_NAME}/${variant}:latest"

  if [[ "${PUSH}" == "true" ]]; then
    docker push "${REGISTRY}/${IMAGE_NAME}/${variant}:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_NAME}/${variant}:latest"
  fi
}

build_image "de" "src/main/docker/single.Dockerfile" "DE"
build_image "eu" "src/main/docker/eu.Dockerfile"
build_image "europe" "src/main/docker/europe.Dockerfile"
build_image "europe-central" "src/main/docker/europe-central.Dockerfile"

echo "============================================================"
echo "Done. Built version: ${VERSION}. Push enabled: ${PUSH}"
echo "============================================================"
