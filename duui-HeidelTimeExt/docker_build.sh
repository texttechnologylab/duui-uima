#!/usr/bin/env bash
set -euo pipefail

# Build and optionally push the DUUI HeidelTimeExt Docker image.
#
# Examples:
#   ./docker_build.sh
#   ./docker_build.sh 1.0
#   PUSH=true ./docker_build.sh 1.0
#
# Optional environment variables:
#   ANNOTATOR_NAME=duui-heideltime-ext
#   DOCKER_REGISTRY=docker.texttechnologylab.org/
#   PUSH=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VERSION="${1:-${ANNOTATOR_VERSION:-1.0}}"
ANNOTATOR_NAME="${ANNOTATOR_NAME:-duui-heideltime-ext}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.texttechnologylab.org/}"
PUSH="${PUSH:-false}"

LOCAL_VERSION_TAG="${ANNOTATOR_NAME}:${VERSION}"
LOCAL_LATEST_TAG="${ANNOTATOR_NAME}:latest"
REMOTE_VERSION_TAG="${DOCKER_REGISTRY}${ANNOTATOR_NAME}:${VERSION}"
REMOTE_LATEST_TAG="${DOCKER_REGISTRY}${ANNOTATOR_NAME}:latest"

printf '\nBuilding %s\n' "${LOCAL_VERSION_TAG}"
docker build -f dockerfile \
  --build-arg ANNOTATOR_VERSION="${VERSION}" \
  -t "${LOCAL_VERSION_TAG}" \
  .

docker tag "${LOCAL_VERSION_TAG}" "${LOCAL_LATEST_TAG}"
docker tag "${LOCAL_VERSION_TAG}" "${REMOTE_VERSION_TAG}"
docker tag "${LOCAL_VERSION_TAG}" "${REMOTE_LATEST_TAG}"

printf '\nBuilt images:\n'
printf '  %s\n' "${LOCAL_VERSION_TAG}" "${LOCAL_LATEST_TAG}" "${REMOTE_VERSION_TAG}" "${REMOTE_LATEST_TAG}"

if [[ "${PUSH}" == "true" ]]; then
  printf '\nPushing images:\n'
  docker push "${REMOTE_VERSION_TAG}"
  docker push "${REMOTE_LATEST_TAG}"
fi
