#!/usr/bin/env bash
set -euo pipefail

# generate a fallback version string based on the current date and time (e.g. snapshot-19700101-123456)
FALLBACK_VERSION="snapshot-$(date +%Y%m%d-%H%M%S)"

# set default values for build args if not provided
ANNOTATOR_NAME="${ANNOTATOR_NAME:-duui-neer-match}"
ANNOTATOR_VERSION="${ANNOTATOR_VERSION:-$FALLBACK_VERSION}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"



# Check if BUILD_TOOL is set, otherwise check for podman or docker
if [ -n "${BUILD_TOOL:-}" ]; then
    echo "Using ${BUILD_TOOL} for building the image"
elif command -v podman &> /dev/null; then
    echo "Using podman for building the image"
    BUILD_TOOL="podman"
elif command -v docker &> /dev/null; then
    echo "Using docker for building the image"
    BUILD_TOOL="docker"
else
    echo "Error: Neither podman nor docker is installed." >&2
    exit 1
fi

${BUILD_TOOL} build \
    --env DUUI_NEER_MATCH_ANNOTATOR_NAME="${ANNOTATOR_NAME}" \
    --env DUUI_NEER_MATCH_ANNOTATOR_VERSION="${ANNOTATOR_VERSION}" \
    --env DUUI_NEER_MATCH_LOG_LEVEL="${LOG_LEVEL}" \
    -t "${ANNOTATOR_NAME}:${ANNOTATOR_VERSION}" \
    -f DOCKERFILE \
    .
