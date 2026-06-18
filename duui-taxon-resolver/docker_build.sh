#!/usr/bin/env bash
set -euo pipefail

# set default values for build args if not provided
ANNOTATOR_NAME="${ANNOTATOR_NAME:-duui-taxon-resolver}"
ANNOTATOR_VERSION="${ANNOTATOR_VERSION:-1.0.0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Check if BUILD_TOOL is set, otherwise check for podman or docker
if [ -n "${BUILD_TOOL:-}" ]; then
    echo "⚙️  Using build tool: ${BUILD_TOOL}"
# Test if docker is available and can be used
elif (command -v docker > /dev/null 2>&1;) && (docker info > /dev/null 2>&1;) then
    BUILD_TOOL="docker"
    echo "⚙️  Using Docker as build tool"
elif (command -v podman > /dev/null 2>&1;) && (podman info > /dev/null 2>&1;) then
    BUILD_TOOL="podman"
    echo "⚙️  Using Podman as build tool"
else
    echo "❌ Error: No build tool found or permissions missing. Please install Docker or Podman and ensure you have permission to run it."
    exit 1
fi

${BUILD_TOOL} build \
    --env TAXON_RESOLVER_ANNOTATOR_NAME="${ANNOTATOR_NAME}" \
    --env TAXON_RESOLVER_ANNOTATOR_VERSION="${ANNOTATOR_VERSION}" \
    --env TAXON_RESOLVER_LOG_LEVEL="${LOG_LEVEL}" \
    -t "${ANNOTATOR_NAME}:${ANNOTATOR_VERSION}" \
    -f DOCKERFILE \
