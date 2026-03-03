#!/bin/bash
set -e

DUUI_POS_AG_ANNOTATOR_NAME="duui-pos-ancient-greek"
DUUI_POS_AG_ANNOTATOR_VERSION="0.1.0"

IMAGE_NAME="${DUUI_POS_AG_ANNOTATOR_NAME}"
IMAGE_TAG="${DUUI_POS_AG_ANNOTATOR_VERSION}"

echo "============================================="
echo "Building: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================="

# Build from project root, using the Dockerfile in src/main/docker/
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    -f src/main/docker/Dockerfile \
    .

echo ""
echo "============================================="
echo "   Build complete"
echo "   Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Run with:"
echo "   docker run -p 9714:9714 ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================="