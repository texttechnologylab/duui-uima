#!/usr/bin/env bash

set -e

MODEL="en_ner_eco_md"

LINKER=(
  "gbif_backbone"
  "taxref"
  "ncbi_taxonomy"
)

IMAGE_NAME="docker.texttechnologylab.org/taxonerd"
TAG_PREFIX="1.0"

TOTAL=${#LINKER[@]}
CURRENT=0

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

# 🧠 optional: einmal sauber starten (UNCOMMENT if needed)
# echo "🧹 Cleaning Docker builder cache..."
# ${BUILD_TOOL} builder prune -f >/dev/null 2>&1 || true

# 🎯 Progress Bar
draw_bar () {
  local progress=$1
  local total=$2
  local width=40

  local percent=$((progress * 100 / total))

  # avoid division issues
  local filled=$(( total > 0 ? width * progress / total : 0 ))
  local empty=$((width - filled))

  printf "\r["

  for ((i=0; i<filled; i++)); do printf "#"; done
  for ((i=0; i<empty; i++)); do printf "-"; done

  printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

# 🔍 check if image exists locally
image_exists () {
  ${BUILD_TOOL} image inspect "$1" > /dev/null 2>&1
}

echo "🚀 Building $TOTAL linker variants for model: $MODEL"
echo ""

for LINK in "${LINKER[@]}"; do
  CURRENT=$((CURRENT + 1))
  TAG="${IMAGE_NAME}-${LINK}:${TAG_PREFIX}"
  LOG="build_${MODEL}_${LINK}.log"

  echo ""
  echo "🔨 [$CURRENT/$TOTAL] Processing $TAG"


  if image_exists "$TAG"; then
      echo "⚡ SKIP: Image already exists ($TAG)"

#      ${BUILD_TOOL} image rm $TAG --force
  else
      echo "🔧 Building $TAG"

      if ${BUILD_TOOL} build \
          --pull \
          --build-arg TAXONERD_MODEL="$MODEL" \
          --build-arg TAXONERD_LINKER="$LINK" \
          -f src/main/docker/Dockerfile-cuda \
          -t "$TAG" \
          . > "$LOG" 2>&1
      then
          echo "✅ Built $TAG"
      else
          echo "❌ Build failed for $TAG (see $LOG)"

          echo ""
          echo "💡 Tip: if this was an apt/GPG error, run:"
          echo "   ${BUILD_TOOL} builder prune -f"
          echo "   then retry"

          exit 1
      fi
  fi

  draw_bar "$CURRENT" "$TOTAL"
done

echo ""
echo ""
echo "🎉 All builds completed!"
