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

# 🧠 optional: einmal sauber starten (UNCOMMENT if needed)
# echo "🧹 Cleaning Docker builder cache..."
# docker builder prune -f >/dev/null 2>&1 || true

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
  docker image inspect "$1" > /dev/null 2>&1
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

#      docker image rm $TAG --force
  else
      echo "🔧 Building $TAG"

      if docker build \
          --pull \
          --quiet \
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
          echo "   docker builder prune -f"
          echo "   then retry"

          exit 1
      fi
  fi

  draw_bar "$CURRENT" "$TOTAL"
done

echo ""
echo ""
echo "🎉 All builds completed!"
