#!/usr/bin/env bash
set -euo pipefail

# Build DUUI TimeX3 Docker images.
# One image = one MODEL_NAME + one MODEL_LANG.
# The Dockerfile always uses requirements.txt.
# This script copies the complete project requirements.txt unchanged into the temporary build context.
# Model artifacts are still downloaded/cached per MODEL_NAME/MODEL_LANG in the Dockerfile.
#
# Examples:
#   ./docker_build.sh microsoft de
#   ./docker_build.sh tei2go de
#   ./docker_build.sh tei2go all
#   ./docker_build.sh timexy de
#   ./docker_build.sh timexy all
#   ./docker_build.sh german-gelectra
#   ./docker_build.sh bert-got-a-date
#   ./docker_build.sh duckling de
#   ./docker_build.sh sutime de
#   ./docker_build.sh all
#   TEI2GO_LANGUAGES="de en es fr it pt" TIMEXY_LANGUAGES="de en fr" ./docker_build.sh all
#   ./docker_build.sh hf-token-classification de satyaalmasian/temporal_tagger_German_GELECTRA

export ANNOTATOR_CUDA="${ANNOTATOR_CUDA:-}"
# export ANNOTATOR_CUDA="-cuda"

export ANNOTATOR_NAME="${ANNOTATOR_NAME:-duui-time}"
export ANNOTATOR_VERSION="${ANNOTATOR_VERSION:-0.1.0}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export MODEL_CACHE_SIZE="${MODEL_CACHE_SIZE:-1}"
export DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.texttechnologylab.org/}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
PYTHON_SRC_DIR="${PYTHON_SRC_DIR:-${PROJECT_ROOT}/src/main/python}"
DOCKERFILE_SRC="${DOCKERFILE_SRC:-${PROJECT_ROOT}/src/main/docker/Dockerfile${ANNOTATOR_CUDA}}"
REQUIREMENTS_SRC="${REQUIREMENTS_SRC:-${PROJECT_ROOT}/requirements.txt}"

# Language variants.
# TEI2GO currently has separate Hugging Face spaCy packages for these six languages.
# Timexy 0.1.3 ships rules for German, English and French.
export DEFAULT_LANGUAGES="${DEFAULT_LANGUAGES:-de}"
export TEI2GO_LANGUAGES="${TEI2GO_LANGUAGES:-de en es fr it pt}"
export TIMEXY_LANGUAGES="${TIMEXY_LANGUAGES:-de en fr}"

# Used by `all`.
export TIME_MODELS="${TIME_MODELS:-microsoft duckling sutime tei2go timexy german-gelectra bert-got-a-date}"

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

sanitize_tag_part() {
  lower "$1" \
    | sed -E 's#[^a-z0-9_.-]+#-#g; s#-+#-#g; s#^-##; s#-$##'
}

contains_word() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

languages_for_model() {
  local model
  model="$(lower "$1")"

  case "${model}" in
    tei2go)
      printf '%s\n' ${TEI2GO_LANGUAGES}
      ;;
    timexy)
      printf '%s\n' ${TIMEXY_LANGUAGES}
      ;;
    bert-got-a-date)
      printf '%s\n' en
      ;;
    german-gelectra)
      printf '%s\n' de
      ;;
    *)
      printf '%s\n' ${DEFAULT_LANGUAGES}
      ;;
  esac
}

validate_model_language() {
  local model="$1"
  local lang="$2"
  local supported

  case "${model}" in
    tei2go)
      read -r -a supported <<< "${TEI2GO_LANGUAGES}"
      ;;
    timexy)
      read -r -a supported <<< "${TIMEXY_LANGUAGES}"
      ;;
    german-gelectra)
      supported=(de)
      ;;
    bert-got-a-date)
      supported=(en)
      ;;
    *)
      return 0
      ;;
  esac

  if ! contains_word "${lang}" "${supported[@]}"; then
    echo "Unsupported language '${lang}' for model '${model}'. Supported: ${supported[*]}" >&2
    exit 1
  fi
}

usage() {
  cat <<USAGE
Usage:
  $0 all
  $0 <model> [language] [model_specname]

Models:
  microsoft
  duckling
  sutime
  tei2go
  timexy
  german-gelectra
  bert-got-a-date
  hf-token-classification

Examples:
  $0 microsoft de
  $0 tei2go de
  $0 tei2go all
  $0 timexy de
  $0 timexy all
  $0 german-gelectra
  $0 bert-got-a-date
  $0 duckling de
  $0 sutime de
  $0 hf-token-classification de satyaalmasian/temporal_tagger_German_GELECTRA

For all default images:
  $0 all

For language-capable variants:
  TEI2GO_LANGUAGES="de en es fr it pt" TIMEXY_LANGUAGES="de en fr" $0 all

Variables:
  DOCKER_REGISTRY=${DOCKER_REGISTRY}
  ANNOTATOR_NAME=${ANNOTATOR_NAME}
  ANNOTATOR_VERSION=${ANNOTATOR_VERSION}
  PROJECT_ROOT=${PROJECT_ROOT}
  PYTHON_SRC_DIR=${PYTHON_SRC_DIR}
  DOCKERFILE_SRC=${DOCKERFILE_SRC}
  REQUIREMENTS_SRC=${REQUIREMENTS_SRC}
  PYTHON_IMAGE=${PYTHON_IMAGE}
  DEFAULT_LANGUAGES=${DEFAULT_LANGUAGES}
  TEI2GO_LANGUAGES=${TEI2GO_LANGUAGES}
  TIMEXY_LANGUAGES=${TIMEXY_LANGUAGES}
USAGE
}

set_model_metadata() {
  local requested_model="$1"
  local requested_lang="${2:-de}"
  local requested_spec="${3:-}"

  MODEL_NAME="$(lower "${requested_model}")"
  MODEL_LANG="$(lower "${requested_lang}")"
  MODEL_SPECNAME="${requested_spec}"
  MODEL_VERSION="${MODEL_VERSION_OVERRIDE:-}"
  MODEL_SOURCE="${MODEL_SOURCE_OVERRIDE:-}"

  validate_model_language "${MODEL_NAME}" "${MODEL_LANG}"

  case "${MODEL_NAME}" in
    microsoft)
      MODEL_SPECNAME="${MODEL_SPECNAME:-recognizers-text-suite}"
      MODEL_VERSION="${MODEL_VERSION:-1.0.2a2}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://github.com/microsoft/Recognizers-Text}"
      ;;
    duckling)
      MODEL_SPECNAME="${MODEL_SPECNAME:-duckling}"
      MODEL_VERSION="${MODEL_VERSION:-latest}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://github.com/facebook/duckling}"
      ;;
    sutime)
      MODEL_SPECNAME="${MODEL_SPECNAME:-stanford-corenlp-sutime}"
      MODEL_VERSION="${MODEL_VERSION:-latest}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://stanfordnlp.github.io/CoreNLP/sutime.html}"
      ;;
    tei2go)
      MODEL_SPECNAME="${MODEL_SPECNAME:-${MODEL_LANG}_tei2go}"
      MODEL_VERSION="${MODEL_VERSION:-0.0.0}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://github.com/hmosousa/tei2go}"
      ;;
    timexy)
      MODEL_SPECNAME="${MODEL_SPECNAME:-timexy-${MODEL_LANG}}"
      MODEL_VERSION="${MODEL_VERSION:-0.1.3}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://pypi.org/project/timexy/}"
      ;;
    german-gelectra)
      MODEL_LANG="de"
      MODEL_SPECNAME="${MODEL_SPECNAME:-satyaalmasian/temporal_tagger_German_GELECTRA}"
      MODEL_VERSION="${MODEL_VERSION:-a523f786c63a5c0542e04d22f4b42364f33ec935}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://huggingface.co/satyaalmasian/temporal_tagger_German_GELECTRA}"
      ;;
    bert-got-a-date)
      MODEL_LANG="en"
      MODEL_SPECNAME="${MODEL_SPECNAME:-satyaalmasian/temporal_tagger_BERT_tokenclassifier}"
      MODEL_VERSION="${MODEL_VERSION:-3b4029b1ec47d4bdc9ef29f6652a44d69410b09f}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://huggingface.co/satyaalmasian/temporal_tagger_BERT_tokenclassifier}"
      ;;
    hf-token-classification)
      if [[ -z "${MODEL_SPECNAME}" ]]; then
        echo "MODEL_SPECNAME is required for hf-token-classification." >&2
        echo "Example: $0 hf-token-classification de satyaalmasian/temporal_tagger_German_GELECTRA" >&2
        exit 1
      fi
      MODEL_VERSION="${MODEL_VERSION:-main}"
      MODEL_SOURCE="${MODEL_SOURCE:-https://huggingface.co/${MODEL_SPECNAME}}"
      ;;
    *)
      echo "Unsupported model: ${MODEL_NAME}" >&2
      usage >&2
      exit 1
      ;;
  esac

  export MODEL_NAME MODEL_LANG MODEL_SPECNAME MODEL_VERSION MODEL_SOURCE
}


create_build_context() {
  local build_context="$1"

  if [[ ! -f "${DOCKERFILE_SRC}" ]]; then
    echo "Dockerfile not found: ${DOCKERFILE_SRC}" >&2
    exit 1
  fi

  if [[ ! -f "${REQUIREMENTS_SRC}" ]]; then
    echo "requirements.txt not found: ${REQUIREMENTS_SRC}" >&2
    exit 1
  fi

  mkdir -p "${build_context}/src/main/docker" "${build_context}/src/main/python"
  cp "${DOCKERFILE_SRC}" "${build_context}/src/main/docker/Dockerfile"

  if [[ -d "${PYTHON_SRC_DIR}" ]]; then
    cp "${PYTHON_SRC_DIR}/TypeSystemTime.xml" "${build_context}/src/main/python/"
    cp "${PYTHON_SRC_DIR}/duui_time.py" "${build_context}/src/main/python/"
    cp "${PYTHON_SRC_DIR}/time_recognition_backend.py" "${build_context}/src/main/python/"
    cp "${PYTHON_SRC_DIR}/duui_time.lua" "${build_context}/src/main/python/"
  else
    echo "Python source dir not found: ${PYTHON_SRC_DIR}" >&2
    exit 1
  fi

  cp "${REQUIREMENTS_SRC}" "${build_context}/requirements.txt"
}

build_one() {
  local requested_model="$1"
  local requested_lang="${2:-de}"
  local requested_spec="${3:-}"

  set_model_metadata "${requested_model}" "${requested_lang}" "${requested_spec}"

  local model_tag
  local lang_tag
  local image
  local build_context

  model_tag="$(sanitize_tag_part "${MODEL_NAME}")"
  lang_tag="$(sanitize_tag_part "${MODEL_LANG}")"

  if [[ "${MODEL_NAME}" == "hf-token-classification" ]]; then
    model_tag="${model_tag}-$(sanitize_tag_part "${MODEL_SPECNAME}")"
  fi

  image="${DOCKER_REGISTRY}${ANNOTATOR_NAME}-${model_tag}-${lang_tag}:${ANNOTATOR_VERSION}${ANNOTATOR_CUDA}"
  build_context="$(mktemp -d -t duui-time-build-${model_tag}-${lang_tag}-XXXXXX)"

  create_build_context "${build_context}"

  echo "============================================================"
  echo "Building ${image}"
  echo "  MODEL_NAME=${MODEL_NAME}"
  echo "  MODEL_SPECNAME=${MODEL_SPECNAME}"
  echo "  MODEL_VERSION=${MODEL_VERSION}"
  echo "  MODEL_SOURCE=${MODEL_SOURCE}"
  echo "  MODEL_LANG=${MODEL_LANG}"
  echo "  PYTHON_IMAGE=${PYTHON_IMAGE}"
  echo "  Dockerfile uses copied requirements: ${REQUIREMENTS_SRC}"
  echo "============================================================"
  echo "Copied requirements.txt:"
  sed 's/^/  /' "${build_context}/requirements.txt"
  echo "============================================================"

  docker build \
    --build-arg PYTHON_IMAGE="${PYTHON_IMAGE}" \
    --build-arg ANNOTATOR_NAME="${ANNOTATOR_NAME}" \
    --build-arg ANNOTATOR_VERSION="${ANNOTATOR_VERSION}" \
    --build-arg LOG_LEVEL="${LOG_LEVEL}" \
    --build-arg MODEL_CACHE_SIZE="${MODEL_CACHE_SIZE}" \
    --build-arg MODEL_NAME="${MODEL_NAME}" \
    --build-arg MODEL_SPECNAME="${MODEL_SPECNAME}" \
    --build-arg MODEL_VERSION="${MODEL_VERSION}" \
    --build-arg MODEL_SOURCE="${MODEL_SOURCE}" \
    --build-arg MODEL_LANG="${MODEL_LANG}" \
    --build-arg TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" \
    -t "${image}" \
    -f "${build_context}/src/main/docker/Dockerfile" \
    "${build_context}"

  docker tag \
    "${image}" \
    "${DOCKER_REGISTRY}${ANNOTATOR_NAME}-${model_tag}-${lang_tag}:latest${ANNOTATOR_CUDA}"

  if [[ "${KEEP_BUILD_CONTEXT:-0}" == "1" ]]; then
    echo "Keeping build context: ${build_context}"
  else
    rm -rf "${build_context}"
  fi
}

build_all() {
  local model
  local lang

  for model in ${TIME_MODELS}; do
    case "$(lower "${model}")" in
      hf-token-classification)
        if [[ -n "${HF_MODEL_SPECNAME:-}" ]]; then
          build_one "hf-token-classification" "${HF_MODEL_LANG:-de}" "${HF_MODEL_SPECNAME}"
        else
          echo "Skipping hf-token-classification in all mode because HF_MODEL_SPECNAME is not set."
        fi
        ;;
      *)
        while IFS= read -r lang; do
          [[ -z "${lang}" ]] && continue
          build_one "${model}" "${lang}"
        done < <(languages_for_model "${model}")
        ;;
    esac
  done
}

build_model_all_languages() {
  local model="$1"
  local spec="${2:-}"
  local lang

  while IFS= read -r lang; do
    [[ -z "${lang}" ]] && continue
    build_one "${model}" "${lang}" "${spec}"
  done < <(languages_for_model "${model}")
}

main() {
  local command="${1:-all}"

  case "${command}" in
    -h|--help|help)
      usage
      ;;
    all)
      build_all
      ;;
    *)
      if [[ "${2:-}" == "all" ]]; then
        build_model_all_languages "${command}" "${3:-}"
      else
        build_one "${command}" "${2:-de}" "${3:-}"
      fi
      ;;
  esac
}

main "$@"