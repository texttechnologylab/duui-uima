#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ANNOTATOR_NAME="duui-coh-metrix"
VERSION="0.1.1"

SPACY_IMAGE="docker.texttechnologylab.org/duui-spacy-benepar-en-de-fr:0.5.1"
SYNTOK_IMAGE="docker.texttechnologylab.org/duui-syntok:0.0.3"
COHMETRIX_IMAGE="docker.texttechnologylab.org/duui-coh-metrix:${VERSION}"
COHMETRIX_IMAGE_WAS_SET=false

SPACY_CONTAINER_NAME="duui-spacy-pipeline"
SYNTOK_CONTAINER_NAME="duui-syntok-pipeline"
COHMETRIX_CONTAINER_NAME="duui-coh-metrix-pipeline"

SPACY_HOST_PORT=32778
SYNTOK_HOST_PORT=32779
COHMETRIX_HOST_PORT=32780

SPACY_CONTAINER_PORT=9714
SYNTOK_CONTAINER_PORT=9714
COHMETRIX_CONTAINER_PORT=9714

LOG_LEVEL="INFO"
LOG_DIR=""
LOG_TAIL_LINES=200
GERMANET_PATH=""
BUILD_COHMETRIX_IMAGE=false
FOREGROUND=true
STOP_ONLY=false
CONTAINER_GERMANET_PATH="/usr/src/app/src/main/resources/germanet"

STARTED_CONTAINERS=()
LOG_FOLLOW_PIDS=()

usage() {
    cat <<'EOF'
Start the three DUUI component containers from pinned images.

Components and default host ports:
  spaCy       docker.texttechnologylab.org/duui-spacy-benepar-en-de-fr:0.5.1  32778 -> 9714
  Syntok     docker.texttechnologylab.org/duui-syntok:0.0.3                32779 -> 9714
  Coh-Metrix docker.texttechnologylab.org/duui-coh-metrix:0.1.1            32780 -> 9714

Usage:
  ./src/main/bash/run_pipeline.sh [options]

Images:
  --spacy-image TAG          spaCy image (default: pinned spaCy 3.7.2 image)
  --syntok-image TAG         Syntok image
  --coh-metrix-image TAG     Coh-Metrix image
  --image TAG                Backward-compatible alias for --coh-metrix-image
  --version VERSION          Coh-Metrix image/build version (default: 0.1.1)
  --build                    Build the selected Coh-Metrix image from this checkout
  --no-build                 Use/pull the selected Coh-Metrix image (default)

Ports:
  --spacy-port PORT          spaCy host port (default: 32778)
  --syntok-port PORT         Syntok host port (default: 32779)
  --coh-metrix-port PORT     Coh-Metrix host port (default: 32780)
  --port PORT                Backward-compatible alias for --coh-metrix-port

Container names:
  --spacy-name NAME          spaCy container name
  --syntok-name NAME         Syntok container name
  --coh-metrix-name NAME     Coh-Metrix container name
  --name NAME                Backward-compatible alias for --coh-metrix-name

Other options:
  --log-level LEVEL          Coh-Metrix log level (default: INFO)
  --log-dir PATH             Also save prefixed component logs below PATH
  --log-tail-lines NUMBER    Lines shown per component after an error (default: 200)
  --germanet PATH            Mount licensed GermaNet read-only into Coh-Metrix
  --foreground               Follow all component logs; stop all three on exit (default)
  --detach                   Start the services without following their logs
  --stop                     Stop all three named containers and exit
  -h, --help                 Show this help

Examples:
  ./src/main/bash/run_pipeline.sh
  ./src/main/bash/run_pipeline.sh --build --coh-metrix-image duui-coh-metrix:local
  ./src/main/bash/run_pipeline.sh --log-dir ./logs
  ./src/main/bash/run_pipeline.sh --detach
  ./src/main/bash/run_pipeline.sh --germanet /absolute/path/to/GermaNet/GN_V190
  ./src/main/bash/run_pipeline.sh --stop

The script starts the component services. A DUUI runner must still connect to
them in this order: spaCy -> Syntok -> Coh-Metrix. Configure Syntok in that
runner with write_sentences=false and write_paragraphs=true.
EOF
}

docker_cli() {
    case "${OSTYPE:-}" in
        msys*|cygwin*|win32*) MSYS_NO_PATHCONV=1 docker "$@" ;;
        *) docker "$@" ;;
    esac
}

container_exists() {
    docker_cli container inspect "$1" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker_cli container inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

stop_container() {
    local name="$1"
    if container_exists "${name}"; then
        docker_cli container rm --force "${name}" >/dev/null
        printf 'Stopped and removed container %s.\n' "${name}"
    else
        printf 'Container %s is not running.\n' "${name}"
    fi
}

cleanup_started_containers() {
    local index name
    for ((index=${#STARTED_CONTAINERS[@]} - 1; index >= 0; index--)); do
        name="${STARTED_CONTAINERS[index]}"
        if container_exists "${name}"; then
            docker_cli container rm --force "${name}" >/dev/null 2>&1 || true
        fi
    done
    STARTED_CONTAINERS=()

    local pid
    for pid in "${LOG_FOLLOW_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
    LOG_FOLLOW_PIDS=()
}

print_component_logs() {
    local label="$1"
    local name="$2"

    container_exists "${name}" || return 0
    printf '\nLast %s log lines from %s:\n' "${LOG_TAIL_LINES}" "${label}" >&2
    docker_cli logs --tail "${LOG_TAIL_LINES}" "${name}" 2>&1 |
        while IFS= read -r line; do
            printf '[%s] %s\n' "${label}" "${line}" >&2
        done || true
}

dump_started_logs() {
    local index
    for ((index=0; index<${#STARTED_CONTAINERS[@]}; index++)); do
        case "${STARTED_CONTAINERS[index]}" in
            "${SPACY_CONTAINER_NAME}")
                print_component_logs "spaCy" "${SPACY_CONTAINER_NAME}"
                ;;
            "${SYNTOK_CONTAINER_NAME}")
                print_component_logs "Syntok" "${SYNTOK_CONTAINER_NAME}"
                ;;
            "${COHMETRIX_CONTAINER_NAME}")
                print_component_logs "Coh-Metrix" "${COHMETRIX_CONTAINER_NAME}"
                ;;
        esac
    done
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    dump_started_logs
    cleanup_started_containers
    exit 1
}

validate_port() {
    local label="$1"
    local port="$2"
    [[ "${port}" =~ ^[0-9]+$ ]] || fail "${label} must be an integer"
    ((port >= 1 && port <= 65535)) || fail "${label} must be between 1 and 65535"
}

start_component() {
    local label="$1"
    local name="$2"
    local image="$3"
    local host_port="$4"
    local container_port="$5"
    shift 5

    local container_id
    printf 'Starting %s from %s on 127.0.0.1:%s ...\n' \
        "${label}" "${image}" "${host_port}"
    if ! container_id="$(docker_cli run \
        --detach \
        --name "${name}" \
        --publish "127.0.0.1:${host_port}:${container_port}" \
        "$@" \
        "${image}")"; then
        fail "could not start ${label} container ${name}"
    fi

    STARTED_CONTAINERS+=("${name}")
    printf 'Started %s (%s).\n' "${name}" "${container_id:0:12}"
}

wait_until_ready() {
    local label="$1"
    local name="$2"
    local host_port="$3"

    if ! command -v curl >/dev/null 2>&1; then
        printf 'curl is unavailable; readiness check for %s was skipped.\n' "${label}"
        return 0
    fi

    local attempt
    for attempt in {1..60}; do
        if curl --silent --fail --max-time 2 \
            "http://127.0.0.1:${host_port}/v1/communication_layer" \
            >/dev/null 2>&1; then
            printf '%s is ready at http://127.0.0.1:%s.\n' "${label}" "${host_port}"
            return 0
        fi

        if ! container_running "${name}"; then
            printf '%s stopped before becoming ready.\n' "${label}" >&2
            return 1
        fi
        sleep 1
    done

    printf '%s did not become ready within 60 seconds.\n' "${label}" >&2
    docker_cli logs "${name}" >&2 || true
    return 1
}

follow_component_logs() {
    local label="$1"
    local name="$2"
    local log_file=""

    if [[ -n "${LOG_DIR}" ]]; then
        log_file="${LOG_DIR}/${name}.log"
    fi

    (
        docker_cli logs --follow --tail all "${name}" 2>&1 |
            while IFS= read -r line; do
                printf '[%s] %s\n' "${label}" "${line}"
                if [[ -n "${log_file}" ]]; then
                    printf '[%s] %s\n' "${label}" "${line}" >> "${log_file}"
                fi
            done
    ) &
    LOG_FOLLOW_PIDS+=("$!")
}

follow_pipeline_logs() {
    if [[ -n "${LOG_DIR}" ]]; then
        printf 'Saving component logs below %s.\n' "${LOG_DIR}"
    fi

    follow_component_logs "spaCy" "${SPACY_CONTAINER_NAME}"
    follow_component_logs "Syntok" "${SYNTOK_CONTAINER_NAME}"
    follow_component_logs "Coh-Metrix" "${COHMETRIX_CONTAINER_NAME}"

    printf '\nFollowing all component logs. Ctrl+C stops the pipeline.\n'
    while true; do
        if ! container_running "${SPACY_CONTAINER_NAME}"; then
            fail "spaCy container stopped unexpectedly"
        fi
        if ! container_running "${SYNTOK_CONTAINER_NAME}"; then
            fail "Syntok container stopped unexpectedly"
        fi
        if ! container_running "${COHMETRIX_CONTAINER_NAME}"; then
            fail "Coh-Metrix container stopped unexpectedly"
        fi
        sleep 1
    done
}

while (($# > 0)); do
    case "$1" in
        --spacy-image)
            (($# >= 2)) || fail "--spacy-image requires a value"
            SPACY_IMAGE="$2"
            shift 2
            ;;
        --syntok-image)
            (($# >= 2)) || fail "--syntok-image requires a value"
            SYNTOK_IMAGE="$2"
            shift 2
            ;;
        --coh-metrix-image|--image)
            (($# >= 2)) || fail "$1 requires a value"
            COHMETRIX_IMAGE="$2"
            COHMETRIX_IMAGE_WAS_SET=true
            shift 2
            ;;
        --version)
            (($# >= 2)) || fail "--version requires a value"
            VERSION="$2"
            shift 2
            ;;
        --spacy-port)
            (($# >= 2)) || fail "--spacy-port requires a value"
            SPACY_HOST_PORT="$2"
            shift 2
            ;;
        --syntok-port)
            (($# >= 2)) || fail "--syntok-port requires a value"
            SYNTOK_HOST_PORT="$2"
            shift 2
            ;;
        --coh-metrix-port|--port)
            (($# >= 2)) || fail "$1 requires a value"
            COHMETRIX_HOST_PORT="$2"
            shift 2
            ;;
        --spacy-name)
            (($# >= 2)) || fail "--spacy-name requires a value"
            SPACY_CONTAINER_NAME="$2"
            shift 2
            ;;
        --syntok-name)
            (($# >= 2)) || fail "--syntok-name requires a value"
            SYNTOK_CONTAINER_NAME="$2"
            shift 2
            ;;
        --coh-metrix-name|--name)
            (($# >= 2)) || fail "$1 requires a value"
            COHMETRIX_CONTAINER_NAME="$2"
            shift 2
            ;;
        --log-level)
            (($# >= 2)) || fail "--log-level requires a value"
            LOG_LEVEL="$2"
            shift 2
            ;;
        --log-dir)
            (($# >= 2)) || fail "--log-dir requires a value"
            LOG_DIR="$2"
            shift 2
            ;;
        --log-tail-lines)
            (($# >= 2)) || fail "--log-tail-lines requires a value"
            LOG_TAIL_LINES="$2"
            shift 2
            ;;
        --germanet)
            (($# >= 2)) || fail "--germanet requires a value"
            GERMANET_PATH="$2"
            shift 2
            ;;
        --no-build)
            BUILD_COHMETRIX_IMAGE=false
            shift
            ;;
        --build)
            BUILD_COHMETRIX_IMAGE=true
            shift
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        --detach)
            FOREGROUND=false
            shift
            ;;
        --stop)
            STOP_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

if [[ "${COHMETRIX_IMAGE_WAS_SET}" == false ]]; then
    COHMETRIX_IMAGE="docker.texttechnologylab.org/${ANNOTATOR_NAME}:${VERSION}"
fi

command -v docker >/dev/null 2>&1 || fail "docker is not installed or not on PATH"
docker_cli info >/dev/null 2>&1 || fail "the Docker daemon is not reachable"

if [[ "${STOP_ONLY}" == true ]]; then
    stop_container "${COHMETRIX_CONTAINER_NAME}"
    stop_container "${SYNTOK_CONTAINER_NAME}"
    stop_container "${SPACY_CONTAINER_NAME}"
    exit 0
fi

if [[ -n "${LOG_DIR}" ]]; then
    FOREGROUND=true
    mkdir -p -- "${LOG_DIR}"
    LOG_DIR="$(cd -- "${LOG_DIR}" && pwd -P)"
fi

validate_port "--spacy-port" "${SPACY_HOST_PORT}"
validate_port "--syntok-port" "${SYNTOK_HOST_PORT}"
validate_port "--coh-metrix-port" "${COHMETRIX_HOST_PORT}"
[[ "${LOG_TAIL_LINES}" =~ ^[1-9][0-9]*$ ]] \
    || fail "--log-tail-lines must be a positive integer"

if [[ "${SPACY_HOST_PORT}" == "${SYNTOK_HOST_PORT}"
      || "${SPACY_HOST_PORT}" == "${COHMETRIX_HOST_PORT}"
      || "${SYNTOK_HOST_PORT}" == "${COHMETRIX_HOST_PORT}" ]]; then
    fail "spaCy, Syntok, and Coh-Metrix must use different host ports"
fi

for name in \
    "${SPACY_CONTAINER_NAME}" \
    "${SYNTOK_CONTAINER_NAME}" \
    "${COHMETRIX_CONTAINER_NAME}"; do
    if container_exists "${name}"; then
        fail "container ${name} already exists; stop the pipeline first with --stop"
    fi
done

if ! docker_cli image inspect "${SPACY_IMAGE}" >/dev/null 2>&1; then
    printf 'Pulling spaCy image %s ...\n' "${SPACY_IMAGE}"
    docker_cli pull "${SPACY_IMAGE}"
fi

if ! docker_cli image inspect "${SYNTOK_IMAGE}" >/dev/null 2>&1; then
    printf 'Pulling Syntok image %s ...\n' "${SYNTOK_IMAGE}"
    docker_cli pull "${SYNTOK_IMAGE}"
fi

if [[ "${BUILD_COHMETRIX_IMAGE}" == true ]]; then
    printf 'Building %s from %s ...\n' "${COHMETRIX_IMAGE}" "${PROJECT_ROOT}"
    docker_cli build \
        --file "${PROJECT_ROOT}/src/main/docker/Dockerfile" \
        --build-arg "DUUI_COH_METRIX_ANNOTATOR_NAME=${ANNOTATOR_NAME}" \
        --build-arg "DUUI_COH_METRIX_ANNOTATOR_VERSION=${VERSION}" \
        --build-arg "DUUI_COH_METRIX_LOG_LEVEL=${LOG_LEVEL}" \
        --tag "${COHMETRIX_IMAGE}" \
        "${PROJECT_ROOT}"
elif ! docker_cli image inspect "${COHMETRIX_IMAGE}" >/dev/null 2>&1; then
    printf 'Pulling Coh-Metrix image %s ...\n' "${COHMETRIX_IMAGE}"
    docker_cli pull "${COHMETRIX_IMAGE}"
fi

COHMETRIX_RUN_ARGS=(
    --env "DUUI_COH_METRIX_LOG_LEVEL=${LOG_LEVEL}"
)

if [[ -n "${GERMANET_PATH}" ]]; then
    [[ -d "${GERMANET_PATH}" ]] || fail "GermaNet path is not a directory: ${GERMANET_PATH}"
    GERMANET_PATH="$(cd -- "${GERMANET_PATH}" && pwd -P)"
    if [[ -z "$(find "${GERMANET_PATH}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        fail "GermaNet directory is empty: ${GERMANET_PATH}"
    fi

    MOUNT_SOURCE="${GERMANET_PATH}"
    case "${OSTYPE:-}" in
        msys*|cygwin*|win32*)
            if command -v cygpath >/dev/null 2>&1; then
                MOUNT_SOURCE="$(cygpath -w "${GERMANET_PATH}")"
            fi
            ;;
    esac

    COHMETRIX_RUN_ARGS+=(
        --mount "type=bind,source=${MOUNT_SOURCE},target=${CONTAINER_GERMANET_PATH},readonly"
        --env "DUUI_COH_METRIX_GERMANET_PATH=${CONTAINER_GERMANET_PATH}"
    )
fi

start_component \
    "spaCy" \
    "${SPACY_CONTAINER_NAME}" \
    "${SPACY_IMAGE}" \
    "${SPACY_HOST_PORT}" \
    "${SPACY_CONTAINER_PORT}"
wait_until_ready "spaCy" "${SPACY_CONTAINER_NAME}" "${SPACY_HOST_PORT}" \
    || fail "spaCy did not become ready"

start_component \
    "Syntok" \
    "${SYNTOK_CONTAINER_NAME}" \
    "${SYNTOK_IMAGE}" \
    "${SYNTOK_HOST_PORT}" \
    "${SYNTOK_CONTAINER_PORT}"
wait_until_ready "Syntok" "${SYNTOK_CONTAINER_NAME}" "${SYNTOK_HOST_PORT}" \
    || fail "Syntok did not become ready"

start_component \
    "Coh-Metrix" \
    "${COHMETRIX_CONTAINER_NAME}" \
    "${COHMETRIX_IMAGE}" \
    "${COHMETRIX_HOST_PORT}" \
    "${COHMETRIX_CONTAINER_PORT}" \
    "${COHMETRIX_RUN_ARGS[@]}"
wait_until_ready \
    "Coh-Metrix" \
    "${COHMETRIX_CONTAINER_NAME}" \
    "${COHMETRIX_HOST_PORT}" \
    || fail "Coh-Metrix did not become ready"

printf '\nAll DUUI component services are ready:\n'
printf '  spaCy:       http://127.0.0.1:%s\n' "${SPACY_HOST_PORT}"
printf '  Syntok:      http://127.0.0.1:%s\n' "${SYNTOK_HOST_PORT}"
printf '  Coh-Metrix:  http://127.0.0.1:%s\n' "${COHMETRIX_HOST_PORT}"
printf '\nRequired runner order: spaCy -> Syntok -> Coh-Metrix\n'
printf 'Syntok parameters: write_sentences=false, write_paragraphs=true\n'
printf 'Stop all containers: %s --stop\n' "$0"
printf 'spaCy logs:      docker logs -f %s\n' "${SPACY_CONTAINER_NAME}"
printf 'Syntok logs:     docker logs -f %s\n' "${SYNTOK_CONTAINER_NAME}"
printf 'Coh-Metrix logs: docker logs -f %s\n' "${COHMETRIX_CONTAINER_NAME}"

if [[ "${FOREGROUND}" == true ]]; then
    trap cleanup_started_containers EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    follow_pipeline_logs
fi
