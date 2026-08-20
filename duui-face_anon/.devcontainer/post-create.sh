#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS="$PROJECT_ROOT/duui/src/python/requirements.txt"

ADOPTIUM_KEYRING="/etc/apt/keyrings/adoptium.gpg"
ADOPTIUM_LIST="/etc/apt/sources.list.d/adoptium.list"

if [[ $EUID -ne 0 ]]; then
    echo "post-create.sh must run as root"
    exit 1
fi

is_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null \
        | grep -q "ok installed"
}

install_apt_packages() {
    local missing=()

    for package in "$@"; do
        if ! is_installed "$package"; then
            missing+=("$package")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        echo "Installing packages: ${missing[*]}"
        apt-get update
        apt-get install -y --no-install-recommends "${missing[@]}"
    fi
}

echo "==> Installing development system dependencies"

install_apt_packages \
    ca-certificates \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    maven


if ! is_installed temurin-21-jdk; then
    echo "==> Configuring Adoptium repository"

    # shellcheck source=/dev/null
    source /etc/os-release

    install -d -m 0755 /etc/apt/keyrings

    if [[ ! -s "$ADOPTIUM_KEYRING" ]]; then
        curl -fsSL \
            https://packages.adoptium.net/artifactory/api/gpg/key/public \
            | gpg --dearmor --batch --yes -o "$ADOPTIUM_KEYRING"
    fi

    repo="deb [signed-by=${ADOPTIUM_KEYRING}] https://packages.adoptium.net/artifactory/deb ${VERSION_CODENAME} main"

    if [[ ! -f "$ADOPTIUM_LIST" ]] ||
       [[ "$(cat "$ADOPTIUM_LIST")" != "$repo" ]]; then
        printf '%s\n' "$repo" > "$ADOPTIUM_LIST"
    fi

    echo "==> Installing Temurin JDK 21"

    apt-get update
    apt-get install -y --no-install-recommends temurin-21-jdk
fi


echo "==> Synchronizing Python dependencies"

python -m pip install \
    --disable-pip-version-check \
    -r "$REQUIREMENTS"


echo "==> Cleaning apt cache"

rm -rf /var/lib/apt/lists/*


echo
echo "==> Development environment ready"
echo

python --version
java -version
mvn -version

python - <<'PY'
import cv2
import torch

print(f"OpenCV: {cv2.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY