#!/usr/bin/env bash
set -euo pipefail

CARLA_VERSION="0.9.16"
ASSET_NAME="CARLA_${CARLA_VERSION}.tar.gz"
INSTALL_ROOT="${CARLA_INSTALL_ROOT:-/home/chetana/opt/carla-${CARLA_VERSION}}"
CACHE_ROOT="${XDG_CACHE_HOME:-/home/chetana/.cache}/carla-downloads"
ARCHIVE_PATH="$CACHE_ROOT/$ASSET_NAME"
REPOSITORY="carla-simulator/carla"

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "FAIL: activate the carla310 environment before installation."
    exit 1
fi

if [[ -x "$INSTALL_ROOT/CarlaUE4.sh" ]]; then
    echo "CARLA $CARLA_VERSION is already installed at $INSTALL_ROOT"
else
    mkdir -p "$CACHE_ROOT" "$INSTALL_ROOT"

    if command -v gh >/dev/null 2>&1; then
        ASSET_URL="$(
            gh api "repos/$REPOSITORY/releases/tags/$CARLA_VERSION" \
                --jq ".assets[] | select(.name == \"$ASSET_NAME\") | .browser_download_url"
        )"
        ASSET_DIGEST="$(
            gh api "repos/$REPOSITORY/releases/tags/$CARLA_VERSION" \
                --jq ".assets[] | select(.name == \"$ASSET_NAME\") | (.digest // \"\")"
        )"
    else
        ASSET_URL="https://github.com/$REPOSITORY/releases/download/$CARLA_VERSION/$ASSET_NAME"
        ASSET_DIGEST=""
    fi

    if [[ -z "$ASSET_URL" ]]; then
        echo "FAIL: could not resolve the official $ASSET_NAME release asset."
        exit 1
    fi

    echo "Downloading official CARLA $CARLA_VERSION packaged server"
    curl --fail --location --continue-at - --output "$ARCHIVE_PATH" "$ASSET_URL"

    if [[ "$ASSET_DIGEST" == sha256:* ]]; then
        EXPECTED_SHA256="${ASSET_DIGEST#sha256:}"
        ACTUAL_SHA256="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"
        if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
            echo "FAIL: CARLA archive SHA-256 mismatch."
            exit 1
        fi
        echo "CARLA archive SHA-256 verified"
    else
        echo "WARNING: GitHub did not publish a SHA-256 digest for this legacy asset."
    fi

    echo "Extracting CARLA to $INSTALL_ROOT"
    tar --extract --gzip --file "$ARCHIVE_PATH" --directory "$INSTALL_ROOT"
    test -x "$INSTALL_ROOT/CarlaUE4.sh"
    rm -f -- "$ARCHIVE_PATH"
fi

python -m pip install --upgrade "carla==$CARLA_VERSION"

python - <<'PY'
import carla

version = getattr(carla, "__version__", "unknown")
print(f"CARLA Python API import passed (reported version: {version})")
PY

printf '\nAdd this to future shells when needed:\n'
printf 'export CARLA_ROOT=%q\n' "$INSTALL_ROOT"
printf '\nInstallation complete. Remaining disk space:\n'
df -h /home/chetana

