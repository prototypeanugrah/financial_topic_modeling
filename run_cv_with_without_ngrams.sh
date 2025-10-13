#!/usr/bin/env bash
# Run cross-validation twice: once with n-gram modeling enabled (default),
# and once with n-grams disabled. Results are stored under separate
# subdirectories beneath the configured outputs directory.

set -euo pipefail

CONFIG_PATH="config.yaml"
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
    if [[ $1 == -* ]]; then
        EXTRA_ARGS=("$@")
    else
        CONFIG_PATH=$1
        EXTRA_ARGS=("${@:2}")
    fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_SUBDIR="comparison_${TIMESTAMP}"

WITH_NGRAMS_SUBDIR="${BASE_SUBDIR}/with_ngrams"
WITHOUT_NGRAMS_SUBDIR="${BASE_SUBDIR}/without_ngrams"

echo "Running with n-grams enabled..."
uv run main.py --config "${CONFIG_PATH}" --output_subdir "${WITH_NGRAMS_SUBDIR}" "${EXTRA_ARGS[@]}"

echo "Running with n-grams disabled..."
uv run main.py --config "${CONFIG_PATH}" --output_subdir "${WITHOUT_NGRAMS_SUBDIR}" --no_ngrams "${EXTRA_ARGS[@]}"

echo "Completed runs. Results stored under:"
echo "  outputs/${WITH_NGRAMS_SUBDIR}"
echo "  outputs/${WITHOUT_NGRAMS_SUBDIR}"
