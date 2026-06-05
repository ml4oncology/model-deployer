#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 IMAGE_OR_SIF" >&2
    echo "Example Docker: $0 mira-services.uhn.ca:5000/model-deployer:20260604" >&2
    echo "Example SIF:    $0 dist/model-deployer_20260604_linux-amd64.sif" >&2
    exit 1
fi

IMAGE="$1"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${PROJECT_DIR}/Outputs}"

mkdir -p "${OUTPUTS_DIR}"

command=(
    python /app/src/main.py
    --start-date 20240904
    --end-date 20260430
    --model-anchor clinic
    --run-on-silent-deployment True
)

if [[ "${IMAGE}" == *.sif ]]; then
    singularity exec \
        --bind "${DATA_DIR}:/app/Data" \
        --bind "${OUTPUTS_DIR}:/app/Outputs" \
        "${IMAGE}" \
        "${command[@]}"
else
    docker run --rm \
        --volume "${DATA_DIR}:/app/Data" \
        --volume "${OUTPUTS_DIR}:/app/Outputs" \
        "${IMAGE}" \
        "${command[@]}"
fi
