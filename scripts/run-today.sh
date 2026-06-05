#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 IMAGE_OR_SIF YYYYMMDD" >&2
    echo "Example Docker: $0 mira-services.uhn.ca:5000/model-deployer:20260604 20260604" >&2
    echo "Example SIF:    $0 dist/model-deployer_20260604_linux-amd64.sif 20260604" >&2
    exit 1
fi

IMAGE="$1"
RUN_DATE="$2"

if [[ ! "${RUN_DATE}" =~ ^[0-9]{8}$ ]]; then
    echo "Run date must use YYYYMMDD format, for example: 20260604" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/Data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${PROJECT_DIR}/Outputs}"

mkdir -p "${OUTPUTS_DIR}"

command=(
    python /app/src/main.py
    --start-date "${RUN_DATE}"
    --end-date "${RUN_DATE}"
    --model-anchor clinic
    --dashboard-font-scale 1.5
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
