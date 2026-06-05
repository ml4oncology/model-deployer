#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-model-deployer}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d)}"
PLATFORM="${PLATFORM:-linux/amd64}"
SAFE_PLATFORM="${PLATFORM//\//-}"
DOCKER_ARCHIVE="${DOCKER_ARCHIVE:-dist/model-deployer_${IMAGE_TAG}_${SAFE_PLATFORM}.docker.tar}"

mkdir -p "$(dirname "${DOCKER_ARCHIVE}")"

docker buildx build \
    --platform "${PLATFORM}" \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file Dockerfile \
    --output "type=docker,dest=${DOCKER_ARCHIVE}" \
    .

echo
echo "Created Docker archive:"
echo "  ${DOCKER_ARCHIVE}"
echo
echo "To load it into Docker:"
echo "  docker load --input ${DOCKER_ARCHIVE}"
echo
echo "To tag and push after loading:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} mira-services.uhn.ca:5000/model-deployer:${IMAGE_TAG}"
echo "  docker push mira-services.uhn.ca:5000/model-deployer:${IMAGE_TAG}"
echo
echo "On a Linux cluster with Singularity or Apptainer, convert it with:"
echo "  singularity build dist/model-deployer_${IMAGE_TAG}.sif docker-archive://${DOCKER_ARCHIVE}"
echo
echo "or:"
echo "  apptainer build dist/model-deployer_${IMAGE_TAG}.sif docker-archive://${DOCKER_ARCHIVE}"
