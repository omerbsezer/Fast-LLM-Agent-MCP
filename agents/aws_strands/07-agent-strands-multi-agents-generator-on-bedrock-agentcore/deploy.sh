#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

REGION="${AWS_REGION:-eu-central-1}"
TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"

echo "==> 1/4  terraform apply (ECR + IAM + Memory)"
terraform -chdir=terraform init -upgrade
terraform -chdir=terraform apply -auto-approve

ECR_URL=$(terraform -chdir=terraform output -raw ecr_repository_url)
REGISTRY="${ECR_URL%/*}"
IMAGE_URI="${ECR_URL}:${TAG}"

echo "==> 2/4  docker buildx build --push (linux/arm64) -> ${IMAGE_URI}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker buildx inspect agentcore-builder >/dev/null 2>&1 || docker buildx create --name agentcore-builder --use
docker buildx use agentcore-builder
docker buildx build --platform linux/arm64 -t "$IMAGE_URI" --push .

echo "==> 3/4  terraform apply (Agent Runtime, image=${IMAGE_URI})"
terraform -chdir=terraform apply -auto-approve -var="container_image_uri=${IMAGE_URI}"

echo "==> 4/4  Done."
terraform -chdir=terraform output
