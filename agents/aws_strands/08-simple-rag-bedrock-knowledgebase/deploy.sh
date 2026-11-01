#!/usr/bin/env bash
# One command for the whole stack:
#   1. Terraform: S3, OpenSearch Serverless, Bedrock Knowledge Base, ECR, IAM
#   2. Build + push the agent image (linux/arm64) and run it on Bedrock AgentCore
#   3. Turn on AgentCore Observability (CloudWatch Transaction Search)
#   4. Write app/.env and agent/.env from the Terraform outputs
# Then start the app yourself:  pip install -r app/requirements.txt && python3 app/app.py
#
# Usage:
#   ./deploy.sh            deploy (or update) everything
#   ./deploy.sh destroy    delete every AWS resource this project created
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$ROOT/terraform"
AGENT_DIR="$ROOT/agent"
IMAGE_VARS="$TF_DIR/agent.auto.tfvars"
BUILDER="agentcore-builder"

tf() { terraform -chdir="$TF_DIR" "$@"; }
step() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

check_tools() {
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null || die "'$cmd' is not installed."
  done
  aws sts get-caller-identity >/dev/null 2>&1 || die "No working AWS credentials (run 'aws configure' or 'aws sso login')."
}

# AgentCore only runs linux/arm64 images. On an x86 machine, install QEMU
# emulation once so buildx can cross-build.
ensure_arm64_builder() {
  if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
  fi
  if ! docker buildx inspect --bootstrap "$BUILDER" | grep -q 'linux/arm64'; then
    step "Installing arm64 emulation (QEMU binfmt) for cross-building"
    docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
    docker buildx rm "$BUILDER" >/dev/null
    docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
    docker buildx inspect --bootstrap "$BUILDER" | grep -q 'linux/arm64' \
      || die "buildx still cannot build linux/arm64 images."
  fi
}

# AgentCore Observability needs CloudWatch Transaction Search: X-Ray spans
# stored as structured logs. It's a one-time, account-wide setting, so it's
# only turned on here (never off, not even by destroy).
enable_transaction_search() {
  local region="$1" account dest
  dest="$(aws xray get-trace-segment-destination --region "$region" --query Destination --output text)"
  if [ "$dest" = "CloudWatchLogs" ]; then
    echo "Transaction Search already enabled."
    return
  fi
  account="$(aws sts get-caller-identity --query Account --output text)"
  aws logs put-resource-policy --region "$region" --policy-name TransactionSearchXRayAccess --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Sid\": \"TransactionSearchXRayAccess\",
      \"Effect\": \"Allow\",
      \"Principal\": {\"Service\": \"xray.amazonaws.com\"},
      \"Action\": \"logs:PutLogEvents\",
      \"Resource\": [
        \"arn:aws:logs:${region}:${account}:log-group:aws/spans:*\",
        \"arn:aws:logs:${region}:${account}:log-group:/aws/application-signals/data:*\"
      ],
      \"Condition\": {
        \"ArnLike\": {\"aws:SourceArn\": \"arn:aws:xray:${region}:${account}:*\"},
        \"StringEquals\": {\"aws:SourceAccount\": \"${account}\"}
      }
    }]
  }" >/dev/null
  aws xray update-trace-segment-destination --region "$region" --destination CloudWatchLogs >/dev/null
  echo "Transaction Search enabled (can take ~10 minutes to become active)."
}

deploy() {
  check_tools terraform docker aws

  step "1/5  terraform init"
  tf init -input=false -upgrade >/dev/null

  step "2/5  ECR repository for the agent image"
  tf apply -input=false -auto-approve -target=aws_ecr_repository.agent >/dev/null
  local ecr_url region tag image_uri
  ecr_url="$(tf output -raw agent_ecr_repository_url)"
  region="$(tf output -raw aws_region 2>/dev/null || echo "${ecr_url}" | cut -d. -f4)"

  # Tag = hash of the agent's source, so an unchanged agent is not rebuilt and
  # the Runtime is only updated when the code actually changes.
  tag="$(cat "$AGENT_DIR/agent.py" "$AGENT_DIR/requirements.txt" "$AGENT_DIR/Dockerfile" | sha256sum | cut -c1-12)"
  image_uri="${ecr_url}:${tag}"

  step "3/5  Agent image ${image_uri}"
  if aws ecr describe-images --region "$region" --repository-name "${ecr_url#*/}" \
       --image-ids imageTag="$tag" >/dev/null 2>&1; then
    echo "Image already in ECR, skipping build."
  else
    ensure_arm64_builder
    aws ecr get-login-password --region "$region" \
      | docker login --username AWS --password-stdin "${ecr_url%%/*}" >/dev/null
    docker buildx build --builder "$BUILDER" --platform linux/arm64 \
      -t "$image_uri" --push "$AGENT_DIR"
  fi
  # Remembered for later plain `terraform apply` runs, so they keep the Runtime.
  printf 'agent_container_image_uri = "%s"\n' "$image_uri" > "$IMAGE_VARS"

  step "4/5  AgentCore Observability (CloudWatch Transaction Search)"
  enable_transaction_search "$region"

  step "5/5  terraform apply (Knowledge Base, OpenSearch, AgentCore Runtime, tracing, .env files)"
  tf apply -input=false -auto-approve

  cat <<EOF

Deployed. Agent runtime: $(tf output -raw agent_runtime_arn)
Start the app:
  pip install -r app/requirements.txt   # first time only
  python3 app/app.py
EOF
}

destroy() {
  check_tools terraform aws
  step "terraform destroy (all project resources, including the AgentCore Runtime)"
  tf destroy -input=false -auto-approve
  rm -f "$IMAGE_VARS" "$ROOT/app/.env" "$AGENT_DIR/.env"
  echo "All resources deleted."
}

case "${1:-}" in
  "")      deploy ;;
  destroy) destroy ;;
  *)       die "Unknown option '$1'. Use: ./deploy.sh [destroy]" ;;
esac
