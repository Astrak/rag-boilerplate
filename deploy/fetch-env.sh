#!/usr/bin/env bash
# Pulls SecureString parameters from AWS SSM Parameter Store and writes them
# to .env for `docker compose` to pick up via `env_file: .env`.
#
# Relies on the EC2 instance's IAM role for AWS auth — no static credentials
# needed here or on disk. Re-run on every boot (see rag-boilerplate.service)
# so a server reset never loses secrets: AWS SSM is the only source of truth.
set -euo pipefail

SSM_PATH="${SSM_PATH:-/rag-boilerplate/prod}"
AWS_REGION="${AWS_REGION:-eu-north-1}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="$REPO_DIR/.env"

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

aws ssm get-parameters-by-path \
  --path "$SSM_PATH" \
  --with-decryption \
  --region "$AWS_REGION" \
  --query "Parameters[*].[Name,Value]" \
  --output text \
  | while IFS=$'\t' read -r name value; do
      printf '%s=%s\n' "${name##*/}" "$value"
    done > "$TMP_FILE"

if [ ! -s "$TMP_FILE" ]; then
  echo "fetch-env: no parameters found under $SSM_PATH in $AWS_REGION — refusing to overwrite $OUT_FILE" >&2
  exit 1
fi

mv "$TMP_FILE" "$OUT_FILE"
chmod 600 "$OUT_FILE"
