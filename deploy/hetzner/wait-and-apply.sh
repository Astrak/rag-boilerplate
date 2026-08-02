#!/usr/bin/env bash
# Retries `terraform apply` until a capacity-constrained server_type (e.g.
# cax21) becomes orderable in one of the given locations, then exits.
#
# This intentionally does NOT do a separate "check availability, then
# create" step -- that races against everyone else's script doing the same
# thing between the check and the create. Attempting the real `apply` IS
# the check: Hetzner returns a "resource_unavailable" error instantly if
# there's no stock, and Terraform state means already-created resources
# (SSH key, firewall) are never recreated, so every retry after the first
# only attempts the one still-missing resource: the server itself.
#
# Safe to Ctrl-C and rerun any time.
set -euo pipefail

cd "$(dirname "$0")"

: "${HCLOUD_TOKEN:?Set HCLOUD_TOKEN first (Cloud Console -> Security -> API Tokens)}"

SERVER_TYPE="${SERVER_TYPE:-cax21}"
LOCATIONS=(${LOCATIONS:-nbg1 fsn1 hel1}) # the only 3 locations that offer the cax* line
INTERVAL="${INTERVAL:-20}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
# No fixed IP to scope the firewall to by default (e.g. tethering off a
# mobile connection) -- security instead relies on key-only SSH + fail2ban
# (see variables.tf/cloud-init.yaml). Set ALLOWED_SSH_CIDR yourself if you
# do have a static IP you'd rather restrict to.
ALLOWED_SSH_CIDR="${ALLOWED_SSH_CIDR:-0.0.0.0/0}"

echo "Watching for '$SERVER_TYPE' across: ${LOCATIONS[*]} (retrying every ${INTERVAL}s, Ctrl-C to stop)"
echo "First attempt's Terraform output is shown in full -- if it fails for a"
echo "reason other than resource_unavailable (bad token, bad SSH key path,"
echo "malformed CIDR, ...), fix that before leaving this looping unattended."

attempt=0
while true; do
  for location in "${LOCATIONS[@]}"; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] attempt $attempt: $SERVER_TYPE in $location"
    if terraform apply -auto-approve \
      -var="server_type=$SERVER_TYPE" \
      -var="location=$location" \
      -var="ssh_public_key=$(cat "$SSH_KEY_FILE")" \
      -var="allowed_ssh_cidrs=[\"$ALLOWED_SSH_CIDR\"]"; then
      echo "Success: $SERVER_TYPE provisioned in $location."
      exit 0
    fi
  done
  sleep "$INTERVAL"
done
