#!/usr/bin/env bash
# One-time bootstrap: obtain the initial Let's Encrypt cert for the domain
# Caddyfile expects at /etc/letsencrypt/live/<domain>/{fullchain,privkey}.pem.
# Run this BEFORE the first `docker compose up` / systemd unit start — the
# api container's healthcheck gates caddy's startup, and caddy fails to
# start at all if the cert files aren't already on disk.
#
# Uses certbot's standalone mode (HTTP-01), which needs port 80 free —
# nothing in docker-compose.yml publishes 80, so this is safe to run
# before the stack is up. Re-running after the stack is already up will
# fail to bind 80 until it's stopped; see certbot-renew.service for the
# renewal path instead, which handles that via --deploy-hook.
set -euo pipefail

DOMAIN="${1:?usage: init-cert.sh <domain> [email]}"
EMAIL="${2:-}"

EMAIL_ARGS=(--register-unsafely-without-email)
if [ -n "$EMAIL" ]; then
  EMAIL_ARGS=(-m "$EMAIL")
fi

certbot certonly --standalone \
  --non-interactive --agree-tos \
  "${EMAIL_ARGS[@]}" \
  -d "$DOMAIN"

echo "Cert obtained for $DOMAIN — now run: systemctl enable --now rag-boilerplate.service"
