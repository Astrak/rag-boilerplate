# Production deploy: Hetzner Cloud

Runs the same `Dockerfile` / `docker-compose.yml` (`api` + `caddy` for TLS +
`dozzle` for log viewing) as before, just on a Hetzner `cax21` box instead
of AWS EC2. See `../OPTIMIZE.md` for the cost/architecture comparison that
motivated this.

Secrets are a hand-placed `.env` file (no SSM, no fetch-on-boot script) —
this repo previously used AWS SSM Parameter Store + an EC2 instance role,
but Hetzner has no instance-role equivalent, and standing up a replacement
secrets service wasn't judged worth it for a single box. Rotating a secret
is a manual edit + restart; a server rebuild means re-placing `.env` by
hand. If this becomes a real pain point later, revisit with a proper tool.

## One-time setup: provision the box

```bash
cd deploy/hetzner
export HCLOUD_TOKEN=xxxxx   # Cloud Console -> Security -> API Tokens
terraform init
terraform apply \
  -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"
```

This provisions the server (`cax21` by default — 4 vCPU ARM64 (Ampere
Altra) / 8 GB, the closer architectural match to the AWS Graviton2 box this
replaces. A live `hcloud server-type list -o columns=name,location_available`
check showed both `cax21` and Hetzner's x86 `cx33` out of stock everywhere
at the same time, so there's no current basis for treating x86 as "more
orderable" — if you want `cx33` anyway (e.g. to sidestep arm64 wheel
availability for some dependency), set `-var="server_type=cx33"` instead;
no code changes needed either way, Docker builds `faiss-cpu`/`numpy` fresh
from source on first `docker compose build` regardless of architecture), an
SSH key, and a firewall open on 22 (SSH — open to `0.0.0.0/0` by default,
since this box is typically administered from a dynamic mobile IP with
nothing fixed to scope it to; security instead relies on SSH being
key-only, per `cloud-init.yaml`'s `ssh_pwauth: false`, plus `fail2ban`
rate-limiting brute-force attempts. If you do have a static IP, restrict it
with `-var='allowed_ssh_cidrs=["YOUR.IP.ADDR.ESS/32"]'`), 80 (ACME HTTP-01
challenges for certbot), and 443 (HTTPS, served by the `caddy` container).
Nothing else is opened — the API port (8000) is never published to the
host, only exposed inside the docker-compose network to `caddy` (see
`expose:` in `docker-compose.yml`). `cloud-init.yaml` runs automatically on
first boot and installs Docker + the `docker-compose-plugin` + `certbot` +
`fail2ban`; wait for it before continuing:

```bash
ssh root@<public_ip> 'cloud-init status --wait'
```

## One-time setup: clone, secrets, cert, start

```bash
ssh root@<public_ip>
git clone <this-repo-url> rag-boilerplate
cd rag-boilerplate
```

Place secrets — write `.env` locally with the same keys as before
(`OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`,
`BUCKET`, `BUCKET_REGION`, `SOURCES`), then from your machine:

```bash
scp .env root@<public_ip>:~/rag-boilerplate/.env
ssh root@<public_ip> 'chmod 600 ~/rag-boilerplate/.env'
```

`BUCKET`/`BUCKET_REGION` still point at the existing AWS S3 bucket used for
knowledge-source backups — that stays on AWS; only compute moved. The
scraper pipeline (`scraper.discover_urls`/`download_articles`/`vectorizer`)
that writes to it runs locally with your own `~/.aws/credentials`, not on
this box, so no AWS credentials are needed here at all.

Obtain the initial TLS cert (must happen before the stack's first start —
`caddy` won't start without a cert already on disk, and the `api`
healthcheck gates `caddy`'s startup):

```bash
sudo deploy/hetzner/init-cert.sh api.polem-ia.com
```

Install the renewal timer (cert renewal isn't automatic otherwise) and the
app's systemd unit, then start everything:

```bash
sudo cp deploy/hetzner/certbot-renew.service deploy/hetzner/certbot-renew.timer /etc/systemd/system/
sudo cp deploy/rag-boilerplate.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now certbot-renew.timer
sudo systemctl enable --now rag-boilerplate.service
```

**Why systemd and not just Docker's `restart: unless-stopped`?** Those are
two different jobs. Docker's own restart policy already handles crash
recovery and "come back after the daemon restarts" fine on its own — no
systemd needed for that. What Docker *can't* do is come back on a full
instance reboot before anything has run `docker compose up` — the systemd
unit is what makes a reboot self-recovering. On AWS this unit also
sequenced an SSM secrets fetch before `docker compose up`; that step is
gone now since `.env` just sits on disk, but the unit still gives a
from-scratch box a single `enable --now` bootstrap instead of a manual step
after every reboot.

## Knowledge-source data

`knowledge-sources/` is bind-mounted into the `api` container and is not
reconstructable from a fresh clone (it's untracked, and the S3 backup sync
in `apps/api/main.py` is intentionally partial — see the comment there). If
migrating from an existing AWS box rather than starting fresh, copy it over
directly:

```bash
rsync -avz ec2-user@<old-aws-ip>:~/rag-boilerplate/knowledge-sources/ \
  root@<new-hetzner-ip>:~/rag-boilerplate/knowledge-sources/
```

## Cutover from an existing AWS deployment

1. Provision the Hetzner box and complete the one-time setup above, but
   don't touch DNS yet.
2. `rsync` `knowledge-sources/` from the AWS box (previous section).
3. Smoke-test against the raw IP before any DNS change:
   ```bash
   curl --resolve api.polem-ia.com:443:<hetzner-ip> https://api.polem-ia.com/
   ```
4. Update the `api.polem-ia.com` DNS A record to the Hetzner IP (at your
   registrar — outside this repo).
5. Once traffic is confirmed flowing to the new box, decommission AWS:
   terminate the EC2 instance, release its Elastic IP, delete the
   `/rag-boilerplate/prod/*` SSM parameters and the IAM role/instance
   profile that granted access to them. **Leave the S3 bucket and its IAM
   policy alone** — still used for knowledge-source backups.

## Day to day

- Secrets change? Edit `.env` on the box by hand, then
  `sudo systemctl restart rag-boilerplate.service` to pick it up.
- `.env` is gitignored and is now the only copy of these secrets on this
  box — back it up somewhere if you'd regret retyping it.

### Deploying a code update

```bash
cd ~/rag-boilerplate
git pull
sudo docker compose build api
sudo docker compose up -d --remove-orphans
```

`docker compose up -d` on its own does **not** rebuild the image — that's
what `docker compose build` is for. Once the image is rebuilt, `up -d`
notices it changed and recreates just the `api` container; `caddy` is left
alone unless you edited the `Caddyfile` too (in which case
`sudo docker compose up -d caddy` picks that up the same way).

Restarting the whole `rag-boilerplate.service` unit does not rebuild the
image either — use the `git pull` + `docker compose build` steps above for
actual code changes, not just a service restart.

Each rebuild leaves the previous image behind as a dangling layer, which
adds up fast on a small instance disk — clean up occasionally with:

```bash
sudo docker image prune -f
```
