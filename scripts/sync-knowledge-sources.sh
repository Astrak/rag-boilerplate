#!/usr/bin/env bash
# Guarded sync of knowledge-sources/ between this machine and the S3 bucket
# used for backup/duplication (see CLAUDE.md). Works from a laptop, the
# Hetzner/AWS box, or (in future) a non-interactive trigger on the server
# itself -- knowledge-sources/ is resolved relative to this script's own
# location, not the caller's cwd, and everything is env-var/exit-code
# driven (no prompts) so it's safe to call from a cron/systemd/API trigger
# later without changes.
#
# Runs `aws` via the official amazon/aws-cli Docker image rather than
# requiring a native install -- neither the dev workstation nor the
# Hetzner box has one, and Docker is already a hard dependency of this
# repo either way.
#
# No marker file is added to knowledge-sources/ itself. "Last changed" is
# read straight from the filesystem (local mtimes) and from S3's own
# LastModified (via `aws s3 ls`) -- but those two clocks can't be compared
# directly: S3's LastModified is upload time, not original edit time, so
# it's always >= the local mtime that produced it. Comparing them head-to
# -head would make every push look "stale" again the moment it finishes.
# Instead each side is compared against a private checkpoint of what THIS
# machine last observed for THAT side, in deploy/.sync-state/<source>
# (gitignored, per-machine, never synced) -- i.e. "has this changed since
# I last looked", not "is my clock later than your clock".
#
# Usage:
#   sync-knowledge-sources.sh status [source...]
#   sync-knowledge-sources.sh push   [source...]  [--force]
#   sync-knowledge-sources.sh pull   [source...]  [--force]
#
# With no [source...] args, operates on every folder under
# knowledge-sources/ (or $SOURCES, comma-separated) -- same convention
# scraper.discover_urls/download_articles/vectorizer already use.
#
# Credentials: exported AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_DEFAULT_REGION, or a gitignored deploy/.aws-sync.env next to this
# script (KEY=value per line) which is sourced automatically if present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KNOWLEDGE_DIR="$REPO_ROOT/knowledge-sources"
STATE_DIR="$SCRIPT_DIR/.sync-state"
BUCKET="rag-faiss-index-bucket"
BUCKET_REGION="eu-north-1"

usage() {
  echo "Usage: $0 {status|push|pull} [source...] [--force]" >&2
  exit 1
}

[ $# -ge 1 ] || usage
cmd="$1"; shift

force=false
sources=()
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    force=true
  else
    sources+=("$arg")
  fi
done
case "$cmd" in status|push|pull) ;; *) usage ;; esac

# Credentials only matter once we know the invocation itself is valid --
# checked here, after arg parsing, so a bad `cmd`/no-args mistake fails
# fast with the usage message instead of a credentials error.
ENV_FILE="$SCRIPT_DIR/.aws-sync.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (env var or deploy/.aws-sync.env)}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY (env var or deploy/.aws-sync.env)}"
: "${AWS_DEFAULT_REGION:=$BUCKET_REGION}"

aws_cli() {
  docker run --rm \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    -v "$KNOWLEDGE_DIR:/knowledge-sources" \
    amazon/aws-cli "$@"
}

if [ ${#sources[@]} -eq 0 ]; then
  if [ -n "${SOURCES:-}" ]; then
    IFS=',' read -ra sources <<< "$SOURCES"
  else
    # Union of local folders and the bucket's own top-level prefixes --
    # local-only doesn't see anything that only exists in S3 yet (e.g. a
    # fresh box with an empty knowledge-sources/), so both sides are
    # checked and de-duplicated.
    mkdir -p "$KNOWLEDGE_DIR"
    declare -A seen=()
    for d in "$KNOWLEDGE_DIR"/*/; do
      [ -d "$d" ] || continue
      seen["$(basename "$d")"]=1
    done
    while IFS= read -r name; do
      [ -n "$name" ] && seen["$name"]=1
    done < <(aws_cli s3 ls "s3://$BUCKET/" | awk '/^ *PRE / {print $2}' | sed 's#/$##')
    mapfile -t sources < <(printf '%s\n' "${!seen[@]}" | sort)
  fi
fi

if [ ${#sources[@]} -eq 0 ]; then
  echo "No sources found under $KNOWLEDGE_DIR and none given on the command line." >&2
  exit 1
fi

# Latest mtime under a source folder, as an integer epoch -- local-knowledge/
# (local-only PDFs, never synced) is excluded so editing it doesn't affect
# sync decisions for the rest. Directories count too (not just -type f), so
# a plain deletion still bumps its parent's mtime and gets noticed.
local_epoch() {
  local source="$1"
  local dir="$KNOWLEDGE_DIR/$source"
  [ -d "$dir" ] || { echo ""; return; }
  find "$dir" -path "*/local-knowledge" -prune -o -printf '%T@\n' \
    | sort -n | tail -1 | cut -d. -f1
}

# Latest LastModified under s3://$BUCKET/$source/, as an integer epoch, via
# plain `aws s3 ls --recursive` (its date/time columns sort correctly as
# text since they're fixed-width ISO). Empty output (nothing pushed yet)
# yields an empty string, not an error.
bucket_epoch() {
  local source="$1"
  local line ts
  line="$(aws_cli s3 ls --recursive "s3://$BUCKET/$source/" | sort | tail -1)"
  [ -z "$line" ] && { echo ""; return; }
  ts="$(echo "$line" | awk '{print $1" "$2}')"
  date -d "$ts" +%s
}

human() {
  [ -n "$1" ] && date -d "@$1" '+%Y-%m-%d %H:%M:%S' || echo "<none>"
}

read_checkpoint() {
  local source="$1" key="$2" f="$STATE_DIR/$source"
  [ -f "$f" ] || { echo ""; return; }
  grep "^$key=" "$f" | tail -1 | cut -d= -f2-
}

write_checkpoint() {
  local source="$1" local_e="$2" bucket_e="$3"
  mkdir -p "$STATE_DIR"
  printf 'local=%s\nbucket=%s\n' "$local_e" "$bucket_e" > "$STATE_DIR/$source"
}

case "$cmd" in
  status)
    printf '%-35s %-21s %-21s\n' "SOURCE" "LOCAL" "BUCKET"
    for source in "${sources[@]}"; do
      l="$(local_epoch "$source")"
      b="$(bucket_epoch "$source")"
      printf '%-35s %-21s %-21s\n' "$source" "$(human "$l")" "$(human "$b")"
    done
    ;;

  push)
    for source in "${sources[@]}"; do
      src_dir="$KNOWLEDGE_DIR/$source"
      if [ ! -d "$src_dir" ]; then
        echo "SKIP $source: no local folder at $src_dir" >&2
        continue
      fi
      b_now="$(bucket_epoch "$source")"
      if [ "$force" != true ] && [ -n "$b_now" ]; then
        ckpt_b="$(read_checkpoint "$source" bucket)"
        if [ -z "$ckpt_b" ]; then
          echo "REFUSE $source: bucket already has data but this machine has no sync history for it -- pull first to establish a baseline, or re-run with --force" >&2
          continue
        elif [ "$b_now" -gt "$ckpt_b" ]; then
          echo "REFUSE $source: bucket changed since this machine's last sync ($(human "$ckpt_b") -> $(human "$b_now")) -- pull first, or re-run with --force" >&2
          continue
        fi
      fi
      echo "PUSH $source -> s3://$BUCKET/$source/"
      aws_cli s3 sync "/knowledge-sources/$source/" "s3://$BUCKET/$source/" \
        --exclude "local-knowledge/*"
      write_checkpoint "$source" "$(local_epoch "$source")" "$(bucket_epoch "$source")"
    done
    ;;

  pull)
    for source in "${sources[@]}"; do
      b_now="$(bucket_epoch "$source")"
      if [ -z "$b_now" ]; then
        echo "SKIP $source: nothing at s3://$BUCKET/$source/ yet" >&2
        continue
      fi
      if [ "$force" != true ] && [ -d "$KNOWLEDGE_DIR/$source" ]; then
        l_now="$(local_epoch "$source")"
        ckpt_l="$(read_checkpoint "$source" local)"
        if [ -z "$ckpt_l" ]; then
          echo "REFUSE $source: local folder already has data but no sync history for it on this machine -- push first if this is the authoritative copy, or re-run with --force to discard it" >&2
          continue
        elif [ "$l_now" -gt "$ckpt_l" ]; then
          echo "REFUSE $source: local changed since this machine's last sync ($(human "$ckpt_l") -> $(human "$l_now")) -- push first, or re-run with --force" >&2
          continue
        fi
      fi
      mkdir -p "$KNOWLEDGE_DIR/$source"
      echo "PULL s3://$BUCKET/$source/ -> $source"
      aws_cli s3 sync "s3://$BUCKET/$source/" "/knowledge-sources/$source/" \
        --exclude "local-knowledge/*"
      write_checkpoint "$source" "$(local_epoch "$source")" "$b_now"
    done
    ;;
esac
