#!/usr/bin/env bash
# One-time setup for Sahil:
# 1. Create a GCP project in Cloud Console (IAM & Admin -> Service Accounts).
# 2. Create a service account (SA) and download its private key as JSON.
# 3. Share the Google Drive weights folder with the SA email address.
# 4. Set DRIVE_SA_KEY to the JSON file path or base64/raw JSON string.
# 5. Never commit the JSON key or DRIVE_SA_KEY to Git (repo is public).

set -euo pipefail

# Check that the required environment variable is set
if [[ -z "${DRIVE_SA_KEY:-}" ]]; then
  echo "Error: DRIVE_SA_KEY environment variable is not set." >&2
  echo "Set DRIVE_SA_KEY to the path of the SA JSON file or its JSON/base64 content." >&2
  exit 1
fi

# Source folder on Google Drive and destination directory
DRIVE_SOURCE="${1:-colab_llm}"
# Destination defaults to /tmp/weights (local disk).
# Note: /dev/shm is 5.7G MEASURED on T4; 75 GB models cannot fit in shm.
# 4 GB copy bench: /dev/shm 248 MB/s vs /tmp 147 MB/s MEASURED. Use disk for large models.
DEST_DIR="${2:-/tmp/weights}"

mkdir -p "$DEST_DIR"

# Temporary files cleanup
TMP_KEY=""
TMP_CONF="$(mktemp /tmp/rclone_sa_XXXXXX.conf)"
cleanup() {
  rm -f "$TMP_CONF"
  if [[ -n "$TMP_KEY" && -f "$TMP_KEY" ]]; then
    rm -f "$TMP_KEY"
  fi
}
trap cleanup EXIT INT TERM

# Resolve service account key (path, raw JSON, or base64-encoded JSON)
if [[ -f "$DRIVE_SA_KEY" ]]; then
  KEY_FILE="$DRIVE_SA_KEY"
else
  TMP_KEY="$(mktemp /tmp/drive_sa_XXXXXX.json)"
  chmod 600 "$TMP_KEY"
  if echo "$DRIVE_SA_KEY" | base64 -d 2>/dev/null | grep -q '"type":'; then
    echo "$DRIVE_SA_KEY" | base64 -d > "$TMP_KEY"
  else
    printf '%s\n' "$DRIVE_SA_KEY" > "$TMP_KEY"
  fi
  KEY_FILE="$TMP_KEY"
fi

# Generate minimal rclone configuration for Google Drive SA
cat <<EOF > "$TMP_CONF"
[gdrive]
type = drive
scope = drive.readonly
service_account_file = $KEY_FILE
EOF

echo "Pulling weights from gdrive:${DRIVE_SOURCE} to ${DEST_DIR}..."

# Fast rclone flags from Optimizing_AI_Agent_Colab_Workflows.md:
# --transfers=12: pull parallel files/shards up to Drive API limit
# --multi-thread-streams=8: parallel chunks for large single files (e.g. GGUF)
# --multi-thread-cutoff=64M: enable multi-threading on files >= 64M
# --drive-chunk-size=128M: reduce HTTP requests on large block transfers
rclone copy "gdrive:${DRIVE_SOURCE}" "$DEST_DIR" \
  --config "$TMP_CONF" \
  --transfers=12 \
  --multi-thread-streams=8 \
  --multi-thread-cutoff=64M \
  --drive-chunk-size=128M \
  --stats=15s

echo "Transfer complete."
