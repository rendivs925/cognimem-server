#!/usr/bin/env bash
# CogniMem capture script for Codex skills
# Install to: ~/.cognimem/capture-codex.sh
# Usage: capture-codex.sh <event_type> [content]

set -euo pipefail

COGNIMEM_URL="${COGNIMEM_CAPTURE_URL:-http://localhost:37778}"
EVENT_TYPE="${1:?Usage: capture-codex.sh <event_type>}"
CONTENT="${2:-}"
TIMESTAMP="$(date +%s)"
PROJECT_PATH="${CODEX_PROJECT_DIR:-${PWD:-}}"

build_json() {
  local event_type="$1"
  local content="${2:-}"
  
  local json="{\"event_type\":\"$event_type\",\"source\":\"codex\",\"timestamp\":$TIMESTAMP"
  
  if [ -n "$PROJECT_PATH" ]; then
    json="$json,\"project_path\":\"$PROJECT_PATH\""
  fi

  if [ -n "$content" ]; then
    local escaped
    escaped="$(echo "$content" | sed 's/"/\\"/g' | head -c 2000)"
    json="$json,\"content\":\"$escaped\""
  fi

  json="$json}"
  echo "$json"
}

send_event() {
  local json="$1"
  local retries=3
  local delay=1

  for i in $(seq 1 $retries); do
    if curl -sf -X POST "$COGNIMEM_URL/capture/events" \
         -H 'Content-Type: application/json' \
         -d "$json" 2>/dev/null; then
      return 0
    fi
    sleep "$delay"
    delay=$((delay * 2))
  done

  echo "[cognimem-capture] Failed to send event after $retries attempts" >&2
  return 1
}

case "$EVENT_TYPE" in
  file_edited|file_write)
    json="$(build_json file_edited "$CONTENT")"
    ;;
  file_created)
    json="$(build_json file_created "$CONTENT")"
    ;;
  task_created|task_create)
    json="$(build_json task_created "$CONTENT")"
    ;;
  task_completed|task_complete)
    json="$(build_json task_completed "$CONTENT")"
    ;;
  *)
    json="$(build_json "$EVENT_TYPE" "$CONTENT")"
    ;;
esac

send_event "$json" || true
