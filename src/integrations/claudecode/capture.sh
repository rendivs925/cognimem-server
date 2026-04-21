#!/usr/bin/env bash
# CogniMem capture script for Claude Code hooks
# Install to: ~/.cognimem/capture-claudecode.sh
# Usage: capture-claudecode.sh <event_type>

set -euo pipefail

COGNIMEM_URL="${COGNIMEM_CAPTURE_URL:-http://localhost:37778}"
EVENT_TYPE="${1:?Usage: capture-claudecode.sh <event_type>}"
TIMESTAMP="$(date +%s)"
SESSION_ID="${CLAUDE_SESSION_ID:-}"
PROJECT_PATH="${CLAUDE_PROJECT_DIR:-}"

build_json() {
  local event_type="$1"
  shift

  local json="{\"event_type\":\"$event_type\",\"source\":\"claude_code\",\"timestamp\":$TIMESTAMP"
  
  if [ -n "$SESSION_ID" ]; then
    json="$json,\"session_id\":\"$SESSION_ID\""
  fi
  if [ -n "$PROJECT_PATH" ]; then
    json="$json,\"project_path\":\"$PROJECT_PATH\""
  fi

  # Add extra fields from arguments as key=value pairs
  while [ $# -gt 0 ]; do
    local key="$1"
    local value="$2"
    shift 2
    # Escape quotes in value
    value="$(echo "$value" | sed 's/"/\\"/g' | head -c 2000)"
    json="$json,\"$key\":\"$value\""
  done

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
  tool_execute_before)
    json="$(build_json tool_execute_before \
      tool_name "${TOOL_NAME:-}" \
      file_path "${FILE_PATH:-}")"
    ;;
  tool_execute_after)
    json="$(build_json tool_execute_after \
      tool_name "${TOOL_NAME:-}" \
      file_path "${FILE_PATH:-}" \
      success "true")"
    ;;
  tool_execute_failure)
    json="$(build_json tool_execute_failure \
      tool_name "${TOOL_NAME:-}" \
      file_path "${FILE_PATH:-}" \
      success "false")"
    ;;
  file_edited)
    json="$(build_json file_edited \
      file_path "${FILE_PATH:-}")"
    ;;
  file_created)
    json="$(build_json file_created \
      file_path "${FILE_PATH:-}")"
    ;;
  file_deleted)
    json="$(build_json file_deleted \
      file_path "${FILE_PATH:-}")"
    ;;
  user_prompt_submitted)
    json="$(build_json user_prompt_submitted)"
    ;;
  task_created)
    json="$(build_json task_created \
      content "${TASK_NAME:-}")"
    ;;
  task_completed)
    json="$(build_json task_completed \
      content "${TASK_NAME:-}")"
    ;;
  subagent_started)
    json="$(build_json subagent_started)"
    ;;
  subagent_stopped)
    json="$(build_json subagent_stopped)"
    ;;
  cwd_changed)
    json="$(build_json cwd_changed \
      content "${PWD:-}")"
    ;;
  instructions_loaded)
    json="$(build_json instructions_loaded)"
    ;;
  config_changed)
    json="$(build_json config_changed)"
    ;;
  worktree_created)
    json="$(build_json worktree_created)"
    ;;
  worktree_removed)
    json="$(build_json worktree_removed)"
    ;;
  pre_compact)
    json="$(build_json pre_compact)"
    ;;
  post_compact)
    json="$(build_json post_compact)"
    ;;
  elicitation)
    json="$(build_json elicitation)"
    ;;
  elicitation_result)
    json="$(build_json elicitation_result)"
    ;;
  stop_failure)
    json="$(build_json stop_failure)"
    ;;
  notification)
    json="$(build_json notification)"
    ;;
  permission_request)
    json="$(build_json permission_asked \
      tool_name "${TOOL_NAME:-}")"
    ;;
  teammate_idle)
    json="$(build_json teammate_idle)"
    ;;
  *)
    echo "[cognimem-capture] Unknown event type: $EVENT_TYPE" >&2
    exit 0
    ;;
esac

send_event "$json" || true
