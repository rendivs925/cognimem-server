# Hook-Based Capture System — Implementation Plan

## Problem

The current capture is unreliable — the OpenCode adapter calls `remember` via MCP, which fails if CogniMem is down. No adapters exist for Claude Code or Codex. Events are stored as raw unenriched sensory memories with no SLM processing.

## Architecture

```
AI Tool (OpenCode/Claude Code/Codex)
  ↓ hook fires
Adapter (JS/Shell)
  ↓ HTTP POST /capture/events
CogniMem Capture Server (Axum)
  ↓ validation
Capture Pipeline
  ├── Suppression filter (noise removal)
  ├── SLM classification (tier/importance/suppress)
  ├── SLM compression (compact representation)
  ├── Event aggregation (tool start+end → single event)
  ├── CanonicalEvent → CognitiveMemoryUnit conversion
  └── Store in graph/search/storage/embeddings
```

## Dependencies to Add

```toml
axum = "0.8"
tower-http = { version = "0.6", features = ["cors", "trace"] }
```

## File Structure

```
src/capture/
  mod.rs          — module exports
  types.rs        — CanonicalEvent, CanonicalEventType, EventSource, IngestResult
  pipeline.rs     — CapturePipeline: validate → suppress → SLM classify → compress → aggregate → convert → store
  server.rs       — Axum capture HTTP server with routes

src/integrations/
  opencode/
    index.js      — rewritten full-event adapter
    package.json  — updated with all event subscriptions
  claudecode/
    hooks.json    — Claude Code hook settings template
    capture.sh    — shell script that POSTs to CogniMem
  codex/
    skills.toml   — Codex skill configuration
    capture.sh    — shell script that POSTs to CogniMem

src/config.rs     — add --capture-port flag
src/main.rs       — wire capture server into daemon run_daemon()
```

---

## Step 1: Canonical Event Schema (`src/capture/types.rs`)

### CanonicalEventType — unified enum covering all 3 platforms

| Canonical | OpenCode | Claude Code | Codex |
|-----------|----------|-------------|-------|
| SessionCreated | session.created | SessionStart | — |
| SessionDeleted | session.deleted | SessionEnd | — |
| SessionIdle | session.idle | — | — |
| ToolExecuteBefore | tool.execute.before | PreToolUse | — |
| ToolExecuteAfter | tool.execute.after | PostToolUse | — |
| ToolExecuteFailure | — | PostToolUseFailure | — |
| FileEdited | file.edited | FileChanged | — |
| FileCreated | — | FileChanged | — |
| FileDeleted | — | FileChanged | — |
| MessageUpdated | message.updated | — | — |
| UserPromptSubmitted | — | UserPromptSubmit | — |
| PermissionAsked | permission.asked | PermissionRequest | — |
| TaskCreated | — | TaskCreated | task.created |
| TaskCompleted | — | TaskCompleted | task.completed |
| SubagentStarted | — | SubagentStart | — |
| SubagentStopped | — | SubagentStop | — |
| CwdChanged | — | CwdChanged | — |
| InstructionsLoaded | — | InstructionsLoaded | — |
| ConfigChanged | — | ConfigChange | — |
| WorktreeCreated | — | WorktreeCreate | — |
| WorktreeRemoved | — | WorktreeRemove | — |
| PreCompact | — | PreCompact | — |
| PostCompact | — | PostCompact | — |
| Elicitation | — | Elicitation | — |
| ElicitationResult | — | ElicitationResult | — |
| Stop | — | Stop | — |
| StopFailure | — | StopFailure | — |
| Notification | — | Notification | — |
| TeammateIdle | — | TeammateIdle | — |

### CanonicalEvent struct

```rust
pub struct CanonicalEvent {
    pub event_type: CanonicalEventType,
    pub timestamp: i64,              // unix epoch seconds
    pub session_id: Option<String>,
    pub project_path: Option<String>,
    pub agent_id: Option<String>,
    pub source: EventSource,         // OpenCode | ClaudeCode | Codex | Manual
    pub tool_name: Option<String>,
    pub tool_input: Option<serde_json::Value>,
    pub tool_output: Option<serde_json::Value>,
    pub file_path: Option<String>,
    pub content: Option<String>,
    pub success: Option<bool>,
    pub duration_ms: Option<u64>,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

### IngestResult

```rust
pub struct IngestResult {
    pub accepted: usize,
    pub suppressed: usize,
    pub stored: usize,
    pub errors: Vec<String>,
}
```

### IngestStats

```rust
pub struct IngestStats {
    pub total_accepted: u64,
    pub total_suppressed: u64,
    pub total_stored: u64,
    pub total_errors: u64,
    pub pending_aggregations: usize,
    pub uptime_secs: u64,
}
```

---

## Step 2: Capture Pipeline (`src/capture/pipeline.rs`)

`CapturePipeline` struct holding a reference to `CogniMemState`.

### Flow per event

1. **Validate** — Reject if `event_type` is unknown, `timestamp` is > 60s in future, `content` exceeds 100KB, missing required fields per event type
2. **Suppress** — Skip noise patterns: heartbeat, ping, idle with no content, duplicate events within 5s window. Also check SLM `classify_memory.suppress` flag
3. **SLM Classify** — Call `slm.classify_memory()` with the event's content (composed from type + tool_name + content + file_path). Get tier, importance, tags, associations, suppress flag
4. **SLM Compress** — Call `slm.compress_memory()` for compact representation
5. **Aggregate** — Combine `ToolExecuteBefore` + `ToolExecuteAfter` into single memory when matching `tool_name` + `session_id`. Hold a bounded `HashMap<(String, String), CanonicalEvent>` for pending pairs (evict after 60s)
6. **Convert** — Build `CognitiveMemoryUnit` from the classified/compressed event:
   - `content` = composed text (event_type label + content + tool/file context)
   - `tier` = SLM-suggested or Sensory as fallback
   - `importance` = SLM-suggested or 0.3 as fallback
   - `scope` = `MemoryScope::Project` if `project_path` present, else `Global`
   - `model` = populate from SLM output (compressed_content, suggested_tier, tags, confidence, etc.)
7. **Store** — Add to `graph`, set `embedding`, `search.index`, `storage.save`, update capacity if needed (evict lowest-activation from tier)

### Batch processing

- `ingest_batch(events: Vec<CanonicalEvent>) -> IngestResult`
- Process events sequentially (pipeline has aggregation state)
- Return aggregated result

---

## Step 3: Capture HTTP Server (`src/capture/server.rs`)

```rust
pub async fn start_capture_server(
    state: Arc<Mutex<CogniMemState>>,
    pipeline: Arc<Mutex<CapturePipeline>>,
    port: u16,
)
```

### Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/capture/events` | `ingest_single` | Accept one `CanonicalEvent`, return `IngestResult` |
| POST | `/capture/events/batch` | `ingest_batch` | Accept `Vec<CanonicalEvent>`, return `IngestResult` |
| GET | `/capture/health` | `health_check` | Return `{"status": "ok", "uptime_secs": ...}` |
| GET | `/capture/stats` | `ingest_stats` | Return `IngestStats` with counters |

### Middleware

- `tower_http::cors::CorsLayer` (allow all origins for local dev)
- `tower_http::trace::TraceLayer`
- JSON extraction via `axum::Json`
- Error handling returning structured JSON errors

---

## Step 4: Wire into Daemon (`src/main.rs` + `src/config.rs`)

### `config.rs` additions

```rust
#[arg(long, default_value_t = 37778, help = "Port for capture HTTP endpoint")]
pub capture_port: u16,
```

### `main.rs` `run_daemon()` additions

- Create `CapturePipeline` with `Arc<Mutex<CapturePipeline>>` sharing state reference
- `tokio::spawn(start_capture_server(state.clone(), pipeline, cli.capture_port))`
- Log capture server startup

---

## Step 5: OpenCode Adapter (`src/integrations/opencode/`)

### `index.js` — rewritten

- `COGNIMEM_CAPTURE_URL` env var (default `http://localhost:37778`)
- For each of the ~8 OpenCode events, construct a `CanonicalEvent` and POST to `/capture/events`
- Event mapping: `onSessionCreate` → `SessionCreated`, `onToolExecute.before` → `ToolExecuteBefore`, etc.
- Retry: exponential backoff (1s, 2s, 4s) up to 3 attempts
- Drop events silently after max retries (log warning)
- Include `source: "opencode"` in every event

### `package.json` — updated event subscriptions

```json
{
  "opencode": {
    "events": {
      "onSessionCreate": true,
      "onSessionDelete": true,
      "onSessionIdle": true,
      "onToolExecuteBefore": true,
      "onToolExecuteAfter": true,
      "onFileEdited": true,
      "onMessageUpdated": true,
      "onPermissionAsked": true
    }
  }
}
```

---

## Step 6: Claude Code Adapter (`src/integrations/claudecode/`)

### `hooks.json` — template

```json
{
  "hooks": {
    "SessionStart": [{"type": "command", "command": "curl -s -X POST http://localhost:37778/capture/events -H 'Content-Type: application/json' -d '{\"event_type\":\"session_created\",\"source\":\"claude_code\",\"timestamp\":'$(( $(date +%s) ))',\"session_id\":\"$CLAUDE_SESSION_ID\"}'}],
    "PostToolUse": [{"type": "command", "command": "/path/to/capture.sh post_tool_use"}],
    ...
  }
}
```

### `capture.sh` — helper script

- Reads event type from `$1`
- Reads Claude Code env vars (`$CLAUDE_SESSION_ID`, `$TOOL_NAME`, `$FILE_PATH`, etc.)
- Constructs canonical JSON
- POSTs to CogniMem capture endpoint
- Retry up to 3 times with backoff

---

## Step 7: Codex Adapter (`src/integrations/codex/`)

### `skills.toml`

```toml
[skill.cognimem-capture]
name = "cognimem-capture"
description = "Capture Codex events into CogniMem"
command = "capture.sh"
events = ["file_write", "task_complete", "task_create"]
```

### `capture.sh` — similar to Claude Code but reads Codex env vars

---

## Step 8: Tests

### Unit tests (`src/capture/`)

- `types.rs`: CanonicalEvent serialization/deserialization roundtrip, all event types parse
- `pipeline.rs`:
  - Suppress noise events (heartbeat, ping)
  - Aggregate tool start+end pairs
  - Convert CanonicalEvent → CognitiveMemoryUnit with NoOpSlm
  - Batch ingestion produces correct IngestResult
  - Content size limit enforcement
  - Missing required fields rejection

### Integration tests (`tests/integration.rs` additions)

- Start Axum capture server with test state
- POST single event → verify stored in graph
- POST batch → verify all stored
- POST suppressed event → verify not stored
- Health check returns 200
- Stats endpoint returns counters

---

## Acceptance Criteria

The hook-based capture phase is **complete** when:

1. Canonical event schema covers all events from OpenCode (8), Claude Code (24), and Codex (3)
2. Capture pipeline processes events through SLM classify → compress → store
3. Axum HTTP server accepts single and batch events on configurable port
4. OpenCode adapter POSTs all ~8 available events to capture endpoint
5. Claude Code adapter templates + scripts translate 24 events
6. Codex adapter templates for skill-based events
7. All new unit + integration tests pass
8. Existing 158 tests still pass (no regressions)
9. `cargo check`, `cargo clippy`, `cargo test` all clean
