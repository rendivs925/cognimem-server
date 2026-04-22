# CogniMem End-to-End System Design

**Version:** 1.0  
**Status:** Draft for Iteration  
**Last Updated:** 2025-04-22  
**Author:** CogniMem System Design

---

## 1. Overview

This document defines the complete end-to-end workflow for CogniMem from the perspective of end users. It serves as the authoritative design reference for implementation and can be iterated upon until expectations are met.

**Goal:** Create the world's best AI coding agent memory system that exactly mimics human brain memory mechanisms.

---

## 2. User Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     END USER VIEWPOINT                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INSTALL                                                      │
│     $ cargo install cognimem-server                                 │
│     OR: brew install cognimem-server (future)                        │
│                                                                  │
│  2. CONFIGURE (one-time)                                          │
│     $ cognimem init --platform opencode    # or claude-code / codex   │
│     → Creates ~/.cognimem/config.toml                            │
│     → Installs hook plugin to OpenCode config                      │
│                                                                  │
│  3. USE AI CODING TOOL NORMALLY                                 │
│     (OpenCode / Claude Code / Codex runs normally)                      │
│     → Tool events captured automatically                         │
│     → Sent to CogniMem HTTP capture endpoint                 │
│                                                                  │
│  4. RECALL MEMORIES (when needed)                                │
│     $ cognimem recall "how did I handle auth?"                   │
│     → Returns relevant memories with context           │
│                                                                  │
│  5. MANUAL TRIGGERS (optional)                                 │
│     $ cognimem consolidate --intensity full    # run consolidation   │
│     $ cognimem persona                    # extract user preferences  │
│                                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow: Tool Usage → Memory Formation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     CAPTURE PIPELINE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                               │
│  USER'S AI TOOL                                                │
│       │                                                         │
│       ▼ (hook events)                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ HOOK ADAPTER                                        │   │
│  │ - OpenCode: index.js (8 events)                      │   │
│  │ - Claude Code: hooks.json + capture.sh (24 events)       │   │
│  │ - Codex: skills.toml + capture.sh (3 events)           │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼ (HTTP POST /capture/events)                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CAPTURE SERVER (Axum)                             │   │
│  │ 1. Validate timestamp (not > 60s future)        │   │
│  │ 2. Deduplicate (5-sec window)                    │   │
│  │ 3. Suppress noise (heartbeat/ping/poll)          │   │
│  │ 4. Aggregate tool_before → tool_after pairs       │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼ (validated events)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SLM PROCESSING (Ollama qwen2.5-coder:3b)         │   │
│  │ - ClassifyMemory: tier, importance, tags, assocs   │   │
│  │ - CompressMemory: summary (≤20 words)              │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼ (classified + compressed)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STORAGE (RocksDB + in-memory graph)            │   │
│  │ - CognitiveMemoryUnit stored                       │   │
│  │ - Indexed for FTS5 search                      │   │
│  │ - Embedded for vector similarity              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Capture Pipeline Detailed Steps

| Step | Module | Description |
|------|-------|-----------|
| 1 | validate_event | Reject timestamp > 60s future, content > 100KB |
| 2 | deduplicate | Cache check with 5-sec window |
| 3 | suppress_noise | Filter heartbeat/ping/pong/health_check |
| 4 | handle_tool_before | Store pending for aggregation |
| 5 | handle_tool_after | Merge with tool_before, compute duration |
| 6 | classify_memory | SLM determines tier/importance/tags |
| 7 | compress_memory | SLM creates summary |
| 8 | store_event | Persist to RocksDB + index |

---

## 4. Query Flow: User Recall

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     RECALL PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                               │
│  USER COMMAND                                                   │
│     $ cognimem recall "how did I handle auth?"                        │
│           │                                                      │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SEARCH ENGINE                                        │   │
│  │ 1. Full-text search (FTS5)                        │   │
│  │ 2. Vector search (embedding similarity)           │   │
│  │ 3. Fuse results (40% FTS + 60% vector)         │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼ (candidates)                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SLM RERANK (Ollama)                                  │   │
│  │ - Query: "how did I handle auth?"                  │   │
│  │ - Re-rank by semantic relevance (LLM)              │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼ (reranked results)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ASSOCIATION EXPANSION                                │   │
│  │ - Spreading activation (3-hop neighbors)           │   │
│  │ - Hebbian strengthen for co-activated              │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     │                                            │
│                     ▼                                            │
│  USER OUTPUT                                                   │
│     ← Top 5-10 memories ranked by relevance + context               │
│                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Recall Pipeline Detailed Steps

| Step | Module | Description |
|------|-------|-----------|
| 1 | fts5_search | Full-text search on content |
| 2 | vector_search | Cosine similarity on embeddings |
| 3 | fuse_scores | Weighted combine (40% FTS + 60% vector) |
| 4 | slm_rerank | LLM re-ranks candidates |
| 5 | associate_expand | Add 3-hop neighbors via spreading activation |
| 6 | update_activation | Record rehearsal timestamps |
| 7 | strengthen_co-activated | Hebbian strengthen for pairs |

---

## 5. Consolidation Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  CONSOLIDATION PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                               │
│ TRIGGERS:                                                       │
│   - Automatic: every 3600 seconds (configurable)                │
│   - Manual:   $ cognimem consolidate --intensity full        │
│                                                               │
│ ┌─────────────────────────────────────────────────────────────┐         │
│ │ 1. DECAY (all tiers)                                        │         │
│ │    - Update activation: a(t) = a(0) * e^(-decay_rate * t) │         │
│ │    - Exponential decay per tier:                                │         │
│ │      Sensory: 0.1/hour                                      │         │
│ │      Working:  0.1/hour                                      │         │
│ │      Episodic: 0.05/hour                                     │         │
│ │      Semantic: 0.02/hour                                    │         │
│ │      Procedural: 0.01/hour                                  ��         │
│ └──────────────────┬──────────────────────────────────────────────┘         │
│                    │                                                      │
│                    ▼                                                      │
│ ┌─────────────────────────────────────────────────────────────┐         │
│ │ 2. PRUNE                                                   │         │
│ │    - Remove Sensory/Working/Episodic below 0.01             │         │
│ │    - Preserve Semantic/Procedural (never pruned)          │         │
│ └──────────────────┬─────────────────────────────────────────────┘         │
│                    │                                                      │
│                    ▼                                                      │
│ ┌─────────────────────────────────────────────────────────────┐         │
│ │ 3. CONFLICT DETECTION                                       │         │
│ │    - Vector similarity > 0.85 = conflict                  │         │
│ │    - Resolution strategies:                             │         │
│ │      latest_wins: Keep newer, discard older              │         │
│ │      keep_both: Preserve both with note                  │         │
│ │      human_decide: Flag for manual review              │         │
│ │    - SLM generates merged summary if needed             │         │
│ └──────────────────┬─────────────────────────────────────────────┘         │
│                    │                                                      │
│                    ▼                                                      │
│ ┌─────────────────────────────────────────────────────────────┐         │
│ │ 4. PROMOTE                                               │         │
│ │    - Episodic > 0.8 → Semantic                             │         │
│ │    - Semantic > 0.9 → Procedural                         │         │
│ │    - Update decay_rate to new tier                       │         │
│ └──────────────────────────────────────────────────────────────┘         │
│                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. User Commands

| Command | Description | Example |
|---------|------------|---------|
| `cognimem init --platform <name>` | One-time setup | `cognimem init --platform opencode` |
| `cognimem recall <query>` | Query memories | `cognimem recall "auth handling"` |
| `cognimem search <query>` | Search memories | `cognimem search "error handling"` |
| `cognimem remember <content>` | Manual store | `cognimem remember "Always use Result"` |
| `cognimem consolidate` | Run consolidation | `cognimem consolidate --intensity full` |
| `cognimem persona` | Extract profiles | `cognimem persona` |
| `cognimem stats` | Show statistics | `cognimem stats` |
| `cognimem daemon` | Start daemon | `cognimem daemon --data-path ~/.cognimem` |
| `cognimem timeline <memory_id>` | Get context | `cognimem timeline abc123` |
| `cognimem complete_pattern <cue>` | Pattern completion | `cognimem complete_pattern "user prefers"` |

---

## 7. MCP Tools Available

| Tool | Description |
|------|------------|
| `remember` | Store a memory with tier/importance |
| `recall` | Retrieve memories by query |
| `search` | Full-text + vector search |
| `associate` | Create association between memories |
| `forget` | Delete or soft-delete a memory |
| `reflect` | Run consolidation cycle |
| `timeline` | Get memories within time window |
| `get_observations` | Get full memory content |
| `execute_skill` | Run a procedural skill |
| `complete_pattern` | Pattern completion via Hebbian |
| `extract_persona` | Extract user profiles |
| `assign_role` | Assign RACI roles |
| `claim_work` | Claim work on memory |
| `release_work` | Release claimed work |
| `find_unclaimed_work` | Find available work |
| `summarize_turn` | Summarize AI turns |
| `summarize_session` | Summarize full session |
| `extract_best_practice` | Extract coding practices |
| `get_project_conventions` | Get project conventions |

---

## 8. HTTP Capture API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/capture/events` | POST | Single event capture |
| `/capture/events/batch` | POST | Batch event capture |
| `/capture/health` | GET | Health check |
| `/capture/stats` | GET | Capture statistics |

### 8.1 Event Payload Schema

```json
{
  "event_type": "file_edited",
  "timestamp": 1716400000,
  "session_id": "uuid",
  "project_path": "/path/to/project",
  "agent_id": null,
  "source": "opencode",
  "tool_name": "edit_file",
  "tool_input": {"file": "/path/file.rs", ...},
  "tool_output": {"success": true, ...},
  "file_path": "/path/file.rs",
  "content": "User edited file.rs",
  "success": true,
  "duration_ms": 150
}
```

---

## 9. Configuration File

Location: `~/.cognimem/config.toml`

```toml
# Core settings
data_path = "~/.cognimem/data"
socket_path = "~/.cognimem/cognimem.sock"

# Ollama settings (required for embeddings + classification)
ollama_model = "qwen2.5-coder:3b"
ollama_url = "http://localhost:11434"

# Capture server
capture_port = 37778

# Consolidation (automatic - runs in background)
decay_interval_secs = 60
consolidation_interval_secs = 3600
prune_threshold = 0.01

# Per-platform hooks
[hooks.opencode]
enabled = true
events = [
  "file_created",
  "file_edited",
  "tool_executed",
  "error",
  "session_created",
  "session_ended",
  "notification",
  "command_executed"
]

[hooks.claude_code]
enabled = true
# 24 event types

[hooks.codex]
enabled = false

# Advanced settings
[advanced]
# Embedding: "miniLM" (better) or "hash" (faster)
embedder = "hash"
decay_exponent = 0.15
conflict_threshold = 0.85
max_memory_age_days = 90
```

---

## 10. File Structure

```
~/.cognimem/
├── cognimem.sock              # Unix socket (MCP)
├── data/
│   ├── memories.db           # RocksDB storage
│   └── search.db             # FTS5 index
├── logs/
│   └── cognimem.log         # Log file
└── config.toml             # User configuration

~/.opencode/extensions/
└── cognimem-capture/       # OpenCode plugin
    ├── index.js            # Hook handler (8 events)
    └── package.json

~/.claude/
└── hooks.json             # Claude Code hooks (24 events)
```

---

## 11. Memory Tiers

| Tier | Purpose | Decay Rate | Capacity | Priority |
|------|--------|-----------|----------|----------|
| Sensory | Raw tool events | 0.1/hr | Unlimited |
| Working | In-context memories | 0.1/hr | 100 |
| Episodic | Session events | 0.05/hr | Unlimited |
| Semantic | Knowledge/facts | 0.02/hr | Unlimited |
| Procedural | Skills/macros | 0.01/hr | Unlimited (high importance) |

---

## 12. Known Issues (To Fix)

| Issue | Severity | Status |
|-------|----------|--------|
| Hash embeddings (poor conflict detection) | Critical | Pending fix |
| Existing data corruption (provenance_ids) | High | Pending fix |
| Near-duplicate guard clause memories | Medium | Pending fix |
| Misclassified persona (biography vs work) | Medium | Pending fix |

---

## 13. Expected Future Enhancements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Real embeddings (MiniLM) | Replace hash for semantic similarity | Critical |
| SM-2 spaced repetition | Schedule reviews by algorithm | High |
| Working memory buffer | Limited-capacity rehearsal | Medium |
| Web dashboard | UI for memory view | Low |
| Publish OpenCode plugin | Deploy to plugin marketplace | Medium |

---

## 14. Iteration Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-04-22 | 1.0 | Initial draft |

---

**Questions for Iteration:**
1. Is the workflow clear and intuitive?
2. Are all user commands intuitive?
3. What's missing from the end-user experience?
4. Should any component work differently?