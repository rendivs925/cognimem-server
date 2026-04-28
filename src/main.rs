mod config;

use clap::Parser;
use cognimem_server::capture::{CapturePipeline, start_capture_server};
use cognimem_server::dashboard::start_dashboard_server;
use cognimem_server::embeddings::fuse_scores;
use cognimem_server::broker::BrokerEvent;
use cognimem_server::watcher::start_file_watcher;
use cognimem_server::memory::{
    AssignRoleArgs, AssignRoleResult, AssociateArgs, AssociateResult, ClaimStatus,
    ClassifyMemoryInput, CognitiveMemoryUnit, ClaimWorkArgs, CompletePatternArgs,
    CompletePatternInput, CompletePatternResult, CompressMemoryInput, DelegateToLlmArgs,
    DiscoverProjectArgs, DiscoverProjectResult, EmotionState, ExecuteSkillArgs, ExecuteSkillResult,
    ExploreModuleArgs, ExtractBestPracticeArgs, ExtractPersonaInput, ExtractPersonaMemoryInput,
    ExtractPersonaResult, FindUnclaimedWorkArgs, ForgetArgs, ForgetResult, GetObservationsArgs,
    GetProjectConventionsArgs, HandoffSummaryArgs, ImagineArgs, ImagineInput, InMemoryStore,
    InjectMemoryArgs, ListMemoriesArgs, ListMemoriesResult, MemoryScope, MemoryStore,
    MemorySummary, MemoryTier, NodeSummaryResult, ObservationsResult, RecallArgs, RecallResult,
    ReflectArgs, ReflectResult, ReleaseWorkArgs, RememberArgs, RememberResult,
    RerankCandidateInput, RerankCandidatesInput, ResolveConflictInput, RocksDbStore, SearchArgs,
    SearchCodebaseArgs, SearchCodebaseResult, SearchResult, SearchResults, SimulatePerspectiveArgs,
    SkillMemory, SlmError, TeachFromDemonstrationArgs, TimelineArgs, TimelineResult, WorkClaim,
};
use cognimem_server::memory::{
    apply_decay_to_all, consolidate, detect_conflicts, promote_memories, prune_below_threshold,
    resolve_conflicts,
};
use cognimem_server::memory::{
    complete_pattern, detect_and_create_skill, execute_skill as run_skill, extract_persona,
    find_skill, rank_by_timescale, strengthen_co_activated, apply_stdp,
};
use cognimem_server::memory::slm_types::SimulatePerspectiveInput;
use cognimem_server::metrics::{
    inc_associate, inc_forget, inc_prune, inc_recall, inc_reflect, inc_remember, set_code_node_count,
    set_memory_count,
};
use cognimem_server::state::CogniMemState;
use config::Cli;
use rmcp::{
    ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ListResourcesResult, PaginatedRequestParams,
        RawResource, ReadResourceRequestParams, ReadResourceResult, Resource, ResourceContents,
    },
    service::{RequestContext, RoleServer},
};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{self, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tracing::error;

#[derive(Clone)]
struct CogniMemServer {
    state: Arc<Mutex<CogniMemState>>,
    rate_limiter: Arc<cognimem_server::rate_limit::RateLimiter>,
}

impl CogniMemServer {
    fn new(state: Arc<Mutex<CogniMemState>>) -> Self {
        Self {
            state,
            rate_limiter: Arc::new(cognimem_server::rate_limit::RateLimiter::new(100, 60)),
        }
    }
}

fn parse_args<T: serde::de::DeserializeOwned>(
    args: serde_json::Map<String, serde_json::Value>,
) -> Result<T, rmcp::ErrorData> {
    let coerced = coerce_arg_types(args);
    serde_json::from_value(serde_json::Value::Object(coerced)).map_err(|e| {
        let msg = e.to_string();
        let hint = if msg.contains("invalid type: string") && msg.contains("expected") {
            let suggestion = if msg.contains("f32") || msg.contains("f64") {
                " — pass a number, not a string (e.g. 0.0 not \"0.0\")"
            } else if msg.contains("integer") || msg.contains("u64") || msg.contains("usize") {
                " — pass an integer, not a string (e.g. 5 not \"5\")"
            } else {
                " — check the parameter type in the tool schema"
            };
            format!("{msg}{suggestion}")
        } else {
            msg
        };
        rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(hint), None)
    })
}

fn coerce_arg_types(
    mut args: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let number_keys = [
        "importance",
        "min_activation",
        "tolerance",
        "strength",
        "prune_threshold",
        "limit",
        "window_secs",
        "hours",
        "confidence",
    ];
    for key in number_keys {
        if let Some(serde_json::Value::String(s)) = args.get(key) {
            if let Ok(f) = s.parse::<f64>() {
                args.insert(key.to_string(), serde_json::Value::from(f));
            } else if let Ok(i) = s.parse::<i64>() {
                args.insert(key.to_string(), serde_json::Value::from(i));
            }
        }
    }
    args
}

fn success_json<T: serde::Serialize>(data: &T) -> CallToolResult {
    match serde_json::to_value(data) {
        Ok(json) => match serde_json::to_string(&json) {
            Ok(text) => CallToolResult::success(vec![rmcp::model::Content::text(text)]),
            Err(e) => {
                tracing::error!("Failed to serialize JSON response: {e}");
                CallToolResult::success(vec![rmcp::model::Content::text(
                    r#"{"error":"serialization failed"}"#.to_string(),
                )])
            }
        },
        Err(e) => {
            tracing::error!("Failed to convert response to JSON value: {e}");
            CallToolResult::success(vec![rmcp::model::Content::text(
                r#"{"error":"serialization failed"}"#.to_string(),
            )])
        }
    }
}

fn invalid_params(msg: &str) -> rmcp::ErrorData {
    rmcp::ErrorData::new(
        rmcp::model::ErrorCode(-32602),
        Cow::Owned(msg.to_string()),
        None,
    )
}

fn tool_not_found() -> rmcp::ErrorData {
    rmcp::ErrorData::new(
        rmcp::model::ErrorCode(-32601),
        Cow::Owned("Unknown tool".to_string()),
        None,
    )
}

fn indexed_content(memory: &CognitiveMemoryUnit) -> String {
    match memory.model.compressed_content.as_deref() {
        Some(compressed) if !compressed.is_empty() => format!("{} {}", memory.content, compressed),
        _ => memory.content.clone(),
    }
}

fn memory_matches_scope(
    memory: &CognitiveMemoryUnit,
    project_path: Option<&str>,
    scope_filter: &str,
) -> bool {
    match scope_filter {
        "global" => memory.scope.is_global(),
        "project" => match memory.scope.project_path() {
            Some(path) => project_path.map_or(true, |expected| expected == path),
            None => false,
        },
        _ => match (project_path, memory.scope.project_path()) {
            (Some(expected), Some(path)) => expected == path,
            (Some(_), None) => memory.scope.is_global(),
            (None, _) => true,
        },
    }
}

fn heuristic_classification(content: &str) -> (MemoryTier, f32) {
    let lower = content.to_lowercase();
    let tier = if lower.contains("always")
        || lower.contains("preference")
        || lower.contains("rule")
        || lower.contains("convention")
    {
        MemoryTier::Semantic
    } else if lower.contains("steps:")
        || lower.contains("process")
        || lower.contains("workflow")
        || lower.contains("how to")
    {
        MemoryTier::Procedural
    } else {
        MemoryTier::Episodic
    };
    let importance = if content.len() > 500 { 0.7 } else { 0.5 };
    (tier, importance)
}

fn heuristic_compression(content: &str) -> String {
    content
        .split_whitespace()
        .take(20)
        .collect::<Vec<_>>()
        .join(" ")
}

const MAX_CONTENT_LEN: usize = 10_000;
const MAX_ASSOCIATIONS: usize = 50;

fn json_schema(
    json: serde_json::Value,
) -> std::sync::Arc<serde_json::Map<String, serde_json::Value>> {
    std::sync::Arc::new(
        json.as_object()
            .cloned()
            .expect("json_schema input must be a JSON object"),
    )
}

const TIER_ENUM: &[&str] = &["sensory", "working", "episodic", "semantic", "procedural"];

impl ServerHandler for CogniMemServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_resources()
                .enable_tools()
                .build(),
        )
    }

    async fn list_tools(
        &self,
        request: Option<rmcp::model::PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let _ = (request, context);
        let tools = vec![
            rmcp::model::Tool::new(
                Cow::Borrowed("remember"),
                Cow::Borrowed("Store a memory with optional tier and importance"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "tier": { "type": "string", "enum": TIER_ENUM },
                        "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                        "scope": { "type": "string", "description": "global or project path" },
                        "associations": { "type": "array", "items": { "type": "string", "format": "uuid" } }
                    },
                    "required": ["content"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("recall"),
                Cow::Borrowed("Retrieve memories relevant to a query"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "tier": { "type": "string", "enum": TIER_ENUM },
                        "project_path": { "type": "string" },
                        "scope_filter": { "type": "string", "enum": ["global", "project", "both"] },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 50 },
                        "min_activation": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Minimum activation threshold" }
                    },
                    "required": ["query"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("associate"),
                Cow::Borrowed(
                    "Create an association between two memories with an optional strength weight",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "string", "format": "uuid" },
                        "to": { "type": "string", "format": "uuid" },
                        "strength": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                    },
                    "required": ["from", "to"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("forget"),
                Cow::Borrowed("Delete or soft-delete a specific memory"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" },
                        "hard_delete": { "type": "boolean", "description": "Permanently remove (true) or set activation near-zero (false, default)" }
                    },
                    "required": ["memory_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("reflect"),
                Cow::Borrowed(
                    "Run a consolidation cycle: decay activation, prune weak memories, promote strong ones, detect conflicts",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "intensity": { "type": "string", "enum": ["light", "full"], "description": "light=decay only, full=decay+prune+promote+consolidate" },
                        "conflict_strategy": { "type": "string", "enum": ["latest_wins", "keep_both", "human_decide"], "description": "How to resolve detected conflicts (default: latest_wins)" }
                    }
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("search"),
                Cow::Borrowed(
                    "Search memories returning compact summaries (id, snippet, tier, activation)",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "tier": { "type": "string", "enum": TIER_ENUM },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 50 }
                    },
                    "required": ["query"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("list_memories"),
                Cow::Borrowed("List memories deterministically with optional filters"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tier": { "type": "string", "enum": TIER_ENUM },
                        "project_path": { "type": "string" },
                        "scope_filter": { "type": "string", "enum": ["global", "project", "both"] },
                        "min_activation": { "type": "number", "minimum": 0.0 },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 200 }
                    }
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("timeline"),
                Cow::Borrowed("Get memories within a time window around a specific memory"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" },
                        "window_secs": { "type": "integer", "description": "Time window in seconds around the memory (default 900 = 15 min)" }
                    },
                    "required": ["memory_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("get_observations"),
                Cow::Borrowed("Get the full content and metadata of a specific memory"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" }
                    },
                    "required": ["memory_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("execute_skill"),
                Cow::Borrowed(
                    "Look up a procedural skill by name and return its distilled pattern and steps",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "skill_name": { "type": "string", "description": "Name of the skill to execute" }
                    },
                    "required": ["skill_name"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("complete_pattern"),
                Cow::Borrowed(
                    "Reconstruct likely full memories from a partial cue using Hebbian associations",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "cue": { "type": "string", "description": "Partial text cue to complete" },
                        "tolerance": { "type": "number", "description": "Minimum similarity threshold (default 0.3)" },
                        "limit": { "type": "integer", "description": "Max candidates (default 5)" }
                    },
                    "required": ["cue"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("extract_persona"),
                Cow::Borrowed(
                    "Scan semantic memories and extract structured persona across 6 domains",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("assign_role"),
                Cow::Borrowed(
                    "Assign RACI roles (responsible, accountable, consulted, informed) to a memory",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" },
                        "responsible": { "type": "string", "description": "Agent ID who does the work" },
                        "accountable": { "type": "string", "description": "Agent ID who owns the outcome" },
                        "consulted": { "type": "array", "items": { "type": "string" }, "description": "Agent IDs consulted before action" },
                        "informed": { "type": "array", "items": { "type": "string" }, "description": "Agent IDs informed after action" }
                    },
                    "required": ["memory_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("claim_work"),
                Cow::Borrowed("Claim exclusive right to work on a memory/task"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" },
                        "claim_type": { "type": "string", "enum": ["research", "implementation", "testing", "review"] },
                        "hours": { "type": "integer", "minimum": 1, "maximum": 72, "description": "Lease duration in hours (default: 24)" }
                    },
                    "required": ["memory_id", "claim_type"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("release_work"),
                Cow::Borrowed("Release a work claim"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "memory_id": { "type": "string", "format": "uuid" },
                        "complete": { "type": "boolean", "description": "Mark as completed (true) or released (false)" }
                    },
                    "required": ["memory_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("find_unclaimed_work"),
                Cow::Borrowed("Find memories or tasks available for claiming"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_path": { "type": "string", "description": "Filter by project path" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 20 }
                    }
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("summarize_turn"),
                Cow::Borrowed("Summarize multiple turns into a concise overview"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "turns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "turn_id": { "type": "string", "format": "uuid" },
                                    "content": { "type": "string" },
                                    "tool_usage": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    },
                                    "decisions": {
                                        "type": "array",
                                        "items": { "type": "string" }
                                    }
                                },
                                "required": ["content"]
                            }
                        }
                    },
                    "required": ["turns"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("summarize_session"),
                Cow::Borrowed("Summarize a complete session with completed and open tasks"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "turns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "turn_id": { "type": "string", "format": "uuid" },
                                    "content": { "type": "string" }
                                },
                                "required": ["content"]
                            }
                        },
                        "completed_tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_id": { "type": "string", "format": "uuid" },
                                    "title": { "type": "string" },
                                    "status": { "type": "string" }
                                },
                                "required": ["title"]
                            }
                        },
                        "open_tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_id": { "type": "string", "format": "uuid" },
                                    "title": { "type": "string" },
                                    "status": { "type": "string" }
                                },
                                "required": ["title"]
                            }
                        }
                    },
                    "required": ["turns"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("extract_best_practice"),
                Cow::Borrowed(
                    "Extract coding best practices from content (DRY, KISS, SOLID, YAGNI, Guard Clauses)",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": { "type": "string", "description": "Content to analyze for best practices" },
                        "context": { "type": "string", "description": "Optional context for the content" }
                    },
                    "required": ["content"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("get_project_conventions"),
                Cow::Borrowed("Get extracted conventions for a project"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_path": { "type": "string" }
                    },
                    "required": ["project_path"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("delegate_to_llm"),
                Cow::Borrowed(
                    "Delegate a query to a larger model when confidence is below threshold",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "context": { "type": "array", "items": { "type": "string" } },
                        "confidence_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                    },
                    "required": ["query"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("teach_from_demonstration"),
                Cow::Borrowed(
                    "Learn a pattern from external demonstration (two-stage: episodic now, procedural after repeat)",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "demonstration": { "type": "string" },
                        "pattern_extracted": { "type": "string" },
                        "domain": { "type": "string" },
                        "source_type": { "type": "string" }
                    },
                    "required": ["demonstration"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("simulate_perspective"),
                Cow::Borrowed(
                    "Answer from a specific perspective (security expert, end user, etc.)",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "perspective_role": { "type": "string" },
                        "situation": { "type": "string" },
                        "question": { "type": "string" }
                    },
                    "required": ["perspective_role", "situation", "question"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("inject_memory"),
                Cow::Borrowed(
                    "Find and inject a highly relevant memory into the current context",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "The current user prompt or context to match against" }
                    },
                    "required": ["query"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("handoff_summary"),
                Cow::Borrowed(
                    "Store a structured handoff summary when transferring context between sessions",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "from_session": { "type": "string", "format": "uuid", "description": "Session ID handing off" },
                        "project_path": { "type": "string", "description": "Project path for the handoff" },
                        "summary": { "type": "string", "description": "Summary of work done and context" },
                        "unresolved": { "type": "array", "items": { "type": "string" }, "description": "List of unresolved issues" },
                        "next_steps": { "type": "array", "items": { "type": "string" }, "description": "Suggested next steps" },
                        "relevant_memories": { "type": "array", "items": { "type": "string", "format": "uuid" }, "description": "Relevant memory IDs to carry forward" }
                    },
                    "required": ["from_session", "summary"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("imagine"),
                Cow::Borrowed(
                    "Simulate a hypothetical scenario using relevant memories as context",
                ),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "scenario": { "type": "string", "description": "The hypothetical scenario to simulate" },
                        "context": { "type": "array", "items": { "type": "string" }, "description": "Optional additional context strings" }
                    },
                    "required": ["scenario"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("discover_project"),
                Cow::Borrowed("Parse source files and discover code structure (functions, structs, traits, etc.)"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_path": { "type": "string", "description": "Path to the project root (defaults to current working directory)" }
                    }
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("explore_module"),
                Cow::Borrowed("Get child nodes of a code graph node or all nodes in a file"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "node_id": { "type": "string", "format": "uuid", "description": "UUID of the parent node to explore" },
                        "file_path": { "type": "string", "description": "File path to list all nodes in that file" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum number of children to return" }
                    }
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("get_node_summary"),
                Cow::Borrowed("Get summary of a specific code node by UUID"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "node_id": { "type": "string", "format": "uuid" }
                    },
                    "required": ["node_id"]
                })),
            ),
            rmcp::model::Tool::new(
                Cow::Borrowed("search_codebase"),
                Cow::Borrowed("Search code graph by name of function, struct, trait, etc."),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "file_path": { "type": "string", "description": "Limit search to a specific file" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 50 }
                    },
                    "required": ["query"]
                })),
            ),
        ];

        Ok(rmcp::model::ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let _ = context;
        if !self.rate_limiter.allow() {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32000),
                Cow::Borrowed("Rate limit exceeded: max 100 requests per minute"),
                None,
            ));
        }
        let args = request.arguments.unwrap_or_default();

        match request.name.as_ref() {
            "remember" => self.handle_remember(args).await,
            "recall" => self.handle_recall(args).await,
            "associate" => self.handle_associate(args).await,
            "forget" => self.handle_forget(args).await,
            "reflect" => self.handle_reflect(args).await,
            "search" => self.handle_search(args).await,
            "list_memories" => self.handle_list_memories(args).await,
            "timeline" => self.handle_timeline(args).await,
            "get_observations" => self.handle_get_observations(args).await,
            "execute_skill" => self.handle_execute_skill(args).await,
            "complete_pattern" => self.handle_complete_pattern(args).await,
            "extract_persona" => self.handle_extract_persona(args).await,
            "assign_role" => self.handle_assign_role(args).await,
            "claim_work" => self.handle_claim_work(args).await,
            "release_work" => self.handle_release_work(args).await,
            "find_unclaimed_work" => self.handle_find_unclaimed_work(args).await,
            "summarize_turn" => self.handle_summarize_turn(args).await,
            "summarize_session" => self.handle_summarize_session(args).await,
            "extract_best_practice" => self.handle_extract_best_practice(args).await,
            "get_project_conventions" => self.handle_get_project_conventions(args).await,
            "delegate_to_llm" => self.handle_delegate_to_llm(args).await,
            "teach_from_demonstration" => self.handle_teach_from_demonstration(args).await,
            "simulate_perspective" => self.handle_simulate_perspective(args).await,
            "inject_memory" => self.handle_inject_memory(args).await,
            "handoff_summary" => self.handle_handoff_summary(args).await,
            "imagine" => self.handle_imagine(args).await,
            "discover_project" => self.handle_discover_project(args).await,
            "explore_module" => self.handle_explore_module(args).await,
            "get_node_summary" => self.handle_get_node_summary(args).await,
            "search_codebase" => self.handle_search_codebase(args).await,
            _ => Err(tool_not_found()),
        }
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, rmcp::ErrorData> {
        let _ = (request, context);
        let guard = self.state.lock().await;
        let mut resources: Vec<Resource> = Vec::new();

        for m in guard.graph.get_all_memories() {
            resources.push(Resource::new(
                RawResource::new(format!("memory://{}/{}", m.tier, m.id), &m.content)
                    .with_description(format!(
                        "Memory in {} tier with activation {:.3}",
                        m.tier, m.metadata.base_activation
                    ))
                    .with_mime_type("application/json"),
                None,
            ));
        }

        resources.push(Resource::new(
            RawResource::new("todo://list", "All todos/tasks in Procedural tier")
                .with_mime_type("application/json"),
            None,
        ));

        resources.push(Resource::new(
            RawResource::new("skill://list", "All learned skills in Procedural tier")
                .with_mime_type("application/json"),
            None,
        ));

        resources.push(Resource::new(
            RawResource::new("context://current", "Current session context")
                .with_mime_type("application/json"),
            None,
        ));

        resources.push(Resource::new(
            RawResource::new("context://history", "Session handoff history")
                .with_mime_type("application/json"),
            None,
        ));

        resources.push(Resource::new(
            RawResource::new("code://list", "All discovered code nodes")
                .with_mime_type("application/json"),
            None,
        ));

        for f in guard.code_graph.all_files().into_iter().take(10) {
            let name = f.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            resources.push(Resource::new(
                RawResource::new(format!("code://{}", name), &format!("Code nodes in {}", name))
                    .with_mime_type("application/json"),
                None,
            ));
        }

        Ok(ListResourcesResult::with_all_items(resources))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, rmcp::ErrorData> {
        let _ = context;
        let uri = &request.uri;
        let guard = self.state.lock().await;

        if uri.starts_with("memory://") {
            let path = &uri["memory://".len()..];
            let id_str = path.split('/').next_back().unwrap_or("");
            let id = uuid::Uuid::parse_str(id_str).map_err(|e| {
                rmcp::ErrorData::new(
                    rmcp::model::ErrorCode(-32602),
                    Cow::Owned(format!("Invalid memory ID: {e}")),
                    None,
                )
            })?;

            let memory = guard.graph.get_memory(&id).ok_or_else(|| {
                rmcp::ErrorData::new(
                    rmcp::model::ErrorCode(-32602),
                    Cow::Owned(format!("Memory not found: {id}")),
                    None,
                )
            })?;

            let json = serde_json::to_string(memory).unwrap_or_else(|e| {
                tracing::error!("Failed to serialize memory for resource read: {e}");
                format!("{{\"error\":\"serialization failed: {e}\"}}")
            });
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
        }

        if uri.starts_with("todo://") {
            let todos: Vec<serde_json::Value> = guard
                .graph
                .get_by_tier(MemoryTier::Procedural)
                .iter()
                .filter(|m| m.content.to_lowercase().contains("todo") || m.content.to_lowercase().contains("task"))
                .take(20)
                .map(|m| {
                    serde_json::json!({
                        "id": m.id,
                        "content": m.content,
                        "status": if m.metadata.base_activation > 0.5 { "active" } else { "pending" },
                        "created_at": m.metadata.created_at,
                    })
                })
                .collect();

            let json = serde_json::to_string(&todos).unwrap_or_default();
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
        }

        if uri.starts_with("skill://") {
            let skills: Vec<serde_json::Value> = guard
                .graph
                .get_by_tier(MemoryTier::Procedural)
                .iter()
                .filter(|m| m.content.starts_with("[skill:") || m.persona.is_some())
                .take(20)
                .map(|m| {
                    serde_json::json!({
                        "id": m.id,
                        "name": m.persona.map(|p| format!("{:?}", p)).unwrap_or_else(|| "unnamed".to_string()),
                        "content": m.content,
                    })
                })
                .collect();

            let json = serde_json::to_string(&skills).unwrap_or_default();
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
        }

        if uri.starts_with("context://") {
            if let Some(ref ctx) = guard.session_context {
                let json = serde_json::to_string(ctx).unwrap_or_default();
                return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
            }

            let sessions: Vec<serde_json::Value> = guard
                .handoffs
                .iter()
                .take(10)
                .map(|h| {
                    serde_json::json!({
                        "from_session": h.from_session,
                        "summary": h.summary,
                        "unresolved": h.unresolved,
                    })
                })
                .collect();

            let json = serde_json::to_string(&sessions).unwrap_or_default();
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
        }

        if uri.starts_with("code://") {
            let path = &uri["code://".len()..];
            let nodes = if path.is_empty() || path == "list" {
                guard.code_graph.all_files()
                    .iter()
                    .flat_map(|f| guard.code_graph.get_nodes_in_file(f))
                    .take(50)
                    .collect::<Vec<_>>()
            } else if let Ok(id) = uuid::Uuid::parse_str(path) {
                guard.code_graph.get_node(&id).into_iter().collect()
            } else {
                guard.code_graph.search_by_name(path)
                    .into_iter()
                    .take(20)
                    .collect()
            };

            let json = serde_json::to_string(&nodes).unwrap_or_default();
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.to_string())]));
        }

        Err(rmcp::ErrorData::new(
            rmcp::model::ErrorCode(-32602),
            Cow::Owned(format!("Invalid resource URI: {uri}")),
            None,
        ))
    }
}

impl CogniMemServer {
    async fn handle_remember(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: RememberArgs = parse_args(args)?;

        if args.content.len() > MAX_CONTENT_LEN {
            return Err(invalid_params(&format!(
                "content exceeds maximum length of {MAX_CONTENT_LEN} bytes"
            )));
        }

        if let Some(ref assoc_ids) = args.associations
            && assoc_ids.len() > MAX_ASSOCIATIONS
        {
            return Err(invalid_params(&format!(
                "associations exceed maximum count of {MAX_ASSOCIATIONS}"
            )));
        }

        let content = args.content;
        let classify = {
            let guard = self.state.lock().await;
            match guard
                .slm
                .classify_memory(ClassifyMemoryInput {
                    content: content.clone(),
                })
                .await
            {
                Ok(classify) => Some(classify),
                Err(e) => {
                    tracing::warn!("Falling back to heuristic memory classification: {e}");
                    None
                }
            }
        };

        let (fallback_tier, fallback_importance) = heuristic_classification(&content);
        let tier = args
            .tier
            .unwrap_or_else(|| classify.as_ref().map(|c| c.tier).unwrap_or(fallback_tier));
        let importance = args.importance.unwrap_or_else(|| {
            classify
                .as_ref()
                .map(|c| c.importance)
                .unwrap_or(fallback_importance)
        });
        let decay_rate = tier.decay_rate();

        let mut memory = CognitiveMemoryUnit::new(content, tier, importance, decay_rate);
        if let Some(scope) = args.scope.as_deref().and_then(MemoryScope::from_str) {
            memory.scope = scope;
        }
        if let Some(assoc_ids) = args.associations {
            memory.associations = assoc_ids;
        }
        if let Some(classify) = &classify {
            for assoc_id in classify
                .associations
                .iter()
                .filter_map(|assoc| assoc.memory_id)
            {
                if !memory.associations.contains(&assoc_id) {
                    memory.associations.push(assoc_id);
                }
            }
            memory.model.suggested_tier = Some(classify.tier);
            memory.model.suggested_importance = Some(classify.importance);
            memory.model.tags = classify.tags.clone();
            let mut provenance_ids: Vec<uuid::Uuid> = classify
                .associations
                .iter()
                .filter_map(|assoc| assoc.memory_id)
                .collect();
            provenance_ids.sort();
            provenance_ids.dedup();
            memory.model.provenance_ids = provenance_ids;
            memory.model.model_name = Some(classify.metadata.model.clone());
            memory.model.confidence = Some(classify.metadata.confidence);
            memory.model.suppress = classify.suppress;
        } else {
            memory.model.suggested_tier = Some(fallback_tier);
            memory.model.suggested_importance = Some(fallback_importance);
            memory.model.model_name = Some("heuristic".to_string());
            memory.model.confidence = Some(0.2);
        }

        let tag_emotion_result = {
            let guard = self.state.lock().await;
            guard
                .slm
                .tag_emotion(cognimem_server::memory::slm_types::TagEmotionInput {
                    content: memory.content.clone(),
                })
                .await
                .ok()
        };
        if let Some(emotion) = tag_emotion_result {
            memory.emotion = Some(EmotionState {
                valence: emotion.valence,
                arousal: emotion.arousal,
            });
        }

        let memory_id = memory.id;
        let mut guard = self.state.lock().await;

        if let Some(limit) = tier.capacity() {
            while guard.graph.count_by_tier(tier) >= limit {
                if let Some(evict_id) = guard.graph.find_lowest_activation_in_tier(tier) {
                    if let Some(evicted) = guard.graph.remove_memory(&evict_id) {
                        guard.search.remove(&evicted.id);
                        if let Err(e) = guard.storage.delete(&evicted.id) {
                            error!("Failed to delete evicted memory {}: {e}", evicted.id);
                        }
                    }
                } else {
                    break;
                }
            }
        }

        guard.graph.add_memory(memory.clone());
        let compressed = match guard
            .slm
            .compress_memory(CompressMemoryInput {
                content: memory.content.clone(),
                tier_hint: Some(memory.tier),
            })
            .await
        {
            Ok(output) => output.summary,
            Err(e) => {
                tracing::warn!("Falling back to heuristic compression: {e}");
                heuristic_compression(&memory.content)
            }
        };
        if let Some(stored_memory) = guard.graph.get_memory_mut(&memory_id) {
            stored_memory.model.compressed_content = Some(compressed.clone());
        }
        memory.model.compressed_content = Some(compressed.clone());
        guard
            .search
            .index(memory_id, &indexed_content(&memory), memory.tier);
        let embedding = guard.embedder.embed(&memory.content);
        guard.graph.set_embedding(memory_id, embedding);
        if let Err(e) = guard.storage.save(&memory) {
            error!("Failed to persist memory {memory_id}: {e}");
        }

        if !matches!(tier, MemoryTier::Procedural) {
            let CogniMemState {
                graph,
                storage,
                search,
                embedder,
                slm,
                work_claims: _,
                session_context: _,
                handoffs: _,
                project_models: _,
                injection: _,
                broker: _,
                code_graph: _,
                c3gan: _,
            } = &mut *guard;
            if let Some(skill_memory) = detect_and_create_skill(
                graph,
                embedder.as_ref(),
                &mut **search,
                slm.as_ref(),
                &memory.content,
            )
            .await
            .map_err(|e| slm_failed("distill_skill", slm.model_name(), e))?
            {
                let skill_compressed = match slm
                    .compress_memory(CompressMemoryInput {
                        content: skill_memory.content.clone(),
                        tier_hint: Some(skill_memory.tier),
                    })
                    .await
                {
                    Ok(output) => output.summary,
                    Err(e) => {
                        tracing::warn!("Falling back to heuristic skill compression: {e}");
                        heuristic_compression(&skill_memory.content)
                    }
                };
                search.index(skill_memory.id, &skill_compressed, skill_memory.tier);
                if let Err(e) = storage.save(&skill_memory) {
                    error!("Failed to persist skill memory {}: {e}", skill_memory.id);
                }
                set_memory_count(graph.len() as u64);
            }
        }

        inc_remember();
        set_memory_count(guard.graph.len() as u64);

        Ok(success_json(&RememberResult::success(&memory)))
    }

    async fn handle_recall(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: RecallArgs = parse_args(args)?;

        if args.query.is_empty() {
            return Err(invalid_params("query must not be empty"));
        }

        if let Some(limit) = args.limit
            && limit == 0
        {
            return Err(invalid_params("limit must be at least 1"));
        }

        let query = args.query.to_lowercase();
        let limit = args.limit.unwrap_or(5);
        let min_activation = args.min_activation.unwrap_or(0.0);
        let now = chrono::Utc::now().timestamp();
        let project_path = args.project_path.clone();
        let scope_filter = args
            .scope_filter
            .clone()
            .unwrap_or_else(|| "both".to_string());

        let mut guard = self.state.lock().await;

        let scope_matches = |memory: &CognitiveMemoryUnit| -> bool {
            memory_matches_scope(memory, project_path.as_deref(), &scope_filter)
        };

        let query_emb = guard.embedder.embed(&query);

        let fts_ids = guard.search.search(&query, args.tier, limit * 4);
        let vec_scores = guard.graph.vector_search(&query_emb, limit * 4, 0.1);

        let fused = if fts_ids.is_empty() && vec_scores.is_empty() {
            let mut fallback: Vec<&CognitiveMemoryUnit> = match args.tier {
                Some(tier) => guard.graph.get_by_tier(tier),
                None => guard.graph.get_all_memories(),
            }
            .into_iter()
            .filter(|m| {
                cognimem_server::search::matches_query(&m.content, &query)
                    && m.metadata.base_activation >= min_activation
                    && scope_matches(m)
            })
            .collect();
            fallback.sort_by(|a, b| {
                b.metadata
                    .base_activation
                    .partial_cmp(&a.metadata.base_activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            fallback.truncate(limit);
            let ids: Vec<uuid::Uuid> = fallback.iter().map(|m| m.id).collect();
            update_activation_for(&mut guard, &ids, now);
            let memories: Vec<MemorySummary> = ids
                .iter()
                .filter_map(|id| guard.graph.get_memory(id).map(MemorySummary::from))
                .collect();
            inc_recall();
            return Ok(success_json(&RecallResult::new(memories)));
        } else {
            fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6)
        };

        let fused_scores: Vec<(uuid::Uuid, f32)> = fused.iter().map(|(id, score)| (*id, *score)).collect();

        let results_before_association: Vec<uuid::Uuid> = fused
            .iter()
            .filter_map(|(id, _)| {
                guard.graph.get_memory(id).map(|m| {
                    if m.metadata.base_activation >= min_activation && scope_matches(m) {
                        Some(m.id)
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .collect();

        let timescaled_scores = rank_by_timescale(&mut guard.graph, fused_scores);
        let timescaled_ids: Vec<uuid::Uuid> = timescaled_scores.iter().map(|(id, _)| *id).collect();

        let direct_ids: Vec<uuid::Uuid> = results_before_association.clone();

        {
            let graph = &mut guard.graph;
            for i in 0..direct_ids.len() {
                for j in (i + 1)..direct_ids.len() {
                    apply_stdp(graph, &direct_ids[i], &direct_ids[j], now);
                }
            }
        }

        let expanded: Vec<&CognitiveMemoryUnit> = {
            let expanded_ids = guard.graph.spreading_activation(&direct_ids, 3, 0.5, 0.1);
            expanded_ids
                .iter()
                .filter_map(|(id, _, _)| guard.graph.get_memory(id))
                .collect()
        };

        let mut results: Vec<&CognitiveMemoryUnit> = results_before_association
            .iter()
            .filter_map(|id| guard.graph.get_memory(id))
            .chain(expanded.into_iter())
            .filter(|m| m.metadata.base_activation >= min_activation && scope_matches(m))
            .collect();

        dedup_memories(&mut results);

        results.sort_by(|a, b| {
            let pos_a = timescaled_ids.iter().position(|id| *id == a.id).unwrap_or(usize::MAX);
            let pos_b = timescaled_ids.iter().position(|id| *id == b.id).unwrap_or(usize::MAX);
            pos_a.cmp(&pos_b)
        });

        let slm_results: Vec<usize> = {
            let candidates: Vec<RerankCandidateInput> = results
                .iter()
                .map(|m| RerankCandidateInput {
                    id: m.id,
                    content: m.content.clone(),
                    initial_score: m.metadata.base_activation,
                })
                .collect();
            let ranked_ids = match guard
                .slm
                .rerank_candidates(RerankCandidatesInput {
                    query: query.clone(),
                    candidates: candidates.clone(),
                    top_n: limit,
                })
                .await
            {
                Ok(output) => output.ranked_ids,
                Err(e) => {
                    tracing::warn!("Falling back to deterministic reranking: {e}");
                    Vec::new()
                }
            };

            if ranked_ids.is_empty() {
                Vec::new()
            } else {
                let candidate_ids: std::collections::HashSet<uuid::Uuid> =
                    candidates.iter().map(|candidate| candidate.id).collect();
                let mut seen = std::collections::HashSet::new();
                let mut is_valid = true;
                for id in &ranked_ids {
                    if !candidate_ids.contains(id) || !seen.insert(*id) {
                        is_valid = false;
                        break;
                    }
                }

                if !is_valid {
                    tracing::warn!("Falling back to deterministic reranking after invalid SLM IDs");
                    Vec::new()
                } else {
                    ranked_ids
                        .iter()
                        .filter_map(|id| {
                            candidates.iter().position(|candidate| &candidate.id == id)
                        })
                        .collect()
                }
            }
        };
        if !slm_results.is_empty() {
            let reranked: Vec<&CognitiveMemoryUnit> = slm_results
                .iter()
                .filter_map(|&i| results.get(i).copied())
                .collect();
            results = reranked;
            results.truncate(limit);
        } else {
            results.truncate(limit);
        }

        let recalled_ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
        update_activation_for(&mut guard, &recalled_ids, now);
        strengthen_co_activated(&mut guard.graph, &recalled_ids);

        let memories: Vec<MemorySummary> = recalled_ids
            .iter()
            .filter_map(|id| guard.graph.get_memory(id).map(MemorySummary::from))
            .collect();

        inc_recall();
        Ok(success_json(&RecallResult::new(memories)))
    }

    async fn handle_associate(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: AssociateArgs = parse_args(args)?;
        let strength = args.strength.unwrap_or(0.5);

        let mut guard = self.state.lock().await;

        if !guard.graph.add_association(&args.from, &args.to, strength) {
            return Err(invalid_params("One or both memory IDs not found"));
        }

        if let Some(mem) = guard.graph.get_memory(&args.from)
            && let Err(e) = guard.storage.save(mem)
        {
            error!("Failed to persist association for {}: {e}", args.from);
        }

        inc_associate();
        Ok(success_json(&AssociateResult::success(
            args.from, args.to, strength,
        )))
    }

    async fn handle_forget(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ForgetArgs = parse_args(args)?;

        let mut guard = self.state.lock().await;

        if !guard.graph.contains(&args.memory_id) {
            return Ok(success_json(&ForgetResult::not_found(args.memory_id)));
        }

        inc_forget();

        if args.hard_delete.unwrap_or(false) {
            if let Some(removed) = guard.graph.remove_memory(&args.memory_id) {
                guard.search.remove(&removed.id);
                if let Err(e) = guard.storage.delete(&removed.id) {
                    error!("Failed to delete memory {}: {e}", removed.id);
                }
            }
            set_memory_count(guard.graph.len() as u64);
            Ok(success_json(&ForgetResult::hard_deleted(args.memory_id)))
        } else {
            if let Some(mem) = guard.graph.get_memory_mut(&args.memory_id) {
                mem.metadata.base_activation = 0.001;
            }
            if let Some(mem) = guard.graph.get_memory(&args.memory_id)
                && let Err(e) = guard.storage.save(mem)
            {
                error!("Failed to persist soft-delete for {}: {e}", args.memory_id);
            }
            Ok(success_json(&ForgetResult::soft_deleted(args.memory_id)))
        }
    }

    async fn handle_reflect(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ReflectArgs = parse_args(args)?;
        let intensity = args.intensity.unwrap_or_else(|| "light".to_string());
        let strategy: cognimem_server::memory::ConflictResolution = args
            .conflict_strategy
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();

        let mut guard = self.state.lock().await;
        let CogniMemState {
            graph,
            storage,
            search,
            embedder,
            slm,
            work_claims: _,
            session_context: _,
            handoffs: _,
            project_models: _,
            injection: _,
            broker: _,
            code_graph: _,
            c3gan: _,
        } = &mut *guard;
        let total = graph.len();

        if intensity == "full" {
            apply_decay_to_all(graph);

            let pruned_ids = prune_below_threshold(graph, 0.01);
            for id in &pruned_ids {
                search.remove(id);
                if let Err(e) = storage.delete(id) {
                    error!("Failed to delete pruned memory {}: {e}", id);
                }
            }

            let conflicts = consolidate(graph, embedder.as_ref());
            let strategy = if args.conflict_strategy.is_some() {
                strategy
            } else if let Some(c) = conflicts.first() {
                let a = graph
                    .get_memory(&c.memory_id_1)
                    .map(|m| m.content.as_str())
                    .unwrap_or("");
                let b = graph
                    .get_memory(&c.memory_id_2)
                    .map(|m| m.content.as_str())
                    .unwrap_or("");
                slm.resolve_conflict(ResolveConflictInput {
                    memory_a_id: c.memory_id_1,
                    memory_a_content: a.to_string(),
                    memory_b_id: c.memory_id_2,
                    memory_b_content: b.to_string(),
                })
                .await
                .map(|output| output.action)
                .unwrap_or_else(|e| {
                    tracing::warn!("Falling back to keep_both conflict resolution: {e}");
                    cognimem_server::memory::ConflictResolution::KeepBoth
                })
            } else {
                strategy
            };
            let resolved = resolve_conflicts(graph, &conflicts, &strategy);
            for id in &resolved {
                search.remove(id);
                if let Err(e) = storage.delete(id) {
                    error!("Failed to delete resolved conflict memory {}: {e}", id);
                }
            }

            let promoted = promote_memories(graph);

            for mem in graph.get_all_memories() {
                search.index(mem.id, &indexed_content(mem), mem.tier);
                if let Err(e) = storage.save(mem) {
                    error!("Failed to persist reflected memory {}: {e}", mem.id);
                }
            }

            inc_reflect();
            inc_prune(pruned_ids.len() as u64 + resolved.len() as u64);
            set_memory_count(graph.len() as u64);

            Ok(success_json(&ReflectResult::new(
                pruned_ids.len() + resolved.len(),
                promoted,
                total,
                conflicts,
            )))
        } else {
            apply_decay_to_all(graph);
            let conflicts = detect_conflicts(graph, embedder.as_ref());

            for mem in graph.get_all_memories() {
                search.index(mem.id, &indexed_content(mem), mem.tier);
                if let Err(e) = storage.save(mem) {
                    error!("Failed to persist reflected memory {}: {e}", mem.id);
                }
            }

            inc_reflect();
            set_memory_count(graph.len() as u64);

            Ok(success_json(&ReflectResult::new(0, 0, total, conflicts)))
        }
    }

    async fn handle_search(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: SearchArgs = parse_args(args)?;

        if args.query.is_empty() {
            return Err(invalid_params("query must not be empty"));
        }

        let limit = args.limit.unwrap_or(10);
        let now = chrono::Utc::now().timestamp();
        let mut guard = self.state.lock().await;

        let query_emb = guard.embedder.embed(&args.query);

        let fts_ids = guard.search.search(&args.query, args.tier, limit * 4);
        let vec_scores = guard.graph.vector_search(&query_emb, limit * 4, 0.1);

        let fused = fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);

        let mut results: Vec<&CognitiveMemoryUnit> = if fused.is_empty() {
            match args.tier {
                Some(tier) => guard.graph.get_by_tier(tier),
                None => guard.graph.get_all_memories(),
            }
            .into_iter()
            .filter(|m| {
                cognimem_server::search::matches_query(&m.content, &args.query.to_lowercase())
            })
            .collect()
        } else {
            fused
                .iter()
                .filter_map(|(id, _)| guard.graph.get_memory(id))
                .collect()
        };

        results.sort_by(|a, b| {
            b.metadata
                .base_activation
                .partial_cmp(&a.metadata.base_activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        let search_results: Vec<SearchResult> = results
            .iter()
            .map(|m| SearchResult {
                id: m.id,
                snippet: m.content.chars().take(80).collect(),
                tier: m.tier,
                activation: m.metadata.base_activation,
            })
            .collect();

        let ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
        update_activation_for(&mut guard, &ids, now);

        inc_recall();
        Ok(success_json(&SearchResults::new(search_results)))
    }

    async fn handle_list_memories(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ListMemoriesArgs = parse_args(args)?;
        let limit = args.limit.unwrap_or(100);
        let min_activation = args.min_activation.unwrap_or(0.0);
        let scope_filter = args.scope_filter.unwrap_or_else(|| "both".to_string());

        let guard = self.state.lock().await;
        let mut memories: Vec<MemorySummary> = match args.tier {
            Some(tier) => guard.graph.get_by_tier(tier),
            None => guard.graph.get_all_memories(),
        }
        .into_iter()
        .filter(|memory| {
            memory.metadata.base_activation >= min_activation
                && memory_matches_scope(memory, args.project_path.as_deref(), &scope_filter)
        })
        .map(MemorySummary::from)
        .collect();

        memories.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        memories.truncate(limit);

        Ok(success_json(&ListMemoriesResult::new(memories)))
    }

    async fn handle_timeline(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: TimelineArgs = parse_args(args)?;
        let window_secs = args.window_secs.unwrap_or(900);

        let guard = self.state.lock().await;
        let memory = guard
            .graph
            .get_memory(&args.memory_id)
            .ok_or_else(|| invalid_params(&format!("Memory not found: {}", args.memory_id)))?;

        let center_time = memory.metadata.last_accessed;
        let center = MemorySummary::from(memory);
        let lower = center_time - window_secs;
        let upper = center_time + window_secs;

        let before: Vec<MemorySummary> = guard
            .graph
            .get_all_memories()
            .iter()
            .filter(|m| {
                m.id != args.memory_id
                    && m.metadata.last_accessed >= lower
                    && m.metadata.last_accessed <= center_time
            })
            .map(|m| MemorySummary::from(*m))
            .collect();

        let after: Vec<MemorySummary> = guard
            .graph
            .get_all_memories()
            .iter()
            .filter(|m| {
                m.id != args.memory_id
                    && m.metadata.last_accessed > center_time
                    && m.metadata.last_accessed <= upper
            })
            .map(|m| MemorySummary::from(*m))
            .collect();

        Ok(success_json(&TimelineResult {
            center,
            before,
            after,
        }))
    }

    async fn handle_get_observations(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: GetObservationsArgs = parse_args(args)?;

        let guard = self.state.lock().await;
        let memory = guard
            .graph
            .get_memory(&args.memory_id)
            .ok_or_else(|| invalid_params(&format!("Memory not found: {}", args.memory_id)))?;

        Ok(success_json(&ObservationsResult {
            memory: memory.clone(),
        }))
    }

    async fn handle_execute_skill(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ExecuteSkillArgs = parse_args(args)?;

        if args.skill_name.is_empty() {
            return Err(invalid_params("skill_name must not be empty"));
        }

        let guard = self.state.lock().await;
        let memory = find_skill(&guard.graph, &args.skill_name)
            .ok_or_else(|| invalid_params(&format!("Skill not found: {}", args.skill_name)))?;

        let skill: SkillMemory = memory
            .content
            .split_once('\n')
            .and_then(|(_, json)| serde_json::from_str(json).ok())
            .ok_or_else(|| invalid_params("Skill memory is malformed"))?;

        let exit_code = run_skill(&skill)
            .map_err(|e| invalid_params(&format!("Skill execution failed: {e}")))?;

        let now = chrono::Utc::now().timestamp();
        if exit_code == 0 {
            skill.record_success(now);
        } else {
            skill.record_failure(now);
        }

        let adjusted = skill.adjust_for_accuracy();

        if adjusted || skill.execution_count > 0 {
            let new_content = format!("{}\n{}", skill.name, serde_json::to_string(&skill).unwrap());
            if let Some(mem) = guard.graph.get_memory_mut(&memory.id) {
                mem.content = new_content;
                mem.metadata.last_accessed = now;
            }
        }

        Ok(success_json(&ExecuteSkillResult {
            skill_id: memory.id,
            skill_name: skill.name,
            pattern: skill.pattern,
            steps: skill.steps,
            source_count: skill.source_ids.len(),
            executed: true,
            exit_code,
        }))
    }

    async fn handle_complete_pattern(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: CompletePatternArgs = parse_args(args)?;

        if args.cue.is_empty() {
            return Err(invalid_params("cue must not be empty"));
        }

        let tolerance = args.tolerance.unwrap_or(0.3);
        let limit = args.limit.unwrap_or(5);

        let guard = self.state.lock().await;
        let mut candidates = complete_pattern(
            &guard.graph,
            guard.embedder.as_ref(),
            &args.cue,
            tolerance,
            limit,
        );

        for candidate in &mut candidates {
            let associated_contents: Vec<&str> = candidate
                .associations
                .iter()
                .filter_map(|a| guard.graph.get_memory(&a.id).map(|m| m.content.as_str()))
                .collect();
            if let Ok(output) = guard
                .slm
                .complete_pattern(CompletePatternInput {
                    cue: candidate.memory.content.clone(),
                    context: associated_contents
                        .iter()
                        .map(|content| (*content).to_string())
                        .collect(),
                })
                .await
            {
                candidate.memory.content = output.completed_text;
            }
        }

        Ok(success_json(&CompletePatternResult { candidates }))
    }

    async fn handle_extract_persona(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let _ = parse_args::<serde_json::Map<String, serde_json::Value>>(args)?;
        let mut guard = self.state.lock().await;

        let semantic_memories: Vec<ExtractPersonaMemoryInput> = guard
            .graph
            .get_by_tier(MemoryTier::Semantic)
            .iter()
            .filter(|m| m.persona.is_none())
            .map(|m| ExtractPersonaMemoryInput {
                id: m.id,
                content: m.content.clone(),
            })
            .collect();

        let profiles = if semantic_memories.len() >= 3 {
            match guard
                .slm
                .extract_persona(ExtractPersonaInput {
                    memories: semantic_memories,
                })
                .await
            {
                Ok(output) => output.profiles,
                Err(e) => {
                    tracing::warn!("Falling back to heuristic persona extraction: {e}");
                    extract_persona(&guard.graph)
                }
            }
        } else {
            extract_persona(&guard.graph)
        };

        persist_persona_profiles(&mut guard, &profiles);

        Ok(success_json(&ExtractPersonaResult { profiles }))
    }

    async fn handle_assign_role(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: AssignRoleArgs = parse_args(args)?;
        let mut guard = self.state.lock().await;

        let memory = guard
            .graph
            .get_memory_mut(&args.memory_id)
            .ok_or_else(|| invalid_params(&format!("Memory not found: {}", args.memory_id)))?;

        if let Some(r) = &args.responsible {
            memory.raci.responsible = Some(r.clone());
        }
        if let Some(a) = &args.accountable {
            memory.raci.accountable = Some(a.clone());
        }
        if let Some(c) = &args.consulted {
            memory.raci.consulted = c.clone();
        }
        if let Some(i) = &args.informed {
            memory.raci.informed = i.clone();
        }

        let raci = memory.raci.clone();

        if let Some(mem) = guard.graph.get_memory(&args.memory_id)
            && let Err(e) = guard.storage.save(mem)
        {
            error!("Failed to persist RACI update for {}: {e}", args.memory_id);
        }

        Ok(success_json(&AssignRoleResult {
            memory_id: args.memory_id,
            raci,
            message: "RACI roles updated successfully".to_string(),
        }))
    }

    async fn handle_claim_work(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ClaimWorkArgs = parse_args(args)?;
        let hours = args.hours.unwrap_or(24);

        let mut guard = self.state.lock().await;

        let session_id = guard
            .session_context
            .as_ref()
            .map(|s| s.session_id)
            .unwrap_or_else(uuid::Uuid::new_v4);

        if let Some(existing) = guard.work_claims.get(&args.memory_id)
            && existing.status == ClaimStatus::Active
            && !existing.is_expired()
        {
            return Err(invalid_params(&format!(
                "Memory {} is already claimed by session {}",
                args.memory_id, existing.session_id
            )));
        }

        let claim = WorkClaim::new(args.memory_id, session_id, args.claim_type, hours);
        let claim_clone = claim.clone();
        guard.work_claims.insert(args.memory_id, claim);

        Ok(success_json(&serde_json::json!({
            "memory_id": args.memory_id,
            "session_id": session_id,
            "claim_type": claim_clone.claim_type.to_string(),
            "leased_until": claim_clone.leased_until,
            "status": claim_clone.status.to_string(),
            "message": format!("Claimed memory {} for {} hours", args.memory_id, hours)
        })))
    }

    async fn handle_release_work(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ReleaseWorkArgs = parse_args(args)?;

        let mut guard = self.state.lock().await;

        let claim = guard
            .work_claims
            .get_mut(&args.memory_id)
            .ok_or_else(|| invalid_params(&format!("No claim found for memory {}", args.memory_id)))?;

        let complete = args.complete.unwrap_or(false);
        if complete {
            claim.complete();
        } else {
            claim.release();
        }

        let status = claim.status.to_string();
        let message = if complete {
            format!("Marked memory {} as completed", args.memory_id)
        } else {
            format!("Released claim on memory {}", args.memory_id)
        };

        Ok(success_json(&serde_json::json!({
            "memory_id": args.memory_id,
            "status": status,
            "message": message
        })))
    }

    async fn handle_find_unclaimed_work(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: FindUnclaimedWorkArgs = parse_args(args)?;

        let guard = self.state.lock().await;

        let mut available: Vec<serde_json::Value> = Vec::new();

        for (id, claim) in &guard.work_claims {
            if claim.status == ClaimStatus::Active
                && claim.is_expired()
                && let Some(mem) = guard.graph.get_memory(id)
            {
                if let Some(ref pp) = args.project_path {
                    if let Some(ref mem_pp) = mem.scope.project_path() {
                        if mem_pp != pp {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                available.push(serde_json::json!({
                    "memory_id": mem.id,
                    "content": mem.content.chars().take(200).collect::<String>(),
                    "tier": mem.tier.to_string(),
                    "claim_type": claim.claim_type.to_string(),
                    "expired_at": claim.leased_until
                }));
            }
        }

        available.truncate(args.limit.unwrap_or(10));

        Ok(success_json(&serde_json::json!({
            "available": available,
            "count": available.len()
        })))
    }

    async fn handle_summarize_turn(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        use cognimem_server::memory::slm_types::SummarizeTurnInput;

        let input: SummarizeTurnInput = parse_args(args)?;
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .summarize_turn(input)
            .await
            .map_err(|e| slm_failed("summarize_turn", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_summarize_session(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        use cognimem_server::memory::slm_types::SummarizeSessionInput;

        let input: SummarizeSessionInput = parse_args(args)?;
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .summarize_session(input)
            .await
            .map_err(|e| slm_failed("summarize_session", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_extract_best_practice(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ExtractBestPracticeArgs = parse_args(args)?;

        let input = cognimem_server::memory::slm_types::ExtractBestPracticeInput {
            content: args.content,
            context: args.context,
        };
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .extract_best_practice(input)
            .await
            .map_err(|e| slm_failed("extract_best_practice", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_get_project_conventions(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: GetProjectConventionsArgs = parse_args(args)?;

        let guard = self.state.lock().await;
        let conventions = guard.project_models.suggest_conventions(&args.project_path);

        Ok(success_json(&serde_json::json!({
            "project_path": args.project_path,
            "conventions": conventions,
            "count": conventions.len()
        })))
    }

    async fn handle_delegate_to_llm(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: DelegateToLlmArgs = parse_args(args)?;

        let input = cognimem_server::memory::slm_types::DelegateInput {
            query: args.query,
            context: args.context.unwrap_or_default(),
            confidence_threshold: args.confidence_threshold.unwrap_or(0.85),
        };
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .delegate_to_llm(input)
            .await
            .map_err(|e| slm_failed("delegate_to_llm", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_teach_from_demonstration(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: TeachFromDemonstrationArgs = parse_args(args)?;

        let input = cognimem_server::memory::slm_types::TeachFromDemonstrationInput {
            demonstration: args.demonstration,
            pattern_extracted: args.pattern_extracted.unwrap_or_default(),
            domain: args.domain,
            source_type: args.source_type,
        };
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .teach_from_demonstration(input)
            .await
            .map_err(|e| slm_failed("teach_from_demonstration", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_inject_memory(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: InjectMemoryArgs = parse_args(args)?;

        if args.query.trim().is_empty() {
            return Err(invalid_params("query must not be empty"));
        }

        let query = args.query;

        let mut guard = self.state.lock().await;

        if guard.injection.injected_count() >= 1 {
            return Ok(success_json(&serde_json::json!({
                "injected": false,
                "reason": "max one injection per session already used",
                "memory": null
            })));
        }

        if guard.injection.has_recent_query(&query) {
            return Ok(success_json(&serde_json::json!({
                "injected": false,
                "reason": "duplicate query, already checked",
                "memory": null
            })));
        }

        let candidates = guard.injection.gather_candidates(&guard.graph);
        if candidates.is_empty() {
            guard.injection.record_query(query);
            return Ok(success_json(&serde_json::json!({
                "injected": false,
                "reason": "no candidates available",
                "memory": null
            })));
        }

        let best = guard
            .injection
            .find_best_candidate(&query, &candidates, guard.slm.as_ref())
            .await
            .map_err(|e| slm_failed("inject_memory", guard.slm.model_name(), e))?;

        match best {
            Some(memory) => {
                let memory_id = memory.id;
                guard.injection.record_injection(memory_id);
                guard.injection.record_query(query);

                guard.broker.publish_event(&BrokerEvent::MemoryUpdated {
                    memory_id,
                    action: "injected".to_string(),
                    agent_id: "cognimem".to_string(),
                });

                Ok(success_json(&serde_json::json!({
                    "injected": true,
                    "reason": null,
                    "memory": {
                        "id": memory.id,
                        "content": memory.content,
                        "tier": memory.tier.to_string(),
                        "importance": memory.metadata.importance,
                        "activation": memory.metadata.base_activation
                    }
                })))
            }
            None => {
                guard.injection.record_query(query);
                Ok(success_json(&serde_json::json!({
                    "injected": false,
                    "reason": "no memory scored above relevance threshold",
                    "memory": null
                })))
            }
        }
    }

    async fn handle_handoff_summary(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: HandoffSummaryArgs = parse_args(args)?;

        if args.summary.trim().is_empty() {
            return Err(invalid_params("summary must not be empty"));
        }

        let handoff = cognimem_server::memory::HandoffSummary::new(
            args.from_session,
            args.project_path,
            args.summary,
            args.unresolved.unwrap_or_default(),
            args.next_steps.unwrap_or_default(),
            args.relevant_memories.unwrap_or_default(),
        );

        let handoff_id = handoff.from_session;
        let mut guard = self.state.lock().await;
        guard.handoffs.push(handoff);

        guard.broker.publish_event(&BrokerEvent::SessionLeft {
            session_id: handoff_id,
            agent_id: "cognimem".to_string(),
        });

        Ok(success_json(&serde_json::json!({
            "handoff_id": handoff_id,
            "created_at": chrono::Utc::now().timestamp(),
            "message": "Handoff summary stored successfully"
        })))
    }

    async fn handle_simulate_perspective(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: SimulatePerspectiveArgs = parse_args(args)?;

        let input = SimulatePerspectiveInput {
            perspective_role: args.perspective_role,
            situation: args.situation,
            question: args.question,
        };
        let guard = self.state.lock().await;
        let output = guard
            .slm
            .simulate_perspective(input)
            .await
            .map_err(|e| slm_failed("simulate_perspective", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_imagine(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ImagineArgs = parse_args(args)?;

        let guard = self.state.lock().await;
        let query_emb = guard.embedder.embed(&args.scenario);
        let similar = guard.graph.vector_search(&query_emb, 5, 0.2);
        let memory_context: Vec<String> = similar
            .iter()
            .filter_map(|(id, _)| guard.graph.get_memory(id))
            .map(|m| format!("[{}] {}", m.tier, m.content.chars().take(200).collect::<String>()))
            .collect();

        let mut context = args.context.unwrap_or_default();
        context.extend(memory_context);

        let input = ImagineInput {
            scenario: args.scenario,
            context,
        };
        let output = guard
            .slm
            .imagine(input)
            .await
            .map_err(|e| slm_failed("imagine", guard.slm.model_name(), e))?;

        Ok(success_json(&output))
    }

    async fn handle_discover_project(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: DiscoverProjectArgs = parse_args(args)?;
        let project_path = args
            .project_path
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let mut guard = self.state.lock().await;
        let count = cognimem_server::memory::discover_project(&project_path, &mut guard.code_graph)
            .map_err(|e| rmcp::ErrorData::new(rmcp::model::ErrorCode(-32000), Cow::Owned(e), None))?;

        let files = guard.code_graph.all_files().len();
        let result = DiscoverProjectResult {
            nodes_discovered: count,
            files_scanned: files,
        };
        Ok(success_json(&result))
    }

    async fn handle_explore_module(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: ExploreModuleArgs = parse_args(args)?;
        let guard = self.state.lock().await;

        let children = if let Some(node_id_str) = &args.node_id {
            let node_id = uuid::Uuid::parse_str(node_id_str).map_err(|e| {
                rmcp::ErrorData::new(
                    rmcp::model::ErrorCode(-32602),
                    Cow::Owned(format!("Invalid node_id: {e}")),
                    None,
                )
            })?;
            guard
                .code_graph
                .get_children(&node_id)
                .iter()
                .take(args.limit.unwrap_or(20))
                .map(|n| NodeSummaryResult {
                    id: n.id,
                    kind: n.kind.to_string(),
                    name: n.name.clone(),
                    file_path: n.file_path.clone(),
                    line_start: n.line_start,
                    line_end: n.line_end,
                    summary: n.summary.clone(),
                    children: n.children.iter().map(|c| c.to_string()).collect(),
                })
                .collect()
        } else if let Some(file_path) = &args.file_path {
            let path = std::path::PathBuf::from(file_path);
            guard
                .code_graph
                .get_nodes_in_file(&path)
                .iter()
                .take(args.limit.unwrap_or(20))
                .map(|n| NodeSummaryResult {
                    id: n.id,
                    kind: n.kind.to_string(),
                    name: n.name.clone(),
                    file_path: n.file_path.clone(),
                    line_start: n.line_start,
                    line_end: n.line_end,
                    summary: n.summary.clone(),
                    children: n.children.iter().map(|c| c.to_string()).collect(),
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(success_json(&children))
    }

    async fn handle_get_node_summary(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: serde_json::Map<String, serde_json::Value> = args;
        let node_id_str = args
            .get("node_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                rmcp::ErrorData::new(
                    rmcp::model::ErrorCode(-32602),
                    Cow::Borrowed("node_id is required"),
                    None,
                )
            })?;

        let node_id = uuid::Uuid::parse_str(node_id_str).map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32602),
                Cow::Owned(format!("Invalid node_id: {e}")),
                None,
            )
        })?;

        let guard = self.state.lock().await;
        let node = guard.code_graph.get_node(&node_id).ok_or_else(|| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32602),
                Cow::Owned(format!("Node not found: {node_id}")),
                None,
            )
        })?;

        let result = NodeSummaryResult {
            id: node.id,
            kind: node.kind.to_string(),
            name: node.name.clone(),
            file_path: node.file_path.clone(),
            line_start: node.line_start,
            line_end: node.line_end,
            summary: node.summary.clone(),
            children: node.children.iter().map(|c| c.to_string()).collect(),
        };
        Ok(success_json(&result))
    }

    async fn handle_search_codebase(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: SearchCodebaseArgs = parse_args(args)?;
        let guard = self.state.lock().await;

        let matching: Vec<NodeSummaryResult> = guard
            .code_graph
            .search_by_name(&args.query)
            .iter()
            .filter(|n| {
                if let Some(ref fp) = args.file_path {
                    n.file_path.to_string_lossy().contains(fp)
                } else {
                    true
                }
            })
            .take(args.limit.unwrap_or(20))
            .map(|n| NodeSummaryResult {
                id: n.id,
                kind: n.kind.to_string(),
                name: n.name.clone(),
                file_path: n.file_path.clone(),
                line_start: n.line_start,
                line_end: n.line_end,
                summary: n.summary.clone(),
                children: n.children.iter().map(|c| c.to_string()).collect(),
            })
            .collect();

        let result = SearchCodebaseResult { matches: matching };
        Ok(success_json(&result))
    }
}

fn dedup_memories(results: &mut Vec<&CognitiveMemoryUnit>) {
    let mut seen = std::collections::HashSet::new();
    results.retain(|memory| seen.insert(memory.id));
}

fn clone_with_embedding(id: uuid::Uuid, content: &str, tier: String, created_at: i64, embedding: Vec<f32>) -> CognitiveMemoryUnit {
    let decay_rate = tier.parse::<MemoryTier>().unwrap_or_default().decay_rate();
    let mut mem = CognitiveMemoryUnit::new(content.to_string(), tier.parse().unwrap_or_default(), 0.5, decay_rate);
    mem.id = id;
    mem.metadata.created_at = created_at;
    mem
}

fn update_activation_for(guard: &mut CogniMemState, ids: &[uuid::Uuid], now: i64) {
    for id in ids {
        if let Some(mem) = guard.graph.get_memory_mut(id) {
            mem.metadata.record_rehearsal(now);
        }
        if let Some(mem) = guard.graph.get_memory(id)
            && let Err(e) = guard.storage.save(mem)
        {
            error!("Failed to persist activation update for {id}: {e}");
        }
    }
}

fn persist_persona_profiles(
    guard: &mut CogniMemState,
    profiles: &[cognimem_server::memory::PersonaProfile],
) {
    let persona_ids: std::collections::HashSet<uuid::Uuid> = guard
        .graph
        .get_by_tier(MemoryTier::Semantic)
        .iter()
        .filter(|m| m.persona.is_some())
        .map(|m| m.id)
        .collect();

    for profile in profiles {
        let clean_source_ids: Vec<uuid::Uuid> = profile
            .source_ids
            .iter()
            .filter(|id| !persona_ids.contains(id))
            .cloned()
            .collect();

        let content = format!("[persona:{}] {}", profile.domain, profile.summary);
        let existing_id = guard
            .graph
            .get_by_tier(MemoryTier::Semantic)
            .into_iter()
            .find(|memory| memory.persona == Some(profile.domain))
            .map(|memory| memory.id);

        if let Some(memory_id) = existing_id {
            if let Some(memory) = guard.graph.get_memory_mut(&memory_id) {
                memory.content = content.clone();
                memory.associations = clean_source_ids.clone();
                memory.persona = Some(profile.domain);
                memory.metadata.importance = profile.confidence.clamp(0.0, 1.0);
                memory.model.model_name = Some(
                    memory
                        .model
                        .model_name
                        .clone()
                        .unwrap_or_else(|| "persona_persist".to_string()),
                );
                memory.model.confidence = Some(profile.confidence.clamp(0.0, 1.0));
                memory.model.provenance_ids = clean_source_ids.clone();
            }
            guard.search.remove(&memory_id);
            guard
                .search
                .index(memory_id, &content, MemoryTier::Semantic);
            let embedding = guard.embedder.embed(&content);
            guard.graph.set_embedding(memory_id, embedding);
            if let Some(memory) = guard.graph.get_memory(&memory_id)
                && let Err(e) = guard.storage.save(memory)
            {
                error!("Failed to persist persona memory {}: {e}", memory_id);
            }
            continue;
        }

        let mut memory = CognitiveMemoryUnit::new(
            content.clone(),
            MemoryTier::Semantic,
            profile.confidence.clamp(0.0, 1.0),
            MemoryTier::Semantic.decay_rate(),
        );
        memory.associations = clean_source_ids.clone();
        memory.persona = Some(profile.domain);
        memory.model.model_name = Some("persona_persist".to_string());
        memory.model.confidence = Some(profile.confidence.clamp(0.0, 1.0));
        memory.model.provenance_ids = clean_source_ids.clone();

        let memory_id = memory.id;
        let embedding = guard.embedder.embed(&content);
        guard.graph.add_memory(memory.clone());
        guard.graph.set_embedding(memory_id, embedding);
        guard
            .search
            .index(memory_id, &content, MemoryTier::Semantic);
        if let Err(e) = guard.storage.save(&memory) {
            error!("Failed to persist persona memory {}: {e}", memory_id);
        }
    }
}

fn slm_failed(operation: &str, model: &str, err: SlmError) -> rmcp::ErrorData {
    let hint = match &err {
        SlmError::RequestFailed(msg) => {
            format!("Ollama request failed: {msg}. Is Ollama running? Try: ollama serve")
        }
        SlmError::InvalidResponse(msg) => format!(
            "Ollama returned unparseable output: {msg}. The model may have emitted thinking tokens or malformed JSON. Try a simpler query or check that model '{model}' is pulled."
        ),
        SlmError::ValidationFailed(msg) => format!(
            "Ollama output failed validation: {msg}. The model returned structurally invalid data for operation '{operation}'."
        ),
    };
    rmcp::ErrorData::new(rmcp::model::ErrorCode(-32000), Cow::Owned(hint), None)
}

async fn decay_task(state: Arc<Mutex<CogniMemState>>, interval: Duration, prune_threshold: f32) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        let mut guard = state.lock().await;
        apply_decay_to_all(&mut guard.graph);
        let pruned = prune_below_threshold(&mut guard.graph, prune_threshold);
        if !pruned.is_empty() {
            for id in &pruned {
                guard.search.remove(id);
                if let Err(e) = guard.storage.delete(id) {
                    error!("Failed to delete pruned memory {}: {e}", id);
                }
            }
            tracing::info!(
                "Pruned {} memories below activation threshold",
                pruned.len()
            );
        }
    }
}

async fn consolidation_task(
    state: Arc<Mutex<CogniMemState>>,
    interval: Duration,
    prune_threshold: f32,
) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        let mut guard = state.lock().await;
        let CogniMemState {
            graph,
            storage,
            search,
            embedder,
            slm,
            c3gan,
            ..
        } = &mut *guard;

        let conflicts = consolidate(graph, embedder.as_ref());
        let resolved = resolve_conflicts(
            graph,
            &conflicts,
            &cognimem_server::memory::ConflictResolution::KeepBoth,
        );
        for id in &resolved {
            search.remove(id);
            if let Err(e) = storage.delete(id) {
                error!("Failed to delete consolidated memory {}: {e}", id);
            }
        }

        let memory_count = graph.len();
        let replay_count = c3gan.get_replay_count(memory_count);
        let anchors = c3gan.sample_anchors(replay_count);
        
        if !anchors.is_empty() {
            tracing::debug!("C3GAN replay: {} anchor memories for replay during consolidation", anchors.len());
            for anchor in anchors {
                let emb = embedder.embed(&anchor.content);
                let rep = clone_with_embedding(anchor.id, &anchor.content, anchor.tier.clone(), anchor.created_at, emb.clone());
                graph.add_memory(rep);
            }
        }

        let dreamt = match cognimem_server::memory::dream::create_dream_memory(
            graph,
            embedder.as_ref(),
            slm.as_ref(),
        )
        .await
        {
            Ok(Some((id, insight))) => {
                tracing::info!("Dreamt new memory {id}: {insight}");
                1
            }
            Ok(None) => 0,
            Err(e) => {
                tracing::warn!("Dreaming failed: {e}");
                0
            }
        };

        let promoted = promote_memories(graph);
        let pruned = prune_below_threshold(graph, prune_threshold);
        for id in &pruned {
            search.remove(id);
            if let Err(e) = storage.delete(id) {
                error!("Failed to delete pruned memory {}: {e}", id);
            }
        }

        for memory in graph.get_all_memories() {
            search.index(memory.id, &indexed_content(memory), memory.tier);
            if let Err(e) = storage.save(memory) {
                error!("Failed to persist consolidated memory {}: {e}", memory.id);
            }
        }

        set_memory_count(graph.len() as u64);
        if !conflicts.is_empty() || dreamt > 0 || promoted > 0 || !pruned.is_empty() {
            tracing::info!(
                "Consolidation cycle complete: conflicts={}, resolved={}, promoted={}, dreamt={}, pruned={}",
                conflicts.len(),
                resolved.len(),
                promoted,
                dreamt,
                pruned.len()
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    if cli.daemon {
        run_daemon(cli).await?;
    } else {
        run_client_bridge(cli).await?;
    }

    Ok(())
}

async fn run_client_bridge(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = socket_path_for(&cli);
    let lock_path = bootstrap_lock_path(&socket_path);

    if !try_connect_existing(&socket_path).await?
        && let Some(_lock_guard) = acquire_bootstrap_lock(&lock_path)?
        && !try_connect_existing(&socket_path).await?
    {
        cleanup_stale_socket(&socket_path)?;
        spawn_daemon(&cli, &socket_path)?;
    }

    let socket = wait_for_daemon_socket_or_retry(&cli, &socket_path, &lock_path).await?;
    bridge_stdio(socket).await?;

    Ok(())
}

async fn run_daemon(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    cognimem_server::metrics::init();

    let socket_path = socket_path_for(&cli);
    ensure_parent_dir(Path::new(&cli.data_path))?;
    ensure_parent_dir(&socket_path)?;
    cleanup_stale_socket(&socket_path)?;

    let _socket_guard = SocketFileGuard(socket_path.clone());
    let listener = UnixListener::bind(&socket_path)?;

    tokio::spawn(start_metrics_server(cli.metrics_port));

    let storage: Box<dyn MemoryStore> = match cli.storage.as_str() {
        "memory" => Box::new(InMemoryStore::new()),
        _ => Box::new(RocksDbStore::open(Path::new(&cli.data_path))?),
    };
    let state = Arc::new(Mutex::new(CogniMemState::new(
        storage,
        cli.ollama_model.clone(),
        cli.ollama_url.clone(),
        cli.redis_url.clone(),
        cli.agent_id.clone(),
    )));
    {
        let guard = state.lock().await;
        set_memory_count(guard.graph.len() as u64);
    }
    if let Some(project) = cli.resolved_project_path() {
        if project.exists() {
            let mut guard = state.lock().await;
            let path_str = project.to_string_lossy();
            match cognimem_server::memory::discover_project(&project, &mut guard.code_graph) {
                Ok(count) => {
                    tracing::info!("Discovered {} code nodes from {}", count, path_str);
                    set_code_node_count(guard.code_graph.len() as u64);
                }
                Err(e) => tracing::warn!("Failed to discover project: {e}"),
            }
        }
    } else {
        tracing::info!("No project path specified, skipping code discovery");
    }
    let server = CogniMemServer::new(state.clone());

    let capture_pipeline = Arc::new(Mutex::new(CapturePipeline::new(state.clone())));
    tokio::spawn(start_capture_server(capture_pipeline, cli.capture_port));
    tokio::spawn(start_dashboard_server(state.clone(), cli.dashboard_port));
    tracing::info!("Dashboard starting on http://localhost:{}", cli.dashboard_port);

    if cli.resolved_project_path().is_some() {
        tokio::spawn(start_file_watcher(
            state.clone(),
            cli.resolved_project_path(),
            2000,
        ));
    }

    tokio::spawn(decay_task(
        state.clone(),
        Duration::from_secs(cli.decay_interval_secs),
        cli.prune_threshold,
    ));
    tokio::spawn(consolidation_task(
        state,
        Duration::from_secs(cli.consolidation_interval_secs),
        cli.prune_threshold,
    ));

    loop {
        let (stream, _) = listener.accept().await?;
        let server = server.clone();
        tokio::spawn(async move {
            match server.serve(stream).await {
                Ok(running_service) => {
                    let _ = running_service.waiting().await;
                }
                Err(err) => {
                    error!("Failed to serve MCP client: {err}");
                }
            }
        });
    }
}

fn socket_path_for(cli: &Cli) -> PathBuf {
    if let Some(path) = &cli.socket_path {
        return PathBuf::from(path);
    }

    let data_path = Path::new(&cli.data_path);
    if let Some(parent) = data_path.parent() {
        return parent.join("cognimem.sock");
    }

    PathBuf::from("./cognimem.sock")
}

fn bootstrap_lock_path(socket_path: &Path) -> PathBuf {
    let mut path = socket_path.as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

fn ensure_parent_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn cleanup_stale_socket(socket_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }
    Ok(())
}

fn acquire_bootstrap_lock(
    lock_path: &Path,
) -> Result<Option<BootstrapLockGuard>, Box<dyn std::error::Error>> {
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path)
    {
        Ok(_) => Ok(Some(BootstrapLockGuard(lock_path.to_path_buf()))),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
        Err(err) => Err(Box::new(err)),
    }
}

async fn try_connect_existing(socket_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    match UnixStream::connect(socket_path).await {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

fn spawn_daemon(cli: &Cli, socket_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let current_exe = std::env::current_exe()?;
    let mut command = std::process::Command::new(current_exe);
    command
        .arg("--daemon")
        .arg("--data-path")
        .arg(&cli.data_path)
        .arg("--decay-interval-secs")
        .arg(cli.decay_interval_secs.to_string())
        .arg("--consolidation-interval-secs")
        .arg(cli.consolidation_interval_secs.to_string())
        .arg("--prune-threshold")
        .arg(cli.prune_threshold.to_string())
        .arg("--storage")
        .arg(&cli.storage)
        .arg("--metrics-port")
        .arg(cli.metrics_port.to_string())
        .arg("--socket-path")
        .arg(socket_path)
        .arg("--capture-port")
        .arg(cli.capture_port.to_string())
        .arg("--dashboard-port")
        .arg(cli.dashboard_port.to_string());
    if !cli.project_path.is_empty() {
        command.arg("--project-path").arg(&cli.project_path);
    }
    if let Some(ref model) = cli.ollama_model {
        command.arg("--ollama-model").arg(model);
    }
    if let Some(ref url) = cli.ollama_url {
        command.arg("--ollama-url").arg(url);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn()?;
    Ok(())
}

async fn wait_for_daemon_socket(
    socket_path: &Path,
) -> Result<UnixStream, Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(5);

    loop {
        match UnixStream::connect(socket_path).await {
            Ok(stream) => return Ok(stream),
            Err(err) if start.elapsed() < timeout => {
                let _ = err;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
}

async fn wait_for_daemon_socket_or_retry(
    cli: &Cli,
    socket_path: &Path,
    lock_path: &Path,
) -> Result<UnixStream, Box<dyn std::error::Error>> {
    match wait_for_daemon_socket(socket_path).await {
        Ok(stream) => Ok(stream),
        Err(_) => {
            let _ = std::fs::remove_file(lock_path);
            if let Some(_lock_guard) = acquire_bootstrap_lock(lock_path)?
                && !try_connect_existing(socket_path).await?
            {
                cleanup_stale_socket(socket_path)?;
                spawn_daemon(cli, socket_path)?;
            }
            wait_for_daemon_socket(socket_path).await
        }
    }
}

async fn bridge_stdio(socket: UnixStream) -> Result<(), Box<dyn std::error::Error>> {
    let (socket_read, mut socket_write) = io::split(socket);
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    let stdin_to_socket = async {
        io::copy(&mut stdin, &mut socket_write).await?;
        socket_write.shutdown().await
    };

    let socket_to_stdout = async {
        let mut reader = socket_read;
        io::copy(&mut reader, &mut stdout).await?;
        stdout.flush().await
    };

    tokio::try_join!(stdin_to_socket, socket_to_stdout)?;
    Ok(())
}

async fn start_metrics_server(metrics_port: u16) {
    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{metrics_port}")).await {
        Ok(listener) => listener,
        Err(err) => {
            error!("Failed to bind metrics port {metrics_port}: {err}");
            return;
        }
    };

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(async move {
                    if let Err(e) = handle_metrics(stream).await {
                        tracing::warn!("Metrics connection handler error: {e}");
                    }
                });
            }
            Err(err) => {
                error!("Failed to accept metrics connection: {err}");
            }
        }
    }
}

async fn handle_metrics(mut stream: tokio::net::TcpStream) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = [0u8; 1024];
    let _n = stream.read(&mut buf).await?;

    let response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n".to_string()
        + &cognimem_server::metrics::encode();

    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

struct SocketFileGuard(PathBuf);

impl Drop for SocketFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct BootstrapLockGuard(PathBuf);

impl Drop for BootstrapLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognimem_server::memory::pattern::CompletePatternResult;
    use cognimem_server::memory::{ExtractPersonaResult, InMemoryStore, PersonaDomain};
    use serde_json::{Map, Value, json};

    fn make_server() -> CogniMemServer {
        let state = Arc::new(Mutex::new(CogniMemState::new(
            Box::new(InMemoryStore::new()),
            None,
            None,
            None,
            None,
        )));
        CogniMemServer::new(state)
    }

    fn object_args(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => unreachable!("expected JSON object args"),
        }
    }

    #[tokio::test]
    async fn handle_extract_persona_falls_back_to_heuristics() {
        let server = make_server();
        {
            let mut guard = server.state.lock().await;
            for content in [
                "I deployed the rust service to production",
                "I fixed a critical auth bug for the backend",
                "I work on API reliability and observability",
            ] {
                let memory = CognitiveMemoryUnit::new(
                    content.to_string(),
                    MemoryTier::Semantic,
                    0.8,
                    MemoryTier::Semantic.decay_rate(),
                );
                let id = guard.graph.add_memory(memory);
                let embedding = guard.embedder.embed(content);
                guard.graph.set_embedding(id, embedding);
            }
        }

        let result = server.handle_extract_persona(Map::new()).await.unwrap();
        let typed: ExtractPersonaResult = result.into_typed().unwrap();

        assert!(!typed.profiles.is_empty());
        assert!(typed.profiles.iter().any(|profile| profile.domain == PersonaDomain::Work));
    }

    #[tokio::test]
    async fn handle_complete_pattern_returns_graph_candidates_when_slm_is_unavailable() {
        let server = make_server();
        {
            let mut guard = server.state.lock().await;
            let memory = CognitiveMemoryUnit::new(
                "use Result for recoverable errors".to_string(),
                MemoryTier::Semantic,
                0.9,
                MemoryTier::Semantic.decay_rate(),
            );
            let id = guard.graph.add_memory(memory);
            let embedding = guard.embedder.embed("use Result for recoverable errors");
            guard.graph.set_embedding(id, embedding);
        }

        let result = server
            .handle_complete_pattern(object_args(json!({
                "cue": "use Result",
                "tolerance": 0.1,
                "limit": 3
            })))
            .await
            .unwrap();
        let typed: CompletePatternResult = result.into_typed().unwrap();

        assert!(!typed.candidates.is_empty());
        assert!(typed.candidates[0].memory.content.contains("Result"));
    }

    #[tokio::test]
    async fn handle_summarize_turn_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_summarize_turn(object_args(json!({
                "turns": [{
                    "turn_id": "00000000-0000-0000-0000-000000000001",
                    "content": "implemented feature X",
                    "tool_usage": ["grep"],
                    "decisions": ["use enum"]
                }]
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }

    #[tokio::test]
    async fn handle_summarize_session_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_summarize_session(object_args(json!({
                "turns": [{
                    "turn_id": "00000000-0000-0000-0000-000000000001",
                    "content": "worked on API"
                }],
                "completed_tasks": [{"title": "task a", "status": "completed"}],
                "open_tasks": [{"title": "task b", "status": "open"}]
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }

    #[tokio::test]
    async fn handle_extract_best_practice_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_extract_best_practice(object_args(json!({
                "content": "Use guard clauses and keep functions small"
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }

    #[tokio::test]
    async fn handle_delegate_to_llm_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_delegate_to_llm(object_args(json!({
                "query": "How should I structure this auth flow?",
                "confidence_threshold": 0.9,
                "context": ["rust", "auth"]
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }

    #[tokio::test]
    async fn handle_teach_from_demonstration_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_teach_from_demonstration(object_args(json!({
                "demonstration": "Prefer Result<T, E> for recoverable failures",
                "pattern_extracted": "error handling",
                "source_type": "code_review"
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }

    #[tokio::test]
    async fn handle_simulate_perspective_returns_slm_error_when_provider_is_missing() {
        let server = make_server();

        let err = server
            .handle_simulate_perspective(object_args(json!({
                "perspective_role": "security_expert",
                "situation": "user login flow",
                "question": "What is the biggest risk?"
            })))
            .await
            .unwrap_err();

        assert!(err.message.contains("Ollama request failed"));
    }
}
