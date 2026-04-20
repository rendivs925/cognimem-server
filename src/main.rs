mod config;

use clap::Parser;
use cognimem_server::embeddings::{EmbeddingEngine, HashEmbedding, fuse_scores};
use cognimem_server::memory::slm::RerankCandidate;
use cognimem_server::memory::{
    AssignRoleArgs, AssignRoleResult, AssociateArgs, AssociateResult, CognitiveMemoryUnit,
    CompletePatternArgs, CompletePatternResult, ExecuteSkillArgs, ExecuteSkillResult,
    ExtractPersonaResult, ForgetArgs, ForgetResult, GetObservationsArgs, InMemoryStore,
    MemoryStore, MemorySummary, MemoryTier, NoOpSlm, OllamaSlm, ObservationsResult, RecallArgs,
    RecallResult, ReflectArgs, ReflectResult, RememberArgs, RememberResult, RocksDbStore,
    SearchArgs, SearchResult, SearchResults, SkillMemory, SlmEngine, TimelineArgs, TimelineResult,
};
use cognimem_server::memory::{
    MemoryGraph, apply_decay_to_all, consolidate, detect_conflicts, promote_memories,
    prune_below_threshold, resolve_conflicts,
};
use cognimem_server::memory::{
    complete_pattern, detect_and_create_skill, extract_persona, find_skill, strengthen_co_activated,
};
use cognimem_server::metrics::{
    inc_associate, inc_forget, inc_prune, inc_recall, inc_reflect, inc_remember, set_memory_count,
};
use cognimem_server::search::{Fts5Search, SearchEngine};
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

struct CogniMemState {
    graph: MemoryGraph,
    storage: Box<dyn MemoryStore>,
    search: Box<dyn SearchEngine + Send>,
    embedder: Box<dyn EmbeddingEngine + Send>,
    slm: Box<dyn SlmEngine + Send>,
}

impl CogniMemState {
    fn new(storage: Box<dyn MemoryStore>, ollama_model: Option<String>, ollama_url: Option<String>) -> Self {
        let mut graph = MemoryGraph::new();
        let mut search: Box<dyn SearchEngine + Send> = match Fts5Search::new() {
            Ok(fts) => Box::new(fts),
            Err(e) => {
                tracing::warn!("Failed to initialize FTS5 search, falling back to substring: {e}");
                Box::new(cognimem_server::search::SubstringSearch)
            }
        };
        let embedder: Box<dyn EmbeddingEngine + Send> = Box::new(HashEmbedding::new());

        let slm: Box<dyn SlmEngine + Send> = if let Some(model) = ollama_model {
            let ollama = OllamaSlm::new(Some(model), ollama_url);
            if ollama.check_available() {
                tracing::info!("Ollama SLM engine initialized successfully");
                Box::new(ollama)
            } else {
                tracing::warn!("Ollama not available, falling back to NoOpSlm");
                Box::new(NoOpSlm)
            }
        } else {
            Box::new(NoOpSlm)
        };
        let memories = match storage.load_all() {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to load memories from storage: {e}");
                Vec::new()
            }
        };
        for m in &memories {
            let emb = embedder.embed(&m.content);
            graph.add_memory(m.clone());
            graph.set_embedding(m.id, emb);
            search.index(m.id, &m.content, m.tier);
        }
        Self {
            graph,
            storage,
            search,
            embedder,
            slm,
        }
    }
}

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
    serde_json::from_value(serde_json::Value::Object(args)).map_err(|e| {
        rmcp::ErrorData::new(
            rmcp::model::ErrorCode(-32602),
            Cow::Owned(e.to_string()),
            None,
        )
    })
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
            "timeline" => self.handle_timeline(args).await,
            "get_observations" => self.handle_get_observations(args).await,
            "execute_skill" => self.handle_execute_skill(args).await,
            "complete_pattern" => self.handle_complete_pattern(args).await,
            "extract_persona" => self.handle_extract_persona(args).await,
            "assign_role" => self.handle_assign_role(args).await,
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
        let resources: Vec<Resource> = guard
            .graph
            .get_all_memories()
            .iter()
            .map(|m| {
                Resource::new(
                    RawResource::new(format!("memory://{}/{}", m.tier, m.id), &m.content)
                        .with_description(format!(
                            "Memory in {} tier with activation {:.3}",
                            m.tier, m.metadata.base_activation
                        ))
                        .with_mime_type("application/json"),
                    None,
                )
            })
            .collect();
        Ok(ListResourcesResult::with_all_items(resources))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, rmcp::ErrorData> {
        let _ = context;
        let uri = &request.uri;
        if !uri.starts_with("memory://") {
            return Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32602),
                Cow::Owned(format!("Invalid resource URI: {uri}")),
                None,
            ));
        }

        let path = &uri["memory://".len()..];
        let id_str = path.split('/').next_back().unwrap_or("");
        let id = uuid::Uuid::parse_str(id_str).map_err(|e| {
            rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32602),
                Cow::Owned(format!("Invalid memory ID: {e}")),
                None,
            )
        })?;

        let guard = self.state.lock().await;
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
        Ok(ReadResourceResult::new(vec![ResourceContents::text(
            json,
            uri.clone(),
        )]))
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

        let tier = args.tier.unwrap_or_default();
        let importance = args.importance.unwrap_or(0.5);
        let decay_rate = tier.decay_rate();

        let mut memory = CognitiveMemoryUnit::new(args.content, tier, importance, decay_rate);
        if let Some(assoc_ids) = args.associations {
            memory.associations = assoc_ids;
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
        let compressed = guard.slm.compress(&memory.content);
        guard.search.index(
            memory_id,
            &format!("{} {}", memory.content, compressed),
            memory.tier,
        );
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
            } = &mut *guard;
            if let Some(skill_memory) =
                detect_and_create_skill(graph, embedder.as_ref(), &mut **search, &memory.content)
            {
                let skill_compressed = slm.compress(&skill_memory.content);
                search.index(skill_memory.id, &skill_compressed, skill_memory.tier);
                if let Err(e) = storage.save(&skill_memory) {
                    error!("Failed to persist skill memory {}: {e}", skill_memory.id);
                }
                set_memory_count(graph.len() as u64);
            }
        }

        inc_remember();
        set_memory_count(guard.graph.len() as u64);

        Ok(success_json(&RememberResult::success(memory_id)))
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

        let mut guard = self.state.lock().await;

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

        let mut results: Vec<&CognitiveMemoryUnit> = fused
            .iter()
            .filter_map(|(id, _)| guard.graph.get_memory(id))
            .filter(|m| m.metadata.base_activation >= min_activation)
            .collect();

        let direct_ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
        expand_with_associations(&direct_ids, &mut results, &guard.graph);

        results.sort_by(|a, b| {
            b.metadata
                .base_activation
                .partial_cmp(&a.metadata.base_activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let slm_results: Vec<usize> = {
            let candidates: Vec<RerankCandidate> = results
                .iter()
                .map(|m| RerankCandidate {
                    id: m.id,
                    content: m.content.clone(),
                    score: m.metadata.base_activation,
                })
                .collect();
            guard.slm.rerank(&candidates, &query, limit)
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
        let strategy = args
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
        } = &mut *guard;
        let total = graph.len();

        if intensity == "full" {
            apply_decay_to_all(graph);

            let pruned_ids = prune_below_threshold(graph, 0.01);
            for id in &pruned_ids {
                search.remove(id);
            }

            let conflicts = consolidate(graph, embedder.as_ref());
            let strategy = if args.conflict_strategy.is_some() {
                strategy
            } else {
                let resolved = conflicts.first().map(|c| {
                    let a = graph
                        .get_memory(&c.memory_id_1)
                        .map(|m| m.content.as_str())
                        .unwrap_or("");
                    let b = graph
                        .get_memory(&c.memory_id_2)
                        .map(|m| m.content.as_str())
                        .unwrap_or("");
                    slm.resolve_conflict(a, b)
                });
                resolved.unwrap_or(strategy)
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

        Ok(success_json(&ExecuteSkillResult {
            skill_id: memory.id,
            skill_name: skill.name,
            pattern: skill.pattern,
            steps: skill.steps,
            source_count: skill.source_ids.len(),
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
            candidate.memory.content = guard
                .slm
                .complete_pattern_hint(&candidate.memory.content, &associated_contents);
        }

        Ok(success_json(&CompletePatternResult { candidates }))
    }

    async fn handle_extract_persona(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let _ = parse_args::<serde_json::Map<String, serde_json::Value>>(args)?;
        let guard = self.state.lock().await;
        let profiles = extract_persona(&guard.graph);
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
}

fn expand_with_associations<'a>(
    direct_ids: &[uuid::Uuid],
    results: &mut Vec<&'a CognitiveMemoryUnit>,
    graph: &'a MemoryGraph,
) {
    let expanded = graph.spreading_activation(direct_ids, 3, 0.5, 0.1);
    for (id, _, _) in &expanded {
        if let Some(mem) = graph.get_memory(id)
            && !direct_ids.contains(id)
        {
            results.push(mem);
        }
    }
}

fn update_activation_for(guard: &mut CogniMemState, ids: &[uuid::Uuid], now: i64) {
    for id in ids {
        if let Some(mem) = guard.graph.get_memory_mut(id) {
            mem.metadata.access_count += 1;
            mem.metadata.last_accessed = now;
            mem.metadata.update_activation(now);
        }
        if let Some(mem) = guard.graph.get_memory(id)
            && let Err(e) = guard.storage.save(mem)
        {
            error!("Failed to persist activation update for {id}: {e}");
        }
    }
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
            }
            tracing::info!(
                "Pruned {} memories below activation threshold",
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
    let state = Arc::new(Mutex::new(CogniMemState::new(storage, cli.ollama_model.clone(), cli.ollama_url.clone())));
    {
        let guard = state.lock().await;
        set_memory_count(guard.graph.len() as u64);
    }
    let server = CogniMemServer::new(state.clone());

    tokio::spawn(decay_task(
        state,
        Duration::from_secs(cli.decay_interval_secs),
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
        .arg("--prune-threshold")
        .arg(cli.prune_threshold.to_string())
        .arg("--storage")
        .arg(&cli.storage)
        .arg("--metrics-port")
        .arg(cli.metrics_port.to_string())
        .arg("--socket-path")
        .arg(socket_path)
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
