mod memory;

use memory::{
    AssociateArgs, AssociateResult, CognitiveMemoryUnit, ForgetArgs, ForgetResult, MemoryStorage,
    MemorySummary, RecallArgs, RecallResult, ReflectArgs, ReflectResult, RememberArgs, RememberResult,
};
use rmcp::{
    ServerHandler, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::{RequestContext, RoleServer},
    transport::stdio,
};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::error;

struct CogniMemState {
    graph: memory::MemoryGraph,
    storage: MemoryStorage,
}

impl CogniMemState {
    fn new(storage: MemoryStorage) -> Self {
        let mut graph = memory::MemoryGraph::new();
        if let Ok(memories) = storage.load_all() {
            for m in memories {
                graph.add_memory(m);
            }
        }
        Self { graph, storage }
    }
}

#[derive(Clone)]
struct CogniMemServer {
    state: Arc<Mutex<CogniMemState>>,
}

impl CogniMemServer {
    fn new(state: Arc<Mutex<CogniMemState>>) -> Self {
        Self { state }
    }
}

fn parse_args<T: serde::de::DeserializeOwned>(
    args: serde_json::Map<String, serde_json::Value>,
) -> Result<T, rmcp::ErrorData> {
    serde_json::from_value(serde_json::Value::Object(args)).map_err(|e| {
        rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(e.to_string()), None)
    })
}

fn success_json<T: serde::Serialize>(data: &T) -> CallToolResult {
    let json = serde_json::to_value(data).unwrap_or_default();
    CallToolResult::success(vec![rmcp::model::Content::text(
        serde_json::to_string(&json).unwrap_or_default(),
    )])
}

fn invalid_params(msg: &str) -> rmcp::ErrorData {
    rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(msg.to_string()), None)
}

fn tool_not_found() -> rmcp::ErrorData {
    rmcp::ErrorData::new(rmcp::model::ErrorCode(-32601), Cow::Owned("Unknown tool".to_string()), None)
}

fn json_schema(json: serde_json::Value) -> std::sync::Arc<serde_json::Map<String, serde_json::Value>> {
    std::sync::Arc::new(json.as_object().cloned().unwrap())
}

const TIER_ENUM: &[&str] = &["sensory", "working", "episodic", "semantic", "procedural"];

impl ServerHandler for CogniMemServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(rmcp::model::ServerCapabilities::default())
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
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
                Cow::Borrowed("Create an association between two memories with an optional strength weight"),
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
                Cow::Borrowed("Run a consolidation cycle: decay activation, prune weak memories, promote strong ones"),
                json_schema(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "intensity": { "type": "string", "enum": ["light", "full"], "description": "light=decay only, full=decay+prune+promote" }
                    }
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
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args = request.arguments.unwrap_or_default();

        match request.name.as_ref() {
            "remember" => self.handle_remember(args).await,
            "recall" => self.handle_recall(args).await,
            "associate" => self.handle_associate(args).await,
            "forget" => self.handle_forget(args).await,
            "reflect" => self.handle_reflect(args).await,
            _ => Err(tool_not_found()),
        }
    }
}

impl CogniMemServer {
    async fn handle_remember(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: RememberArgs = parse_args(args)?;

        let tier = args.tier.unwrap_or_default();
        let importance = args.importance.unwrap_or(0.5);
        let decay_rate = tier.decay_rate();

        let mut memory = CognitiveMemoryUnit::new(args.content, tier, importance, decay_rate);
        if let Some(assoc_ids) = args.associations {
            memory.associations = assoc_ids;
        }

        let memory_id = memory.id;
        let mut guard = self.state.lock().await;
        guard.graph.add_memory(memory.clone());
        if let Err(e) = guard.storage.save(&memory) {
            error!("Failed to persist memory {memory_id}: {e}");
        }

        Ok(success_json(&RememberResult::success(memory_id)))
    }

    async fn handle_recall(
        &self,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args: RecallArgs = parse_args(args)?;

        let query = args.query.to_lowercase();
        let limit = args.limit.unwrap_or(5);
        let min_activation = args.min_activation.unwrap_or(0.0);
        let now = chrono::Utc::now().timestamp();

        let mut guard = self.state.lock().await;

        let mut results: Vec<&CognitiveMemoryUnit> = guard
            .graph
            .get_all_memories()
            .into_iter()
            .filter(|m| matches_tier(m.tier, args.tier)
                && matches_query(&m.content, &query)
                && m.metadata.base_activation >= min_activation)
            .collect();

        expand_with_associations(&mut results, &guard.graph);

        results.sort_by(|a, b| {
            b.metadata.base_activation.partial_cmp(&a.metadata.base_activation).unwrap()
        });
        results.truncate(limit);

        let recalled_ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
        update_activation_for(&mut guard, &recalled_ids, now);

        let memories: Vec<MemorySummary> = recalled_ids
            .iter()
            .filter_map(|id| guard.graph.get_memory(id).map(MemorySummary::from))
            .collect();

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

        Ok(success_json(&AssociateResult::success(args.from, args.to, strength)))
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

        if args.hard_delete.unwrap_or(false) {
            if let Some(removed) = guard.graph.remove_memory(&args.memory_id)
                && let Err(e) = guard.storage.delete(&removed.id)
            {
                error!("Failed to delete memory {}: {e}", removed.id);
            }
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

        let mut guard = self.state.lock().await;
        let total = guard.graph.len();

        memory::apply_decay_to_all(&mut guard.graph);

        let pruned = if intensity == "full" {
            memory::prune_below_threshold(&mut guard.graph, 0.01)
        } else {
            0
        };

        let promoted = memory::promote_memories(&mut guard.graph);

        for mem in guard.graph.get_all_memories() {
            if let Err(e) = guard.storage.save(mem) {
                error!("Failed to persist reflected memory {}: {e}", mem.id);
            }
        }

        Ok(success_json(&ReflectResult::new(pruned, promoted, total)))
    }
}

fn matches_tier(memory_tier: memory::MemoryTier, filter: Option<memory::MemoryTier>) -> bool {
    filter.is_none_or(|t| memory_tier == t)
}

fn matches_query(content: &str, query: &str) -> bool {
    let lower = content.to_lowercase();
    lower.contains(query) || query.split_whitespace().any(|w| lower.contains(w))
}

fn expand_with_associations<'a>(results: &mut Vec<&'a CognitiveMemoryUnit>, graph: &'a memory::MemoryGraph) {
    let direct_ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
    let mut expanded = std::collections::HashSet::new();

    for id in &direct_ids {
        for (assoc_id, _) in graph.get_associations(id) {
            if !direct_ids.contains(&assoc_id) {
                expanded.insert(assoc_id);
            }
        }
    }

    for id in &expanded {
        if let Some(mem) = graph.get_memory(id) {
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

async fn decay_task(state: Arc<Mutex<CogniMemState>>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        let mut guard = state.lock().await;
        memory::apply_decay_to_all(&mut guard.graph);
        let removed = memory::prune_below_threshold(&mut guard.graph, 0.01);
        if removed > 0 {
            tracing::info!("Pruned {removed} memories below activation threshold");
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let storage = MemoryStorage::open(std::path::Path::new("./cognimem-data"))?;
    let state = Arc::new(Mutex::new(CogniMemState::new(storage)));
    let server = CogniMemServer::new(state.clone());

    tokio::spawn(decay_task(state, Duration::from_secs(300)));

    let running_service = server.serve(stdio()).await?;
    running_service.waiting().await?;

    Ok(())
}