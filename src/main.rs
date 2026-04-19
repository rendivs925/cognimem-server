use rmcp::{
    ServerHandler, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::{RequestContext, RoleServer},
    transport::stdio,
};
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::Mutex;

mod memory;
use memory::{
    CognitiveMemoryUnit, MemorySummary, RecallArgs, RecallResult, RememberArgs, RememberResult,
};

struct CogniMemState {
    graph: memory::MemoryGraph,
}

impl CogniMemState {
    fn new() -> Self {
        Self {
            graph: memory::MemoryGraph::new(),
        }
    }
}

#[derive(Clone)]
struct CogniMemServer {
    state: Arc<Mutex<CogniMemState>>,
}

impl CogniMemServer {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(CogniMemState::new())),
        }
    }
}

impl ServerHandler for CogniMemServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(rmcp::model::ServerCapabilities::default())
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let schema_remember = Arc::new(serde_json::json!({
            "type": "object",
            "properties": {
                "content": { "type": "string" },
                "tier": { 
                    "type": "string", 
                    "enum": ["sensory", "working", "episodic", "semantic", "procedural"] 
                },
                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                "associations": { "type": "array", "items": { "type": "string", "format": "uuid" } }
            },
            "required": ["content"]
        }).as_object().cloned().unwrap());

        let schema_recall = Arc::new(serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "tier": { "type": "string" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 50 }
            },
            "required": ["query"]
        }).as_object().cloned().unwrap());

        let tool_remember = rmcp::model::Tool::new(
            Cow::Borrowed("remember"),
            Cow::Borrowed("Store a memory with optional tier and importance"),
            schema_remember,
        );

        let tool_recall = rmcp::model::Tool::new(
            Cow::Borrowed("recall"),
            Cow::Borrowed("Retrieve memories relevant to a query"),
            schema_recall,
        );

        Ok(rmcp::model::ListToolsResult {
            tools: vec![tool_remember, tool_recall],
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let name = request.name.as_ref();
        let args = request.arguments.unwrap_or_default();

        match name {
            "remember" => {
                let args: RememberArgs = serde_json::from_value(serde_json::Value::Object(args))
                    .map_err(|e| rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(e.to_string()), None))?;

                let tier = args.tier.unwrap_or_default();
                let importance = args.importance.unwrap_or(0.5);
                let decay_rate = tier.decay_rate();

                let memory = CognitiveMemoryUnit::new(args.content, tier, importance, decay_rate);
                let memory_id = memory.id;

                let mut guard = self.state.lock().await;
                guard.graph.add_memory(memory);

                let result = RememberResult::success(memory_id);
                let json = serde_json::to_value(&result).unwrap_or_default();

                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    serde_json::to_string(&json).unwrap_or_default()
                )]))
            }
            "recall" => {
                let args: RecallArgs = serde_json::from_value(serde_json::Value::Object(args))
                    .map_err(|e| rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(e.to_string()), None))?;

                let query = args.query.to_lowercase();
                let tier_filter = args.tier;
                let limit = args.limit.unwrap_or(5);

                let guard = self.state.lock().await;
                let results: Vec<&CognitiveMemoryUnit> = guard
                    .graph
                    .get_all_memories()
                    .into_iter()
                    .filter(|m| {
                        if let Some(tier) = tier_filter {
                            if m.tier != tier {
                                return false;
                            }
                        }
                        m.content.to_lowercase().contains(&query)
                            || query.split_whitespace().any(|w| m.content.to_lowercase().contains(w))
                    })
                    .collect();

                let mut sorted_results: Vec<&CognitiveMemoryUnit> = results;
                sorted_results.sort_by(|a, b| {
                    b.metadata.base_activation.partial_cmp(&a.metadata.base_activation).unwrap()
                });
                sorted_results.truncate(limit);

                let memories: Vec<MemorySummary> = sorted_results
                    .iter()
                    .map(|m| MemorySummary::from(*m))
                    .collect();

                let result = RecallResult::new(memories);
                let json = serde_json::to_value(&result).unwrap_or_default();

                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    serde_json::to_string(&json).unwrap_or_default()
                )]))
            }
            _ => Err(rmcp::ErrorData::new(
                rmcp::model::ErrorCode(-32601),
                Cow::Borrowed("Unknown tool"),
                None,
            )),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let server = CogniMemServer::new();
    server.serve(stdio()).await?;

    Ok(())
}