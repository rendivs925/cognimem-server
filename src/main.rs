use rmcp::{
    ServerHandler, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::{RequestContext, RoleServer},
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, Map};
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::Mutex;

mod memory;
use memory::*;

struct CogniMemState {
    graph: MemoryGraph,
}

impl CogniMemState {
    fn new() -> Self {
        Self {
            graph: MemoryGraph::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RememberArgs {
    content: String,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    importance: Option<f32>,
    #[serde(default)]
    associations: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecallArgs {
    query: String,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
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
        request: Option<rmcp::model::PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let schema_remember = Arc::new(serde_json::json!({
            "type": "object",
            "properties": {
                "content": { "type": "string" },
                "tier": { "type": "string", "enum": ["Sensory","Working","Episodic","Semantic","Procedural"] },
                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                "associations": { "type": "array", "items": { "type": "string" } }
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

        let tools = vec![tool_remember, tool_recall];

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
        let name = request.name.as_ref();
        let args: Map<String, Value> = request.arguments.unwrap_or_default();

        match name {
            "remember" => {
                let remember_args: RememberArgs = serde_json::from_value(Value::Object(args))
                    .map_err(|e| rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(e.to_string()), None))?;
                let tier_str = remember_args.tier.unwrap_or_else(|| "Episodic".to_string());
                let tier = match tier_str.as_str() {
                    "Sensory" => MemoryTier::Sensory,
                    "Working" => MemoryTier::Working,
                    "Episodic" => MemoryTier::Episodic,
                    "Semantic" => MemoryTier::Semantic,
                    "Procedural" => MemoryTier::Procedural,
                    _ => MemoryTier::Episodic,
                };

                let importance = remember_args.importance.unwrap_or(0.5);

                let decay_rate = match tier {
                    MemoryTier::Sensory => 2.0,
                    MemoryTier::Working => 1.0,
                    MemoryTier::Episodic => 0.5,
                    MemoryTier::Semantic => 0.2,
                    MemoryTier::Procedural => 0.1,
                };

                let memory = CognitiveMemoryUnit::new(remember_args.content, tier, importance, decay_rate);
                let id = memory.id;

                let mut guard = self.state.lock().await;
                guard.graph.add_memory(memory);

                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    json!({
                        "memory_id": id.to_string(),
                        "message": "Memory stored successfully"
                    }).to_string()
                )]))
            }
            "recall" => {
                let recall_args: RecallArgs = serde_json::from_value(Value::Object(args))
                    .map_err(|e| rmcp::ErrorData::new(rmcp::model::ErrorCode(-32602), Cow::Owned(e.to_string()), None))?;
                let query = recall_args.query.to_lowercase();
                let tier_filter = recall_args.tier;
                let limit = recall_args.limit.unwrap_or(5);

                let guard = self.state.lock().await;
                let results: Vec<&CognitiveMemoryUnit> = guard
                    .graph
                    .get_all_memories()
                    .into_iter()
                    .filter(|m| {
                        if let Some(ref tier_str) = tier_filter {
                            let matches_tier = match tier_str.as_str() {
                                "Sensory" => matches!(m.tier, MemoryTier::Sensory),
                                "Working" => matches!(m.tier, MemoryTier::Working),
                                "Episodic" => matches!(m.tier, MemoryTier::Episodic),
                                "Semantic" => matches!(m.tier, MemoryTier::Semantic),
                                "Procedural" => matches!(m.tier, MemoryTier::Procedural),
                                _ => true,
                            };
                            if !matches_tier {
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

                let memories_json: Vec<Value> = sorted_results
                    .iter()
                    .map(|m| {
                        json!({
                            "id": m.id.to_string(),
                            "content": m.content,
                            "tier": format!("{:?}", m.tier),
                            "activation": m.metadata.base_activation,
                        })
                    })
                    .collect();

                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    json!({ "memories": memories_json }).to_string()
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