use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::memory::MemoryTier;
use crate::state::CogniMemState;

#[derive(Clone)]
pub struct DashboardState {
    pub state: Arc<Mutex<CogniMemState>>,
}

pub async fn start_dashboard_server(state: Arc<Mutex<CogniMemState>>, port: u16) {
    let app_state = DashboardState { state };
    let app = Router::new()
        .route("/", get(dashboard_index))
        .route("/api/memories", get(api_memories))
        .route("/api/memories", post(api_remember))
        .route("/api/graph", get(api_code_graph))
        .route("/api/stats", get(api_stats))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind dashboard on port {port}: {e}");
            return;
        }
    };
    tracing::info!("Dashboard available at http://localhost:{port}");
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("Dashboard server error: {e}");
    }
}

#[derive(Serialize)]
struct MemoryRow {
    id: String,
    tier: String,
    content: String,
    activation: f32,
    strength: f32,
    created_at: String,
}

fn generate_dashboard_html() -> String {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n");
    html.push_str("<html lang=\"en\">\n");
    html.push_str("<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    html.push_str("<title>CogniMem Dashboard</title>\n");
    html.push_str("<script src=\"https://unpkg.com/htmx.org@1.9.10\"></script>\n");
    html.push_str("<style>\n");
    html.push_str("* { box-sizing: border-box; margin: 0; padding: 0; }\n");
    html.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }\n");
    html.push_str(".container { max-width: 1200px; margin: 0 auto; padding: 20px; }\n");
    html.push_str("h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.5rem; }\n");
    html.push_str(".tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }\n");
    html.push_str(".tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; background: transparent; color: #8b949e; border: none; font-size: 14px; transition: all 0.2s; }\n");
    html.push_str(".tab:hover { background: #21262d; color: #c9d1d9; }\n");
    html.push_str(".tab.active { background: #21262d; color: #58a6ff; }\n");
    html.push_str(".stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }\n");
    html.push_str(".stat-card { background: #161b22; padding: 16px; border-radius: 8px; border: 1px solid #30363d; }\n");
    html.push_str(".stat-value { font-size: 24px; font-weight: 600; color: #58a6ff; }\n");
    html.push_str(".stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; }\n");
    html.push_str(".card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }\n");
    html.push_str(".card-header { padding: 12px 16px; background: #21262d; border-bottom: 1px solid #30363d; font-weight: 500; }\n");
    html.push_str("table { width: 100%; border-collapse: collapse; }\n");
    html.push_str("th, td { padding: 10px 16px; text-align: left; border-bottom: 1px solid #30363d; font-size: 13px; }\n");
    html.push_str("th { background: #21262d; color: #8b949e; font-weight: 500; text-transform: uppercase; font-size: 11px; }\n");
    html.push_str("tr:hover { background: #21262d; }\n");
    html.push_str(".tier { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }\n");
    html.push_str(".tier-sensory { background: #1f3a5f; color: #58a6ff; }\n");
    html.push_str(".tier-working { background: #3d1f5f; color: #a371f7; }\n");
    html.push_str(".tier-episodic { background: #1f5f3d; color: #3fb950; }\n");
    html.push_str(".tier-semantic { background: #5f3d1f; color: #d29922; }\n");
    html.push_str(".tier-procedural { background: #5f1f3d; color: #f85149; }\n");
    html.push_str(".activation { color: #3fb950; }\n");
    html.push_str(".strength { color: #d29922; }\n");
    html.push_str("</style>\n");
    html.push_str("</head>\n");
    html.push_str("<body>\n");
    html.push_str("<div class=\"container\">\n");
    html.push_str("<h1>CogniMem Dashboard</h1>\n");
    html.push_str("<div class=\"tabs\">\n");
    html.push_str("<button class=\"tab active\" hx-get=\"/api/memories\" hx-target=\"#content\" hx-swap=\"innerHTML\">Memories</button>\n");
    html.push_str("<button class=\"tab\" hx-get=\"/api/graph\" hx-target=\"#content\" hx-swap=\"innerHTML\">Code Graph</button>\n");
    html.push_str("<button class=\"tab\" hx-get=\"/api/stats\" hx-target=\"#stats\" hx-swap=\"innerHTML\">Stats</button>\n");
    html.push_str("</div>\n");
    html.push_str("<div id=\"stats\" class=\"stats\" hx-get=\"/api/stats\" hx-trigger=\"load\"></div>\n");
    html.push_str("<div id=\"content\" class=\"card\">\n");
    html.push_str("<div class=\"card-header\">Memories</div>\n");
    html.push_str("<div hx-get=\"/api/memories\" hx-trigger=\"load\"></div>\n");
    html.push_str("</div>\n");
    html.push_str("</div>\n");
    html.push_str("</body>\n");
    html.push_str("</html>\n");
    html
}

async fn dashboard_index() -> Html<String> {
    Html(generate_dashboard_html())
}

async fn api_stats(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let guard = state.state.lock().await;
    let memories = guard.graph.get_all_memories();

    let by_tier: std::collections::HashMap<String, usize> = memories
        .iter()
        .fold(std::collections::HashMap::new(), |mut acc, m| {
            *acc.entry(m.tier.to_string()).or_insert(0) += 1;
            acc
        });

    let avg_activation = if memories.is_empty() {
        0.0
    } else {
        memories.iter().map(|m| m.metadata.base_activation).sum::<f32>() / memories.len() as f32
    };

    let avg_strength = if memories.is_empty() {
        0.0
    } else {
        memories.iter().map(|m| m.metadata.strength).sum::<f32>() / memories.len() as f32
    };

    Json(serde_json::json!({
        "total": memories.len(),
        "by_tier": by_tier,
        "avg_activation": format!("{:.3}", avg_activation),
        "avg_strength": format!("{:.3}", avg_strength),
        "code_nodes": guard.code_graph.len(),
    }))
}

fn tier_class(tier: &str) -> &'static str {
    match tier {
        "sensory" => "tier-sensory",
        "working" => "tier-working",
        "episodic" => "tier-episodic",
        "semantic" => "tier-semantic",
        "procedural" => "tier-procedural",
        _ => "tier-episodic",
    }
}

async fn api_memories(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let memories = guard.graph.get_all_memories();

    let mut html = String::from("<table><thead><tr><th>ID</th><th>Tier</th><th>Content</th><th>Act</th><th>Str</th><th>Created</th></tr></thead><tbody>");

    for m in memories.iter().take(50) {
        let id = &m.id.to_string()[..8];
        let tier_str = m.tier.to_string();
        let content = m.content.chars().take(80).collect::<String>();
        let content_escaped = content.replace('<', "&lt;").replace('>', "&gt;");
        let created = chrono::DateTime::from_timestamp(m.metadata.created_at, 0)
            .map(|dt| dt.format("%H:%M").to_string())
            .unwrap_or_default();

        html.push_str(&format!(
            "<tr><td>{}</td><td><span class=\"tier {}\">{}</span></td><td>{}</td><td class=\"activation\">{:.3}</td><td class=\"strength\">{:.3}</td><td>{}</td></tr>",
            id,
            tier_class(&tier_str),
            tier_str,
            content_escaped,
            m.metadata.base_activation,
            m.metadata.strength,
            created
        ));
    }

    html.push_str("</tbody></table>");
    Html(html)
}

async fn api_remember(
    State(state): State<DashboardState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Html<String>) {
    let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if content.is_empty() {
        return (StatusCode::BAD_REQUEST, Html("Content required".to_string()));
    }

    let memory = crate::memory::CognitiveMemoryUnit::new(
        content.to_string(),
        MemoryTier::Episodic,
        0.5,
        MemoryTier::Episodic.decay_rate(),
    );

    {
        let mut guard = state.state.lock().await;
        let emb = guard.embedder.embed(&memory.content);
        guard.graph.add_memory(memory.clone());
        guard.graph.set_embedding(memory.id, emb);
        let _ = guard.storage.save(&memory);
    }

    (StatusCode::OK, Html("Memory added".to_string()))
}

async fn api_code_graph(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let files = guard.code_graph.all_files();

    let mut html = String::new();
    for file in files.iter().take(20) {
        let nodes = guard.code_graph.get_nodes_in_file(file);
        if nodes.is_empty() {
            continue;
        }
        let filename = file.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
        html.push_str(&format!("<div style=\"padding:8px;border-bottom:1px solid #30363d;\"><code>{}</code></div>", filename));
        for node in nodes.iter().take(10) {
            html.push_str(&format!(
                "<div style=\"padding:4px 16px 4px 32px;font-size:12px;color:#8b949e;\">{:?}: {}</div>",
                node.kind,
                node.name
            ));
        }
    }

    if html.is_empty() {
        html = "No code discovered. Run with --project-path".to_string();
    }

    Html(html)
}