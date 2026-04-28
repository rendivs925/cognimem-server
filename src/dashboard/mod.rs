use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
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

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    tracing::info!("Dashboard at http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

const HTML: &str = include_str!("index.html");

async fn dashboard_index() -> Html<String> {
    Html(HTML.to_string())
}

async fn api_stats(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();
    
    let mut by_tier: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for tier in ["sensory", "working", "episodic", "semantic", "procedural"] {
        let count = ms.iter().filter(|m| m.tier.to_string() == tier).count();
        by_tier.insert(tier.to_string(), serde_json::json!(count));
    }
    
    Json(serde_json::json!({
        "total": ms.len(),
        "by_tier": by_tier,
        "code_nodes": guard.code_graph.len(),
    }))
}

fn tier_color(tier: &str) -> &'static str {
    match tier {
        "sensory" => "tier-sensory", "working" => "tier-working",
        "episodic" => "tier-episodic", "semantic" => "tier-semantic",
        _ => "tier-procedural",
    }
}

async fn api_memories(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();
    
    if ms.is_empty() {
        return Html(r#"<div class="empty">No memories yet</div>"#.to_string());
    }
    
    let rows: Vec<String> = ms.iter().take(50).map(|m| {
        let id = m.id.to_string();
        let short_id = id.get(..8).unwrap_or(&id);
        let tier_str = m.tier.to_string();
        
        format!(r#"<tr>
            <td><code>{}</code></td>
            <td><span class="tier {}">{}</span></td>
            <td>{}</td>
            <td class="metric">{:.2}</td>
            <td class="metric">{:.2}</td>
        </tr>"#,
            short_id,
            tier_color(&tier_str),
            tier_str,
            m.content.chars().take(60).collect::<String>(),
            m.metadata.base_activation,
            m.metadata.strength
        )
    }).collect();
    
    Html(format!(r#"<table class="table">
        <thead><tr><th>ID</th><th>Tier</th><th>Content</th><th>Activation</th><th>Strength</th></tr></thead>
        <tbody>{}</tbody>
    </table>"#, rows.join("\n")))
}

async fn api_code_graph(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let files = guard.code_graph.all_files();
    
    if files.is_empty() {
        return Html(r#"<div class="empty">No code discovered. Run with --project-path</div>"#.to_string());
    }
    
    let blocks: Vec<String> = files.iter().take(20).map(|f| {
        let name = f.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        
        let nodes = guard.code_graph.get_nodes_in_file(f);
        let items: Vec<String> = nodes.iter().take(10).map(|n| {
            format!(r#"<div style="padding:8px 24px;font-size:13px;opacity:0.7">{:?} {}</div>"#, n.kind, n.name)
        }).collect();
        
        format!(r#"<div style="padding:12px 16px;border-bottom:1px solid var(--muted)"><code>{}</code></div>{}"#,
            name, items.join("\n"))
    }).collect();
    
    Html(blocks.join("\n"))
}

async fn api_remember(
    State(state): State<DashboardState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Html<String>) {
    let content = payload
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
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