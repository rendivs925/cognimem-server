mod theme;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::memory::{CodeNodeKind, MemoryTier};
use crate::state::CogniMemState;

pub use theme::get_theme;

#[derive(Clone)]
pub struct DashboardState {
    pub state: Arc<Mutex<CogniMemState>>,
    pub theme: theme::Theme,
}

pub async fn start_dashboard_server(state: Arc<Mutex<CogniMemState>>, port: u16) {
    let theme = get_theme("onyx");
    let app_state = DashboardState { state, theme };
    let app = Router::new()
        .route("/", get(dashboard_index))
        .route("/api/memories", get(api_memories))
        .route("/api/memories", post(api_remember))
        .route("/api/graph", get(api_code_graph))
        .route("/api/graph/nodes", get(api_graph_nodes))
        .route("/api/graph/file/:path", get(api_graph_file))
        .route("/api/stats", get(api_stats))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    tracing::info!("Dashboard at http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

const TEMPLATE: &str = include_str!("index.html");

fn render_template(theme: &theme::Theme) -> String {
    TEMPLATE
        .replace("{{TITLE}}", "CogniMem")
        .replace("{{CSS}}", &theme.css())
        .replace("{{LOGO}}", "CogniMem")
}

async fn dashboard_index(State(state): State<DashboardState>) -> Html<String> {
    Html(render_template(&state.theme))
}

async fn api_stats(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();
    
    let total = ms.len();
    let sensory = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Sensory)).count();
    let working = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Working)).count();
    let episodic = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Episodic)).count();
    let semantic = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Semantic)).count();
    let procedural = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Procedural)).count();
    
    let avg_activation = if !ms.is_empty() {
        ms.iter().map(|m| m.metadata.base_activation as f64).sum::<f64>() / ms.len() as f64
    } else { 0.0 };
    
    let avg_strength = if !ms.is_empty() {
        ms.iter().map(|m| m.metadata.strength as f64).sum::<f64>() / ms.len() as f64
    } else { 0.0 };
    
    let recent = ms.iter().take(10).map(|m| {
        format!(r#"<div style="padding:8px 12px;border-bottom:1px solid var(--muted)">
            <span class="tier tier-{}">{}</span>
            <span style="margin-left:12px">{}</span>
        </div>"#,
            m.tier.to_string().to_lowercase(),
            m.tier,
            m.content.chars().take(50).collect::<String>()
        )
    }).collect::<Vec<_>>().join("\n");
    
    let tier_data = serde_json::json!({
        "sensory": sensory,
        "working": working,
        "episodic": episodic,
        "semantic": semantic,
        "procedural": procedural
    });
    
    Html(format!(r#"<div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Total Memories</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{}</div>
            <div class="stat-label">Code Nodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{:.2}</div>
            <div class="stat-label">Avg Activation</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{:.2}</div>
            <div class="stat-label">Avg Strength</div>
        </div>
    </div>
    
    <div class="card" style="margin-top:24px">
        <div class="card-header">Memory Distribution</div>
        <div class="tier-chart">
            <div class="tier-bar">
                <div class="tier-bar-label">Sensory</div>
                <div class="tier-bar-track">
                    <div class="tier-bar-fill" style="width:{}%;background:#0FC5ED"></div>
                </div>
                <div class="tier-bar-value">{}</div>
            </div>
            <div class="tier-bar">
                <div class="tier-bar-label">Working</div>
                <div class="tier-bar-track">
                    <div class="tier-bar-fill" style="width:{}%;background:#a277ff"></div>
                </div>
                <div class="tier-bar-value">{}</div>
            </div>
            <div class="tier-bar">
                <div class="tier-bar-label">Episodic</div>
                <div class="tier-bar-track">
                    <div class="tier-bar-fill" style="width:{}%;background:#47FF9C"></div>
                </div>
                <div class="tier-bar-value">{}</div>
            </div>
            <div class="tier-bar">
                <div class="tier-bar-label">Semantic</div>
                <div class="tier-bar-track">
                    <div class="tier-bar-fill" style="width:{}%;background:#FFE073"></div>
                </div>
                <div class="tier-bar-value">{}</div>
            </div>
            <div class="tier-bar">
                <div class="tier-bar-label">Procedural</div>
                <div class="tier-bar-track">
                    <div class="tier-bar-fill" style="width:{}%;background:#E52E2E"></div>
                </div>
                <div class="tier-bar-value">{}</div>
            </div>
        </div>
    </div>
    
    <div class="card" style="margin-top:24px">
        <div class="card-header">Recent Memories</div>
        <div>{}</div>
    </div>"#,
        total,
        guard.code_graph.len(),
        avg_activation,
        avg_strength,
        if total > 0 { sensory as f64 / total as f64 * 100.0 } else { 0.0 }, sensory,
        if total > 0 { working as f64 / total as f64 * 100.0 } else { 0.0 }, working,
        if total > 0 { episodic as f64 / total as f64 * 100.0 } else { 0.0 }, episodic,
        if total > 0 { semantic as f64 / total as f64 * 100.0 } else { 0.0 }, semantic,
        if total > 0 { procedural as f64 / total as f64 * 100.0 } else { 0.0 }, procedural,
        recent
    ))
}

fn tier_class(tier: &str) -> &'static str {
    match tier {
        "sensory" => "tier-sensory",
        "working" => "tier-working",
        "episodic" => "tier-episodic",
        "semantic" => "tier-semantic",
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
            tier_class(&tier_str),
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

#[derive(serde::Serialize)]
struct FileTreeEntry {
    name: String,
    path: String,
    is_dir: bool,
    children: Option<Vec<FileTreeEntry>>,
}

fn build_file_tree(files: &[&PathBuf]) -> Vec<FileTreeEntry> {
    let mut dirs: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for f in files {
        let path = f.to_string_lossy().to_string();
        if let Some(pos) = path.rfind('/') {
            let dir = &path[..pos];
            let name = &path[pos+1..];
            dirs.entry(dir.to_string()).or_default().push(name.to_string());
        }
    }
    
    let mut entries: Vec<FileTreeEntry> = dirs.iter().map(|(d, files_in_dir)| {
        let dir_name = d.rfind('/').map(|p| &d[p+1..]).unwrap_or(d).to_string();
        let mut children: Vec<FileTreeEntry> = files_in_dir.iter().map(|n| {
            let mut full_path = d.to_string();
            full_path.push('/');
            full_path.push_str(n);
            FileTreeEntry {
                name: n.clone(),
                path: full_path,
                is_dir: false,
                children: None,
            }
        }).collect();
        children.sort_by(|a, b| a.name.cmp(&b.name));
        FileTreeEntry {
            name: dir_name,
            path: d.clone(),
            is_dir: true,
            children: Some(children),
        }
    }).collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

async fn api_graph_nodes(State(state): State<DashboardState>) -> Json<Vec<FileTreeEntry>> {
    let guard = state.state.lock().await;
    let files = guard.code_graph.all_files();
    Json(build_file_tree(&files))
}

async fn api_graph_file(
    Path(path): Path<String>,
    State(state): State<DashboardState>,
) -> Html<String> {
    let guard = state.state.lock().await;
    let file_path = PathBuf::from(&path);
    let nodes = guard.code_graph.get_nodes_in_file(&file_path);
    
    if nodes.is_empty() {
        return Html(r#"<div class="empty">No nodes in this file</div>"#.to_string());
    }
    
    let blocks: Vec<String> = nodes.iter().map(|n| {
        let kind_icon = match n.kind {
            CodeNodeKind::Function => "fn",
            CodeNodeKind::Struct => "struct",
            CodeNodeKind::Trait => "trait",
            CodeNodeKind::Impl => "impl",
            CodeNodeKind::Enum => "enum",
            CodeNodeKind::Macro => "macro",
            _ => "code",
        };
        format!(r#"<div class="node-item">
            <span class="node-kind">{}</span>
            <span class="node-name">{}</span>
            <span class="node-lines">{}:{}</span>
        </div>"#,
            kind_icon,
            n.name,
            n.line_start,
            n.line_end
        )
    }).collect();
    
    Html(format!(r#"<div class="file-nodes">{}</div>"#, blocks.join("\n")))
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
