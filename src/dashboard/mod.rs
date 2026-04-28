mod theme;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::memory::{CodeNodeKind, MemoryTier};
use crate::state::CogniMemState;

pub use theme::get_theme;

#[derive(Clone)]
pub struct DashboardState {
    pub state: Arc<Mutex<CogniMemState>>,
    pub theme: theme::Theme,
}

// --- Helpers ---

fn card(title: &str, body: &str) -> String {
    format!(r#"<div class="card"><div class="card-header">{}</div>{}</div>"#, title, body)
}

fn empty(msg: &str) -> Html<String> {
    Html(format!(r#"<div class="empty">{}</div>"#, msg))
}

fn badge(tier: &str) -> String {
    format!(r#"<span class="badge badge-{}">{}</span>"#, tier.to_lowercase(), tier)
}

fn list_item(body: &str) -> String {
    format!(r#"<div class="list-item">{}</div>"#, body)
}

fn stat_card(value: &str, label: &str) -> String {
    format!(r#"<div class="stat-card"><div class="stat-value">{}</div><div class="stat-label">{}</div></div>"#, value, label)
}

fn tier_bar(label: &str, pct: f64, count: usize, color: &str) -> String {
    format!(r#"<div class="tier-bar">
        <div class="tier-bar-label">{}</div>
        <div class="tier-bar-track"><div class="tier-bar-fill" style="width:{}%;background:{}"></div></div>
        <div class="tier-bar-value">{}</div>
    </div>"#, label, pct, color, count)
}

fn esc_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn esc_attr(s: &str) -> String {
    s.replace('\'', "\\'")
}

// --- Server ---

pub async fn start_dashboard_server(state: Arc<Mutex<CogniMemState>>, port: u16) {
    let theme = get_theme("onyx");
    let app_state = DashboardState { state, theme };
    let app = Router::new()
        .route("/", get(dashboard_index))
        .route("/api/theme", get(api_theme))
        .route("/api/memories", get(api_memories))
        .route("/api/memories", post(api_remember))
        .route("/api/search", get(api_search))
        .route("/api/graph/nodes", get(api_graph_nodes))
        .route("/api/graph/file/{path}", get(api_graph_file))
        .route("/api/stats", get(api_stats))
        .route("/api/skills", get(api_skills))
        .route("/api/persona", get(api_persona))
        .route("/api/work", get(api_work))
        .route("/api/timeline", get(api_timeline))
        .nest_service("/static", ServeDir::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/dashboard/static")
        ))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    tracing::info!("Dashboard at http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

async fn dashboard_index(State(state): State<DashboardState>) -> Html<String> {
    let html = include_str!("static/index.html")
        .replace("{{CSS_VARS}}", &state.theme.to_css_vars());
    Html(html)
}

async fn api_theme(State(state): State<DashboardState>) -> Json<theme::Theme> {
    Json(state.theme)
}

// --- Memories ---

async fn api_memories(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();

    if ms.is_empty() {
        return empty("No memories yet. Click + Add to create one.");
    }

    let rows: Vec<String> = ms.iter().take(100).map(|m| {
        let id = m.id.to_string();
        let short_id = id.get(..8).unwrap_or(&id);
        let tier = m.tier.to_string();
        let content = m.content.chars().take(80).collect::<String>();
        let content_esc = esc_html(&content);
        let content_attr = esc_attr(&content);
        let a = m.metadata.base_activation;
        let s = m.metadata.strength;

        format!(r#"<tr onclick="viewMemory(this.dataset.id,this.dataset.tier,this.dataset.content,this.dataset.activation,this.dataset.strength)" data-id="{}" data-tier="{}" data-content="{}" data-activation="{}" data-strength="{}">
            <td><code>{}</code></td>
            <td>{}</td>
            <td class="text-sm">{}</td>
            <td class="font-mono text-xs">{:.2}</td>
            <td class="font-mono text-xs">{:.2}</td>
        </tr>"#,
            id, tier.to_lowercase(), content_attr, a, s,
            short_id,
            badge(&tier),
            content_esc,
            a, s
        )
    }).collect();

    Html(format!(r#"<table class="w-full border-collapse">
        <thead><tr class="text-[11px] uppercase tracking-wider opacity-40 border-b" style="border-color:var(--border)">
            <th class="p-3 text-left font-semibold">ID</th>
            <th class="p-3 text-left font-semibold">Tier</th>
            <th class="p-3 text-left font-semibold">Content</th>
            <th class="p-3 text-left font-semibold">Activation</th>
            <th class="p-3 text-left font-semibold">Strength</th>
        </tr></thead>
        <tbody>{}</tbody>
    </table>""#, rows.join("\n")))
}

async fn api_search(
    State(state): State<DashboardState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Html<String> {
    let query = params.get("query").map(|s| s.as_str()).unwrap_or("");
    let tier_str = params.get("tier").map(|s| s.as_str());
    let limit: usize = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50);

    if query.is_empty() {
        return empty("Enter a search query");
    }

    let guard = state.state.lock().await;
    let tier_filter = tier_str.and_then(|t| match t {
        "sensory" => Some(MemoryTier::Sensory),
        "working" => Some(MemoryTier::Working),
        "episodic" => Some(MemoryTier::Episodic),
        "semantic" => Some(MemoryTier::Semantic),
        "procedural" => Some(MemoryTier::Procedural),
        _ => None,
    });

    let ids = guard.search.search(query, tier_filter, limit);
    let results: Vec<_> = ids.iter()
        .filter_map(|id| guard.graph.get_memory(id))
        .collect();

    if results.is_empty() {
        return empty("No results found");
    }

    let items: Vec<String> = results.iter().map(|m| {
        let tier = m.tier.to_string();
        let content = m.content.chars().take(80).collect::<String>();
        let content_esc = esc_html(&content);
        let a = m.metadata.base_activation;
        let s = m.metadata.strength;
        list_item(&format!(r#"data-id="{}" data-tier="{}" data-content="{}" data-activation="{}" data-strength="{}" onclick="viewMemory(this.dataset.id,this.dataset.tier,this.dataset.content,this.dataset.activation,this.dataset.strength)"
            <span>{}</span>
            <span class="flex-1">{}</span>
            <span class="font-mono text-xs opacity-50">{:.2}</span>"#,
            m.id, tier.to_lowercase(), esc_attr(&content), a, s,
            badge(&tier),
            content_esc,
            a
        ))
    }).collect();

    Html(format!(r#"<div class="text-xs opacity-40 px-4 py-2">{} results</div>{}"#, results.len(), items.join("\n")))
}

async fn api_remember(
    State(state): State<DashboardState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Html<String>) {
    let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let tier = match payload.get("tier").and_then(|v| v.as_str()) {
        Some("sensory") => MemoryTier::Sensory,
        Some("working") => MemoryTier::Working,
        Some("semantic") => MemoryTier::Semantic,
        Some("procedural") => MemoryTier::Procedural,
        _ => MemoryTier::Episodic,
    };

    if content.is_empty() {
        return (StatusCode::BAD_REQUEST, Html("Content required".to_string()));
    }

    let memory = crate::memory::CognitiveMemoryUnit::new(
        content.to_string(), tier, 0.5, tier.decay_rate(),
    );

    {
        let mut guard = state.state.lock().await;
        let emb = guard.embedder.embed(&memory.content);
        let id = guard.graph.add_memory(memory);
        guard.graph.set_embedding(id, emb);
        if let Some(m) = guard.graph.get_memory(&id) {
            guard.storage.save(m).ok();
        }
    }

    (StatusCode::OK, Html("OK".to_string()))
}

// --- Code Graph ---

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
            dirs.entry(path[..pos].to_string()).or_default()
                .push(path[pos + 1..].to_string());
        }
    }

    let mut entries: Vec<FileTreeEntry> = dirs.iter().map(|(d, files_in_dir)| {
        let dir_name = d.rfind('/').map(|p| &d[p + 1..]).unwrap_or(d).to_string();
        let mut children: Vec<FileTreeEntry> = files_in_dir.iter().map(|n| {
            let mut full_path = d.clone();
            full_path.push('/');
            full_path.push_str(n);
            FileTreeEntry { name: n.clone(), path: full_path, is_dir: false, children: None }
        }).collect();
        children.sort_by(|a, b| a.name.cmp(&b.name));
        FileTreeEntry { name: dir_name, path: d.clone(), is_dir: true, children: Some(children) }
    }).collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

async fn api_graph_nodes(State(state): State<DashboardState>) -> Json<Vec<FileTreeEntry>> {
    let guard = state.state.lock().await;
    Json(build_file_tree(&guard.code_graph.all_files()))
}

async fn api_graph_file(
    Path(path): Path<String>,
    State(state): State<DashboardState>,
) -> Html<String> {
    let guard = state.state.lock().await;
    let nodes = guard.code_graph.get_nodes_in_file(&PathBuf::from(&path));

    if nodes.is_empty() {
        return empty("No nodes in this file");
    }

    let blocks: Vec<String> = nodes.iter().map(|n| {
        let kind = match n.kind {
            CodeNodeKind::Function => "fn",
            CodeNodeKind::Struct => "struct",
            CodeNodeKind::Trait => "trait",
            CodeNodeKind::Impl => "impl",
            CodeNodeKind::Enum => "enum",
            CodeNodeKind::Macro => "macro!",
            _ => "code",
        };
        format!(r#"<div class="flex items-center gap-3 px-4 py-2.5 border-b text-sm" style="border-color:var(--border)">
            <span class="text-[10px] uppercase px-1.5 py-0.5 rounded font-mono" style="background:var(--border)">{}</span>
            <span class="flex-1 font-mono">{}</span>
            <span class="text-xs opacity-40 font-mono">{}:{}</span>
        </div>"#, kind, n.name, n.line_start, n.line_end)
    }).collect();

    Html(format!(r#"<div class="py-1">{}"#, blocks.join("\n")))
}

// --- Stats ---

async fn api_stats(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();

    let total = ms.len();
    let counts: Vec<(usize, &str, &str)> = vec![
        (ms.iter().filter(|m| matches!(m.tier, MemoryTier::Sensory)).count(), "Sensory", "#0FC5ED"),
        (ms.iter().filter(|m| matches!(m.tier, MemoryTier::Working)).count(), "Working", "#a277ff"),
        (ms.iter().filter(|m| matches!(m.tier, MemoryTier::Episodic)).count(), "Episodic", "#47FF9C"),
        (ms.iter().filter(|m| matches!(m.tier, MemoryTier::Semantic)).count(), "Semantic", "#FFE073"),
        (ms.iter().filter(|m| matches!(m.tier, MemoryTier::Procedural)).count(), "Procedural", "#F43F5E"),
    ];

    let avg_a = if !ms.is_empty() {
        ms.iter().map(|m| m.metadata.base_activation as f64).sum::<f64>() / ms.len() as f64
    } else { 0.0 };

    let avg_s = if !ms.is_empty() {
        ms.iter().map(|m| m.metadata.strength as f64).sum::<f64>() / ms.len() as f64
    } else { 0.0 };

    let pct = |n: usize| -> f64 { if total > 0 { n as f64 / total as f64 * 100.0 } else { 0.0 } };

    let tier_bars: Vec<String> = counts.iter().map(|(c, name, color)| {
        tier_bar(name, pct(*c), *c, color)
    }).collect();

    let recent: Vec<String> = ms.iter().take(10).map(|m| {
        list_item(&format!("{}<span>{}</span>",
            badge(&m.tier.to_string()),
            esc_html(&m.content.chars().take(60).collect::<String>())))
    }).collect();

    Html(format!(r#"
<div class="grid grid-cols-4 gap-4 mb-6">
    {}
    {}
    {}
    {}
</div>
<div class="card mb-6"><div class="card-header">Memory Distribution</div><div class="p-5">{}</div></div>
{}"#,
        stat_card(&total.to_string(), "Total Memories"),
        stat_card(&guard.code_graph.len().to_string(), "Code Nodes"),
        stat_card(&format!("{:.2}", avg_a), "Avg Activation"),
        stat_card(&format!("{:.2}", avg_s), "Avg Strength"),
        tier_bars.join("\n"),
        card("Recent Memories", &recent.join("\n"))
    ))
}

// --- Skills ---

async fn api_skills(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let skills: Vec<_> = guard.graph.get_all_memories().into_iter()
        .filter(|m| matches!(m.tier, MemoryTier::Procedural))
        .collect();

    if skills.is_empty() {
        return empty("No skills learned yet. Skills are created from procedural memories.");
    }

    let items: Vec<String> = skills.iter().map(|s| {
        list_item(&format!(r#"<span class="badge badge-procedural">skill</span>
            <span class="flex-1 font-mono">{}</span>
            <span class="text-xs opacity-40">{:.2}</span>"#,
            esc_html(&s.content.chars().take(60).collect::<String>()),
            s.metadata.strength))
    }).collect();

    Html(card("Procedural Skills", &items.join("\n")))
}

// --- Persona ---

async fn api_persona(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let persona: Vec<_> = guard.graph.get_all_memories().into_iter()
        .filter(|m| m.persona.is_some())
        .collect();

    if persona.is_empty() {
        return empty("No persona data extracted. Use the extract_persona tool.");
    }

    let items: Vec<String> = persona.iter().map(|m| {
        list_item(&format!(r#"<span class="badge badge-episodic">{:?}</span>
            <span class="flex-1">{}</span>"#,
            m.persona.as_ref().unwrap(),
            esc_html(&m.content.chars().take(80).collect::<String>())))
    }).collect();

    Html(card("Persona Annotations", &items.join("\n")))
}

// --- Work Items ---

async fn api_work(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;

    if guard.work_claims.is_empty() {
        return empty("No claimed work items. Use the claim_work tool.");
    }

    let now = chrono::Utc::now().timestamp();
    let items: Vec<String> = guard.work_claims.iter().map(|(id, c)| {
        let status = if c.leased_until < now { "expired" } else { "active" };
        let status_class = if status == "active" { "badge-episodic" } else { "badge-semantic" };
        list_item(&format!(r#"<span class="badge {}">{}</span>
            <code class="text-xs opacity-40">..{}</code>
            <span class="flex-1">{}</span>"#,
            status_class, status,
            id.to_string().get(..6).unwrap_or(""),
            c.claim_type))
    }).collect();

    Html(card("Work Claims", &items.join("\n")))
}

// --- Timeline ---

async fn api_timeline(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();

    if ms.is_empty() {
        return empty("No memories yet");
    }

    let items: Vec<String> = ms.iter().take(50).enumerate().map(|(i, m)| {
        list_item(&format!(r#"<div class="text-xs font-mono opacity-30 w-8 shrink-0">#{}</div>
            {}<span class="flex-1">{}</span>"#,
            i + 1,
            badge(&m.tier.to_string()),
            esc_html(&m.content.chars().take(60).collect::<String>())))
    }).collect();

    Html(card("Memory Timeline", &items.join("\n")))
}
