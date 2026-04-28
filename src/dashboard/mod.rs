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

pub async fn start_dashboard_server(state: Arc<Mutex<CogniMemState>>, port: u16) {
    let theme = get_theme("onyx");
    let app_state = DashboardState { state, theme };
    let app = Router::new()
        .route("/", get(dashboard_index))
        .route("/api/theme", get(api_theme))
        .route("/api/memories", get(api_memories))
        .route("/api/memories", post(api_remember))
        .route("/api/search", get(api_search))
        .route("/api/graph", get(api_code_graph))
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
        return Html(r#"<div class="empty">No memories yet. Click + Add to create one.</div>"#.to_string());
    }

    let rows: Vec<String> = ms.iter().take(100).map(|m| {
        let id = m.id.to_string();
        let short_id = id.get(..8).unwrap_or(&id);
        let tier_str = m.tier.to_string().to_lowercase();
        let content_escaped = m.content
            .chars().take(80).collect::<String>()
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;");

        format!(r#"<tr data-id="{}" data-tier="{}" data-content="{}" data-activation="{}" data-strength="{}" onclick="viewMemory('{}','{}','{}',{},{})">
            <td><code>{}</code></td>
            <td><span class="badge badge-{}">{}</span></td>
            <td class="text-sm">{}</td>
            <td class="font-mono text-xs">{:.2}</td>
            <td class="font-mono text-xs">{:.2}</td>
        </tr>"#,
            id, tier_str, content_escaped, m.metadata.base_activation, m.metadata.strength,
            id, tier_str, content_escaped.replace('\'', "\\'"), m.metadata.base_activation, m.metadata.strength,
            short_id,
            tier_str, m.tier,
            content_escaped,
            m.metadata.base_activation,
            m.metadata.strength
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
    let tier_filter = params.get("tier").map(|s| s.as_str());
    let limit: usize = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50);

    if query.is_empty() {
        return Html(r#"<div class="empty">Enter a search query</div>"#.to_string());
    }

    let guard = state.state.lock().await;
    let mut results: Vec<_> = guard.graph.get_all_memories().into_iter()
        .filter(|m| {
            let content_match = m.content.to_lowercase().contains(&query.to_lowercase());
            let tier_match = tier_filter.map(|t| m.tier.to_string().to_lowercase() == t).unwrap_or(true);
            content_match && tier_match
        })
        .take(limit)
        .collect();

    results.sort_by(|a, b| b.metadata.base_activation.partial_cmp(&a.metadata.base_activation).unwrap());

    if results.is_empty() {
        return Html(r#"<div class="empty">No results found</div>"#.to_string());
    }

    let items: Vec<String> = results.iter().map(|m| {
        let tier_str = m.tier.to_string().to_lowercase();
        format!(r#"<div class="flex items-center gap-3 px-4 py-3 border-b text-sm" style="border-color:var(--border);cursor:pointer" onclick="viewMemory('{}','{}','{}',{},{})">
            <span class="badge badge-{}">{}</span>
            <span class="flex-1">{}</span>
            <span class="font-mono text-xs opacity-50">{:.2}</span>
        </div>"#,
            m.id, tier_str, m.content.replace('\'', "\\'"), m.metadata.base_activation, m.metadata.strength,
            tier_str, m.tier,
            m.content.chars().take(80).collect::<String>(),
            m.metadata.base_activation
        )
    }).collect();

    Html(format!(r#"<div class="text-xs opacity-40 px-4 py-2">{} results</div>{}"#, results.len(), items.join("\n")))
}

async fn api_remember(
    State(state): State<DashboardState>,
    Json(payload): Json<serde_json::Value>,
) -> (StatusCode, Html<String>) {
    let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let tier_str = payload.get("tier").and_then(|v| v.as_str()).unwrap_or("episodic");
    let tier = match tier_str {
        "sensory" => MemoryTier::Sensory,
        "working" => MemoryTier::Working,
        "semantic" => MemoryTier::Semantic,
        "procedural" => MemoryTier::Procedural,
        _ => MemoryTier::Episodic,
    };

    if content.is_empty() {
        return (StatusCode::BAD_REQUEST, Html("Content required".to_string()));
    }

    let memory = crate::memory::CognitiveMemoryUnit::new(
        content.to_string(),
        tier,
        0.5,
        tier.decay_rate(),
    );

    {
        let mut guard = state.state.lock().await;
        let emb = guard.embedder.embed(&memory.content);
        guard.graph.add_memory(memory.clone());
        guard.graph.set_embedding(memory.id, emb);
        let _ = guard.storage.save(&memory);
    }

    (StatusCode::OK, Html("OK".to_string()))
}

// --- Code Graph ---

async fn api_code_graph(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let files = guard.code_graph.all_files();

    if files.is_empty() {
        return Html(r#"<div class="empty">No code discovered. Run with --project-path</div>"#.to_string());
    }

    let blocks: Vec<String> = files.iter().take(50).map(|f| {
        let name = f.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let nodes = guard.code_graph.get_nodes_in_file(f);
        let items: Vec<String> = nodes.iter().take(20).map(|n| {
            format!(r#"<div class="flex items-center gap-3 px-6 py-2 text-xs opacity-60 border-b" style="border-color:var(--border)">
                <span class="text-[10px] uppercase px-1.5 py-0.5 rounded" style="background:var(--border)">{:?}</span>
                <span>{}</span>
            </div>"#, n.kind, n.name)
        }).collect();
        format!(r#"<div class="px-4 py-3 border-b text-sm font-mono" style="border-color:var(--border)">{}</div>{}"#,
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
        let kind_label = match n.kind {
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
        </div>"#, kind_label, n.name, n.line_start, n.line_end)
    }).collect();

    Html(format!(r#"<div class="py-1">{}"#, blocks.join("\n")))
}

// --- Stats ---

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

    let pct = |n: usize| -> f64 { if total > 0 { n as f64 / total as f64 * 100.0 } else { 0.0 } };

    let recent = ms.iter().take(10).map(|m| {
        let t = m.tier.to_string().to_lowercase();
        format!(r#"<div class="flex items-center gap-3 px-4 py-2.5 border-b text-sm" style="border-color:var(--border)">
            <span class="badge badge-{}">{}</span>
            <span>{}</span>
        </div>"#, t, m.tier, m.content.chars().take(60).collect::<String>())
    }).collect::<Vec<_>>().join("\n");

    Html(format!(r#"
<div class="grid grid-cols-4 gap-4 mb-6">
    <div class="rounded-xl border p-5" style="background:var(--surface);border-color:var(--border)">
        <div class="text-2xl font-bold font-mono" style="color:var(--accent)">{}</div>
        <div class="text-[11px] uppercase tracking-wider opacity-50 mt-2">Total Memories</div>
    </div>
    <div class="rounded-xl border p-5" style="background:var(--surface);border-color:var(--border)">
        <div class="text-2xl font-bold font-mono" style="color:var(--accent)">{}</div>
        <div class="text-[11px] uppercase tracking-wider opacity-50 mt-2">Code Nodes</div>
    </div>
    <div class="rounded-xl border p-5" style="background:var(--surface);border-color:var(--border)">
        <div class="text-2xl font-bold font-mono" style="color:var(--accent)">{:.2}</div>
        <div class="text-[11px] uppercase tracking-wider opacity-50 mt-2">Avg Activation</div>
    </div>
    <div class="rounded-xl border p-5" style="background:var(--surface);border-color:var(--border)">
        <div class="text-2xl font-bold font-mono" style="color:var(--accent)">{:.2}</div>
        <div class="text-[11px] uppercase tracking-wider opacity-50 mt-2">Avg Strength</div>
    </div>
</div>

<div class="rounded-xl border mb-6" style="background:var(--surface);border-color:var(--border)">
    <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Memory Distribution</div>
    <div class="p-5">
        <div class="flex items-center gap-3 mb-2.5">
            <div class="w-20 text-xs">Sensory</div>
            <div class="flex-1 h-2 rounded overflow-hidden" style="background:var(--hover)"><div class="h-full rounded" style="width:{}%;background:#0FC5ED"></div></div>
            <div class="w-8 text-right text-xs font-mono">{}</div>
        </div>
        <div class="flex items-center gap-3 mb-2.5">
            <div class="w-20 text-xs">Working</div>
            <div class="flex-1 h-2 rounded overflow-hidden" style="background:var(--hover)"><div class="h-full rounded" style="width:{}%;background:#a277ff"></div></div>
            <div class="w-8 text-right text-xs font-mono">{}</div>
        </div>
        <div class="flex items-center gap-3 mb-2.5">
            <div class="w-20 text-xs">Episodic</div>
            <div class="flex-1 h-2 rounded overflow-hidden" style="background:var(--hover)"><div class="h-full rounded" style="width:{}%;background:#47FF9C"></div></div>
            <div class="w-8 text-right text-xs font-mono">{}</div>
        </div>
        <div class="flex items-center gap-3 mb-2.5">
            <div class="w-20 text-xs">Semantic</div>
            <div class="flex-1 h-2 rounded overflow-hidden" style="background:var(--hover)"><div class="h-full rounded" style="width:{}%;background:#FFE073"></div></div>
            <div class="w-8 text-right text-xs font-mono">{}</div>
        </div>
        <div class="flex items-center gap-3 mb-2.5">
            <div class="w-20 text-xs">Procedural</div>
            <div class="flex-1 h-2 rounded overflow-hidden" style="background:var(--hover)"><div class="h-full rounded" style="width:{}%;background:#F43F5E"></div></div>
            <div class="w-8 text-right text-xs font-mono">{}</div>
        </div>
    </div>
</div>

<div class="rounded-xl border" style="background:var(--surface);border-color:var(--border)">
    <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Recent Memories</div>
    <div>{}</div>
</div>"#,
        total, guard.code_graph.len(), avg_activation, avg_strength,
        pct(sensory), sensory, pct(working), working, pct(episodic), episodic,
        pct(semantic), semantic, pct(procedural), procedural,
        recent
    ))
}

// --- Skills ---

async fn api_skills(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();
    let skills: Vec<_> = ms.iter().filter(|m| matches!(m.tier, MemoryTier::Procedural)).collect();

    if skills.is_empty() {
        return Html(r#"<div class="empty">No skills learned yet. Skills are created from procedural memories.</div>"#.to_string());
    }

    let items: Vec<String> = skills.iter().map(|s| {
        format!(r#"<div class="flex items-center gap-3 px-4 py-3 border-b text-sm" style="border-color:var(--border)">
            <span class="badge badge-procedural">skill</span>
            <span class="flex-1 font-mono">{}</span>
            <span class="text-xs opacity-40">{:.2}</span>
        </div>"#, s.content.chars().take(60).collect::<String>(), s.metadata.strength)
    }).collect();

    Html(format!(r#"<div class="rounded-xl border" style="background:var(--surface);border-color:var(--border)">
        <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Procedural Skills</div>
        {}
    </div>"#, items.join("\n")))
}

// --- Persona ---

async fn api_persona(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();
    let persona_memories: Vec<_> = ms.iter().filter(|m| m.persona.is_some()).collect();

    if persona_memories.is_empty() {
        return Html(r#"<div class="empty">No persona data extracted yet. Use the extract_persona tool to generate profiles.</div>"#.to_string());
    }

    let items: Vec<String> = persona_memories.iter().map(|m| {
        let domain = m.persona.as_ref().unwrap();
        format!(r#"<div class="flex items-center gap-3 px-4 py-3 border-b text-sm" style="border-color:var(--border)">
            <span class="badge badge-episodic">{:?}</span>
            <span class="flex-1">{}</span>
        </div>"#, domain, m.content.chars().take(80).collect::<String>())
    }).collect();

    Html(format!(r#"<div class="rounded-xl border" style="background:var(--surface);border-color:var(--border)">
        <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Persona Annotations</div>
        {}
    </div>"#, items.join("\n")))
}

// --- Work Items ---

async fn api_work(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;

    if guard.work_claims.is_empty() {
        return Html(r#"<div class="empty">No claimed work items. Use the claim_work tool to claim tasks.</div>"#.to_string());
    }

    let items: Vec<String> = guard.work_claims.iter().map(|(id, claim)| {
        let now = chrono::Utc::now().timestamp();
        let status = if claim.leased_until < now { "expired" } else { "active" };
        format!(r#"<div class="flex items-center gap-3 px-4 py-3 border-b text-sm" style="border-color:var(--border)">
            <span class="badge {}">{}</span>
            <code class="text-xs opacity-40">..{}</code>
            <span class="flex-1">{}</span>
        </div>"#,
            if status == "active" { "badge-episodic" } else { "badge-semantic" },
            status,
            id.to_string().get(..6).unwrap_or(""),
            claim.claim_type)
    }).collect();

    Html(format!(r#"<div class="rounded-xl border" style="background:var(--surface);border-color:var(--border)">
        <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Work Claims</div>
        {}
    </div>"#, items.join("\n")))
}

// --- Timeline ---

async fn api_timeline(State(state): State<DashboardState>) -> Html<String> {
    let guard = state.state.lock().await;
    let ms = guard.graph.get_all_memories();

    if ms.is_empty() {
        return Html(r#"<div class="empty">No memories yet</div>"#.to_string());
    }

    let items: Vec<String> = ms.iter().take(50).enumerate().map(|(i, m)| {
        let t = m.tier.to_string().to_lowercase();
        format!(r#"<div class="flex gap-4 px-4 py-3 border-b text-sm" style="border-color:var(--border)">
            <div class="text-xs font-mono opacity-30 w-8 shrink-0">#{}</div>
            <span class="badge badge-{} shrink-0">{}</span>
            <span class="flex-1">{}</span>
        </div>"#, i + 1, t, m.tier, m.content.chars().take(60).collect::<String>())
    }).collect();

    Html(format!(r#"<div class="rounded-xl border" style="background:var(--surface);border-color:var(--border)">
        <div class="px-5 py-4 border-b font-semibold text-sm" style="border-color:var(--border)">Memory Timeline</div>
        {}
    </div>"#, items.join("\n")))
}
