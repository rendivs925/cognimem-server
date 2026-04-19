mod config;
mod memory;
mod metrics;

use clap::Parser;
use config::Cli;
use memory::{
    AssociateArgs, AssociateResult, CognitiveMemoryUnit, ForgetArgs, ForgetResult, InMemoryStore,
    MemoryStore, MemorySummary, RecallArgs, RecallResult, ReflectArgs, ReflectResult, RememberArgs,
    RememberResult, RocksDbStore,
};
use metrics::{
    inc_associate, inc_forget, inc_prune, inc_recall, inc_remember, inc_reflect,
    set_memory_count,
};
use rmcp::{
    ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ListResourcesResult, PaginatedRequestParams,
        ReadResourceRequestParams, ReadResourceResult, ResourceContents, Resource, RawResource,
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
    graph: memory::MemoryGraph,
    storage: Box<dyn MemoryStore>,
}

impl CogniMemState {
    fn new(storage: Box<dyn MemoryStore>) -> Self {
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
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let _ = context;
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
                    RawResource::new(
                        format!("memory://{}/{}", m.tier, m.id),
                        &m.content,
                    )
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

        let json = serde_json::to_string(memory).unwrap_or_default();
        Ok(ReadResourceResult::new(vec![ResourceContents::text(json, uri.clone())]))
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

        if let Some(limit) = tier.capacity() {
            while guard.graph.count_by_tier(tier) >= limit {
                if let Some(evict_id) = guard.graph.find_lowest_activation_in_tier(tier) {
                    if let Some(evicted) = guard.graph.remove_memory(&evict_id)
                        && let Err(e) = guard.storage.delete(&evicted.id)
                    {
                        error!("Failed to delete evicted memory {}: {e}", evicted.id);
                    }
                } else {
                    break;
                }
            }
        }

        guard.graph.add_memory(memory.clone());
        if let Err(e) = guard.storage.save(&memory) {
            error!("Failed to persist memory {memory_id}: {e}");
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

        let query = args.query.to_lowercase();
        let limit = args.limit.unwrap_or(5);
        let min_activation = args.min_activation.unwrap_or(0.0);
        let now = chrono::Utc::now().timestamp();

        let mut guard = self.state.lock().await;

        let mut results: Vec<&CognitiveMemoryUnit> = match args.tier {
            Some(tier) => guard.graph.get_by_tier(tier),
            None => guard.graph.get_all_memories(),
        }
        .into_iter()
        .filter(|m| matches_query(&m.content, &query) && m.metadata.base_activation >= min_activation)
        .collect();

        let direct_ids: Vec<uuid::Uuid> = results.iter().map(|m| m.id).collect();
        expand_with_associations(&direct_ids, &mut results, &guard.graph);

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

        inc_forget();

        if args.hard_delete.unwrap_or(false) {
            if let Some(removed) = guard.graph.remove_memory(&args.memory_id)
                && let Err(e) = guard.storage.delete(&removed.id)
            {
                error!("Failed to delete memory {}: {e}", removed.id);
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

        inc_reflect();
        inc_prune(pruned as u64);
        set_memory_count(guard.graph.len() as u64);

        Ok(success_json(&ReflectResult::new(pruned, promoted, total)))
    }
}

fn matches_query(content: &str, query: &str) -> bool {
    let lower = content.to_lowercase();
    lower.contains(query) || query.split_whitespace().any(|w| lower.contains(w))
}

fn expand_with_associations<'a>(direct_ids: &[uuid::Uuid], results: &mut Vec<&'a CognitiveMemoryUnit>, graph: &'a memory::MemoryGraph) {
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
        memory::apply_decay_to_all(&mut guard.graph);
        let removed = memory::prune_below_threshold(&mut guard.graph, prune_threshold);
        if removed > 0 {
            tracing::info!("Pruned {removed} memories below activation threshold");
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
    metrics::init();

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
    let state = Arc::new(Mutex::new(CogniMemState::new(storage)));
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
                    let _ = handle_metrics(stream).await;
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
        + &metrics::encode();

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
