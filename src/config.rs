use clap::Parser;
use cognimem_server::memory::DEFAULT_SLM_MODEL;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "cognimem-server",
    about = "Cognitive memory MCP server for AI coding agents"
)]
pub struct Cli {
    #[arg(
        long,
        default_value = "./cognimem-data",
        help = "Path to RocksDB data directory"
    )]
    pub data_path: String,

    #[arg(long, default_value_t = 300, help = "Decay interval in seconds")]
    pub decay_interval_secs: u64,

    #[arg(
        long,
        default_value_t = 900,
        help = "Consolidation interval in seconds"
    )]
    pub consolidation_interval_secs: u64,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Activation threshold below which memories are pruned"
    )]
    pub prune_threshold: f32,

    #[arg(
        long,
        default_value = "rocksdb",
        help = "Storage backend: 'rocksdb' or 'memory'"
    )]
    pub storage: String,

    #[arg(long, default_value_t = 9090, help = "Port for metrics endpoint")]
    pub metrics_port: u16,

    #[arg(long, hide = true, help = "Run as the shared CogniMem daemon")]
    pub daemon: bool,

    #[arg(long, hide = true, help = "Unix socket path for daemon/client bridge")]
    pub socket_path: Option<String>,

    #[arg(
        long,
        default_value = DEFAULT_SLM_MODEL,
        help = "Default model to use for all SLM-backed operations"
    )]
    pub ollama_model: Option<String>,

    #[arg(
        long,
        default_value = "http://localhost:11434",
        help = "Ollama base URL"
    )]
    pub ollama_url: Option<String>,

    #[arg(long, default_value_t = 37778, help = "Port for capture HTTP endpoint")]
    pub capture_port: u16,

    #[arg(long, default_value_t = 37779, help = "Port for web dashboard HTTP endpoint")]
    pub dashboard_port: u16,

    #[arg(
        long,
        default_value = "redis://localhost:6379",
        help = "Redis connection URL for message broker"
    )]
    pub redis_url: Option<String>,

    #[arg(
        long,
        help = "Unique agent ID for this client (used in broker events)"
    )]
    pub agent_id: Option<String>,

    #[arg(long, default_value = "", help = "Path to project to auto-discover (default: current dir if .rs files exist)")]
    pub project_path: String,

    #[arg(long, help = "Enable TLS with certificate file")]
    pub tls_cert: Option<String>,

    #[arg(long, help = "Enable TLS with private key file")]
    pub tls_key: Option<String>,

    #[arg(long, help = "Password for encrypted TLS key or data encryption")]
    pub password: Option<String>,

    #[arg(long, default_value_t = false, help = "Require authentication for all MCP requests")]
    pub require_auth: bool,
}

impl Cli {
    pub fn resolved_project_path(&self) -> Option<std::path::PathBuf> {
        let p = self.project_path.trim();
        if p.is_empty() || p == "." {
            std::env::current_dir().ok()
        } else {
            Some(std::path::PathBuf::from(p))
        }
    }
}
