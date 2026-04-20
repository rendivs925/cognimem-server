use clap::Parser;

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
        default_value = "qwen2.5-coder:3b",
        help = "Ollama model to use for SLM operations (e.g., qwen2.5-coder:3b)"
    )]
    pub ollama_model: Option<String>,

    #[arg(
        long,
        default_value = "http://localhost:11434",
        help = "Ollama base URL"
    )]
    pub ollama_url: Option<String>,
}
