# CogniMem Server

CogniMem is a local MCP server in Rust that gives coding agents a persistent cognitive memory layer.

It supports:
- tiered memories: `sensory`, `working`, `episodic`, `semantic`, `procedural`
- memory storage and recall
- associations between memories
- multi-hop spreading activation during recall
- background decay and pruning
- reflection and promotion between tiers
- RocksDB or in-memory storage
- Prometheus-style metrics over HTTP

## Tools

The MCP server exposes these tools:

- `remember`: store a memory
- `recall`: search and retrieve memories
- `associate`: link two memories with a strength value
- `forget`: soft-delete or hard-delete a memory
- `reflect`: run a decay/prune/promote cycle

## Build

```bash
cargo build --release
```

The binary will be available at:

```bash
target/release/cognimem-server
```

If you install it with Cargo:

```bash
cargo install --path .
```

it will typically be available as:

```bash
~/.cargo/bin/cognimem-server
```

## Run

Default run:

```bash
cognimem-server
```

Example with explicit options:

```bash
cognimem-server \
  --data-path /path/to/cognimem-data \
  --decay-interval-secs 300 \
  --prune-threshold 0.01 \
  --metrics-port 9090
```

### CLI Options

- `--data-path <PATH>`: RocksDB storage path
- `--decay-interval-secs <SECONDS>`: background decay interval
- `--prune-threshold <FLOAT>`: prune memories below this activation threshold
- `--storage <rocksdb|memory>`: select backend
- `--metrics-port <PORT>`: metrics HTTP port

## OpenCode MCP Setup

OpenCode uses the `mcp` section in `opencode.json`.

Adjust the example paths below to match your machine.

Example config using the installed binary name:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cognimem": {
      "type": "local",
      "command": [
        "cognimem-server",
        "--data-path",
        "/path/to/cognimem-data",
        "--decay-interval-secs",
        "300",
        "--metrics-port",
        "9090"
      ],
      "enabled": true,
      "timeout": 5000
    }
  }
}
```

If OpenCode cannot resolve the binary from `PATH`, use the absolute path instead:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cognimem": {
      "type": "local",
      "command": [
        "/absolute/path/to/cognimem-server",
        "--data-path",
        "/path/to/cognimem-data",
        "--decay-interval-secs",
        "300",
        "--metrics-port",
        "9090"
      ],
      "enabled": true,
      "timeout": 5000
    }
  }
}
```

## Verify OpenCode Integration

Check that OpenCode sees the server:

```bash
opencode mcp list
```

Expected result should include:

```text
cognimem connected
```

Try a real tool call in OpenCode:

```text
use cognimem to remember that the project uses RocksDB
```

Then:

```text
use cognimem to recall what storage the project uses
```

## Metrics

Metrics are exposed over HTTP on the configured `--metrics-port`.

Example:

```bash
curl http://127.0.0.1:9090
```

Sample output:

```text
# HELP cognimem_memory_count Total number of memories
# TYPE cognimem_memory_count gauge
cognimem_memory_count 1
```

Available metrics:

- `cognimem_memory_count`
- `cognimem_remember_total`
- `cognimem_recall_total`
- `cognimem_forget_total`
- `cognimem_reflect_total`
- `cognimem_prune_total`
- `cognimem_associate_total`

## Memory Model

### Tiers

- `sensory`: fast decay, small capacity
- `working`: moderate decay, bounded capacity
- `episodic`: default memory tier
- `semantic`: durable generalized knowledge
- `procedural`: most durable patterns and routines

### Implemented Behavior

- ACT-R-style activation updates on recall
- multi-hop spreading activation during recall
- association strengths between memories
- tier capacity enforcement for `sensory` and `working`
- automatic eviction of the lowest-activation memory when a bounded tier is full
- promotion rules during reflection:
  - `episodic -> semantic` above `0.8`
  - `semantic -> procedural` above `0.9`

## Storage Backends

### RocksDB

Default backend:

```bash
cognimem-server --storage rocksdb
```

### In-memory

Ephemeral backend for testing:

```bash
cognimem-server --storage memory
```

## Development

Run checks:

```bash
cargo clippy -- -D warnings
cargo test
cargo build --release
```

## Notes

- MCP communication is over stdio
- metrics are separate HTTP traffic on `--metrics-port`
- using `cognimem-server` directly in OpenCode config is fine if it is on `PATH`
