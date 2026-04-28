#!/bin/bash
set -e

# Install CogniMem to ~/.local/bin and config

 mkdir -p ~/.local/bin
 mkdir -p ~/.cognimem-server

# Build
cargo build --release --bin cognimem-server

# Install binary
cp target/release/cognimem-server ~/.local/bin/

# Create default config for OpenCode
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/cognimem.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cognimem": {
      "type": "local",
      "command": ["cognimem-server", "--data-path", "~/.cognimem-server/cognimem-data"],
      "enabled": true
    }
  }
}
EOF

echo "Installed to ~/.local/bin/cognimem-server"
echo "Config written to ~/.config/opencode/cognimem.json"
echo ""
echo "Restart OpenCode to use CogniMem"