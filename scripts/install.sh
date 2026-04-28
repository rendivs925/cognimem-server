#!/bin/bash
set -e

# CogniMem Installation Script
# Supports: Direct install, Homebrew tap, Docker
# Usage: ./install.sh [--path ~/.local] [--brew] [--docker]

INSTALL_PATH="${INSTALL_PATH:-$HOME/.local/bin}"
DATA_PATH="${DATA_PATH:-$HOME/.cognimem-server}"
MODE="direct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path) INSTALL_PATH="$2"; shift 2 ;;
    --brew) MODE="brew"; shift ;;
    --docker) MODE="docker"; shift ;;
    *) shift ;;
  esac
done

install_direct() {
  echo "Installing CogniMem to $INSTALL_PATH..."

  mkdir -p "$INSTALL_PATH"
  mkdir -p "$DATA_PATH"

  echo "Building..."
  cargo build --release --bin cognimem-server

  echo "Installing binary..."
  cp target/release/cognimem-server "$INSTALL_PATH/"

  mkdir -p ~/.config/opencode
  cat > ~/.config/opencode/cognimem.json << EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cognimem": {
      "type": "local",
      "command": ["cognimem-server", "--data-path", "$DATA_PATH"],
      "enabled": true
    }
  }
}
EOF

  echo "Installed to $INSTALL_PATH/cognimem-server"
  echo "Config written to ~/.config/opencode/cognimem.json"
  echo ""
  echo "Restart OpenCode to use CogniMem"
}

install_homebrew() {
  local TAP_NAME="cognimem/home-cognimem"
  echo "Installing via Homebrew tap: $TAP_NAME"

  if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh"
    exit 1
  fi

  brew tap $TAP_NAME
  brew install cognimem-server

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

  echo "Restart OpenCode to use CogniMem"
}

install_docker() {
  echo "Running CogniMem via Docker..."
  docker run -d --name cognimem-server \
    -p 37778:37778 \
    -v "$DATA_PATH:/data" \
    cognimem/server:latest

  mkdir -p ~/.config/opencode
  cat > ~/.config/opencode/cognimem.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cognimem": {
      "type": "stdio",
      "command": ["docker", "exec", "cognimem-server", "cognimem-server"],
      "enabled": true
    }
  }
}
EOF

  echo "CogniMem running at port 37778"
  echo "Restart OpenCode to use CogniMem"
}

case "$MODE" in
  brew) install_homebrew ;;
  docker) install_docker ;;
  *) install_direct ;;
esac