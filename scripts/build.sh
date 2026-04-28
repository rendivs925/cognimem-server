#!/bin/bash
set -e

echo "Building CogniMem server..."

# Build release
cargo build --release --bin cognimem-server

# Create dist directory
mkdir -p dist

# Copy binary
cp target/release/cognimem-server dist/

echo "Built: dist/cognimem-server"
echo ""
echo "To run:"
echo "  ./dist/cognimem-server --data-path ~/.cognimem-data"