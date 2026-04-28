# CogniMem OpenCode Plugin

This plugin integrates CogniMem memory system with OpenCode.

## Quick Start

1. **Install CogniMem Server:**
   ```bash
   cargo install cognimem-server
   ```

2. **For Project-Level Plugin:**
   Copy this folder to `.opencode/plugins/cognimem/` in your project.

3. **For Global Plugin:**
   Copy to `~/.config/opencode/plugins/cognimem/`

4. **Add to Your opencode.json:**
   ```json
   {
     "plugin": ["cognimem-opencode-plugin"]
   }
   ```

## Environment Variables

- `COGNIMEM_HOST` - Server host (default: localhost)
- `COGNIMEM_PORT` - Server port (default: 37778)

## Features

### Event Hooks

- `session.created` - Initialize session context
- `session.idle` - Trigger consolidation
- `experimental.session.compacting` - Inject memory context

### Custom Tools

- `cognimem_recall` - Recall memories
- `cognimem_inject` - Inject new memory
- `cognimem_search` - Search codebase
- `cognimem_consolidate` - Run consolidation
- `cognimem_dream` - Trigger dreaming
- `cognimem_discover` - Discover code graph
- `cognimem_imagine` - Imagine scenarios

## MCP Fallback

If using MCP tools instead of the plugin:

```json
{
  "mcp": {
    "cognimem": {
      "type": "local",
      "command": ["cognimem-server"]
    }
  }
}
```