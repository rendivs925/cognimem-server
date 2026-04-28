import type { Plugin, tool } from "@opencode-ai/plugin";

const COGNIMEM_HOST = process.env.COGNIMEM_HOST || "localhost";
const COGNIMEM_PORT = process.env.COGNIMEM_PORT || "37778";

async function callCogniMem(method: string, args: Record<string, unknown> = {}) {
  const response = await fetch(`http://${COGNIMEM_HOST}:${COGNIMEM_PORT}/mcp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "tools/call",
      params: { name: method, arguments: args },
      id: 1,
    }),
  });
  return response.json();
}

export const CogniMemPlugin: Plugin = async ({ project, directory }) => {
  console.log("[CogniMem] Plugin initialized for:", directory);

  return {
    "session.created": async ({ session }) => {
      console.log("[CogniMem] Session created, project:", project?.path);
    },

    "session.idle": async ({ session }) => {
      console.log("[CogniMem] Session idle, consolidating...");
      try {
        await callCogniMem("consolidate", { tier: "working" });
      } catch (e) {
        console.error("[CogniMem] Consolidate failed:", e);
      }
    },

    "experimental.session.compacting": async ({ session, output }) => {
      console.log("[CogniMem] Compacting session, fetching context...");
      try {
        const result = await callCogniMem("recall", {
          query: `session ${session.id}`,
          limit: 3,
          min_activation: 0.5,
        });
        if (result.result?.content) {
          output.context.push(`## CogniMem Context\n${result.result.content.join("\n")}`);
        }
      } catch (e) {
        console.error("[CogniMem] Context fetch failed:", e);
      }
    },

    "tool.execute.after": async ({ input, output }) => {
      if (input.tool === "bash" || input.tool === "edit") {
        console.log("[CogniMem] Tool executed:", input.tool);
      }
    },

    tool: {
      cognimem_recall: tool({
        description: "Recall relevant memories from CogniMem",
        args: {
          query: tool.schema.string(),
          limit: tool.schema.number().default(5),
          min_activation: tool.schema.number().default(0.3),
        },
        async execute({ query, limit = 5, min_activation = 0.3 }) {
          const result = await callCogniMem("recall", { query, limit, min_activation });
          return result.result?.content || "No memories found";
        },
      }),

      cognimem_inject: tool({
        description: "Inject a new memory into CogniMem",
        args: {
          content: tool.schema.string(),
          tier: tool.schema.enum("sensory", "working", "episodic", "semantic", "procedural").default("working"),
          associations: tool.schema.array(tool.schema.string()).optional(),
        },
        async execute({ content, tier = "working", associations = [] }) {
          const result = await callCogniMem("inject", { content, tier, associations });
          return result.result?.id || "Injected";
        },
      }),

      cognimem_search: tool({
        description: "Search memories by query",
        args: {
          query: tool.schema.string(),
          tier: tool.schema.enum("sensory", "working", "episodic", "semantic", "procedural").optional(),
        },
        async execute({ query, tier }) {
          const result = await callCogniMem("search_codebase", { query, tier });
          return result.result?.memories || "No results";
        },
      }),

      cognimem_consolidate: tool({
        description: "Trigger memory consolidation",
        args: {
          tier: tool.schema.enum("sensory", "working", "episodic", "semantic", "procedural").default("working"),
        },
        async execute({ tier = "working" }) {
          const result = await callCogniMem("consolidate", { tier });
          return result.result || "Consolidated";
        },
      }),

      cognimem_dream: tool({
        description: "Trigger dreaming/consolidation cycle",
        args: {},
        async execute() {
          const result = await callCogniMem("dream", {});
          return result.result || "Dream complete";
        },
      }),

      cognimem_discover: tool({
        description: "Discover code graph in a project",
        args: {
          project_path: tool.schema.string(),
        },
        async execute({ project_path }) {
          const result = await callCogniMem("discover_project", { project_path });
          return result.result?.node_count + " nodes discovered" || "Done";
        },
      }),

      cognimem_imagine: tool({
        description: "Imagine a hypothetical scenario using memories",
        args: {
          scenario: tool.schema.string(),
        },
        async execute({ scenario }) {
          const result = await callCogniMem("imagine", { scenario });
          return result.result?.content || "No imagination result";
        },
      }),
    },
  };
};