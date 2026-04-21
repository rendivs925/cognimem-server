const http = require('http');

const COGNIMEM_URL = process.env.COGNIMEM_URL || 'http://localhost:3000';

function callCogniMem(method, params) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      jsonrpc: '2.0',
      id: Date.now(),
      method,
      params
    });

    const url = new URL('/mcp', COGNIMEM_URL);
    const req = http.request({
      hostname: url.hostname,
      port: url.port || 80,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    }, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function remember(content, projectPath = null, tier = 'sensory') {
  try {
    const scope = projectPath || 'global';
    await callCogniMem('tools/remember', {
      content,
      tier,
      scope
    });
  } catch (e) {
    console.error('Failed to remember:', e.message);
  }
}

module.exports = {
  name: 'cognimem-capture',
  version: '0.1.0',

  async onSessionCreate(session) {
    const projectPath = session.workspace || null;
    await remember(`Session started for project: ${projectPath || 'unknown'}`, projectPath);
  },

  async onSessionEnd(session) {
    await remember('Session ended', session.workspace);
  },

  async onToolExecute(tool) {
    const success = tool.exitCode === 0;
    const content = `Tool '${tool.name}' ${success ? 'succeeded' : 'failed'}`;
    await remember(content, tool.workspace, 'sensory');
  },

  async onMessage(message) {
    if (message.role === 'assistant' && message.content) {
      await remember(message.content.substring(0, 500), message.workspace, 'sensory');
    }
  }
};