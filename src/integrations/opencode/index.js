const http = require('http');

const COGNIMEM_URL = process.env.COGNIMEM_CAPTURE_URL || 'http://localhost:37778';
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];

function postEvent(event) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(event);
    const url = new URL('/capture/events', COGNIMEM_URL);
    const req = http.request({
      hostname: url.hostname,
      port: url.port || 80,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
      },
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function sendEvent(event) {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      await postEvent(event);
      return;
    } catch (e) {
      if (attempt < MAX_RETRIES - 1) {
        await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
      } else {
        console.error(`[cognimem-capture] Failed after ${MAX_RETRIES} attempts:`, e.message);
      }
    }
  }
}

async function sendBatch(events) {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const data = JSON.stringify(events);
      const url = new URL('/capture/events/batch', COGNIMEM_URL);
      await new Promise((resolve, reject) => {
        const req = http.request({
          hostname: url.hostname,
          port: url.port || 80,
          path: url.pathname,
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(data),
          },
        }, (res) => {
          let body = '';
          res.on('data', (chunk) => { body += chunk; });
          res.on('end', resolve);
        });
        req.on('error', reject);
        req.write(data);
        req.end();
      });
      return;
    } catch (e) {
      if (attempt < MAX_RETRIES - 1) {
        await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
      } else {
        console.error(`[cognimem-capture] Batch failed after ${MAX_RETRIES} attempts:`, e.message);
      }
    }
  }
}

function makeEvent(eventType, data = {}) {
  return {
    event_type: eventType,
    timestamp: Math.floor(Date.now() / 1000),
    session_id: data.session_id || null,
    project_path: data.project_path || null,
    agent_id: data.agent_id || null,
    source: 'opencode',
    tool_name: data.tool_name || null,
    tool_input: data.tool_input || null,
    tool_output: data.tool_output || null,
    file_path: data.file_path || null,
    content: data.content || null,
    success: data.success !== undefined ? data.success : null,
    duration_ms: data.duration_ms || null,
    metadata: data.metadata || {},
  };
}

module.exports = {
  name: 'cognimem-capture',
  version: '0.2.0',

  async onSessionCreated(session) {
    await sendEvent(makeEvent('session_created', {
      session_id: session.id,
      project_path: session.workspace || null,
    }));
  },

  async onSessionDeleted(session) {
    await sendEvent(makeEvent('session_deleted', {
      session_id: session.id,
      project_path: session.workspace || null,
    }));
  },

  async onSessionIdle(session) {
    await sendEvent(makeEvent('session_idle', {
      session_id: session.id,
      project_path: session.workspace || null,
      content: session.idleReason || null,
    }));
  },

  async onToolExecuteBefore(tool) {
    await sendEvent(makeEvent('tool_execute_before', {
      session_id: tool.sessionId || null,
      project_path: tool.workspace || null,
      tool_name: tool.name,
      tool_input: tool.input || null,
      file_path: tool.filePath || null,
    }));
  },

  async onToolExecuteAfter(tool) {
    await sendEvent(makeEvent('tool_execute_after', {
      session_id: tool.sessionId || null,
      project_path: tool.workspace || null,
      tool_name: tool.name,
      tool_output: tool.output || null,
      file_path: tool.filePath || null,
      content: typeof tool.output === 'string' ? tool.output.substring(0, 2000) : null,
      success: tool.exitCode === 0,
      duration_ms: tool.durationMs || null,
    }));
  },

  async onFileEdited(file) {
    await sendEvent(makeEvent('file_edited', {
      session_id: file.sessionId || null,
      project_path: file.workspace || null,
      file_path: file.path,
      content: file.content ? file.content.substring(0, 2000) : null,
    }));
  },

  async onMessageUpdated(message) {
    if (message.role === 'assistant' && message.content) {
      await sendEvent(makeEvent('message_updated', {
        session_id: message.sessionId || null,
        project_path: message.workspace || null,
        content: message.content.substring(0, 2000),
      }));
    }
  },

  async onPermissionAsked(permission) {
    await sendEvent(makeEvent('permission_asked', {
      session_id: permission.sessionId || null,
      project_path: permission.workspace || null,
      tool_name: permission.toolName || null,
      content: permission.question || null,
    }));
  },
};
