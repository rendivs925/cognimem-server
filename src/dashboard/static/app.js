let allMemories = [];
let currentView = 'memories';

const views = {
    memories: { title: 'Memories', subtitle: 'Browse and manage cognitive memory units', toolbar: true },
    codegraph: { title: 'Code Graph', subtitle: 'Explore source code structure', toolbar: false },
    search: { title: 'Search', subtitle: 'Full-text search across all memories', toolbar: false },
    stats: { title: 'Statistics', subtitle: 'System metrics and memory distribution', toolbar: false },
    skills: { title: 'Skills', subtitle: 'Learned procedural skills', toolbar: false },
    persona: { title: 'Persona', subtitle: 'Extracted user persona profiles', toolbar: false },
    work: { title: 'Work Items', subtitle: 'Claimed and available work tasks', toolbar: false },
    timeline: { title: 'Timeline', subtitle: 'Memory activity timeline', toolbar: false },
};

function switchView(view) {
    currentView = view;
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`[data-view="${view}"]`)?.classList.add('active');
    const v = views[view];
    document.getElementById('page-title').textContent = v.title;
    document.getElementById('page-subtitle').textContent = v.subtitle;
    const tb = document.getElementById('toolbar');
    tb.innerHTML = '';
    if (v.toolbar) {
        tb.innerHTML = `<input type="text" id="search-input" class="search-input" placeholder="Search memories..." onkeyup="searchMemories(this.value)" style="flex:1;max-width:360px;background:var(--hover);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;color:var(--fg);font-size:14px;font-family:Outfit,sans-serif;transition:border-color .15s">
<button onclick="showAddModal()" class="px-4 py-2 rounded-lg text-sm font-semibold border-0 cursor-pointer" style="background:var(--accent);color:var(--bg)">+ Add</button>`;
    }
    loadView(view);
}

function showAddModal() {
    document.getElementById('add-modal').classList.remove('hidden');
}

function loadView(view) {
    const c = document.getElementById('content');
    switch (view) {
        case 'memories': fetchMemories(); break;
        case 'codegraph': loadCodeGraph(); break;
        case 'search': showSearchUI(); break;
        case 'stats': fetchStats(); break;
        case 'skills': fetchSkills(); break;
        case 'persona': fetchPersona(); break;
        case 'work': fetchWork(); break;
        case 'timeline': fetchTimeline(); break;
    }
}

async function fetchMemories() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/memories');
        const html = await res.text();
        c.innerHTML = html;
        // Store memories for client-side search
        const rows = c.querySelectorAll('tr[data-id]');
        allMemories = Array.from(rows).map(r => ({
            id: r.dataset.id,
            tier: r.dataset.tier,
            content: r.dataset.content,
            activation: parseFloat(r.dataset.activation),
            strength: parseFloat(r.dataset.strength),
        }));
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load memories</div>';
    }
}

function searchMemories(query) {
    if (!query || !allMemories.length) {
        document.querySelectorAll('tr[data-id]').forEach(r => r.style.display = '');
        return;
    }
    const q = query.toLowerCase();
    document.querySelectorAll('tr[data-id]').forEach(r => {
        const match = r.dataset.content.toLowerCase().includes(q) || r.dataset.tier.toLowerCase().includes(q);
        r.style.display = match ? '' : 'none';
    });
}

async function loadCodeGraph() {
    const c = document.getElementById('content');
    c.innerHTML = `<div class="flex gap-0 min-h-[400px]">
        <div class="w-[280px] flex-shrink-0 border-r overflow-y-auto max-h-[600px] p-3" style="border-color:var(--border)" id="tree-panel">
            <div class="text-[11px] uppercase tracking-wider opacity-40 px-2.5 pb-3">Files</div>
            <div id="file-tree"></div>
        </div>
        <div class="flex-1 overflow-y-auto max-h-[600px]" id="node-panel">
            <div class="empty">Select a file to view its nodes</div>
        </div>
    </div>`;
    try {
        const res = await fetch('/api/graph/nodes');
        const tree = await res.json();
        renderTree(tree, document.getElementById('file-tree'));
    } catch (e) {
        document.getElementById('file-tree').innerHTML = '<div class="empty">No code graph data</div>';
    }
}

function renderTree(entries, parent) {
    entries.forEach(entry => {
        const div = document.createElement('div');
        if (entry.is_dir) {
            div.className = 'tree-item';
            div.innerHTML = `<div class="tree-folder" onclick="toggleFolder(this)">
                <span class="tree-arrow">&#9654;</span>
                <span class="tree-folder-icon">&#128193;</span>
                <span>${entry.name}</span>
            </div>`;
            const children = document.createElement('div');
            children.className = 'tree-children';
            children.style.display = 'none';
            if (entry.children) renderTree(entry.children, children);
            div.appendChild(children);
        } else {
            div.className = 'tree-item';
            div.innerHTML = `<div class="tree-file" onclick="loadFileNodes('${entry.path}')">
                <span class="tree-file-icon">&#128196;</span>
                <span>${entry.name}</span>
            </div>`;
        }
        parent.appendChild(div);
    });
}

function toggleFolder(el) {
    const children = el.parentElement.querySelector('.tree-children');
    const arrow = el.querySelector('.tree-arrow');
    if (children.style.display === 'block') {
        children.style.display = 'none';
        arrow.classList.remove('open');
    } else {
        children.style.display = 'block';
        arrow.classList.add('open');
    }
}

async function loadFileNodes(path) {
    const panel = document.getElementById('node-panel');
    panel.innerHTML = '<div class="px-4 py-3 text-sm opacity-50">Loading...</div>';
    try {
        const res = await fetch(`/api/graph/file/${encodeURIComponent(path)}`);
        const html = await res.text();
        panel.innerHTML = html || '<div class="empty">No nodes</div>';
    } catch (e) {
        panel.innerHTML = '<div class="empty">Error loading nodes</div>';
    }
}

function showSearchUI() {
    const c = document.getElementById('content');
    c.innerHTML = `<div class="mb-4">
        <input type="text" id="search-query" class="w-full rounded-lg border px-4 py-3 text-sm" style="background:var(--hover);border-color:var(--border);color:var(--fg)" placeholder="Search memories..." onkeydown="if(event.key==='Enter') doSearch()">
    </div>
    <div class="flex gap-2 mb-4">
        <select id="search-tier" class="rounded-lg border px-3 py-2 text-sm" style="background:var(--hover);border-color:var(--border);color:var(--fg)">
            <option value="">All Tiers</option>
            <option value="sensory">Sensory</option>
            <option value="working">Working</option>
            <option value="episodic">Episodic</option>
            <option value="semantic">Semantic</option>
            <option value="procedural">Procedural</option>
        </select>
        <button onclick="doSearch()" class="px-4 py-2 rounded-lg text-sm font-semibold border-0 cursor-pointer" style="background:var(--accent);color:var(--bg)">Search</button>
    </div>
    <div id="search-results"></div>`;
}

async function doSearch() {
    const query = document.getElementById('search-query').value;
    const tier = document.getElementById('search-tier').value;
    const r = document.getElementById('search-results');
    if (!query) return;
    r.innerHTML = '<div class="py-8 text-center opacity-50">Searching...</div>';
    try {
        const params = new URLSearchParams({ query, limit: 50 });
        if (tier) params.set('tier', tier);
        const res = await fetch(`/api/search?${params}`);
        const html = await res.text();
        r.innerHTML = html;
    } catch (e) {
        r.innerHTML = '<div class="empty">Search failed</div>';
    }
}

async function fetchStats() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/stats');
        c.innerHTML = await res.text();
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load stats</div>';
    }
}

async function fetchSkills() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/skills');
        c.innerHTML = await res.text();
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load skills</div>';
    }
}

async function fetchPersona() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/persona');
        c.innerHTML = await res.text();
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load persona</div>';
    }
}

async function fetchWork() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/work');
        c.innerHTML = await res.text();
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load work items</div>';
    }
}

async function fetchTimeline() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/timeline');
        c.innerHTML = await res.text();
    } catch (e) {
        c.innerHTML = '<div class="empty">Failed to load timeline</div>';
    }
}

async function addMemory() {
    const content = document.getElementById('new-memory-content').value;
    const tier = document.getElementById('new-memory-tier').value;
    if (!content) return;
    try {
        await fetch('/api/memories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, tier })
        });
        document.getElementById('new-memory-content').value = '';
        document.getElementById('add-modal').classList.add('hidden');
        if (currentView === 'memories') fetchMemories();
    } catch (e) {}
}

function viewMemory(id, tier, content, activation, strength) {
    document.getElementById('modal-content').innerHTML = `
        <div class="flex items-center gap-3 mb-4">
            <span class="badge badge-${tier}">${tier}</span>
            <code class="text-xs opacity-60">${id}</code>
        </div>
        <div class="text-sm leading-relaxed mb-4">${content}</div>
        <div class="flex gap-6 text-xs opacity-60">
            <span>Activation: ${activation.toFixed(2)}</span>
            <span>Strength: ${strength.toFixed(2)}</span>
        </div>
        <button onclick="closeModal()" class="mt-4 px-4 py-2 rounded-lg text-sm border cursor-pointer" style="border-color:var(--border);color:var(--fg)">Close</button>
    `;
    document.getElementById('modal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('modal').classList.add('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    switchView('memories');
});
