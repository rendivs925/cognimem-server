let allMemories = [];
let currentView = 'memories';
let selectedTreeNode = null;

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
    const titleEl = document.getElementById('page-title');
    const subtitleEl = document.getElementById('page-subtitle');
    titleEl.style.opacity = '0';
    titleEl.style.transform = 'translateY(4px)';
    subtitleEl.style.opacity = '0';
    setTimeout(() => {
        titleEl.textContent = v.title;
        subtitleEl.textContent = v.subtitle;
        titleEl.style.transition = 'all 0.2s ease';
        subtitleEl.style.transition = 'all 0.2s ease';
        titleEl.style.opacity = '1';
        titleEl.style.transform = 'translateY(0)';
        subtitleEl.style.opacity = '1';
    }, 50);
    const tb = document.getElementById('toolbar');
    tb.innerHTML = '';
    if (v.toolbar) {
        const input = document.createElement('input');
        input.type = 'text';
        input.id = 'search-input';
        input.placeholder = 'Search memories...';
        input.className = 'animate-fade-in';
        input.style.cssText = 'flex:1;max-width:360px;background:var(--hover);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;color:var(--fg);font-size:14px;font-family:Outfit,sans-serif';
        input.onkeyup = () => searchMemories(input.value);
        tb.appendChild(input);
        const btn = document.createElement('button');
        btn.textContent = '+ Add';
        btn.className = 'px-4 py-2 rounded-lg text-sm font-semibold border-0 cursor-pointer btn animate-fade-in';
        btn.style.cssText = 'background:var(--accent);color:var(--bg)';
        btn.onclick = showAddModal;
        tb.appendChild(btn);
    }
    loadView(view);
}

function showLoading(container) {
    container.innerHTML = '<div class="space-y-3 p-4"><div class="skeleton h-10 w-full"></div><div class="skeleton h-10 w-3/4"></div><div class="skeleton h-10 w-full"></div><div class="skeleton h-10 w-1/2"></div></div>';
}

function applyStagger(container) {
    const children = container.children;
    if (children.length > 0 && children.length <= 20) {
        container.classList.add('animate-stagger');
    }
}

function showAddModal() {
    document.getElementById('add-modal').classList.remove('hidden');
}

function loadView(view) {
    const c = document.getElementById('content');
    showLoading(c);
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
        applyStagger(c.querySelector('tbody') || c);
        const rows = c.querySelectorAll('tr[data-id]');
        allMemories = Array.from(rows).map(r => ({
            id: r.dataset.id,
            tier: r.dataset.tier,
            content: r.dataset.content,
            activation: parseFloat(r.dataset.activation),
            strength: parseFloat(r.dataset.strength),
        }));
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load memories</div>';
    }
}

function searchMemories(query) {
    if (!query || !allMemories.length) {
        document.querySelectorAll('tr[data-id]').forEach(r => { r.style.display = ''; });
        return;
    }
    const q = query.toLowerCase();
    document.querySelectorAll('tr[data-id]').forEach(r => {
        const match = r.dataset.content.toLowerCase().includes(q) || r.dataset.tier.toLowerCase().includes(q);
        r.style.display = match ? '' : 'none';
        if (match) r.classList.add('animate-fade-in');
    });
}

async function loadCodeGraph() {
    const c = document.getElementById('content');
    c.innerHTML = `
    <div class="flex gap-0 rounded-xl border overflow-hidden min-h-[450px]" style="background:var(--surface);border-color:var(--border)">
        <div class="w-[280px] flex-shrink-0 border-r overflow-y-auto" style="border-color:var(--border)" id="tree-panel">
            <div class="text-[11px] uppercase tracking-wider opacity-40 px-4 py-3 border-b select-none" style="border-color:var(--border)">Files</div>
            <div id="file-tree" class="pt-1"></div>
        </div>
        <div class="flex-1 overflow-y-auto" id="node-panel">
            <div class="flex items-center justify-center h-full text-sm opacity-30 select-none">Select a file to view its nodes</div>
        </div>
    </div>`;
    try {
        const res = await fetch('/api/graph/nodes');
        const tree = await res.json();
        const ft = document.getElementById('file-tree');
        renderTree(tree, ft);
    } catch (e) {
        document.getElementById('file-tree').innerHTML = '<div class="empty animate-fade-in">No code graph data</div>';
    }
}

function renderTree(entries, parent) {
    entries.forEach((entry, i) => {
        const div = document.createElement('div');
        div.style.animationDelay = `${i * 20}ms`;
        div.className = 'animate-fade-in';
        if (entry.is_dir) {
            div.innerHTML = `<div class="tree-folder" onclick="toggleFolder(this)">
                <span class="tree-arrow">&#9654;</span>
                <span class="opacity-60 text-xs">&#128193;</span>
                <span class="text-sm">${escHtml(entry.name)}</span>
            </div>`;
            const children = document.createElement('div');
            children.className = 'tree-children';
            if (entry.children) renderTree(entry.children, children);
            div.appendChild(children);
        } else {
            div.innerHTML = `<div class="tree-file flex items-center gap-2 px-2 py-1 rounded" onclick="selectFileNode(this, '${escHtml(entry.path)}')">
                <span class="opacity-50 text-xs">&#128196;</span>
                <span class="text-sm">${escHtml(entry.name)}</span>
            </div>`;
        }
        parent.appendChild(div);
    });
}

function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function toggleFolder(el) {
    const children = el.parentElement.querySelector('.tree-children');
    const arrow = el.querySelector('.tree-arrow');
    if (children.classList.contains('open')) {
        children.classList.remove('open');
        arrow.classList.remove('open');
    } else {
        children.classList.add('open');
        arrow.classList.add('open');
    }
}

function selectFileNode(el, path) {
    document.querySelectorAll('.tree-file').forEach(f => f.classList.remove('selected'));
    el.classList.add('selected');
    loadFileNodes(path);
}

async function loadFileNodes(path) {
    const panel = document.getElementById('node-panel');
    panel.innerHTML = '<div class="skeleton h-8 mx-4 mt-4"></div><div class="skeleton h-8 mx-4 mt-2"></div><div class="skeleton h-8 mx-4 mt-2"></div>';
    try {
        const res = await fetch(`/api/graph/file/${encodeURIComponent(path)}`);
        const html = await res.text();
        panel.innerHTML = html || '<div class="flex items-center justify-center h-full text-sm opacity-30">No nodes in this file</div>';
        applyStagger(panel);
    } catch (e) {
        panel.innerHTML = '<div class="flex items-center justify-center h-full text-sm opacity-30">Error loading nodes</div>';
    }
}

function showSearchUI() {
    const c = document.getElementById('content');
    c.innerHTML = `
    <div class="rounded-xl border overflow-hidden" style="background:var(--surface);border-color:var(--border)">
        <div class="p-5 border-b" style="border-color:var(--border)">
            <div class="flex gap-3">
                <input type="text" id="search-query" class="flex-1 rounded-lg border px-4 py-3 text-sm animate-fade-in" style="background:var(--hover);border-color:var(--border);color:var(--fg)" placeholder="Search memories..." autofocus onkeydown="if(event.key==='Enter') doSearch()">
                <button onclick="doSearch()" class="px-5 py-3 rounded-lg text-sm font-semibold border-0 cursor-pointer btn" style="background:var(--accent);color:var(--bg)">Search</button>
            </div>
            <div class="mt-3">
                <select id="search-tier" class="rounded-lg border px-3 py-2 text-sm" style="background:var(--hover);border-color:var(--border);color:var(--fg)">
                    <option value="">All Tiers</option>
                    <option value="sensory">Sensory</option>
                    <option value="working">Working</option>
                    <option value="episodic">Episodic</option>
                    <option value="semantic">Semantic</option>
                    <option value="procedural">Procedural</option>
                </select>
            </div>
        </div>
        <div id="search-results" class="min-h-[100px]"></div>
    </div>`;
    document.getElementById('search-query').focus();
}

async function doSearch() {
    const query = document.getElementById('search-query').value;
    const tier = document.getElementById('search-tier').value;
    const r = document.getElementById('search-results');
    if (!query) return;
    r.innerHTML = '<div class="flex items-center justify-center py-10"><div class="animate-pulse opacity-50 text-sm">Searching...</div></div>';
    try {
        const params = new URLSearchParams({ query, limit: 50 });
        if (tier) params.set('tier', tier);
        const res = await fetch(`/api/search?${params}`);
        const html = await res.text();
        r.innerHTML = html;
        applyStagger(r);
    } catch (e) {
        r.innerHTML = '<div class="empty animate-fade-in">Search failed</div>';
    }
}

async function fetchStats() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/stats');
        c.innerHTML = await res.text();
        applyStagger(c);
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load stats</div>';
    }
}

async function fetchSkills() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/skills');
        c.innerHTML = await res.text();
        applyStagger(c);
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load skills</div>';
    }
}

async function fetchPersona() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/persona');
        c.innerHTML = await res.text();
        applyStagger(c);
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load persona</div>';
    }
}

async function fetchWork() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/work');
        c.innerHTML = await res.text();
        applyStagger(c);
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load work items</div>';
    }
}

async function fetchTimeline() {
    const c = document.getElementById('content');
    try {
        const res = await fetch('/api/timeline');
        c.innerHTML = await res.text();
        applyStagger(c);
    } catch (e) {
        c.innerHTML = '<div class="empty animate-fade-in">Failed to load timeline</div>';
    }
}

async function addMemory() {
    const content = document.getElementById('new-memory-content').value;
    const tier = document.getElementById('new-memory-tier').value;
    if (!content) return;
    const btn = document.querySelector('#add-modal .btn');
    btn.textContent = 'Saving...';
    btn.disabled = true;
    btn.style.opacity = '0.6';
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
    btn.textContent = 'Save';
    btn.disabled = false;
    btn.style.opacity = '1';
}

function viewMemory(id, tier, content, activation, strength) {
    document.getElementById('modal-content').innerHTML = `
        <div class="flex items-center gap-3 mb-4">
            <span class="badge badge-${escHtml(tier)}">${escHtml(tier)}</span>
            <code class="text-xs opacity-60">${escHtml(id)}</code>
        </div>
        <div class="text-sm leading-relaxed mb-4">${content}</div>
        <div class="flex gap-6 text-xs opacity-60">
            <span>Activation: ${parseFloat(activation).toFixed(2)}</span>
            <span>Strength: ${parseFloat(strength).toFixed(2)}</span>
        </div>
        <button onclick="closeModal()" class="mt-4 px-4 py-2 rounded-lg text-sm border cursor-pointer btn" style="border-color:var(--border);color:var(--fg)">Close</button>
    `;
    document.getElementById('modal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('modal').classList.add('hidden');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
        document.getElementById('add-modal').classList.add('hidden');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    switchView('memories');
});
