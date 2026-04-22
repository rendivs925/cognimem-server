use crate::embeddings::{EmbeddingEngine, HashEmbedding};
use crate::memory::types::{HandoffSummary, SessionContext, WorkClaim};
use crate::memory::{MemoryGraph, MemoryStore, NoOpSlm, OllamaSlm, ProjectModelManager, SlmEngine};
use crate::search::{Fts5Search, SearchEngine, SubstringSearch};
use std::collections::HashMap;

fn indexed_content(memory: &crate::memory::CognitiveMemoryUnit) -> String {
    match memory.model.compressed_content.as_deref() {
        Some(compressed) if !compressed.is_empty() => {
            format!("{} {}", memory.content, compressed)
        }
        _ => memory.content.clone(),
    }
}

pub struct CogniMemState {
    pub graph: MemoryGraph,
    pub storage: Box<dyn MemoryStore>,
    pub search: Box<dyn SearchEngine + Send>,
    pub embedder: Box<dyn EmbeddingEngine + Send + Sync>,
    pub slm: Box<dyn SlmEngine + Send>,
    pub work_claims: HashMap<uuid::Uuid, WorkClaim>,
    pub session_context: Option<SessionContext>,
    pub handoffs: Vec<HandoffSummary>,
    pub project_models: ProjectModelManager,
}

impl CogniMemState {
    pub fn new(
        storage: Box<dyn MemoryStore>,
        ollama_model: Option<String>,
        ollama_url: Option<String>,
    ) -> Self {
        let mut graph = MemoryGraph::new();
        let mut search: Box<dyn SearchEngine + Send> = match Fts5Search::new() {
            Ok(fts) => Box::new(fts),
            Err(e) => {
                tracing::warn!("Failed to initialize FTS5 search, falling back to substring: {e}");
                Box::new(SubstringSearch)
            }
        };
        let embedder: Box<dyn EmbeddingEngine + Send + Sync> = Box::new(HashEmbedding::new());

        let slm: Box<dyn SlmEngine + Send> = if let Some(model) = ollama_model {
            let ollama = OllamaSlm::new(Some(model), ollama_url);
            if ollama.check_available() {
                tracing::info!("Ollama SLM engine initialized successfully");
                Box::new(ollama)
            } else {
                tracing::warn!("Ollama not available, falling back to NoOpSlm");
                Box::new(NoOpSlm)
            }
        } else {
            Box::new(NoOpSlm)
        };
        let memories = match storage.load_all() {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to load memories from storage: {e}");
                Vec::new()
            }
        };
        for m in &memories {
            let emb = embedder.embed(&m.content);
            graph.add_memory(m.clone());
            graph.set_embedding(m.id, emb);
        }
        graph.rebuild_associations();
        for m in &memories {
            search.index(m.id, &indexed_content(m), m.tier);
        }
        Self {
            graph,
            storage,
            search,
            embedder,
            slm,
            work_claims: HashMap::new(),
            session_context: None,
            handoffs: Vec::new(),
            project_models: ProjectModelManager::new(),
        }
    }
}
