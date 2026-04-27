use super::graph::MemoryGraph;
use super::slm::{SlmEngine, SlmError};
use super::slm_types::DreamInput;
use super::types::{CognitiveMemoryUnit, MemorySource, MemoryTier};
use crate::embeddings::EmbeddingEngine;
use uuid::Uuid;

const DREAM_CONFIDENCE: f32 = 0.15;
const DREAM_DECAY_RATE: f32 = 0.3;
const DREAM_SIMILARITY_MAX: f32 = 0.3;

pub fn pick_dream_candidates<'a>(
    graph: &'a MemoryGraph,
    _embedder: &dyn EmbeddingEngine,
) -> Option<(&'a CognitiveMemoryUnit, &'a CognitiveMemoryUnit)> {
    let candidates: Vec<&CognitiveMemoryUnit> = graph
        .get_all_memories()
        .into_iter()
        .filter(|m| matches!(m.tier, MemoryTier::Episodic | MemoryTier::Semantic))
        .collect();

    if candidates.len() < 2 {
        return None;
    }

    let a = candidates.first()?;
    let emb_a = graph.get_embedding(&a.id)?;

    for b in candidates.iter().skip(1) {
        let emb_b = match graph.get_embedding(&b.id) {
            Some(e) => e,
            None => continue,
        };
        let sim = crate::embeddings::cosine_similarity(emb_a, emb_b);
        if sim < DREAM_SIMILARITY_MAX {
            return Some((a, *b));
        }
    }

    None
}

pub async fn create_dream_memory(
    graph: &mut MemoryGraph,
    embedder: &dyn EmbeddingEngine,
    slm: &dyn SlmEngine,
) -> Result<Option<(Uuid, String)>, SlmError> {
    let pair = pick_dream_candidates(graph, embedder);
    let (a, b) = match pair {
        Some(p) => p,
        None => return Ok(None),
    };

    let output = slm
        .dream(DreamInput {
            memory_a: a.content.clone(),
            memory_b: b.content.clone(),
        })
        .await?;

    let mut memory = CognitiveMemoryUnit::new(
        output.dreamt_content.clone(),
        MemoryTier::Episodic,
        0.2,
        DREAM_DECAY_RATE,
    );
    memory.source = MemorySource::Dreamt;
    memory.model.confidence = Some(DREAM_CONFIDENCE);
    memory.model.model_name = Some(output.metadata.model.clone());
    memory.associations = vec![a.id, b.id];

    let id = memory.id;
    let emb = embedder.embed(&memory.content);
    graph.add_memory(memory);
    graph.set_embedding(id, emb);

    Ok(Some((id, output.insight)))
}
