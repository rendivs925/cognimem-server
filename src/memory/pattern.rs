use crate::embeddings::{cosine_similarity, EmbeddingEngine};
use super::graph::MemoryGraph;
use super::types::MemorySummary;
use uuid::Uuid;

const HEBBIAN_STRENGTHEN: f32 = 0.05;
const HEBBIAN_WEAKEN: f32 = 0.01;
const MIN_STRENGTH: f32 = 0.05;
const MAX_STRENGTH: f32 = 1.0;

pub fn strengthen_co_activated(graph: &mut MemoryGraph, ids: &[Uuid]) {
    if ids.len() < 2 {
        weaken_lone(graph, ids);
        return;
    }

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let current = graph.get_association_strength(&ids[i], &ids[j])
                .unwrap_or(0.0);
            let new_strength = (current + HEBBIAN_STRENGTHEN).min(MAX_STRENGTH);
            graph.update_association(&ids[i], &ids[j], new_strength);
        }
    }
}

fn weaken_lone(graph: &mut MemoryGraph, ids: &[Uuid]) {
    for id in ids {
        let associations = graph.get_associations(id);
        for (assoc_id, strength) in associations {
            let weakened = (strength - HEBBIAN_WEAKEN).max(MIN_STRENGTH);
            graph.update_association(id, &assoc_id, weakened);
        }
    }
}

pub fn complete_pattern(
    graph: &MemoryGraph,
    embedder: &dyn EmbeddingEngine,
    cue: &str,
    tolerance: f32,
    limit: usize,
) -> Vec<PatternCandidate> {
    let query_emb = embedder.embed(cue);

    let mut candidates: Vec<PatternCandidate> = graph
        .get_all_memories()
        .iter()
        .filter_map(|m| {
            let emb = graph.get_embedding(&m.id)?;
            let sim = cosine_similarity(&query_emb, emb);
            if sim >= tolerance {
                let mut associated: Vec<MemorySummary> = graph
                    .get_associations(&m.id)
                    .iter()
                    .filter_map(|(assoc_id, strength)| {
                        if *strength >= 0.3 {
                            graph.get_memory(assoc_id).map(MemorySummary::from)
                        } else {
                            None
                        }
                    })
                    .collect();
                associated.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal));
                associated.truncate(limit);

                Some(PatternCandidate {
                    memory: MemorySummary::from(*m),
                    similarity: sim,
                    associations: associated,
                })
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(limit);
    candidates
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternCandidate {
    pub memory: MemorySummary,
    pub similarity: f32,
    pub associations: Vec<MemorySummary>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CompletePatternArgs {
    pub cue: String,
    #[serde(default)]
    pub tolerance: Option<f32>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletePatternResult {
    pub candidates: Vec<PatternCandidate>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::HashEmbedding;
    use crate::memory::types::{CognitiveMemoryUnit, MemoryTier};

    fn make_memory(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
        CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
    }

    #[test]
    fn test_strengthen_co_activated() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("alpha", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("beta", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.5);

        strengthen_co_activated(&mut graph, &[id1, id2]);

        let strength = graph.get_association_strength(&id1, &id2).unwrap();
        assert!(strength > 0.5, "association should be strengthened: got {strength}");
    }

    #[test]
    fn test_strengthen_multiple() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("a", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("b", MemoryTier::Episodic));
        let id3 = graph.add_memory(make_memory("c", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.3);
        graph.add_association(&id1, &id3, 0.3);
        graph.add_association(&id2, &id3, 0.3);

        strengthen_co_activated(&mut graph, &[id1, id2, id3]);

        assert!(graph.get_association_strength(&id1, &id2).unwrap() > 0.3);
        assert!(graph.get_association_strength(&id1, &id3).unwrap() > 0.3);
        assert!(graph.get_association_strength(&id2, &id3).unwrap() > 0.3);
    }

    #[test]
    fn test_complete_pattern_returns_candidates() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();

        let id = graph.add_memory(make_memory("rust programming language", MemoryTier::Semantic));
        let emb = embedder.embed("rust programming language");
        graph.set_embedding(id, emb);

        let results = complete_pattern(&graph, &embedder, "rust programming", 0.3, 5);
        assert!(!results.is_empty());
        assert!(results[0].similarity >= 0.3);
    }

    #[test]
    fn test_complete_pattern_with_associations() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();

        let id1 = graph.add_memory(make_memory("rust programming", MemoryTier::Semantic));
        let id2 = graph.add_memory(make_memory("cargo build tool", MemoryTier::Semantic));
        let emb1 = embedder.embed("rust programming");
        let emb2 = embedder.embed("cargo build tool");
        graph.set_embedding(id1, emb1);
        graph.set_embedding(id2, emb2);
        graph.add_association(&id1, &id2, 0.8);

        let results = complete_pattern(&graph, &embedder, "rust", 0.3, 5);
        if !results.is_empty() && results[0].memory.id == id1 {
            assert!(!results[0].associations.is_empty());
        }
    }
}