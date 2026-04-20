use super::graph::MemoryGraph;
use super::types::{Conflict, ConflictResolution, MemoryTier};
use crate::embeddings::{EmbeddingEngine, cosine_similarity};
use chrono::Utc;

const REPLAY_BOOST: f32 = 0.05;
const ASSOCIATED_BOOST: f32 = 0.02;
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.75;
const RECENT_WINDOW_SECS: i64 = 3600;

pub fn consolidate(graph: &mut MemoryGraph, embedder: &dyn EmbeddingEngine) -> Vec<Conflict> {
    replay_recent(graph);
    detect_conflicts_from_embeddings(graph, compute_all_embeddings(graph, embedder))
}

fn replay_recent(graph: &mut MemoryGraph) {
    let now = Utc::now().timestamp();
    let cutoff = now - RECENT_WINDOW_SECS;

    let recent: Vec<uuid::Uuid> = graph
        .get_all_memories()
        .iter()
        .filter(|m| {
            m.metadata.last_accessed >= cutoff
                && matches!(m.tier, MemoryTier::Episodic | MemoryTier::Working)
        })
        .map(|m| m.id)
        .collect();

    for id in &recent {
        if let Some(mem) = graph.get_memory_mut(id) {
            mem.metadata.base_activation = (mem.metadata.base_activation + REPLAY_BOOST).min(1.0);
            mem.metadata.access_count += 1;
        }

        let associated: Vec<(uuid::Uuid, f32)> = graph.get_associations(id);
        for (assoc_id, strength) in &associated {
            let boost = ASSOCIATED_BOOST * strength;
            if let Some(mem) = graph.get_memory_mut(assoc_id) {
                mem.metadata.base_activation = (mem.metadata.base_activation + boost).min(1.0);
            }
        }
    }
}

fn compute_all_embeddings(
    graph: &MemoryGraph,
    embedder: &dyn EmbeddingEngine,
) -> Vec<(uuid::Uuid, Vec<f32>, MemoryTier)> {
    graph
        .get_all_memories()
        .iter()
        .filter(|m| !matches!(m.tier, MemoryTier::Sensory))
        .map(|m| {
            let emb = graph
                .get_embedding(&m.id)
                .cloned()
                .unwrap_or_else(|| embedder.embed(&m.content));
            (m.id, emb, m.tier)
        })
        .collect()
}

pub fn detect_conflicts(graph: &MemoryGraph, embedder: &dyn EmbeddingEngine) -> Vec<Conflict> {
    detect_conflicts_from_embeddings(graph, compute_all_embeddings(graph, embedder))
}

fn detect_conflicts_from_embeddings(
    _graph: &MemoryGraph,
    embeddings: Vec<(uuid::Uuid, Vec<f32>, MemoryTier)>,
) -> Vec<Conflict> {
    let mut conflicts = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let (id_a, emb_a, tier_a) = &embeddings[i];
            let (id_b, emb_b, tier_b) = &embeddings[j];

            if tier_a != tier_b {
                continue;
            }

            let key = if id_a < id_b {
                (*id_a, *id_b)
            } else {
                (*id_b, *id_a)
            };
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            let similarity = cosine_similarity(emb_a, emb_b);
            if (CONFLICT_SIMILARITY_THRESHOLD..0.98).contains(&similarity) {
                conflicts.push(Conflict {
                    memory_id_1: *id_a,
                    memory_id_2: *id_b,
                    similarity,
                    tier: *tier_a,
                });
            }
        }
    }

    conflicts
}

pub fn resolve_conflicts(
    graph: &mut MemoryGraph,
    conflicts: &[Conflict],
    strategy: &ConflictResolution,
) -> Vec<uuid::Uuid> {
    match strategy {
        ConflictResolution::LatestWins => {
            let mut removed = Vec::new();
            for c in conflicts {
                let a_time = graph
                    .get_memory(&c.memory_id_1)
                    .map(|m| m.metadata.created_at);
                let b_time = graph
                    .get_memory(&c.memory_id_2)
                    .map(|m| m.metadata.created_at);
                let to_remove = match (a_time, b_time) {
                    (Some(a), Some(b)) if a >= b => c.memory_id_2,
                    (Some(_), Some(_)) => c.memory_id_1,
                    (Some(_), None) => c.memory_id_2,
                    (None, Some(_)) => c.memory_id_1,
                    _ => continue,
                };
                graph.remove_memory(&to_remove);
                removed.push(to_remove);
            }
            removed
        }
        ConflictResolution::KeepBoth => Vec::new(),
        ConflictResolution::HumanDecide => {
            for c in conflicts {
                if let Some(mem) = graph.get_memory_mut(&c.memory_id_1) {
                    mem.metadata.importance = (mem.metadata.importance + 0.1).min(1.0);
                }
                if let Some(mem) = graph.get_memory_mut(&c.memory_id_2) {
                    mem.metadata.importance = (mem.metadata.importance + 0.1).min(1.0);
                }
            }
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::HashEmbedding;
    use crate::memory::types::CognitiveMemoryUnit;

    fn make_memory_with_content(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
        CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
    }

    #[test]
    fn test_consolidate_replays_recent() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory_with_content(
            "recent memory",
            MemoryTier::Episodic,
        ));
        graph.get_memory_mut(&id).unwrap().metadata.base_activation = 0.3;

        let embedder = HashEmbedding::new();
        let emb = embedder.embed("recent memory");
        graph.set_embedding(id, emb);

        let _conflicts = consolidate(&mut graph, &embedder);
        let mem = graph.get_memory(&id).unwrap();
        assert!(
            mem.metadata.base_activation > 0.3,
            "activation should increase after replay"
        );
    }

    #[test]
    fn test_conflict_detection_high_similarity() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();

        let id1 = graph.add_memory(make_memory_with_content(
            "rust programming language tutorial",
            MemoryTier::Episodic,
        ));
        let id2 = graph.add_memory(make_memory_with_content(
            "rust programming language guide",
            MemoryTier::Episodic,
        ));

        let emb1 = embedder.embed("rust programming language tutorial");
        let emb2 = embedder.embed("rust programming language guide");
        graph.set_embedding(id1, emb1);
        graph.set_embedding(id2, emb2);

        let conflicts = detect_conflicts(&graph, &embedder);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].memory_id_1, id1);
        assert_eq!(conflicts[0].memory_id_2, id2);
    }

    #[test]
    fn test_no_conflict_different_tiers() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();

        graph.add_memory(make_memory_with_content(
            "similar content",
            MemoryTier::Episodic,
        ));
        graph.add_memory(make_memory_with_content(
            "similar content here",
            MemoryTier::Semantic,
        ));

        let conflicts = detect_conflicts(&graph, &embedder);
        assert!(
            conflicts.is_empty(),
            "memories in different tiers should not conflict"
        );
    }

    #[test]
    fn test_resolve_latest_wins() {
        let mut graph = MemoryGraph::new();

        let id1 = graph.add_memory(make_memory_with_content("older", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory_with_content("newer", MemoryTier::Episodic));

        graph.get_memory_mut(&id1).unwrap().metadata.created_at = 1000;
        graph.get_memory_mut(&id2).unwrap().metadata.created_at = 2000;

        let conflict = Conflict {
            memory_id_1: id1,
            memory_id_2: id2,
            similarity: 0.85,
            tier: MemoryTier::Episodic,
        };

        let removed = resolve_conflicts(&mut graph, &[conflict], &ConflictResolution::LatestWins);
        assert_eq!(removed.len(), 1);
        assert!(graph.get_memory(&id1).is_none());
        assert!(graph.get_memory(&id2).is_some());
    }

    #[test]
    fn test_resolve_keep_both() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory_with_content("a", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory_with_content("b", MemoryTier::Episodic));

        let conflict = Conflict {
            memory_id_1: id1,
            memory_id_2: id2,
            similarity: 0.85,
            tier: MemoryTier::Episodic,
        };

        let removed = resolve_conflicts(&mut graph, &[conflict], &ConflictResolution::KeepBoth);
        assert!(removed.is_empty());
        assert!(graph.get_memory(&id1).is_some());
        assert!(graph.get_memory(&id2).is_some());
    }

    #[test]
    fn test_resolve_human_decide_boosts_importance() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory_with_content("a", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory_with_content("b", MemoryTier::Episodic));

        let conflict = Conflict {
            memory_id_1: id1,
            memory_id_2: id2,
            similarity: 0.85,
            tier: MemoryTier::Episodic,
        };

        let _removed = resolve_conflicts(&mut graph, &[conflict], &ConflictResolution::HumanDecide);
        assert_eq!(graph.get_memory(&id1).unwrap().metadata.importance, 0.6);
        assert_eq!(graph.get_memory(&id2).unwrap().metadata.importance, 0.6);
    }
}
