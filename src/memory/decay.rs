use super::graph::MemoryGraph;
use super::types::MemoryTier;
use chrono::Utc;

pub fn apply_decay_to_all(graph: &mut MemoryGraph) {
    let now = Utc::now().timestamp();
    let ids: Vec<uuid::Uuid> = graph.get_all_memories().iter().map(|m| m.id).collect();

    for id in ids {
        if let Some(mem) = graph.get_memory_mut(&id) {
            mem.metadata.update_activation(now);
        }
    }
}

pub fn prune_below_threshold(graph: &mut MemoryGraph, threshold: f32) -> usize {
    let to_remove: Vec<uuid::Uuid> = graph
        .get_all_memories()
        .into_iter()
        .filter(|m| is_prunable(m.tier, m.metadata.base_activation, threshold))
        .map(|m| m.id)
        .collect();

    let count = to_remove.len();
    for id in &to_remove {
        graph.remove_memory(id);
    }
    count
}

fn is_prunable(tier: MemoryTier, activation: f32, threshold: f32) -> bool {
    if activation >= threshold {
        return false;
    }
    matches!(tier, MemoryTier::Sensory | MemoryTier::Working | MemoryTier::Episodic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::CognitiveMemoryUnit;

    fn make_memory(tier: MemoryTier, activation: f32) -> CognitiveMemoryUnit {
        let mut mem = CognitiveMemoryUnit::new("test".into(), tier, 0.5, tier.decay_rate());
        mem.metadata.base_activation = activation;
        mem
    }

    #[test]
    fn test_prune_removes_low_activation() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory(MemoryTier::Sensory, 0.005));
        assert!(graph.get_memory(&id).is_some());
        let removed = prune_below_threshold(&mut graph, 0.01);
        assert_eq!(removed, 1);
        assert!(graph.get_memory(&id).is_none());
    }

    #[test]
    fn test_prune_preserves_semantic() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory(MemoryTier::Semantic, 0.005));
        let removed = prune_below_threshold(&mut graph, 0.01);
        assert_eq!(removed, 0);
        assert!(graph.get_memory(&id).is_some());
    }

    #[test]
    fn test_prune_preserves_high_activation() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory(MemoryTier::Episodic, 0.5));
        let removed = prune_below_threshold(&mut graph, 0.01);
        assert_eq!(removed, 0);
        assert!(graph.get_memory(&id).is_some());
    }

    #[test]
    fn test_decay_updates_activation() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory(MemoryTier::Episodic, 1.0));
        // Simulate time passing by modifying last_accessed
        if let Some(mem) = graph.get_memory_mut(&id) {
            mem.metadata.last_accessed = Utc::now().timestamp() - 3600; // 1 hour ago
        }
        apply_decay_to_all(&mut graph);
        // Activation should have decreased from 1.0
        let mem = graph.get_memory(&id).unwrap();
        assert!(mem.metadata.base_activation < 1.0);
    }
}