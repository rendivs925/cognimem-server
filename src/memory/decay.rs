use crate::memory::MemoryTier;

use super::graph::MemoryGraph;
use chrono::Utc;
use uuid::Uuid;

pub fn apply_decay_to_all(graph: &mut MemoryGraph) {
    let now = Utc::now().timestamp();
    let mut updates = Vec::new();

    for mem in graph.get_all_memories() {
        let id = mem.id;
        let mut updated = mem.clone();
        updated.metadata.update_activation(now);
        updates.push((id, updated));
    }

    for (id, updated_mem) in updates {
        if let Some(mut_ref) = graph.get_memory_mut(&id) {
            *mut_ref = updated_mem;
        }
    }
}

pub fn prune_below_threshold(graph: &mut MemoryGraph, threshold: f32) -> usize {
    let to_remove: Vec<Uuid> = graph
        .get_all_memories()
            .iter()
            .filter(|m| {
                m.metadata.base_activation < threshold
                    && matches!(
                        m.tier,
                        MemoryTier::Sensory | MemoryTier::Working | MemoryTier::Episodic
                    )
            })
            .map(|m| m.id)
            .collect();

    for id in &to_remove {
        // We need a proper removal method, but for now skip (would require graph node removal)
        // In full impl, we'd use graph.remove_node(index)
    }

    to_remove.len()
}
