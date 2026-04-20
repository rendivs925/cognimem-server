use super::types::{CognitiveMemoryUnit, MemoryTier};
use slotmap::{new_key_type, SlotMap};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

new_key_type! {
    pub struct MemoryKey;
}

pub struct MemoryGraph {
    nodes: SlotMap<MemoryKey, CognitiveMemoryUnit>,
    id_to_key: HashMap<Uuid, MemoryKey>,
    edges: HashMap<(Uuid, Uuid), f32>,
    by_tier: HashMap<MemoryTier, HashSet<Uuid>>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            id_to_key: HashMap::new(),
            edges: HashMap::new(),
            by_tier: HashMap::new(),
        }
    }

    pub fn add_memory(&mut self, memory: CognitiveMemoryUnit) -> Uuid {
        let id = memory.id;
        let tier = memory.tier;
        let associations = memory.associations.clone();
        let key = self.nodes.insert(memory);
        self.id_to_key.insert(id, key);
        self.by_tier.entry(tier).or_default().insert(id);

        for assoc_id in &associations {
            if self.id_to_key.contains_key(assoc_id) {
                self.edges.insert((id, *assoc_id), 0.5);
            }
        }

        id
    }

    pub fn get_memory(&self, id: &Uuid) -> Option<&CognitiveMemoryUnit> {
        let key = self.id_to_key.get(id)?;
        self.nodes.get(*key)
    }

    pub fn get_memory_mut(&mut self, id: &Uuid) -> Option<&mut CognitiveMemoryUnit> {
        let key = self.id_to_key.get(id).copied()?;
        self.nodes.get_mut(key)
    }

    pub fn remove_memory(&mut self, id: &Uuid) -> Option<CognitiveMemoryUnit> {
        let key = self.id_to_key.remove(id)?;
        let memory = self.nodes.remove(key)?;
        self.by_tier.get_mut(&memory.tier).map(|s| s.remove(id));
        self.edges.retain(|(from, to), _| from != id && to != id);
        Some(memory)
    }

    pub fn add_association(&mut self, from: &Uuid, to: &Uuid, strength: f32) -> bool {
        if !self.contains(from) || !self.contains(to) {
            return false;
        }

        self.edges.insert((*from, *to), strength.clamp(0.0, 1.0));
        if let Some(mem) = self.get_memory_mut(from)
            && !mem.associations.contains(to)
        {
            mem.associations.push(*to);
        }
        true
    }

    pub fn get_associations(&self, id: &Uuid) -> Vec<(Uuid, f32)> {
        self.edges
            .iter()
            .filter(|((from, _), _)| from == id)
            .map(|((_, to), &strength)| (*to, strength))
            .collect()
    }

    pub fn spreading_activation(&self, start_ids: &[Uuid], max_depth: usize, decay_factor: f32, min_strength: f32) -> Vec<(Uuid, f32, usize)> {
        let mut visited: HashSet<Uuid> = start_ids.iter().copied().collect();
        let mut results: Vec<(Uuid, f32, usize)> = Vec::new();
        let mut frontier: Vec<(Uuid, f32, usize)> = start_ids.iter().map(|id| (*id, 1.0, 0)).collect();

        while let Some((id, accumulated, depth)) = frontier.pop() {
            if depth >= max_depth {
                continue;
            }
            for (neighbor, edge_strength) in self.get_associations(&id) {
                if visited.contains(&neighbor) {
                    continue;
                }
                let strength = accumulated * edge_strength * decay_factor.powi(depth as i32 + 1);
                if strength < min_strength {
                    continue;
                }
                visited.insert(neighbor);
                results.push((neighbor, strength, depth + 1));
                frontier.push((neighbor, strength, depth + 1));
            }
        }

        results
    }

    pub fn get_all_memories(&self) -> Vec<&CognitiveMemoryUnit> {
        self.nodes.iter().map(|(_, v)| v).collect()
    }

    pub fn get_by_tier(&self, tier: MemoryTier) -> Vec<&CognitiveMemoryUnit> {
        self.by_tier
            .get(&tier)
            .map(|ids| ids.iter().filter_map(|id| self.get_memory(id)).collect())
            .unwrap_or_default()
    }

    pub fn contains(&self, id: &Uuid) -> bool {
        self.id_to_key.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn count_by_tier(&self, tier: MemoryTier) -> usize {
        self.by_tier.get(&tier).map(|s| s.len()).unwrap_or(0)
    }

    pub fn find_lowest_activation_in_tier(&self, tier: MemoryTier) -> Option<Uuid> {
        self.by_tier
            .get(&tier)
            .and_then(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_memory(id).map(|m| (m.id, m.metadata.base_activation)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(id, _)| id)
            })
    }

    pub fn change_tier(&mut self, id: &Uuid, old_tier: MemoryTier, new_tier: MemoryTier) {
        self.by_tier.entry(old_tier).or_default().remove(id);
        self.by_tier.entry(new_tier).or_default().insert(*id);
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::CognitiveMemoryUnit;

    fn make_memory(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
        CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
    }

    #[test]
    fn add_and_get_memory() {
        let mut graph = MemoryGraph::new();
        let id = graph.add_memory(make_memory("hello", MemoryTier::Episodic));
        let mem = graph.get_memory(&id).unwrap();
        assert_eq!(mem.content, "hello");
        assert_eq!(mem.tier, MemoryTier::Episodic);
    }

    #[test]
    fn remove_memory_stable_indices() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Sensory));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Working));
        let id3 = graph.add_memory(make_memory("third", MemoryTier::Episodic));

        graph.remove_memory(&id2);

        // Remaining memories still accessible
        assert!(graph.get_memory(&id1).is_some());
        assert!(graph.get_memory(&id3).is_some());
        assert!(graph.get_memory(&id2).is_none());
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn tier_index_works() {
        let mut graph = MemoryGraph::new();
        let _id1 = graph.add_memory(make_memory("a", MemoryTier::Sensory));
        let id2 = graph.add_memory(make_memory("b", MemoryTier::Episodic));
        let id3 = graph.add_memory(make_memory("c", MemoryTier::Episodic));

        let episodic = graph.get_by_tier(MemoryTier::Episodic);
        assert_eq!(episodic.len(), 2);

        graph.remove_memory(&id2);
        let episodic = graph.get_by_tier(MemoryTier::Episodic);
        assert_eq!(episodic.len(), 1);
        assert_eq!(episodic[0].id, id3);
    }

    #[test]
    fn associations_are_tracked() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));

        assert!(graph.add_association(&id1, &id2, 0.8));
        let assocs = graph.get_associations(&id1);
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].0, id2);
        assert!((assocs[0].1 - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn association_with_missing_memory_fails() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("exists", MemoryTier::Episodic));
        let fake = Uuid::new_v4();
        assert!(!graph.add_association(&id1, &fake, 0.5));
    }

    #[test]
    fn remove_cleans_edges() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.5);

        graph.remove_memory(&id2);
        assert!(graph.get_associations(&id1).is_empty());
    }

    #[test]
    fn add_with_initial_associations() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));

        let mut mem2 = CognitiveMemoryUnit::new("second".into(), MemoryTier::Episodic, 0.5, 0.5);
        mem2.associations.push(id1);
        let id2 = graph.add_memory(mem2);

        let assocs = graph.get_associations(&id2);
        assert_eq!(assocs.len(), 1);
        assert_eq!(assocs[0].0, id1);
    }

    #[test]
    fn spreading_activation_single_hop() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.8);

        let results = graph.spreading_activation(&[id1], 3, 0.5, 0.1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id2);
        assert_eq!(results[0].2, 1);
    }

    #[test]
    fn spreading_activation_multi_hop() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));
        let id3 = graph.add_memory(make_memory("third", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.9);
        graph.add_association(&id2, &id3, 0.9);

        let results = graph.spreading_activation(&[id1], 3, 0.5, 0.01);
        assert_eq!(results.len(), 2);
        let id2_result = results.iter().find(|(id, _, _)| id == &id2).unwrap();
        assert_eq!(id2_result.2, 1);
        let id3_result = results.iter().find(|(id, _, _)| id == &id3).unwrap();
        assert_eq!(id3_result.2, 2);
    }

    #[test]
    fn spreading_activation_no_cycles() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
        let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));
        graph.add_association(&id1, &id2, 0.8);
        graph.add_association(&id2, &id1, 0.8);

        let results = graph.spreading_activation(&[id1], 3, 0.5, 0.1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn count_by_tier_tracks_removals() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("a", MemoryTier::Sensory));
        let id2 = graph.add_memory(make_memory("b", MemoryTier::Sensory));
        graph.add_memory(make_memory("c", MemoryTier::Episodic));

        assert_eq!(graph.count_by_tier(MemoryTier::Sensory), 2);
        graph.remove_memory(&id1);
        assert_eq!(graph.count_by_tier(MemoryTier::Sensory), 1);
        assert_eq!(graph.count_by_tier(MemoryTier::Episodic), 1);
        assert_eq!(graph.count_by_tier(MemoryTier::Procedural), 0);
        let _ = id2;
    }

    #[test]
    fn find_lowest_activation_returns_correct() {
        let mut graph = MemoryGraph::new();
        let id1 = graph.add_memory(make_memory("high", MemoryTier::Sensory));
        graph.get_memory_mut(&id1).unwrap().metadata.base_activation = 0.9;
        let id2 = graph.add_memory(make_memory("low", MemoryTier::Sensory));
        graph.get_memory_mut(&id2).unwrap().metadata.base_activation = 0.1;
        let id3 = graph.add_memory(make_memory("mid", MemoryTier::Sensory));
        graph.get_memory_mut(&id3).unwrap().metadata.base_activation = 0.5;

        let lowest = graph.find_lowest_activation_in_tier(MemoryTier::Sensory);
        assert_eq!(lowest, Some(id2));
    }

    #[test]
    fn find_lowest_activation_empty_tier() {
        let graph = MemoryGraph::new();
        assert_eq!(graph.find_lowest_activation_in_tier(MemoryTier::Sensory), None);
    }
}