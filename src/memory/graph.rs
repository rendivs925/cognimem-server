use super::types::{CognitiveMemoryUnit, MemoryTier};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use uuid::Uuid;

pub struct MemoryGraph {
    graph: DiGraph<CognitiveMemoryUnit, f32>,
    index_map: HashMap<Uuid, NodeIndex>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index_map: HashMap::new(),
        }
    }

    pub fn add_memory(&mut self, mut memory: CognitiveMemoryUnit) -> Uuid {
        let id = memory.id;
        let node = self.graph.add_node(memory);
        self.index_map.insert(id, node);
        id
    }

    pub fn get_memory(&self, id: &Uuid) -> Option<&CognitiveMemoryUnit> {
        self.index_map
            .get(id)
            .and_then(|idx| self.graph.node_weight(*idx))
    }

    pub fn get_memory_mut(&mut self, id: &Uuid) -> Option<&mut CognitiveMemoryUnit> {
        self.index_map
            .get(id)
            .and_then(|idx| self.graph.node_weight_mut(*idx))
    }

    pub fn add_association(&mut self, from: &Uuid, to: &Uuid, strength: f32) -> bool {
        let from_idx = match self.index_map.get(from) {
            Some(i) => *i,
            None => return false,
        };
        let to_idx = match self.index_map.get(to) {
            Some(i) => *i,
            None => return false,
        };
        self.graph.add_edge(from_idx, to_idx, strength);

        true
    }

    pub fn get_all_memories(&self) -> Vec<&CognitiveMemoryUnit> {
        self.graph.node_weights().collect()
    }

    pub fn len(&self) -> usize {
        self.graph.node_count()
    }
}
