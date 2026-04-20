use super::store::MemoryStore;
use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;
use uuid::Uuid;

/// A thread-safe, in-memory store backed by a `RwLock<HashMap>`.
///
/// Suitable for testing and single-process usage. Data is lost on process exit.
pub struct InMemoryStore {
    memories: RwLock<HashMap<Uuid, CognitiveMemoryUnit>>,
}

impl InMemoryStore {
    /// Creates a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            memories: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStore for InMemoryStore {
    fn save(&self, memory: &CognitiveMemoryUnit) -> Result<()> {
        self.memories
            .write()
            .expect("in-memory store write lock not poisoned")
            .insert(memory.id, memory.clone());
        Ok(())
    }

    fn delete(&self, id: &Uuid) -> Result<()> {
        self.memories
            .write()
            .expect("in-memory store write lock not poisoned")
            .remove(id);
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<CognitiveMemoryUnit>> {
        Ok(self
            .memories
            .read()
            .expect("in-memory store read lock not poisoned")
            .values()
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryTier;

    #[test]
    fn in_memory_store_save_and_load() {
        let store = InMemoryStore::new();
        let memory = CognitiveMemoryUnit::new("test".into(), MemoryTier::Episodic, 0.5, 0.5);
        store.save(&memory).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "test");
    }

    #[test]
    fn in_memory_store_delete() {
        let store = InMemoryStore::new();
        let memory = CognitiveMemoryUnit::new("test".into(), MemoryTier::Episodic, 0.5, 0.5);
        let id = memory.id;
        store.save(&memory).unwrap();
        store.delete(&id).unwrap();
        assert!(store.load_all().unwrap().is_empty());
    }

    #[test]
    fn in_memory_store_round_trip() {
        let store = InMemoryStore::new();
        let m1 = CognitiveMemoryUnit::new("first".into(), MemoryTier::Sensory, 0.3, 2.0);
        let m2 = CognitiveMemoryUnit::new("second".into(), MemoryTier::Semantic, 0.8, 0.2);
        store.save(&m1).unwrap();
        store.save(&m2).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 2);

        store.delete(&m1.id).unwrap();
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "second");
    }
}
