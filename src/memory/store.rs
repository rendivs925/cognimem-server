use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use uuid::Uuid;

/// Persistent storage backend for cognitive memory units.
///
/// Implementations must be `Send + Sync` to allow shared access
/// across async tasks.
pub trait MemoryStore: Send + Sync {
    /// Persists a memory unit. Overwrites if the ID already exists.
    fn save(&self, memory: &CognitiveMemoryUnit) -> Result<()>;

    /// Deletes the memory with the given ID. No-op if the ID doesn't exist.
    fn delete(&self, id: &Uuid) -> Result<()>;

    /// Loads all stored memories. Used during startup to rebuild the in-memory graph.
    fn load_all(&self) -> Result<Vec<CognitiveMemoryUnit>>;
}
