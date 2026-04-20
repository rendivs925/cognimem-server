use super::store::MemoryStore;
use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use rocksdb::DB;
use std::path::Path;
use tracing::warn;
use uuid::Uuid;

/// A RocksDB-backed persistent store for cognitive memory units.
///
/// Each memory is serialized as JSON and stored with its UUID as the key.
pub struct RocksDbStore {
    db: DB,
}

impl RocksDbStore {
    /// Opens a RocksDB database at the given path, creating it if necessary.
    pub fn open(path: &Path) -> Result<Self> {
        let db = DB::open_default(path)?;
        Ok(Self { db })
    }
}

impl MemoryStore for RocksDbStore {
    fn save(&self, memory: &CognitiveMemoryUnit) -> Result<()> {
        self.db
            .put(memory.id.as_bytes(), serde_json::to_vec(memory)?)?;
        Ok(())
    }

    fn delete(&self, id: &Uuid) -> Result<()> {
        self.db.delete(id.as_bytes())?;
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<CognitiveMemoryUnit>> {
        let mut memories = Vec::new();
        for item in self.db.iterator(rocksdb::IteratorMode::Start) {
            let (_, value) = item?;
            match serde_json::from_slice::<CognitiveMemoryUnit>(&value) {
                Ok(memory) => memories.push(memory),
                Err(e) => {
                    let key_hint = String::from_utf8_lossy(&value[..value.len().min(64)]);
                    warn!("Skipping corrupted memory entry: {e} (data: {key_hint}...)");
                }
            }
        }
        Ok(memories)
    }
}
