use super::store::MemoryStore;
use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use rocksdb::DB;
use std::path::Path;
use uuid::Uuid;

pub struct RocksDbStore {
    db: DB,
}

impl RocksDbStore {
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
            if let Ok(memory) = serde_json::from_slice::<CognitiveMemoryUnit>(&value) {
                memories.push(memory);
            }
        }
        Ok(memories)
    }
}
