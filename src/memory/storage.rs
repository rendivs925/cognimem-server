use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use rocksdb::DB;
use std::path::Path;

pub struct MemoryStorage {
    db: DB,
}

impl MemoryStorage {
    pub fn open(path: &Path) -> Result<Self> {
        let db = DB::open_default(path)?;
        Ok(Self { db })
    }

    pub fn save(&self, memory: &CognitiveMemoryUnit) -> Result<()> {
        self.db.put(memory.id.as_bytes(), serde_json::to_vec(memory)?)?;
        Ok(())
    }

    pub fn delete(&self, id: &uuid::Uuid) -> Result<()> {
        self.db.delete(id.as_bytes())?;
        Ok(())
    }

    pub fn load_all(&self) -> Result<Vec<CognitiveMemoryUnit>> {
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