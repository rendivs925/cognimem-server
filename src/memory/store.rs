use super::types::CognitiveMemoryUnit;
use anyhow::Result;
use uuid::Uuid;

pub trait MemoryStore: Send + Sync {
    fn save(&self, memory: &CognitiveMemoryUnit) -> Result<()>;
    fn delete(&self, id: &Uuid) -> Result<()>;
    fn load_all(&self) -> Result<Vec<CognitiveMemoryUnit>>;
}