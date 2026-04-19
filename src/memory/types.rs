use chrono::Utc;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString, IntoStaticStr};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, IntoStaticStr, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum MemoryTier {
    Sensory,
    Working,
    Episodic,
    Semantic,
    Procedural,
}

impl Default for MemoryTier {
    fn default() -> Self {
        Self::Episodic
    }
}

impl MemoryTier {
    pub fn decay_rate(&self) -> f32 {
        match self {
            MemoryTier::Sensory => 2.0,
            MemoryTier::Working => 1.0,
            MemoryTier::Episodic => 0.5,
            MemoryTier::Semantic => 0.2,
            MemoryTier::Procedural => 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub created_at: i64,
    pub last_accessed: i64,
    pub access_count: u32,
    pub importance: f32,
    pub base_activation: f32,
    pub decay_rate: f32,
}

impl Default for MemoryMetadata {
    fn default() -> Self {
        Self::new(0.5, 0.5)
    }
}

impl MemoryMetadata {
    pub fn new(importance: f32, decay_rate: f32) -> Self {
        let now = Utc::now().timestamp();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 1,
            importance: importance.clamp(0.0, 1.0),
            base_activation: 1.0,
            decay_rate,
        }
    }

    pub fn update_activation(&mut self, now: i64) {
        let elapsed_secs = (now - self.last_accessed) as f64;
        let decayed = (elapsed_secs + 1.0).powf(-self.decay_rate as f64);
        let new_activation = (self.access_count as f64 * decayed).ln();
        self.base_activation = new_activation.max(0.01) as f32;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMemoryUnit {
    pub id: Uuid,
    pub tier: MemoryTier,
    pub content: String,
    pub metadata: MemoryMetadata,
    pub associations: Vec<Uuid>,
}

impl Default for CognitiveMemoryUnit {
    fn default() -> Self {
        Self::new(String::new(), MemoryTier::default(), 0.5, 0.5)
    }
}

impl CognitiveMemoryUnit {
    pub fn new(content: String, tier: MemoryTier, importance: f32, decay_rate: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            tier,
            content,
            metadata: MemoryMetadata::new(importance, decay_rate),
            associations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberArgs {
    pub content: String,
    pub tier: Option<MemoryTier>,
    pub importance: Option<f32>,
    pub associations: Option<Vec<Uuid>>,
}

impl Default for RememberArgs {
    fn default() -> Self {
        Self {
            content: String::new(),
            tier: None,
            importance: None,
            associations: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallArgs {
    pub query: String,
    pub tier: Option<MemoryTier>,
    pub limit: Option<usize>,
}

impl Default for RecallArgs {
    fn default() -> Self {
        Self {
            query: String::new(),
            tier: None,
            limit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberResult {
    pub memory_id: Uuid,
    pub message: String,
}

impl RememberResult {
    pub fn success(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            message: "Memory stored successfully".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub memories: Vec<MemorySummary>,
}

impl RecallResult {
    pub fn new(memories: Vec<MemorySummary>) -> Self {
        Self { memories }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    pub id: Uuid,
    pub content: String,
    pub tier: MemoryTier,
    pub activation: f32,
}

impl From<&CognitiveMemoryUnit> for MemorySummary {
    fn from(memory: &CognitiveMemoryUnit) -> Self {
        Self {
            id: memory.id,
            content: memory.content.clone(),
            tier: memory.tier,
            activation: memory.metadata.base_activation,
        }
    }
}