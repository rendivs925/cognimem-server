use chrono::Utc;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString, IntoStaticStr};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, IntoStaticStr, EnumString, Default)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum MemoryTier {
    Sensory,
    Working,
    #[default]
    Episodic,
    Semantic,
    Procedural,
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

    pub fn capacity(&self) -> Option<usize> {
        match self {
            MemoryTier::Sensory => Some(50),
            MemoryTier::Working => Some(200),
            MemoryTier::Episodic => None,
            MemoryTier::Semantic => None,
            MemoryTier::Procedural => None,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RememberArgs {
    pub content: String,
    pub tier: Option<MemoryTier>,
    pub importance: Option<f32>,
    pub associations: Option<Vec<Uuid>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecallArgs {
    pub query: String,
    pub tier: Option<MemoryTier>,
    pub limit: Option<usize>,
    #[serde(default)]
    pub min_activation: Option<f32>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssociateArgs {
    #[serde(default)]
    pub from: Uuid,
    #[serde(default)]
    pub to: Uuid,
    pub strength: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociateResult {
    pub from: Uuid,
    pub to: Uuid,
    pub strength: f32,
    pub message: String,
}

impl AssociateResult {
    pub fn success(from: Uuid, to: Uuid, strength: f32) -> Self {
        Self {
            from,
            to,
            strength,
            message: "Association created successfully".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetArgs {
    pub memory_id: Uuid,
    #[serde(default)]
    pub hard_delete: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetResult {
    pub memory_id: Uuid,
    pub deleted: bool,
    pub message: String,
}

impl ForgetResult {
    pub fn hard_deleted(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: true,
            message: "Memory permanently deleted".to_string(),
        }
    }

    pub fn soft_deleted(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: false,
            message: "Memory marked for pruning (activation set to near-zero)".to_string(),
        }
    }

    pub fn not_found(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: false,
            message: "Memory not found".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub memory_id_1: Uuid,
    pub memory_id_2: Uuid,
    pub similarity: f32,
    pub tier: MemoryTier,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    #[default]
    LatestWins,
    KeepBoth,
    HumanDecide,
}

impl std::fmt::Display for ConflictResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictResolution::LatestWins => write!(f, "latest_wins"),
            ConflictResolution::KeepBoth => write!(f, "keep_both"),
            ConflictResolution::HumanDecide => write!(f, "human_decide"),
        }
    }
}

impl std::str::FromStr for ConflictResolution {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "latest_wins" => Ok(ConflictResolution::LatestWins),
            "keep_both" => Ok(ConflictResolution::KeepBoth),
            "human_decide" => Ok(ConflictResolution::HumanDecide),
            _ => Err(format!("unknown conflict resolution: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectArgs {
    #[serde(default)]
    pub intensity: Option<String>,
    #[serde(default)]
    pub conflict_strategy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectResult {
    pub pruned_count: usize,
    pub promoted_count: usize,
    pub decayed_count: usize,
    pub conflicts: Vec<Conflict>,
}

impl ReflectResult {
    pub fn new(pruned_count: usize, promoted_count: usize, decayed_count: usize, conflicts: Vec<Conflict>) -> Self {
        Self { pruned_count, promoted_count, decayed_count, conflicts }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchArgs {
    pub query: String,
    pub tier: Option<MemoryTier>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub snippet: String,
    pub tier: MemoryTier,
    pub activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
}

impl SearchResults {
    pub fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineArgs {
    pub memory_id: Uuid,
    #[serde(default)]
    pub window_secs: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineResult {
    pub center: MemorySummary,
    pub before: Vec<MemorySummary>,
    pub after: Vec<MemorySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GetObservationsArgs {
    pub memory_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationsResult {
    pub memory: CognitiveMemoryUnit,
}