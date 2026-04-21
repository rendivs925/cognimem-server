use chrono::Utc;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString, IntoStaticStr};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMemoryMetadata {
    #[serde(default)]
    pub compressed_content: Option<String>,
    #[serde(default)]
    pub suggested_tier: Option<MemoryTier>,
    #[serde(default)]
    pub suggested_importance: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub provenance_ids: Vec<Uuid>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub suppress: bool,
}

/// Classification tiers for memories, modeled after human memory systems.
///
/// Each tier has different decay rates and capacity limits, controlling
/// how long and how many memories persist at that level.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Display,
    IntoStaticStr,
    EnumString,
    Default,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum MemoryTier {
    /// Fleeting, high-volume sensory impressions. Fast decay, small capacity.
    Sensory,
    /// Short-term active processing buffer. Moderate decay, limited capacity.
    Working,
    /// Autobiographical event memories. Default tier.
    #[default]
    Episodic,
    /// Generalized, abstracted knowledge. Slow decay, unlimited capacity.
    Semantic,
    /// Learned procedures and skills. Slowest decay, unlimited capacity.
    Procedural,
}

impl MemoryTier {
    /// Returns the exponential decay rate for this tier.
    ///
    /// Higher values cause faster forgetting. Sensory decays fastest (2.0),
    /// Procedural slowest (0.1).
    pub fn decay_rate(&self) -> f32 {
        match self {
            MemoryTier::Sensory => 2.0,
            MemoryTier::Working => 1.0,
            MemoryTier::Episodic => 0.5,
            MemoryTier::Semantic => 0.2,
            MemoryTier::Procedural => 0.1,
        }
    }

    /// Returns the maximum number of memories allowed in this tier, or `None` if unlimited.
    ///
    /// Only Sensory (50) and Working (200) have capacity limits.
    /// Episodic, Semantic, and Procedural tiers are unlimited.
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

/// Domains for persona classification of memories.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Display,
    IntoStaticStr,
    EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum PersonaDomain {
    /// Life history and background information.
    Biography,
    /// Events and encounters.
    Experiences,
    /// Likes, dislikes, and stated preferences.
    Preferences,
    /// Relationships and social connections.
    Social,
    /// Professional and project-related information.
    Work,
    /// Behavioral patterns and tendencies.
    Psychometrics,
}

/// RACI role assignments for a memory, following the Responsible-Accountable-Consulted-Informed model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaciRoles {
    /// The person responsible for performing the work.
    #[serde(default)]
    pub responsible: Option<String>,
    /// The person ultimately accountable for the outcome.
    #[serde(default)]
    pub accountable: Option<String>,
    /// People whose input is sought before acting.
    #[serde(default)]
    pub consulted: Vec<String>,
    /// People who are kept informed of progress.
    #[serde(default)]
    pub informed: Vec<String>,
}

/// Metadata tracking activation dynamics and access patterns for a memory.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Unix timestamp when the memory was created.
    pub created_at: i64,
    /// Unix timestamp of the most recent access.
    pub last_accessed: i64,
    /// Number of times the memory has been accessed.
    pub access_count: u32,
    /// User-assigned importance score in [0, 1].
    pub importance: f32,
    /// Current activation level, computed from access count and decay.
    pub base_activation: f32,
    /// Exponential decay rate controlling activation falloff.
    pub decay_rate: f32,
    /// Rehearsal timestamps used for ACT-R style base-level activation.
    #[serde(default)]
    pub rehearsal_timestamps: [i64; 8],
    /// Number of valid entries in `rehearsal_timestamps`.
    #[serde(default)]
    pub rehearsal_count: u8,
}

impl Default for MemoryMetadata {
    fn default() -> Self {
        Self::new(0.5, 0.5)
    }
}

impl MemoryMetadata {
    /// Creates new metadata with the given importance and decay rate.
    ///
    /// `importance` is clamped to [0, 1]. `base_activation` starts at 1.0
    /// and timestamps are set to the current time.
    pub fn new(importance: f32, decay_rate: f32) -> Self {
        let now = Utc::now().timestamp();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 1,
            importance: importance.clamp(0.0, 1.0),
            base_activation: 1.0,
            decay_rate,
            rehearsal_timestamps: {
                let mut timestamps = [0; 8];
                timestamps[0] = now;
                timestamps
            },
            rehearsal_count: 1,
        }
    }

    /// Recomputes `base_activation` based on elapsed time since last access.
    ///
    /// Uses a power-law decay model: `ln(access_count * (elapsed+1)^(-decay_rate))`,
    /// with a floor of 0.01.
    pub fn update_activation(&mut self, now: i64) {
        let count = self.rehearsal_count.max(1) as usize;
        let decay = self.decay_rate as f64;
        let summed: f64 = self.rehearsal_timestamps[..count]
            .iter()
            .map(|&timestamp| {
                let elapsed = (now - timestamp).max(0) as f64 + 1.0;
                elapsed.powf(-decay)
            })
            .sum();
        let new_activation = summed.ln();
        self.base_activation = new_activation.max(0.01) as f32;
    }

    pub fn record_rehearsal(&mut self, now: i64) {
        self.last_accessed = now;
        self.access_count = self.access_count.saturating_add(1);

        if self.rehearsal_count as usize >= self.rehearsal_timestamps.len() {
            self.rehearsal_timestamps.rotate_left(1);
            let last = self.rehearsal_timestamps.len() - 1;
            self.rehearsal_timestamps[last] = now;
        } else {
            self.rehearsal_timestamps[self.rehearsal_count as usize] = now;
            self.rehearsal_count += 1;
        }

        self.update_activation(now);
    }
}

/// The core memory unit in the cognitive memory system.
///
/// Combines content, tier classification, activation metadata,
/// associations, persona domain, RACI roles, and model-derived metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMemoryUnit {
    /// Unique identifier for this memory.
    pub id: Uuid,
    /// Memory tier controlling decay and capacity behavior.
    pub tier: MemoryTier,
    /// The textual content of the memory.
    pub content: String,
    /// Activation and access metadata.
    pub metadata: MemoryMetadata,
    /// IDs of memories this unit is associated with.
    pub associations: Vec<Uuid>,
    /// Optional persona domain classification.
    #[serde(default)]
    pub persona: Option<PersonaDomain>,
    /// RACI role assignments.
    #[serde(default)]
    pub raci: RaciRoles,
    /// Persisted model-derived metadata for this memory.
    #[serde(default)]
    pub model: ModelMemoryMetadata,
}

impl Default for CognitiveMemoryUnit {
    fn default() -> Self {
        Self::new(String::new(), MemoryTier::default(), 0.5, 0.5)
    }
}

impl CognitiveMemoryUnit {
    /// Creates a new memory unit with the given content, tier, importance, and decay rate.
    ///
    /// A new UUID is generated. Associations, persona, and RACI roles start empty.
    pub fn new(content: String, tier: MemoryTier, importance: f32, decay_rate: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            tier,
            content,
            metadata: MemoryMetadata::new(importance, decay_rate),
            associations: Vec::new(),
            persona: None,
            raci: RaciRoles::default(),
            model: ModelMemoryMetadata::default(),
        }
    }
}

/// Arguments for the `remember` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RememberArgs {
    /// The content to store as a memory.
    pub content: String,
    /// Optional tier override; defaults to `Episodic`.
    pub tier: Option<MemoryTier>,
    /// Optional importance override in [0, 1].
    pub importance: Option<f32>,
    /// Optional pre-existing association IDs.
    pub associations: Option<Vec<Uuid>>,
}

/// Arguments for the `recall` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecallArgs {
    /// The query string to match against memories.
    pub query: String,
    /// Optional tier filter.
    pub tier: Option<MemoryTier>,
    /// Optional maximum number of results.
    pub limit: Option<usize>,
    /// Optional minimum activation threshold for returned memories.
    #[serde(default)]
    pub min_activation: Option<f32>,
}

/// Result of a successful `remember` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberResult {
    /// The ID of the newly stored memory.
    pub memory_id: Uuid,
    /// The final tier used to store the memory.
    pub tier: MemoryTier,
    /// The final importance used to store the memory.
    pub importance: f32,
    /// Persisted model-derived metadata captured during remember.
    pub model: ModelMemoryMetadata,
    /// Human-readable confirmation message.
    pub message: String,
}

impl RememberResult {
    /// Creates a success result for the given memory ID.
    pub fn success(memory: &CognitiveMemoryUnit) -> Self {
        Self {
            memory_id: memory.id,
            tier: memory.tier,
            importance: memory.metadata.importance,
            model: memory.model.clone(),
            message: "Memory stored successfully".to_string(),
        }
    }
}

/// Result of a `recall` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// The recalled memories, ordered by relevance.
    pub memories: Vec<MemorySummary>,
}

impl RecallResult {
    /// Creates a recall result from a list of memory summaries.
    pub fn new(memories: Vec<MemorySummary>) -> Self {
        Self { memories }
    }
}

/// A lightweight summary of a memory, used in recall and search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// The memory's unique identifier.
    pub id: Uuid,
    /// The memory's content text.
    pub content: String,
    /// The memory's tier.
    pub tier: MemoryTier,
    /// The memory's current activation level.
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

/// Arguments for the `associate` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssociateArgs {
    /// Source memory ID.
    #[serde(default)]
    pub from: Uuid,
    /// Target memory ID.
    #[serde(default)]
    pub to: Uuid,
    /// Optional association strength in [0, 1].
    pub strength: Option<f32>,
}

/// Result of a successful `associate` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociateResult {
    /// Source memory ID.
    pub from: Uuid,
    /// Target memory ID.
    pub to: Uuid,
    /// The association strength.
    pub strength: f32,
    /// Human-readable confirmation message.
    pub message: String,
}

impl AssociateResult {
    /// Creates a success result for the given association.
    pub fn success(from: Uuid, to: Uuid, strength: f32) -> Self {
        Self {
            from,
            to,
            strength,
            message: "Association created successfully".to_string(),
        }
    }
}

/// Arguments for the `forget` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetArgs {
    /// The memory to forget.
    pub memory_id: Uuid,
    /// If `true`, permanently delete; if `false` or `None`, soft-delete by setting activation near-zero.
    #[serde(default)]
    pub hard_delete: Option<bool>,
}

/// Result of a `forget` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetResult {
    /// The memory ID that was targeted.
    pub memory_id: Uuid,
    /// Whether the memory was permanently deleted.
    pub deleted: bool,
    /// Human-readable description of the outcome.
    pub message: String,
}

impl ForgetResult {
    /// Creates a result for a hard-deleted memory.
    pub fn hard_deleted(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: true,
            message: "Memory permanently deleted".to_string(),
        }
    }

    /// Creates a result for a soft-deleted memory (activation set to near-zero).
    pub fn soft_deleted(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: false,
            message: "Memory marked for pruning (activation set to near-zero)".to_string(),
        }
    }

    /// Creates a result when the target memory was not found.
    pub fn not_found(memory_id: Uuid) -> Self {
        Self {
            memory_id,
            deleted: false,
            message: "Memory not found".to_string(),
        }
    }
}

/// A detected conflict between two similar memories in the same tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// First conflicting memory ID.
    pub memory_id_1: Uuid,
    /// Second conflicting memory ID.
    pub memory_id_2: Uuid,
    /// Cosine similarity between the two memories' embeddings.
    pub similarity: f32,
    /// The tier in which the conflict was detected.
    pub tier: MemoryTier,
}

/// Strategy for resolving memory conflicts.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConflictResolution {
    /// Keep the most recently created memory, remove the older one.
    #[default]
    LatestWins,
    /// Keep both conflicting memories.
    KeepBoth,
    /// Boost importance of both and defer to human judgment.
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

/// Arguments for the `reflect` (consolidation) operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectArgs {
    /// Optional intensity: `"light"` for decay only, `"full"` for decay + prune + promote.
    #[serde(default)]
    pub intensity: Option<String>,
    /// Optional conflict resolution strategy name.
    #[serde(default)]
    pub conflict_strategy: Option<String>,
}

/// Result of a `reflect` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectResult {
    /// Number of memories pruned (removed).
    pub pruned_count: usize,
    /// Number of memories promoted to a higher tier.
    pub promoted_count: usize,
    /// Number of memories that had their activation decayed.
    pub decayed_count: usize,
    /// Conflicts detected during consolidation.
    pub conflicts: Vec<Conflict>,
}

impl ReflectResult {
    /// Creates a reflect result with the given counts and conflicts.
    pub fn new(
        pruned_count: usize,
        promoted_count: usize,
        decayed_count: usize,
        conflicts: Vec<Conflict>,
    ) -> Self {
        Self {
            pruned_count,
            promoted_count,
            decayed_count,
            conflicts,
        }
    }
}

/// Arguments for the `search` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchArgs {
    /// The search query string.
    pub query: String,
    /// Optional tier filter.
    pub tier: Option<MemoryTier>,
    /// Optional maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// A single search result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The memory's unique identifier.
    pub id: Uuid,
    /// A text snippet from the memory content.
    pub snippet: String,
    /// The memory's tier.
    pub tier: MemoryTier,
    /// The memory's current activation level.
    pub activation: f32,
}

/// Aggregated search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// The matched search results.
    pub results: Vec<SearchResult>,
}

impl SearchResults {
    /// Creates search results from a list of result items.
    pub fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}

/// Arguments for the `timeline` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineArgs {
    /// The center memory ID for the timeline.
    pub memory_id: Uuid,
    /// Optional time window in seconds; defaults to 3600 (1 hour).
    #[serde(default)]
    pub window_secs: Option<i64>,
}

/// Result of a `timeline` operation, showing memories before and after a center point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineResult {
    /// The center memory.
    pub center: MemorySummary,
    /// Memories created before the center memory within the window.
    pub before: Vec<MemorySummary>,
    /// Memories created after the center memory within the window.
    pub after: Vec<MemorySummary>,
}

/// Arguments for the `get_observations` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GetObservationsArgs {
    /// The memory ID to retrieve full details for.
    pub memory_id: Uuid,
}

/// Result of a `get_observations` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationsResult {
    /// The full memory unit.
    pub memory: CognitiveMemoryUnit,
}

/// Arguments for the `execute_skill` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecuteSkillArgs {
    /// Name of the skill to execute.
    pub skill_name: String,
}

/// Result of an `execute_skill` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteSkillResult {
    /// The skill's unique identifier.
    pub skill_id: Uuid,
    /// The skill's name.
    pub skill_name: String,
    /// The extracted pattern common to source memories.
    pub pattern: String,
    /// The distilled procedural steps.
    pub steps: Vec<String>,
    /// Number of source memories that contributed to this skill.
    pub source_count: usize,
    /// Whether the skill was executed successfully.
    pub executed: bool,
    /// Exit code returned by the sandboxed skill runtime.
    pub exit_code: i32,
}

/// A procedural skill distilled from repeated patterns across memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMemory {
    /// The skill's name.
    pub name: String,
    /// The common pattern extracted from source memories.
    pub pattern: String,
    /// Procedural steps distilled from source memories.
    pub steps: Vec<String>,
    /// IDs of the source memories this skill was derived from.
    pub source_ids: Vec<Uuid>,
}

impl SkillMemory {
    /// Creates a new skill memory.
    pub fn new(name: String, pattern: String, steps: Vec<String>, source_ids: Vec<Uuid>) -> Self {
        Self {
            name,
            pattern,
            steps,
            source_ids,
        }
    }
}

/// Arguments for the `assign_role` operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssignRoleArgs {
    /// The memory ID to assign roles to.
    pub memory_id: Uuid,
    /// The responsible party.
    #[serde(default)]
    pub responsible: Option<String>,
    /// The accountable party.
    #[serde(default)]
    pub accountable: Option<String>,
    /// Parties to consult.
    #[serde(default)]
    pub consulted: Option<Vec<String>>,
    /// Parties to inform.
    #[serde(default)]
    pub informed: Option<Vec<String>>,
}

/// Result of an `assign_role` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignRoleResult {
    /// The memory ID that was updated.
    pub memory_id: Uuid,
    /// The assigned RACI roles.
    pub raci: RaciRoles,
    /// Human-readable confirmation message.
    pub message: String,
}

/// A persona profile summarizing memories in a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaProfile {
    /// The persona domain.
    pub domain: PersonaDomain,
    /// A textual summary of the profile.
    pub summary: String,
    /// Source memory IDs contributing to this profile.
    pub source_ids: Vec<Uuid>,
    /// Confidence score in [0, 1] based on keyword match density.
    pub confidence: f32,
}

/// Result of an `extract_persona` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractPersonaResult {
    /// The extracted persona profiles, one per detected domain.
    pub profiles: Vec<PersonaProfile>,
}
