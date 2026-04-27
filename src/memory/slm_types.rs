use super::types::{ConflictResolution, MemoryTier, PersonaDomain, PersonaProfile};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlmMetadata {
    pub model: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressMemoryInput {
    pub content: String,
    pub tier_hint: Option<MemoryTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressMemoryOutput {
    pub summary: String,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyMemoryInput {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationSuggestion {
    pub memory_id: Option<Uuid>,
    pub label: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyMemoryOutput {
    pub tier: MemoryTier,
    pub importance: f32,
    pub suppress: bool,
    pub tags: Vec<String>,
    pub associations: Vec<AssociationSuggestion>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidatesInput {
    pub query: String,
    pub candidates: Vec<RerankCandidateInput>,
    pub top_n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidateInput {
    pub id: Uuid,
    pub content: String,
    pub initial_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidatesOutput {
    pub ranked_ids: Vec<Uuid>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    Duplicate,
    Contradiction,
    Complement,
    Unrelated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveConflictInput {
    pub memory_a_id: Uuid,
    pub memory_a_content: String,
    pub memory_b_id: Uuid,
    pub memory_b_content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveConflictOutput {
    pub kind: ConflictKind,
    pub action: ConflictResolution,
    pub merged_summary: Option<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractPersonaMemoryInput {
    pub id: Uuid,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractPersonaInput {
    pub memories: Vec<ExtractPersonaMemoryInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaFact {
    pub domain: PersonaDomain,
    pub summary: String,
    pub source_ids: Vec<Uuid>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractPersonaOutput {
    pub profiles: Vec<PersonaProfile>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillSkillInput {
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillSkillOutput {
    pub name: String,
    pub pattern: String,
    pub steps: Vec<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletePatternInput {
    pub cue: String,
    pub context: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletePatternOutput {
    pub completed_text: String,
    pub evidence: Vec<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub turn_id: Uuid,
    pub content: String,
    pub tool_usage: Vec<String>,
    pub decisions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeTurnInput {
    pub turns: Vec<TurnSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeTurnOutput {
    pub summary: String,
    pub key_decisions: Vec<String>,
    pub key_actions: Vec<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSummary {
    pub task_id: Option<Uuid>,
    pub title: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeSessionInput {
    pub turns: Vec<TurnSummary>,
    pub completed_tasks: Vec<TaskSummary>,
    pub open_tasks: Vec<TaskSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeSessionOutput {
    pub summary: String,
    pub completed: Vec<String>,
    pub unresolved: Vec<String>,
    pub next_steps: Vec<String>,
    pub handoff_context: Option<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractBestPracticeInput {
    pub content: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub principle: String,
    pub description: String,
    pub applies_to: Vec<String>,
    pub example: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractBestPracticeOutput {
    pub practices: Vec<BestPractice>,
    pub confidence: f32,
    pub should_persist: bool,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegateInput {
    pub query: String,
    #[serde(default)]
    pub context: Vec<String>,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegateOutput {
    pub response: String,
    pub delegated: bool,
    pub confidence: f32,
    pub model_used: String,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachFromDemonstrationInput {
    pub demonstration: String,
    pub pattern_extracted: String,
    pub domain: Option<String>,
    pub source_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeachFromDemonstrationOutput {
    pub episodic_memory_id: Uuid,
    pub skill_pending: bool,
    pub promotion_candidates: Vec<Uuid>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagEmotionInput {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagEmotionOutput {
    pub valence: f32,
    pub arousal: f32,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRelevanceInput {
    pub query: String,
    pub candidate_content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRelevanceOutput {
    pub relevance: f32,
    pub reasoning: Option<String>,
    pub metadata: SlmMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatePerspectiveInput {
    pub perspective_role: String,
    pub situation: String,
    pub question: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatePerspectiveOutput {
    pub reasoning: String,
    pub recommendation: String,
    pub confidence: f32,
    pub alternative_perspectives: Vec<String>,
}
