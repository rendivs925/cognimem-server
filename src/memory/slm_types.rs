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
