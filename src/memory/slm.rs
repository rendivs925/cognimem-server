use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, ConflictKind, DistillSkillInput,
    DistillSkillOutput, ExtractBestPracticeInput, ExtractBestPracticeOutput, ExtractPersonaInput,
    ExtractPersonaOutput, RerankCandidatesInput, RerankCandidatesOutput, ResolveConflictInput,
    ResolveConflictOutput, SlmMetadata, SummarizeSessionInput, SummarizeSessionOutput,
    SummarizeTurnInput, SummarizeTurnOutput,
};
use super::types::{ConflictResolution, MemoryTier};
use std::fmt;

pub const DEFAULT_SLM_MODEL: &str = "qwen2.5-coder:3b";

#[derive(Debug, Clone)]
pub enum SlmError {
    RequestFailed(String),
    InvalidResponse(String),
    ValidationFailed(String),
}

impl fmt::Display for SlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlmError::RequestFailed(message) => write!(f, "request failed: {message}"),
            SlmError::InvalidResponse(message) => write!(f, "invalid response: {message}"),
            SlmError::ValidationFailed(message) => write!(f, "validation failed: {message}"),
        }
    }
}

impl std::error::Error for SlmError {}

pub trait SlmEngine: Send {
    fn model_name(&self) -> &str;
    fn compress_memory(&self, input: CompressMemoryInput) -> Result<CompressMemoryOutput, SlmError>;
    fn classify_memory(&self, input: ClassifyMemoryInput) -> Result<ClassifyMemoryOutput, SlmError>;
    fn rerank_candidates(
        &self,
        input: RerankCandidatesInput,
    ) -> Result<RerankCandidatesOutput, SlmError>;
    fn resolve_conflict(
        &self,
        input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError>;
    fn extract_persona(
        &self,
        input: ExtractPersonaInput,
    ) -> Result<ExtractPersonaOutput, SlmError>;
    fn distill_skill(&self, input: DistillSkillInput) -> Result<DistillSkillOutput, SlmError>;
    fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError>;
    fn summarize_turn(&self, input: SummarizeTurnInput) -> Result<SummarizeTurnOutput, SlmError>;
    fn summarize_session(&self, input: SummarizeSessionInput) -> Result<SummarizeSessionOutput, SlmError>;
    fn extract_best_practice(
        &self,
        input: ExtractBestPracticeInput,
    ) -> Result<ExtractBestPracticeOutput, SlmError>;
}

pub struct NoOpSlm;

impl SlmEngine for NoOpSlm {
    fn model_name(&self) -> &str {
        "noop"
    }

    fn compress_memory(&self, input: CompressMemoryInput) -> Result<CompressMemoryOutput, SlmError> {
        Ok(CompressMemoryOutput {
            summary: input
                .content
                .split_whitespace()
                .take(20)
                .collect::<Vec<_>>()
                .join(" "),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.3,
            },
        })
    }

    fn classify_memory(&self, _input: ClassifyMemoryInput) -> Result<ClassifyMemoryOutput, SlmError> {
        Ok(ClassifyMemoryOutput {
            tier: MemoryTier::Episodic,
            importance: 0.5,
            suppress: false,
            tags: Vec::new(),
            associations: Vec::new(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
            },
        })
    }

    fn rerank_candidates(
        &self,
        input: RerankCandidatesInput,
    ) -> Result<RerankCandidatesOutput, SlmError> {
        let mut candidates = input.candidates;
        candidates.sort_by(|a, b| {
            b.initial_score
                .partial_cmp(&a.initial_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(RerankCandidatesOutput {
            ranked_ids: candidates
                .into_iter()
                .take(input.top_n)
                .map(|c| c.id)
                .collect(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.25,
            },
        })
    }

    fn resolve_conflict(
        &self,
        _input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError> {
        Ok(ResolveConflictOutput {
            kind: ConflictKind::Unrelated,
            action: ConflictResolution::LatestWins,
            merged_summary: None,
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.1,
            },
        })
    }

    fn extract_persona(
        &self,
        _input: ExtractPersonaInput,
    ) -> Result<ExtractPersonaOutput, SlmError> {
        Ok(ExtractPersonaOutput {
            profiles: Vec::new(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.1,
            },
        })
    }

    fn distill_skill(&self, input: DistillSkillInput) -> Result<DistillSkillOutput, SlmError> {
        let pattern = input.examples.first().cloned().unwrap_or_default();
        Ok(DistillSkillOutput {
            name: pattern
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join("_")
                .to_lowercase(),
            pattern,
            steps: input.examples,
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
            },
        })
    }

    fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError> {
        Ok(CompletePatternOutput {
            completed_text: input.cue,
            evidence: input.context.into_iter().take(3).collect(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
            },
        })
    }

    fn summarize_turn(&self, input: SummarizeTurnInput) -> Result<SummarizeTurnOutput, SlmError> {
        let summary = input.turns.first()
            .map(|t| t.content.chars().take(200).collect())
            .unwrap_or_default();
        Ok(SummarizeTurnOutput {
            summary,
            key_decisions: Vec::new(),
            key_actions: Vec::new(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
            },
        })
    }

    fn summarize_session(&self, input: SummarizeSessionInput) -> Result<SummarizeSessionOutput, SlmError> {
        let summary = input.turns.first()
            .map(|t| t.content.chars().take(200).collect())
            .unwrap_or_default();
        Ok(SummarizeSessionOutput {
            summary,
            completed: input.completed_tasks.iter().map(|t| t.title.clone()).collect(),
            unresolved: input.open_tasks.iter().map(|t| t.title.clone()).collect(),
            next_steps: Vec::new(),
            handoff_context: None,
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
            },
        })
    }

    fn extract_best_practice(
        &self,
        input: ExtractBestPracticeInput,
    ) -> Result<ExtractBestPracticeOutput, SlmError> {
        let content_lower = input.content.to_lowercase();
        let mut practices = Vec::new();

        let keywords = [
            ("DRY", "Don't Repeat Yourself - extract common patterns"),
            ("KISS", "Keep It Simple - prefer simple over clever"),
            ("YAGNI", "You Aren't Gonna Need It - don't add features until needed"),
            ("SOLID", "Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion"),
            ("guard clause", "Reject invalid inputs early at function entrance"),
        ];

        for (keyword, principle) in keywords {
            if content_lower.contains(&keyword.to_lowercase()) {
                practices.push(super::slm_types::BestPractice {
                    principle: keyword.to_string(),
                    description: principle.to_string(),
                    applies_to: vec!["current context".to_string()],
                    example: None,
                });
            }
        }

        let should_persist = !practices.is_empty();
        Ok(ExtractBestPracticeOutput {
            practices,
            confidence: 0.3,
            should_persist,
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.3,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::slm_types::RerankCandidateInput;

    #[test]
    fn test_noop_compress_truncates() {
        let slm = NoOpSlm;
        let long = "this is a very long sentence that should be compressed down to just the first twenty words because the no op implementation simply truncates";
        let compressed = slm
            .compress_memory(CompressMemoryInput {
                content: long.to_string(),
                tier_hint: None,
            })
            .unwrap();
        assert!(compressed.summary.split_whitespace().count() <= 20);
    }

    #[test]
    fn test_noop_rerank_sorts_by_score() {
        let slm = NoOpSlm;
        let candidates = vec![
            RerankCandidateInput {
                id: uuid::Uuid::new_v4(),
                content: "low".into(),
                initial_score: 0.2,
            },
            RerankCandidateInput {
                id: uuid::Uuid::new_v4(),
                content: "high".into(),
                initial_score: 0.9,
            },
            RerankCandidateInput {
                id: uuid::Uuid::new_v4(),
                content: "mid".into(),
                initial_score: 0.5,
            },
        ];
        let result = slm
            .rerank_candidates(RerankCandidatesInput {
                query: "test".into(),
                candidates: candidates.clone(),
                top_n: 3,
            })
            .unwrap();
        assert_eq!(result.ranked_ids[0], candidates[1].id);
        assert_eq!(result.ranked_ids[1], candidates[2].id);
        assert_eq!(result.ranked_ids[2], candidates[0].id);
    }

    #[test]
    fn test_noop_resolve_conflict_returns_latest() {
        let slm = NoOpSlm;
        assert_eq!(
            slm.resolve_conflict(ResolveConflictInput {
                memory_a_id: uuid::Uuid::new_v4(),
                memory_a_content: "a".into(),
                memory_b_id: uuid::Uuid::new_v4(),
                memory_b_content: "b".into(),
            })
            .unwrap()
            .action,
            ConflictResolution::LatestWins
        );
    }

    #[test]
    fn test_noop_complete_pattern_returns_input() {
        let slm = NoOpSlm;
        assert_eq!(
            slm.complete_pattern(CompletePatternInput {
                cue: "partial cue".into(),
                context: Vec::new(),
            })
            .unwrap()
            .completed_text,
            "partial cue"
        );
    }
}
