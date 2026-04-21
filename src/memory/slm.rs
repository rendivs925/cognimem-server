use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, ConflictKind, DistillSkillInput,
    DistillSkillOutput, ExtractPersonaInput, ExtractPersonaOutput, RerankCandidatesInput,
    RerankCandidatesOutput, ResolveConflictInput, ResolveConflictOutput, SlmMetadata,
};
use super::types::{ConflictResolution, MemoryTier};

pub const DEFAULT_SLM_MODEL: &str = "qwen2.5-coder:3b";

pub trait SlmEngine: Send {
    fn model_name(&self) -> &str;
    fn compress_memory(&self, input: CompressMemoryInput) -> Option<CompressMemoryOutput>;
    fn classify_memory(&self, input: ClassifyMemoryInput) -> Option<ClassifyMemoryOutput>;
    fn rerank_candidates(&self, input: RerankCandidatesInput) -> Option<RerankCandidatesOutput>;
    fn resolve_conflict(&self, input: ResolveConflictInput) -> Option<ResolveConflictOutput>;
    fn extract_persona(&self, input: ExtractPersonaInput) -> Option<ExtractPersonaOutput>;
    fn distill_skill(&self, input: DistillSkillInput) -> Option<DistillSkillOutput>;
    fn complete_pattern(&self, input: CompletePatternInput) -> Option<CompletePatternOutput>;
}

pub struct NoOpSlm;

impl SlmEngine for NoOpSlm {
    fn model_name(&self) -> &str {
        "noop"
    }

    fn compress_memory(&self, input: CompressMemoryInput) -> Option<CompressMemoryOutput> {
        Some(CompressMemoryOutput {
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

    fn classify_memory(&self, _input: ClassifyMemoryInput) -> Option<ClassifyMemoryOutput> {
        Some(ClassifyMemoryOutput {
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

    fn rerank_candidates(&self, input: RerankCandidatesInput) -> Option<RerankCandidatesOutput> {
        let mut candidates = input.candidates;
        candidates.sort_by(|a, b| {
            b.initial_score
                .partial_cmp(&a.initial_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(RerankCandidatesOutput {
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

    fn resolve_conflict(&self, _input: ResolveConflictInput) -> Option<ResolveConflictOutput> {
        Some(ResolveConflictOutput {
            kind: ConflictKind::Unrelated,
            action: ConflictResolution::LatestWins,
            merged_summary: None,
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.1,
            },
        })
    }

    fn extract_persona(&self, _input: ExtractPersonaInput) -> Option<ExtractPersonaOutput> {
        Some(ExtractPersonaOutput {
            profiles: Vec::new(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.1,
            },
        })
    }

    fn distill_skill(&self, input: DistillSkillInput) -> Option<DistillSkillOutput> {
        let pattern = input.examples.first().cloned().unwrap_or_default();
        Some(DistillSkillOutput {
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

    fn complete_pattern(&self, input: CompletePatternInput) -> Option<CompletePatternOutput> {
        Some(CompletePatternOutput {
            completed_text: input.cue,
            evidence: input.context.into_iter().take(3).collect(),
            metadata: SlmMetadata {
                model: self.model_name().to_string(),
                confidence: 0.2,
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
