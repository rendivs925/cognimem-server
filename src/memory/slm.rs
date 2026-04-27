use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, DelegateInput, DelegateOutput,
    DistillSkillInput, DistillSkillOutput, DreamInput, DreamOutput,
    ExtractBestPracticeInput, ExtractBestPracticeOutput, ImagineInput, ImagineOutput,
    ExtractPersonaInput, ExtractPersonaOutput, ScoreRelevanceInput, ScoreRelevanceOutput,
    SimulatePerspectiveInput, SimulatePerspectiveOutput,
    TagEmotionInput, TagEmotionOutput, TeachFromDemonstrationInput, TeachFromDemonstrationOutput,
    RerankCandidatesInput, RerankCandidatesOutput, ResolveConflictInput, ResolveConflictOutput,
    SummarizeSessionInput, SummarizeSessionOutput, SummarizeTurnInput,
    SummarizeTurnOutput,
};
use async_trait::async_trait;
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

#[async_trait]
pub trait SlmEngine: Send + Sync {
    fn model_name(&self) -> &str;
    async fn compress_memory(
        &self,
        input: CompressMemoryInput,
    ) -> Result<CompressMemoryOutput, SlmError>;
    async fn classify_memory(
        &self,
        input: ClassifyMemoryInput,
    ) -> Result<ClassifyMemoryOutput, SlmError>;
    async fn rerank_candidates(
        &self,
        input: RerankCandidatesInput,
    ) -> Result<RerankCandidatesOutput, SlmError>;
    async fn resolve_conflict(
        &self,
        input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError>;
    async fn extract_persona(
        &self,
        input: ExtractPersonaInput,
    ) -> Result<ExtractPersonaOutput, SlmError>;
    async fn distill_skill(&self, input: DistillSkillInput)
    -> Result<DistillSkillOutput, SlmError>;
    async fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError>;
    async fn summarize_turn(
        &self,
        input: SummarizeTurnInput,
    ) -> Result<SummarizeTurnOutput, SlmError>;
    async fn summarize_session(
        &self,
        input: SummarizeSessionInput,
    ) -> Result<SummarizeSessionOutput, SlmError>;
    async fn extract_best_practice(
        &self,
        input: ExtractBestPracticeInput,
    ) -> Result<ExtractBestPracticeOutput, SlmError>;
    async fn delegate_to_llm(
        &self,
        input: DelegateInput,
    ) -> Result<DelegateOutput, SlmError>;
    async fn teach_from_demonstration(
        &self,
        input: TeachFromDemonstrationInput,
    ) -> Result<TeachFromDemonstrationOutput, SlmError>;
    async fn simulate_perspective(
        &self,
        input: SimulatePerspectiveInput,
    ) -> Result<SimulatePerspectiveOutput, SlmError>;
    async fn tag_emotion(&self, input: TagEmotionInput) -> Result<TagEmotionOutput, SlmError>;
    async fn score_relevance(
        &self,
        input: ScoreRelevanceInput,
    ) -> Result<ScoreRelevanceOutput, SlmError>;
    async fn dream(&self, input: DreamInput) -> Result<DreamOutput, SlmError>;
    async fn imagine(&self, input: ImagineInput) -> Result<ImagineOutput, SlmError>;
}

pub struct NoOpSlm;

const NOOP_ERROR: &str = "NoOpSlm requires a configured LLM provider (Ollama)";

#[async_trait]
impl SlmEngine for NoOpSlm {
    fn model_name(&self) -> &str {
        "noop"
    }

    async fn compress_memory(
        &self,
        input: CompressMemoryInput,
    ) -> Result<CompressMemoryOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn classify_memory(
        &self,
        input: ClassifyMemoryInput,
    ) -> Result<ClassifyMemoryOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn rerank_candidates(
        &self,
        input: RerankCandidatesInput,
    ) -> Result<RerankCandidatesOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn resolve_conflict(
        &self,
        input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn extract_persona(
        &self,
        input: ExtractPersonaInput,
    ) -> Result<ExtractPersonaOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn distill_skill(
        &self,
        input: DistillSkillInput,
    ) -> Result<DistillSkillOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn summarize_turn(
        &self,
        input: SummarizeTurnInput,
    ) -> Result<SummarizeTurnOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn summarize_session(
        &self,
        input: SummarizeSessionInput,
    ) -> Result<SummarizeSessionOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn extract_best_practice(
        &self,
        input: ExtractBestPracticeInput,
    ) -> Result<ExtractBestPracticeOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn delegate_to_llm(
        &self,
        input: DelegateInput,
    ) -> Result<DelegateOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn teach_from_demonstration(
        &self,
        input: TeachFromDemonstrationInput,
    ) -> Result<TeachFromDemonstrationOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn simulate_perspective(
        &self,
        input: SimulatePerspectiveInput,
    ) -> Result<SimulatePerspectiveOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn tag_emotion(&self, input: TagEmotionInput) -> Result<TagEmotionOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn score_relevance(
        &self,
        input: ScoreRelevanceInput,
    ) -> Result<ScoreRelevanceOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn dream(&self, input: DreamInput) -> Result<DreamOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }

    async fn imagine(&self, input: ImagineInput) -> Result<ImagineOutput, SlmError> {
        drop(input);
        Err(SlmError::RequestFailed(NOOP_ERROR.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_compress_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .compress_memory(CompressMemoryInput {
                content: "test content".into(),
                tier_hint: None,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_classify_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .classify_memory(ClassifyMemoryInput {
                content: "test".into(),
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_rerank_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .rerank_candidates(RerankCandidatesInput {
                query: "test".into(),
                candidates: vec![],
                top_n: 3,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_resolve_conflict_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .resolve_conflict(ResolveConflictInput {
                memory_a_id: uuid::Uuid::new_v4(),
                memory_a_content: "a".into(),
                memory_b_id: uuid::Uuid::new_v4(),
                memory_b_content: "b".into(),
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_extract_persona_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .extract_persona(ExtractPersonaInput { memories: vec![] })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_distill_skill_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .distill_skill(DistillSkillInput {
                examples: vec!["test".into()],
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_complete_pattern_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .complete_pattern(CompletePatternInput {
                cue: "partial cue".into(),
                context: vec![],
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_summarize_turn_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .summarize_turn(SummarizeTurnInput { turns: vec![] })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_summarize_session_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .summarize_session(SummarizeSessionInput {
                turns: vec![],
                completed_tasks: vec![],
                open_tasks: vec![],
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_extract_best_practice_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .extract_best_practice(ExtractBestPracticeInput {
                content: "test".into(),
                context: None,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_delegate_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .delegate_to_llm(DelegateInput {
                query: "test".into(),
                context: vec![],
                confidence_threshold: 0.7,
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_teach_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .teach_from_demonstration(TeachFromDemonstrationInput {
                demonstration: "test".into(),
                pattern_extracted: "pattern".to_string(),
                domain: None,
                source_type: Some("code_review".to_string()),
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_simulate_returns_error() {
        let slm = NoOpSlm;
        let result = slm
            .simulate_perspective(SimulatePerspectiveInput {
                perspective_role: "security_expert".to_string(),
                question: "?".to_string(),
                situation: "test".to_string(),
            })
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_slm_error_display() {
        let err = SlmError::RequestFailed("test error".to_string());
        assert_eq!(err.to_string(), "request failed: test error");
    }

    #[test]
    fn test_slm_error_validation() {
        let err = SlmError::ValidationFailed("invalid".to_string());
        assert_eq!(err.to_string(), "validation failed: invalid");
    }

    #[test]
    fn test_slm_error_invalid_response() {
        let err = SlmError::InvalidResponse("bad response".to_string());
        assert_eq!(err.to_string(), "invalid response: bad response");
    }
}
