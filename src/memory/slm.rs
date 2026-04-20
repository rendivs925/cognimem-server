use super::types::{ConflictResolution, PersonaProfile};

/// A small-language-model interface for advanced memory operations.
///
/// Implementations provide compression, reranking, conflict resolution,
/// and pattern completion powered by an LLM.
pub trait SlmEngine: Send {
    /// Compresses the given content into a shorter representation.
    fn compress(&self, content: &str) -> String;
    /// Reranks candidates by relevance to the query, returning indices of the top `top_n`.
    fn rerank(&self, candidates: &[RerankCandidate], query: &str, top_n: usize) -> Vec<usize>;
    /// Decides how to resolve a conflict between two memory contents.
    fn resolve_conflict(&self, content_a: &str, content_b: &str) -> ConflictResolution;
    /// Generates a completion hint for a partial pattern cue given surrounding context.
    fn complete_pattern_hint(&self, partial: &str, context: &[&str]) -> String;
    /// Extracts structured persona profiles from memory content.
    fn extract_persona(&self, memories: &[String]) -> Vec<PersonaProfile>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// A candidate for reranking, carrying an ID, content, and initial score.
pub struct RerankCandidate {
    pub id: uuid::Uuid,
    pub content: String,
    pub score: f32,
}

/// A no-op SLM implementation that provides baseline behavior without an actual model.
///
/// - `compress`: truncates to the first 20 words.
/// - `rerank`: sorts candidates by existing score in descending order.
/// - `resolve_conflict`: always returns `LatestWins`.
/// - `complete_pattern_hint`: returns the partial cue unchanged.
pub struct NoOpSlm;

impl SlmEngine for NoOpSlm {
    fn compress(&self, content: &str) -> String {
        content
            .split_whitespace()
            .take(20)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn rerank(&self, candidates: &[RerankCandidate], _query: &str, top_n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.score))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(top_n).map(|(i, _)| i).collect()
    }

    fn resolve_conflict(&self, _content_a: &str, _content_b: &str) -> ConflictResolution {
        ConflictResolution::LatestWins
    }

    fn complete_pattern_hint(&self, partial: &str, _context: &[&str]) -> String {
        partial.to_string()
    }

    fn extract_persona(&self, _memories: &[String]) -> Vec<PersonaProfile> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_compress_truncates() {
        let slm = NoOpSlm;
        let long = "this is a very long sentence that should be compressed down to just the first twenty words because the no op implementation simply truncates";
        let compressed = slm.compress(long);
        assert!(compressed.split_whitespace().count() <= 20);
    }

    #[test]
    fn test_noop_rerank_sorts_by_score() {
        let slm = NoOpSlm;
        let candidates = vec![
            RerankCandidate {
                id: uuid::Uuid::new_v4(),
                content: "low".into(),
                score: 0.2,
            },
            RerankCandidate {
                id: uuid::Uuid::new_v4(),
                content: "high".into(),
                score: 0.9,
            },
            RerankCandidate {
                id: uuid::Uuid::new_v4(),
                content: "mid".into(),
                score: 0.5,
            },
        ];
        let result = slm.rerank(&candidates, "test", 3);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 0);
    }

    #[test]
    fn test_noop_resolve_conflict_returns_latest() {
        let slm = NoOpSlm;
        assert_eq!(
            slm.resolve_conflict("a", "b"),
            ConflictResolution::LatestWins
        );
    }

    #[test]
    fn test_noop_complete_pattern_returns_input() {
        let slm = NoOpSlm;
        assert_eq!(slm.complete_pattern_hint("partial cue", &[]), "partial cue");
    }
}
