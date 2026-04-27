use super::graph::MemoryGraph;
use super::slm::{SlmEngine, SlmError};
use super::slm_types::ScoreRelevanceInput;
use super::types::{CognitiveMemoryUnit, MemoryTier};
use chrono::Utc;
use std::collections::HashSet;
use uuid::Uuid;

const INJECTION_RELEVANCE_THRESHOLD: f32 = 0.85;
const INJECTION_CANDIDATE_POOL_SIZE: usize = 10;
const HIGH_IMPORTANCE_THRESHOLD: f32 = 0.6;
const RECENT_WINDOW_SECS: i64 = 3600;

pub struct InjectionDecider {
    injected_this_session: HashSet<Uuid>,
    recent_queries: Vec<String>,
}

impl Default for InjectionDecider {
    fn default() -> Self {
        Self::new()
    }
}

impl InjectionDecider {
    pub fn new() -> Self {
        Self {
            injected_this_session: HashSet::new(),
            recent_queries: Vec::new(),
        }
    }

    pub fn gather_candidates<'a>(
        &self,
        graph: &'a MemoryGraph,
    ) -> Vec<&'a CognitiveMemoryUnit> {
        let now = Utc::now().timestamp();
        let mut candidates: Vec<&CognitiveMemoryUnit> = graph
            .get_all_memories()
            .into_iter()
            .filter(|m| {
                if self.injected_this_session.contains(&m.id) {
                    return false;
                }
                matches!(m.tier, MemoryTier::Episodic | MemoryTier::Semantic)
                    && (m.metadata.importance >= HIGH_IMPORTANCE_THRESHOLD
                        || now - m.metadata.last_accessed < RECENT_WINDOW_SECS)
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.metadata
                .importance
                .partial_cmp(&a.metadata.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.metadata
                        .base_activation
                        .partial_cmp(&a.metadata.base_activation)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        candidates.truncate(INJECTION_CANDIDATE_POOL_SIZE);
        candidates
    }

    pub async fn find_best_candidate(
        &self,
        query: &str,
        candidates: &[&CognitiveMemoryUnit],
        slm: &dyn SlmEngine,
    ) -> Result<Option<CognitiveMemoryUnit>, SlmError> {
        for candidate in candidates {
            let output = slm
                .score_relevance(ScoreRelevanceInput {
                    query: query.to_string(),
                    candidate_content: candidate.content.clone(),
                })
                .await?;
            if output.relevance >= INJECTION_RELEVANCE_THRESHOLD {
                return Ok(Some((*candidate).clone()));
            }
        }
        Ok(None)
    }

    pub fn record_injection(&mut self, memory_id: Uuid) {
        self.injected_this_session.insert(memory_id);
    }

    pub fn record_query(&mut self, query: String) {
        self.recent_queries.push(query);
        if self.recent_queries.len() > 20 {
            self.recent_queries.remove(0);
        }
    }

    pub fn has_recent_query(&self, query: &str) -> bool {
        self.recent_queries
            .iter()
            .any(|q| q.eq_ignore_ascii_case(query))
    }

    pub fn reset_session(&mut self) {
        self.injected_this_session.clear();
        self.recent_queries.clear();
    }

    pub fn injected_count(&self) -> usize {
        self.injected_this_session.len()
    }
}
