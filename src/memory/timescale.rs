use super::graph::MemoryGraph;
use super::types::{CognitiveMemoryUnit, MemoryTier};
use chrono::Utc;
use uuid::Uuid;

const EXPLORE_BONUS_BASE: f32 = 0.3;
const EXPLOIT_BONUS_BASE: f32 = 0.7;
const FAST_TIMESCALE_ACCESS_WINDOW: i64 = 3600;
const STDP_STRENGTHEN_WINDOW: i64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimescaleKind {
    Fast,
    Slow,
}

impl TimescaleKind {
    pub fn from_tier(tier: MemoryTier) -> Self {
        match tier {
            MemoryTier::Sensory | MemoryTier::Working => TimescaleKind::Fast,
            MemoryTier::Episodic | MemoryTier::Semantic | MemoryTier::Procedural => {
                TimescaleKind::Slow
            }
        }
    }
}

pub struct DualTimescaleManager {
    explore_weight: f32,
    exploit_weight: f32,
}

impl Default for DualTimescaleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DualTimescaleManager {
    pub fn new() -> Self {
        Self {
            explore_weight: EXPLORE_BONUS_BASE,
            exploit_weight: EXPLOIT_BONUS_BASE,
        }
    }

    pub fn with_weights(explore: f32, exploit: f32) -> Self {
        Self {
            explore_weight: explore,
            exploit_weight: exploit,
        }
    }

    pub fn compute_recall_score(&self, memory: &CognitiveMemoryUnit, now: i64) -> f32 {
        let timescale = TimescaleKind::from_tier(memory.tier);
        let base = memory.metadata.base_activation;

        match timescale {
            TimescaleKind::Fast => self.exploit_bonus(base, &memory.metadata, now),
            TimescaleKind::Slow => self.explore_bonus(base, &memory.metadata, now),
        }
    }

    fn exploit_bonus(&self, base: f32, metadata: &super::types::MemoryMetadata, now: i64) -> f32 {
        let recency_bonus = if now - metadata.last_accessed < FAST_TIMESCALE_ACCESS_WINDOW {
            1.0
        } else {
            0.5
        };
        let access_count_factor = (metadata.access_count as f32).min(5.0) / 5.0;
        (self.exploit_weight * base * recency_bonus * (0.5 + 0.5 * access_count_factor))
            + (1.0 - self.exploit_weight) * base
    }

    fn explore_bonus(&self, base: f32, metadata: &super::types::MemoryMetadata, now: i64) -> f32 {
        let access_count_factor = 1.0 / (1.0 + metadata.access_count as f32 * 0.2);
        let age_bonus = if now - metadata.created_at > 86400 {
            1.2
        } else {
            1.0
        };
        (self.explore_weight * base * access_count_factor * age_bonus)
            + (1.0 - self.explore_weight) * base
    }
}

pub fn apply_stdp(graph: &mut MemoryGraph, source_id: &Uuid, target_id: &Uuid, now: i64) {
    let source_ts = {
        let m = match graph.get_memory(source_id) {
            Some(m) => m,
            None => return,
        };
        m.metadata.last_accessed
    };

    if graph.get_memory(target_id).is_none() {
        return;
    }

    let time_diff = (now - source_ts).abs();
    let strengthen = time_diff < STDP_STRENGTHEN_WINDOW;
    let current = graph.get_association_strength(source_id, target_id).unwrap_or(0.5);

    let new_strength = if strengthen {
        (current + 0.1).min(1.0)
    } else if time_diff > STDP_STRENGTHEN_WINDOW * 10 {
        (current - 0.05).max(0.05)
    } else {
        current
    };

    graph.update_association(source_id, target_id, new_strength);
}

pub fn rank_by_timescale(graph: &mut MemoryGraph, query_weights: Vec<(Uuid, f32)>) -> Vec<(Uuid, f32)> {
    let now = Utc::now().timestamp();
    let manager = DualTimescaleManager::new();

    let mut scored: Vec<(Uuid, f32)> = query_weights
        .into_iter()
        .map(|(id, base_score)| {
            let memory = match graph.get_memory(&id) {
                Some(m) => m,
                None => return (id, base_score),
            };
            let timescale_score = manager.compute_recall_score(&memory, now);
            let final_score = base_score * 0.6 + timescale_score * 0.4;
            (id, final_score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timescale_from_tier() {
        assert_eq!(TimescaleKind::from_tier(MemoryTier::Sensory), TimescaleKind::Fast);
        assert_eq!(TimescaleKind::from_tier(MemoryTier::Working), TimescaleKind::Fast);
        assert_eq!(TimescaleKind::from_tier(MemoryTier::Episodic), TimescaleKind::Slow);
        assert_eq!(TimescaleKind::from_tier(MemoryTier::Semantic), TimescaleKind::Slow);
    }

    #[test]
    fn test_manager_default_weights() {
        let manager = DualTimescaleManager::new();
        assert_eq!(manager.explore_weight, EXPLORE_BONUS_BASE);
        assert_eq!(manager.exploit_weight, EXPLOIT_BONUS_BASE);
    }

    #[test]
    fn test_manager_custom_weights() {
        let manager = DualTimescaleManager::with_weights(0.4, 0.6);
        assert_eq!(manager.explore_weight, 0.4);
        assert_eq!(manager.exploit_weight, 0.6);
    }
}