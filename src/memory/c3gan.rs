use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorMemory {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub tier: String,
    pub created_at: i64,
}

pub struct C3GAN {
    anchors: VecDeque<AnchorMemory>,
    max_anchors: usize,
    replay_ratio: f32,
}

impl C3GAN {
    pub fn new(max_anchors: usize, replay_ratio: f32) -> Self {
        Self {
            anchors: VecDeque::with_capacity(max_anchors),
            max_anchors,
            replay_ratio,
        }
    }

    pub fn add_anchor(&mut self, id: Uuid, content: String, embedding: Vec<f32>, tier: String, created_at: i64) {
        if self.anchors.len() >= self.max_anchors {
            self.anchors.pop_front();
        }
        self.anchors.push_back(AnchorMemory {
            id,
            content,
            embedding,
            tier,
            created_at,
        });
    }

    pub fn get_replay_count(&self, new_count: usize) -> usize {
        ((new_count as f32) * self.replay_ratio).ceil() as usize
    }

    pub fn sample_anchors(&self, count: usize) -> Vec<&AnchorMemory> {
        if self.anchors.is_empty() || count == 0 {
            return Vec::new();
        }
        let step = (self.anchors.len() / count.max(1)).max(1);
        self.anchors
            .iter()
            .step_by(step)
            .take(count)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }
}

impl Default for C3GAN {
    fn default() -> Self {
        Self::new(100, 0.1)
    }
}