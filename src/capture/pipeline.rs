use super::types::{CanonicalEvent, CanonicalEventType, IngestResult};
use crate::memory::slm_types::{ClassifyMemoryInput, CompressMemoryInput, TagEmotionInput};
use crate::memory::types::{CognitiveMemoryUnit, EmotionState, MemoryScope, MemorySource, MemoryTier};
use crate::state::CogniMemState;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

const MAX_CONTENT_BYTES: usize = 100_000;
const MAX_FUTURE_SECS: i64 = 60;
const AGGREGATION_TIMEOUT_SECS: i64 = 60;
const MAX_PENDING_AGGREGATIONS: usize = 500;
const DEDUP_WINDOW_SECS: i64 = 5;

static TOTAL_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static TOTAL_SUPPRESSED: AtomicU64 = AtomicU64::new(0);
static TOTAL_STORED: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);

pub fn get_ingest_stats(pending: usize, uptime_secs: u64) -> super::types::IngestStats {
    super::types::IngestStats {
        total_accepted: TOTAL_ACCEPTED.load(Ordering::Relaxed),
        total_suppressed: TOTAL_SUPPRESSED.load(Ordering::Relaxed),
        total_stored: TOTAL_STORED.load(Ordering::Relaxed),
        total_errors: TOTAL_ERRORS.load(Ordering::Relaxed),
        pending_aggregations: pending,
        uptime_secs,
    }
}

struct PendingAggregation {
    before_event: CanonicalEvent,
    received_at: i64,
}

pub struct CapturePipeline {
    state: Arc<Mutex<CogniMemState>>,
    pending_aggregations: HashMap<(String, String), PendingAggregation>,
    suppress_patterns: Vec<String>,
    dedup_cache: HashMap<(CanonicalEventType, String, String), i64>,
    started_at: i64,
}

impl CapturePipeline {
    pub fn new(state: Arc<Mutex<CogniMemState>>) -> Self {
        Self {
            state,
            pending_aggregations: HashMap::new(),
            suppress_patterns: vec![
                "heartbeat".to_string(),
                "ping".to_string(),
                "pong".to_string(),
                "health_check".to_string(),
            ],
            dedup_cache: HashMap::new(),
            started_at: Utc::now().timestamp(),
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending_aggregations.len()
    }

    pub fn uptime_secs(&self) -> u64 {
        let now = Utc::now().timestamp();
        (now - self.started_at).max(0) as u64
    }

    pub async fn ingest_batch(&mut self, events: Vec<CanonicalEvent>) -> IngestResult {
        let mut result = IngestResult::default();
        self.evict_stale_aggregations();

        for event in events {
            match self.process_event(event).await {
                Ok(stored) => {
                    result.accepted += 1;
                    if stored {
                        result.stored += 1;
                    } else {
                        result.suppressed += 1;
                    }
                }
                Err(err) => {
                    result.errors.push(err);
                }
            }
        }

        TOTAL_ACCEPTED.fetch_add(result.accepted as u64, Ordering::Relaxed);
        TOTAL_SUPPRESSED.fetch_add(result.suppressed as u64, Ordering::Relaxed);
        TOTAL_STORED.fetch_add(result.stored as u64, Ordering::Relaxed);
        TOTAL_ERRORS.fetch_add(result.errors.len() as u64, Ordering::Relaxed);

        result
    }

    async fn process_event(&mut self, event: CanonicalEvent) -> Result<bool, String> {
        self.validate_event(&event)?;

        if self.should_suppress(&event) {
            return Ok(false);
        }

        if self.is_duplicate(&event) {
            return Ok(false);
        }

        if event.event_type.is_tool_before() {
            self.handle_tool_before(event);
            return Ok(false);
        }

        let event = if event.event_type.is_tool_after() {
            match self.handle_tool_after(event) {
                Some(merged) => merged,
                None => return Ok(true),
            }
        } else {
            event
        };

        self.store_event(&event).await
    }

    fn validate_event(&self, event: &CanonicalEvent) -> Result<(), String> {
        let now = Utc::now().timestamp();
        if event.timestamp > now + MAX_FUTURE_SECS {
            return Err(format!(
                "timestamp {} is too far in the future (now={})",
                event.timestamp, now
            ));
        }

        if let Some(ref content) = event.content {
            if content.len() > MAX_CONTENT_BYTES {
                return Err(format!(
                    "content exceeds maximum size of {} bytes",
                    MAX_CONTENT_BYTES
                ));
            }
        }

        if let Some(ref input) = event.tool_input {
            let input_str = input.to_string();
            if input_str.len() > MAX_CONTENT_BYTES {
                return Err("tool_input exceeds maximum size".to_string());
            }
        }

        if let Some(ref output) = event.tool_output {
            let output_str = output.to_string();
            if output_str.len() > MAX_CONTENT_BYTES {
                return Err("tool_output exceeds maximum size".to_string());
            }
        }

        Ok(())
    }

    fn should_suppress(&self, event: &CanonicalEvent) -> bool {
        if let Some(ref tool) = event.tool_name {
            let tool_lower = tool.to_lowercase();
            for pattern in &self.suppress_patterns {
                if tool_lower.contains(&pattern.to_lowercase()) {
                    return true;
                }
            }
        }

        if let Some(ref content) = event.content {
            let content_lower = content.to_lowercase();
            for pattern in &self.suppress_patterns {
                if content_lower.contains(&pattern.to_lowercase()) {
                    return true;
                }
            }
        }

        if event.event_type.is_noisy() && event.content.is_none() {
            return true;
        }

        false
    }

    fn is_duplicate(&mut self, event: &CanonicalEvent) -> bool {
        let tool_key = event.tool_name.clone().unwrap_or_default();
        let file_key = event.file_path.clone().unwrap_or_default();
        let cache_key = (event.event_type, tool_key, file_key);
        let now = Utc::now().timestamp();

        if let Some(&prev_ts) = self.dedup_cache.get(&cache_key) {
            if now - prev_ts < DEDUP_WINDOW_SECS {
                return true;
            }
        }

        self.dedup_cache.insert(cache_key, now);

        if self.dedup_cache.len() > 2000 {
            let cutoff = now - DEDUP_WINDOW_SECS;
            self.dedup_cache.retain(|_, ts| *ts > cutoff);
        }

        false
    }

    fn handle_tool_before(&mut self, event: CanonicalEvent) {
        if let Some(key) = event.aggregation_key() {
            if self.pending_aggregations.len() >= MAX_PENDING_AGGREGATIONS {
                self.evict_stale_aggregations();
            }
            self.pending_aggregations.insert(
                key,
                PendingAggregation {
                    before_event: event,
                    received_at: Utc::now().timestamp(),
                },
            );
        }
    }

    fn handle_tool_after(&mut self, after: CanonicalEvent) -> Option<CanonicalEvent> {
        if let Some(key) = after.aggregation_key() {
            if let Some(pending) = self.pending_aggregations.remove(&key) {
                let mut merged = after;
                let duration = if pending.before_event.timestamp > 0 && merged.timestamp > 0 {
                    Some((merged.timestamp - pending.before_event.timestamp).max(0) as u64)
                } else {
                    None
                };
                merged.duration_ms = duration.or(merged.duration_ms);
                merged.tool_input = merged.tool_input.or(pending.before_event.tool_input);
                return Some(merged);
            }
        }
        Some(after)
    }

    fn evict_stale_aggregations(&mut self) {
        let now = Utc::now().timestamp();
        let cutoff = now - AGGREGATION_TIMEOUT_SECS;
        self.pending_aggregations
            .retain(|_, pending| pending.received_at > cutoff);
    }

    async fn store_event(&mut self, event: &CanonicalEvent) -> Result<bool, String> {
        let content = event.compose_content();

        let (tier, importance, tags, suppress, compressed, model_name, confidence, emotion) = {
            let guard = self.state.lock().await;
            let classify_result = guard
                .slm
                .classify_memory(ClassifyMemoryInput {
                    content: content.clone(),
                })
                .await
                .ok();

            let compress_result = guard
                .slm
                .compress_memory(CompressMemoryInput {
                    content: content.clone(),
                    tier_hint: classify_result.as_ref().map(|c| c.tier),
                })
                .await
                .ok();

            let emotion_result = guard
                .slm
                .tag_emotion(TagEmotionInput {
                    content: content.clone(),
                })
                .await
                .ok();

            let tier = classify_result
                .as_ref()
                .map(|c| c.tier)
                .unwrap_or(MemoryTier::Sensory);
            let importance = classify_result
                .as_ref()
                .map(|c| c.importance)
                .unwrap_or(0.3);
            let tags = classify_result
                .as_ref()
                .map(|c| c.tags.clone())
                .unwrap_or_default();
            let suppress = classify_result
                .as_ref()
                .map(|c| c.suppress)
                .unwrap_or(false);
            let compressed = compress_result.map(|c| c.summary).unwrap_or_default();
            let model_name = classify_result
                .as_ref()
                .map(|c| c.metadata.model.clone())
                .unwrap_or_else(|| guard.slm.model_name().to_string());
            let confidence = classify_result
                .as_ref()
                .map(|c| c.metadata.confidence)
                .unwrap_or(0.0);
            let emotion = emotion_result.map(|e| EmotionState {
                valence: e.valence,
                arousal: e.arousal,
            });

            (
                tier, importance, tags, suppress, compressed, model_name, confidence, emotion,
            )
        };

        if suppress {
            return Ok(false);
        }

        let scope = match &event.project_path {
            Some(path) if !path.is_empty() => MemoryScope::Project {
                project_path: path.clone(),
            },
            _ => MemoryScope::Global,
        };

        let decay_rate = tier.decay_rate();
        let mut memory = CognitiveMemoryUnit::new(content.clone(), tier, importance, decay_rate);
        memory.scope = scope;
        memory.source = MemorySource::External;
        memory.emotion = emotion;
        memory.model.tags = tags;
        memory.model.model_name = Some(model_name);
        memory.model.confidence = Some(confidence);
        memory.model.compressed_content = if compressed.is_empty() {
            None
        } else {
            Some(compressed.clone())
        };
        memory.model.suggested_tier = Some(tier);
        memory.model.suggested_importance = Some(importance);

        let memory_id = memory.id;
        let mut guard = self.state.lock().await;

        if let Some(limit) = tier.capacity() {
            while guard.graph.count_by_tier(tier) >= limit {
                if let Some(evict_id) = guard.graph.find_lowest_activation_in_tier(tier) {
                    if let Some(evicted) = guard.graph.remove_memory(&evict_id) {
                        guard.search.remove(&evicted.id);
                        let _ = guard.storage.delete(&evicted.id);
                    }
                } else {
                    break;
                }
            }
        }

        guard.graph.add_memory(memory.clone());
        let index_content = match &memory.model.compressed_content {
            Some(comp) => format!("{} {}", &memory.content, comp),
            None => memory.content.clone(),
        };
        guard.search.index(memory_id, &index_content, memory.tier);
        let embedding = guard.embedder.embed(&memory.content);
        guard.graph.set_embedding(memory_id, embedding);

        if let Err(e) = guard.storage.save(&memory) {
            tracing::error!("Failed to persist captured memory {}: {e}", memory_id);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{InMemoryStore, MemoryGraph};

    fn make_test_state() -> Arc<Mutex<CogniMemState>> {
        Arc::new(Mutex::new(CogniMemState {
            graph: MemoryGraph::new(),
            storage: Box::new(InMemoryStore::new()),
            search: Box::new(crate::search::SubstringSearch),
            embedder: Box::new(crate::embeddings::HashEmbedding::new()),
            slm: Box::new(crate::memory::NoOpSlm),
            work_claims: HashMap::new(),
            session_context: None,
            handoffs: Vec::new(),
            project_models: crate::memory::ProjectModelManager::new(),
            injection: crate::memory::InjectionDecider::new(),
            broker: Box::new(crate::broker::SimpleBroker::new()),
            code_graph: crate::memory::CodeGraph::new(),
            c3gan: crate::memory::C3GAN::new(100, 0.1),
        }))
    }

    fn make_event(event_type: CanonicalEventType) -> CanonicalEvent {
        CanonicalEvent {
            event_type,
            timestamp: Utc::now().timestamp(),
            session_id: Some("test-session".into()),
            project_path: Some("/test/project".into()),
            agent_id: None,
            source: super::super::types::EventSource::Opencode,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            file_path: None,
            content: Some("test content".into()),
            success: None,
            duration_ms: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_ingest_single_event() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::FileEdited);
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.accepted, 1);
        assert_eq!(result.stored, 1);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_suppress_heartbeat() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::ToolExecuteAfter);
        event.tool_name = Some("heartbeat".into());
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.suppressed, 1);
        assert_eq!(result.stored, 0);
    }

    #[tokio::test]
    async fn test_suppress_noisy_no_content() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::SessionIdle);
        event.content = None;
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.suppressed, 1);
    }

    #[tokio::test]
    async fn test_noisy_with_content_stored() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::Notification);
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.stored, 1);
    }

    #[tokio::test]
    async fn test_aggregate_tool_events() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let now = Utc::now().timestamp();

        let mut before = make_event(CanonicalEventType::ToolExecuteBefore);
        before.tool_name = Some("bash".into());
        before.timestamp = now - 5;

        let mut after = make_event(CanonicalEventType::ToolExecuteAfter);
        after.tool_name = Some("bash".into());
        after.timestamp = now;
        after.success = Some(true);
        after.content = Some("ran tests".into());

        let result = pipeline.ingest_batch(vec![before, after]).await;
        assert_eq!(result.accepted, 2);
        assert_eq!(result.stored, 1);
    }

    #[tokio::test]
    async fn test_batch_ingest() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let events = vec![
            make_event(CanonicalEventType::SessionCreated),
            make_event(CanonicalEventType::FileEdited),
            make_event(CanonicalEventType::FileCreated),
        ];
        let result = pipeline.ingest_batch(events).await;
        assert_eq!(result.accepted, 3);
        assert_eq!(result.stored, 3);
    }

    #[tokio::test]
    async fn test_validation_future_timestamp() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::FileEdited);
        event.timestamp = Utc::now().timestamp() + 120;
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("future"));
    }

    #[tokio::test]
    async fn test_validation_content_too_large() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::FileEdited);
        event.content = Some("x".repeat(200_000));
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("maximum size"));
    }

    #[tokio::test]
    async fn test_deduplication() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::FileEdited);
        let events = vec![event.clone(), event];
        let result = pipeline.ingest_batch(events).await;
        assert_eq!(result.stored, 1);
        assert_eq!(result.suppressed, 1);
    }

    #[tokio::test]
    async fn test_project_scope() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::FileEdited);
        event.project_path = Some("/my/project".into());
        pipeline.ingest_batch(vec![event]).await;
        let guard = state.lock().await;
        let memories = guard.graph.get_all_memories();
        let mem = memories.first().unwrap();
        assert_eq!(
            mem.scope,
            MemoryScope::Project {
                project_path: "/my/project".into()
            }
        );
    }

    #[tokio::test]
    async fn test_global_scope() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::FileEdited);
        event.project_path = None;
        pipeline.ingest_batch(vec![event]).await;
        let guard = state.lock().await;
        let memories = guard.graph.get_all_memories();
        let mem = memories.first().unwrap();
        assert_eq!(mem.scope, MemoryScope::Global);
    }
}
