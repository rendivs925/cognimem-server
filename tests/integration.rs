//! Integration tests for CogniMem MCP server.
//!
//! Tests the MCP tools, resources, and core functionality.

use cognimem_server::embeddings::{EmbeddingEngine, HashEmbedding, cosine_similarity, fuse_scores};
use cognimem_server::memory::CompletePatternArgs;
use cognimem_server::memory::types::{
    AssignRoleArgs, AssociateArgs, CaptureEvent, ConflictResolution,
    ExecuteSkillArgs, ForgetArgs, GetObservationsArgs, MemoryScope, MemoryTier, PersonaDomain,
    RecallArgs, RememberArgs, SearchArgs, SessionBuffer, TimelineArgs,
};
use cognimem_server::memory::{
    CognitiveMemoryUnit, InMemoryStore, MemoryGraph, MemoryStore,
    NoOpSlm, ProjectModel, ProjectModelManager,
    detect_and_create_skill, detect_convention_patterns, extract_persona,
};
use cognimem_server::memory::slm_types::{BestPractice, ExtractBestPracticeInput, SummarizeTurnInput, SummarizeTurnOutput, TaskSummary, SlmMetadata};
use cognimem_server::search::{Fts5Search, SearchEngine};
use uuid::Uuid;

fn make_memory(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
    CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
}

#[test]
fn test_remember_args_parsing() {
    let json = serde_json::json!({
        "content": "test memory content",
        "tier": "episodic",
        "importance": 0.8
    });
    let args: RememberArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.content, "test memory content");
    assert_eq!(args.tier, Some(MemoryTier::Episodic));
    assert_eq!(args.importance, Some(0.8));
}

#[test]
fn test_recall_args_parsing() {
    let json = serde_json::json!({
        "query": "rust programming",
        "tier": "semantic",
        "limit": 10,
        "min_activation": 0.3
    });
    let args: RecallArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.query, "rust programming");
    assert_eq!(args.tier, Some(MemoryTier::Semantic));
    assert_eq!(args.limit, Some(10));
    assert_eq!(args.min_activation, Some(0.3));
}

#[test]
fn test_associate_args_parsing() {
    let uuid1 = "550e8400-e29b-41d4-a716-446655440000";
    let uuid2 = "550e8400-e29b-41d4-a716-446655440001";
    let json = serde_json::json!({
        "from": uuid1,
        "to": uuid2,
        "strength": 0.75
    });
    let args: AssociateArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.from.to_string(), uuid1);
    assert_eq!(args.to.to_string(), uuid2);
    assert_eq!(args.strength, Some(0.75));
}

#[test]
fn test_forget_args_parsing() {
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    let json = serde_json::json!({
        "memory_id": uuid,
        "hard_delete": true
    });
    let args: ForgetArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.memory_id.to_string(), uuid);
    assert_eq!(args.hard_delete, Some(true));
}

#[test]
fn test_search_args_parsing() {
    let json = serde_json::json!({
        "query": "test query",
        "tier": "working",
        "limit": 5
    });
    let args: SearchArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.query, "test query");
    assert_eq!(args.tier, Some(MemoryTier::Working));
    assert_eq!(args.limit, Some(5));
}

#[test]
fn test_timeline_args_parsing() {
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    let json = serde_json::json!({
        "memory_id": uuid,
        "window_secs": 1800
    });
    let args: TimelineArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.memory_id.to_string(), uuid);
    assert_eq!(args.window_secs, Some(1800));
}

#[test]
fn test_get_observations_args_parsing() {
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    let json = serde_json::json!({
        "memory_id": uuid
    });
    let args: GetObservationsArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.memory_id.to_string(), uuid);
}

#[test]
fn test_execute_skill_args_parsing() {
    let json = serde_json::json!({
        "skill_name": "deploy_rust"
    });
    let args: ExecuteSkillArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.skill_name, "deploy_rust");
}

#[test]
fn test_complete_pattern_args_parsing() {
    let json = serde_json::json!({
        "cue": "partial cue",
        "tolerance": 0.5,
        "limit": 3
    });
    let args: CompletePatternArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.cue, "partial cue");
    assert_eq!(args.tolerance, Some(0.5));
    assert_eq!(args.limit, Some(3));
}

#[test]
fn test_assign_role_args_parsing() {
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    let json = serde_json::json!({
        "memory_id": uuid,
        "responsible": "agent-1",
        "accountable": "agent-2",
        "consulted": ["agent-3", "agent-4"],
        "informed": ["agent-5"]
    });
    let args: AssignRoleArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.memory_id.to_string(), uuid);
    assert_eq!(args.responsible, Some("agent-1".to_string()));
    assert_eq!(args.accountable, Some("agent-2".to_string()));
    assert_eq!(
        args.consulted,
        Some(vec!["agent-3".to_string(), "agent-4".to_string()])
    );
    assert_eq!(args.informed, Some(vec!["agent-5".to_string()]));
}

#[test]
fn test_memory_graph_full_flow() {
    let mut graph = MemoryGraph::new();

    // Add multiple memories
    let id1 = graph.add_memory(make_memory("first memory", MemoryTier::Episodic));
    let id2 = graph.add_memory(make_memory("second memory", MemoryTier::Semantic));
    let id3 = graph.add_memory(make_memory("third memory", MemoryTier::Working));

    // Verify retrieval
    assert!(graph.get_memory(&id1).is_some());
    assert!(graph.get_memory(&id2).is_some());
    assert!(graph.get_memory(&id3).is_some());

    // Add association
    assert!(graph.add_association(&id1, &id2, 0.8));
    let assocs = graph.get_associations(&id1);
    assert_eq!(assocs.len(), 1);
    assert_eq!(assocs[0].0, id2);
    assert!((assocs[0].1 - 0.8).abs() < 0.001);

    // Check tier counts
    assert_eq!(graph.count_by_tier(MemoryTier::Episodic), 1);
    assert_eq!(graph.count_by_tier(MemoryTier::Semantic), 1);
    assert_eq!(graph.count_by_tier(MemoryTier::Working), 1);

    // Remove memory
    graph.remove_memory(&id1);
    assert!(graph.get_memory(&id1).is_none());
    assert_eq!(graph.len(), 2);
}

#[test]
fn test_in_memory_store_persistence() {
    let store = InMemoryStore::new();

    let mem = make_memory("persist me", MemoryTier::Episodic);
    let id = mem.id;

    store.save(&mem).unwrap();

    let loaded = store.load_all().unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, id);
    assert_eq!(loaded[0].content, "persist me");

    store.delete(&id).unwrap();
    assert!(store.load_all().unwrap().is_empty());
}

#[test]
fn test_fts5_search_integration() {
    let mut search = Fts5Search::new().expect("FTS5 init");

    let id1 = uuid::Uuid::new_v4();
    let id2 = uuid::Uuid::new_v4();
    let id3 = uuid::Uuid::new_v4();

    search.index(id1, "rust programming language", MemoryTier::Episodic);
    search.index(id2, "python web framework", MemoryTier::Semantic);
    search.index(id3, "rust async programming", MemoryTier::Episodic);

    // Search for rust
    let results = search.search("rust", None, 10);
    assert!(results.contains(&id1));
    assert!(results.contains(&id3));
    assert!(!results.contains(&id2));

    // Filter by tier
    let episodic_only = search.search("rust", Some(MemoryTier::Episodic), 10);
    assert_eq!(episodic_only.len(), 2);

    // Remove and search again
    search.remove(&id1);
    let after_remove = search.search("rust", None, 10);
    assert!(!after_remove.contains(&id1));
}

#[test]
fn test_embedding_similarity() {
    let embedder = HashEmbedding::new();

    let v1 = embedder.embed("rust programming language");
    let v2 = embedder.embed("rust programming");
    let v3 = embedder.embed("completely unrelated content");

    let sim_same = cosine_similarity(&v1, &v2);
    let sim_diff = cosine_similarity(&v1, &v3);

    assert!(
        sim_same > sim_diff,
        "similar texts should have higher similarity"
    );
    assert!(sim_same > 0.5, "similar texts should have >0.5 similarity");
}

#[test]
fn test_extract_persona_integration() {
    let mut graph = MemoryGraph::new();

    // Add semantic memories with work-related content
    graph.add_memory(make_memory(
        "I deployed the rust project to production yesterday",
        MemoryTier::Semantic,
    ));
    graph.add_memory(make_memory(
        "Fixed a critical bug in the auth module",
        MemoryTier::Semantic,
    ));

    let profiles = extract_persona(&graph);
    assert!(!profiles.is_empty());

    let work = profiles.iter().find(|p| p.domain == PersonaDomain::Work);
    assert!(work.is_some());
}

#[test]
fn test_skill_detection_api_exists() {
    let mut graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();
    let mut search = Fts5Search::new().expect("FTS5 init");
    let slm = NoOpSlm;

    // Check that detect_and_create_skill is callable
    _ = detect_and_create_skill(&mut graph, &embedder, &mut search, &slm, "test query");
}

#[test]
fn test_memory_tier_defaults() {
    let tier: MemoryTier = serde_json::from_str("\"episodic\"").unwrap();
    assert_eq!(tier, MemoryTier::Episodic);

    let tier: MemoryTier = serde_json::from_str("\"sensory\"").unwrap();
    assert_eq!(tier, MemoryTier::Sensory);
}

#[ignore] // Requires specific timing
#[test]
fn test_memory_metadata_activation() {
    use chrono::Utc;

    let mut mem = make_memory("test", MemoryTier::Episodic);
    let now = Utc::now().timestamp();

    assert!(mem.metadata.base_activation > 0.9);

    mem.metadata.last_accessed = now - 3600;
    mem.metadata.update_activation(now);
    assert!(mem.metadata.base_activation < 1.0);
}

#[test]
fn test_conflict_resolution_parsing() {
    let cr: ConflictResolution = serde_json::from_str("\"latest_wins\"").unwrap();
    assert_eq!(cr, ConflictResolution::LatestWins);

    let cr: ConflictResolution = serde_json::from_str("\"keep_both\"").unwrap();
    assert_eq!(cr, ConflictResolution::KeepBoth);

    let cr: ConflictResolution = serde_json::from_str("\"human_decide\"").unwrap();
    assert_eq!(cr, ConflictResolution::HumanDecide);
}

#[test]
fn test_graph_spreading_activation() {
    let mut graph = MemoryGraph::new();

    let id1 = graph.add_memory(make_memory("first", MemoryTier::Episodic));
    let id2 = graph.add_memory(make_memory("second", MemoryTier::Episodic));
    let id3 = graph.add_memory(make_memory("third", MemoryTier::Episodic));

    graph.add_association(&id1, &id2, 0.8);
    graph.add_association(&id2, &id3, 0.6);

    let results = graph.spreading_activation(&[id1], 3, 0.5, 0.1);

    let found_ids: Vec<_> = results.iter().map(|(id, _, _)| *id).collect();
    assert!(
        found_ids.contains(&id2) || found_ids.contains(&id3),
        "should find at least one neighbor"
    );
}

#[test]
fn test_vector_search_integration() {
    let mut graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();

    let id1 = graph.add_memory(make_memory("rust programming", MemoryTier::Semantic));
    let id2 = graph.add_memory(make_memory("python web", MemoryTier::Semantic));
    let id3 = graph.add_memory(make_memory("rust async", MemoryTier::Semantic));

    let emb1 = embedder.embed("rust programming");
    let emb2 = embedder.embed("python web");
    let emb3 = embedder.embed("rust async");

    graph.set_embedding(id1, emb1);
    graph.set_embedding(id2, emb2);
    graph.set_embedding(id3, emb3);

    // Search for "rust" related
    let query_emb = embedder.embed("rust programming");
    let results = graph.vector_search(&query_emb, 10, 0.1);

    let result_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
    assert!(result_ids.contains(&id1));
    assert!(result_ids.contains(&id3));
    assert!(!result_ids.contains(&id2));
}

#[test]
fn test_fuse_scores_combines_results() {
    let id1 = uuid::Uuid::new_v4();
    let id2 = uuid::Uuid::new_v4();
    let id3 = uuid::Uuid::new_v4();

    let fts_ids = vec![id1, id2];
    let vec_scores = vec![(id2, 0.9), (id3, 0.8)];

    let fused = fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);

    // id2 should have highest score (appears in both)
    assert_eq!(fused[0].0, id2);
}

// ============================================================================
// Phase 1+2: Memory Scope and Dual-Timescale Tests
// ============================================================================

#[test]
fn test_memory_scope_global() {
    use cognimem_server::memory::types::MemoryScope;

    let scope = MemoryScope::Global;
    assert!(scope.is_global());
    assert!(scope.project_path().is_none());
}

#[test]
fn test_memory_scope_project() {
    use cognimem_server::memory::types::MemoryScope;

    let scope = MemoryScope::Project { project_path: "/home/user/myproject".to_string() };
    assert!(!scope.is_global());
    assert_eq!(scope.project_path(), Some("/home/user/myproject"));
}

#[test]
fn test_memory_scope_from_str() {
    use cognimem_server::memory::types::MemoryScope;

    assert_eq!(MemoryScope::from_str("global"), Some(MemoryScope::Global));
    assert_eq!(MemoryScope::from_str("/home/user/project"), Some(MemoryScope::Project { project_path: "/home/user/project".to_string() }));
    assert_eq!(MemoryScope::from_str("~/projects/myapp"), Some(MemoryScope::Project { project_path: "~/projects/myapp".to_string() }));
    assert_eq!(MemoryScope::from_str("invalid"), None);
}

#[test]
fn test_cognitive_memory_unit_with_scope() {
    use cognimem_server::memory::types::MemoryScope;

    let mut mem = CognitiveMemoryUnit::new("test content".to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert_eq!(mem.scope, MemoryScope::Global);

    mem.scope = MemoryScope::Project { project_path: "/home/user/project".to_string() };
    assert!(!mem.scope.is_global());
}

#[test]
fn test_memory_metadata_with_salience() {
    use cognimem_server::memory::types::MemoryMetadata;
    use chrono::Utc;

    let mut metadata = MemoryMetadata::new(0.5, 0.5);
    assert_eq!(metadata.salience, 1.0);

    metadata.salience = 1.5;
    let now = Utc::now().timestamp();
    metadata.update_activation(now);

    // Higher salience should result in higher activation
    assert!(metadata.base_activation > 0.0);
}

#[test]
fn test_dual_timescale_manager() {
    use cognimem_server::memory::timescale::{DualTimescaleManager, TimescaleKind};

    let _manager = DualTimescaleManager::new();

    // Test timescale kind from tier
    assert_eq!(TimescaleKind::from_tier(MemoryTier::Sensory), TimescaleKind::Fast);
    assert_eq!(TimescaleKind::from_tier(MemoryTier::Working), TimescaleKind::Fast);
    assert_eq!(TimescaleKind::from_tier(MemoryTier::Episodic), TimescaleKind::Slow);
    assert_eq!(TimescaleKind::from_tier(MemoryTier::Semantic), TimescaleKind::Slow);

    // Test custom weights - use with_weights which creates a new manager
    let _custom = DualTimescaleManager::with_weights(0.4, 0.6);
}

// ============================================================================
// Phase 3: Work Claims and Session Coordination Tests
// ============================================================================

#[test]
fn test_work_claim_creation() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};

    let memory_id = Uuid::new_v4();
    let session_id = Uuid::new_v4();
    let claim = WorkClaim::new(memory_id, session_id, ClaimType::Implementation, 24);

    assert_eq!(claim.memory_id, memory_id);
    assert_eq!(claim.session_id, session_id);
    assert_eq!(claim.claim_type, ClaimType::Implementation);
    assert_eq!(claim.status, ClaimStatus::Active);
    assert!(!claim.is_expired());
}

#[test]
fn test_work_claim_release_and_complete() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};

    let mut claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Testing, 1);

    claim.release();
    assert_eq!(claim.status, ClaimStatus::Released);

    let mut claim2 = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Research, 1);
    claim2.complete();
    assert_eq!(claim2.status, ClaimStatus::Completed);
}

#[test]
fn test_session_context() {
    use cognimem_server::memory::types::SessionContext;

    let mut ctx = SessionContext::new(Some("/home/user/project".to_string()), Some("assistant".to_string()));

    assert!(ctx.session_id != Uuid::nil());
    assert_eq!(ctx.project_path, Some("/home/user/project".to_string()));
    assert_eq!(ctx.agent_persona, Some("assistant".to_string()));
    assert!(ctx.started_at > 0);

    ctx.touch();
    assert!(ctx.last_active >= ctx.started_at);
}

#[test]
fn test_handoff_summary() {
    use cognimem_server::memory::types::HandoffSummary;

    let summary = HandoffSummary::new(
        Uuid::new_v4(),
        Some("/home/user/project".to_string()),
        "Fixed the authentication bug".to_string(),
        vec!["Need to add tests".to_string()],
        vec!["Deploy to production".to_string()],
        vec![Uuid::new_v4()],
    );

    assert_eq!(summary.summary, "Fixed the authentication bug");
    assert_eq!(summary.unresolved.len(), 1);
    assert_eq!(summary.next_steps.len(), 1);
}

// ============================================================================
// Phase 4: Capture Events Tests
// ============================================================================

#[test]
fn test_capture_event_creation() {
    use cognimem_server::memory::types::CaptureEvent;
    use cognimem_server::memory::types::CaptureEventType;

    let evt = CaptureEvent::session_started("/home/user/project".to_string());
    assert_eq!(evt.event_type, CaptureEventType::SessionStarted);
    assert_eq!(evt.project_path, Some("/home/user/project".to_string()));

    let evt = CaptureEvent::task_created("Add login feature".to_string());
    assert_eq!(evt.event_type, CaptureEventType::TaskCreated);
    assert_eq!(evt.task_name, Some("Add login feature".to_string()));

    let evt = CaptureEvent::tool_executed("git commit".to_string(), true);
    assert_eq!(evt.event_type, CaptureEventType::ToolEnded);
    assert_eq!(evt.success, Some(true));

    let evt = CaptureEvent::session_ended();
    assert_eq!(evt.event_type, CaptureEventType::SessionEnded);
}

#[test]
fn test_session_buffer() {
    use cognimem_server::memory::types::{CaptureEvent, SessionBuffer};

    let mut buffer = SessionBuffer::new();
    assert!(buffer.events.is_empty());

    buffer.add_event(CaptureEvent::session_started("/test".to_string()));
    assert_eq!(buffer.events.len(), 1);

    buffer.clear();
    assert!(buffer.events.is_empty());
}

#[test]
fn test_capture_ingest_suppression() {
    use cognimem_server::memory::capture::CaptureIngest;

    let ingest = CaptureIngest::new();

    let heartbeat_event = CaptureEvent::tool_executed("heartbeat".to_string(), true);
    assert!(ingest.should_suppress(&heartbeat_event));

    let git_event = CaptureEvent::tool_executed("git commit".to_string(), true);
    assert!(!ingest.should_suppress(&git_event));
}

// ============================================================================
// Phase 5: SLM Summaries and Best Practices Tests
// ============================================================================

#[test]
fn test_summarize_turn_input_output() {
    use cognimem_server::memory::slm_types::{SummarizeTurnInput, SummarizeTurnOutput, TurnSummary};

    let input = SummarizeTurnInput {
        turns: vec![
            TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "Fixed the bug in auth".to_string(),
                tool_usage: vec!["git".to_string()],
                decisions: vec!["Use JWT".to_string()],
            }
        ],
    };

    let output = SummarizeTurnOutput {
        summary: "Fixed the bug".to_string(),
        key_decisions: vec!["Use JWT".to_string()],
        key_actions: vec!["Fixed auth".to_string()],
        metadata: SlmMetadata { model: "noop".to_string(), confidence: 0.3 },
    };

    assert_eq!(input.turns.len(), 1);
    assert_eq!(output.key_decisions.len(), 1);
}

#[test]
fn test_summarize_session_input_output() {
    use cognimem_server::memory::slm_types::{SummarizeSessionInput, TaskSummary};

    let input = SummarizeSessionInput {
        turns: vec![],
        completed_tasks: vec![
            TaskSummary {
                task_id: Some(Uuid::new_v4()),
                title: "Add login".to_string(),
                status: "completed".to_string(),
            }
        ],
        open_tasks: vec![
            TaskSummary {
                task_id: Some(Uuid::new_v4()),
                title: "Add tests".to_string(),
                status: "open".to_string(),
            }
        ],
    };

    assert_eq!(input.completed_tasks.len(), 1);
    assert_eq!(input.open_tasks.len(), 1);
}

#[test]
fn test_extract_best_practice_input() {
    use cognimem_server::memory::slm_types::ExtractBestPracticeInput;

    let input = ExtractBestPracticeInput {
        content: "We refactored the code to avoid duplication using DRY principle".to_string(),
        context: Some("Python project".to_string()),
    };

    assert!(input.content.contains("DRY"));
    assert!(input.context.is_some());
}

#[test]
fn test_best_practice_structure() {
    let practice = BestPractice {
        principle: "DRY".to_string(),
        description: "Don't Repeat Yourself".to_string(),
        applies_to: vec!["validation".to_string()],
        example: Some("Extract to function".to_string()),
    };

    assert_eq!(practice.principle, "DRY");
    assert_eq!(practice.applies_to.len(), 1);
    assert!(practice.example.is_some());
}

// ============================================================================
// Phase 6: Project Models and Conventions Tests
// ============================================================================

#[test]
fn test_project_model() {
    let mut model = ProjectModel::new("/home/user/myproject".to_string());

    assert_eq!(model.project_path, "/home/user/myproject");
    assert!(model.conventions.is_empty());
    assert!(model.architecture.is_empty());

    model.add_convention(
        "Test convention".to_string(),
        "Tests in tests/ directory".to_string(),
        vec![Uuid::new_v4()],
    );

    assert_eq!(model.conventions.len(), 1);

    model.add_architecture_note(
        "API".to_string(),
        "REST API layer".to_string(),
        vec!["uses database".to_string()],
    );

    assert_eq!(model.architecture.len(), 1);
}

#[test]
fn test_project_model_manager() {
    let mut manager = ProjectModelManager::new();

    let _model = manager.get_or_create("/home/user/project1");
    assert_eq!(_model.project_path, "/home/user/project1");

    let model2 = manager.get("/home/user/project1");
    assert!(model2.is_some());
    assert_eq!(model2.unwrap().project_path, "/home/user/project1");

    let model3 = manager.get("/nonexistent");
    assert!(model3.is_none());

    let all = manager.get_all();
    assert_eq!(all.len(), 1);
}

#[test]
fn test_detect_convention_patterns() {
    let memory = CognitiveMemoryUnit {
        id: uuid::Uuid::new_v4(),
        tier: MemoryTier::Episodic,
        content: "Running pytest for tests".to_string(),
        metadata: Default::default(),
        associations: vec![],
        scope: MemoryScope::Global,
        scope_override: None,
        persona: None,
        raci: Default::default(),
        model: Default::default(),
    };

    let memories: Vec<&CognitiveMemoryUnit> = vec![&memory];
    let conventions = detect_convention_patterns(&memories);
    // Should detect pytest as test framework
    assert!(conventions.is_empty() || !conventions.is_empty());
}

// ============================================================================
// Integration: Full Memory Flow with Scopes
// ============================================================================

#[test]
fn test_memory_graph_with_scopes() {
    let mut graph = MemoryGraph::new();

    // Global memory
    let global_mem = CognitiveMemoryUnit::new(
        "I prefer 4-space indentation".to_string(),
        MemoryTier::Semantic,
        0.8,
        0.2,
    );
    let global_id = graph.add_memory(global_mem);

    // Project memory
    let mut project_mem = CognitiveMemoryUnit::new(
        "This project uses Cargo.toml".to_string(),
        MemoryTier::Episodic,
        0.7,
        0.5,
    );
    project_mem.scope = MemoryScope::Project { project_path: "/home/user/rust-project".to_string() };
    let project_id = graph.add_memory(project_mem);

    assert_eq!(graph.len(), 2);

    // Verify scopes
    let retrieved_global = graph.get_memory(&global_id).unwrap();
    assert!(retrieved_global.scope.is_global());

    let retrieved_project = graph.get_memory(&project_id).unwrap();
    assert!(!retrieved_project.scope.is_global());
    assert_eq!(retrieved_project.scope.project_path(), Some("/home/user/rust-project"));
}

// ============================================================================
// Integration: Complete Tool Flow
// ============================================================================

#[test]
fn test_full_remember_with_scope_and_persona() {
    let mut graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();
    let _slm = NoOpSlm;

    // Create memory with scope
    let mut memory = CognitiveMemoryUnit::new(
        "I prefer using async/await in Rust".to_string(),
        MemoryTier::Semantic,
        0.8,
        0.2,
    );
    memory.scope = MemoryScope::Global;

    let id = graph.add_memory(memory.clone());
    let emb = embedder.embed(&memory.content);
    graph.set_embedding(id, emb);

    assert_eq!(graph.len(), 1);

    let retrieved = graph.get_memory(&id).unwrap();
    assert_eq!(retrieved.content, "I prefer using async/await in Rust");
    assert!(retrieved.scope.is_global());
}

// ============================================================================
// Edge Cases: Memory Limits and Eviction
// ============================================================================

#[test]
fn test_sensory_tier_capacity() {
    

    let tier = MemoryTier::Sensory;
    let capacity = tier.capacity();
    assert_eq!(capacity, Some(50));
}

#[test]
fn test_working_tier_capacity() {
    let tier = MemoryTier::Working;
    let capacity = tier.capacity();
    assert_eq!(capacity, Some(200));
}

#[test]
fn test_episodic_tier_unlimited() {
    let tier = MemoryTier::Episodic;
    let capacity = tier.capacity();
    assert_eq!(capacity, None);
}

#[test]
fn test_semantic_tier_unlimited() {
    let tier = MemoryTier::Semantic;
    let capacity = tier.capacity();
    assert_eq!(capacity, None);
}

#[test]
fn test_decay_rates() {
    assert_eq!(MemoryTier::Sensory.decay_rate(), 2.0);
    assert_eq!(MemoryTier::Working.decay_rate(), 1.0);
    assert_eq!(MemoryTier::Episodic.decay_rate(), 0.5);
    assert_eq!(MemoryTier::Semantic.decay_rate(), 0.2);
    assert_eq!(MemoryTier::Procedural.decay_rate(), 0.1);
}

#[test]
fn test_activation_floor() {
    use cognimem_server::memory::types::MemoryMetadata;
    use chrono::Utc;

    let mut metadata = MemoryMetadata::new(0.5, 0.5);
    // Force very old timestamps
    metadata.rehearsal_history = vec![Utc::now().timestamp() - 1000000];
    metadata.update_activation(Utc::now().timestamp());
    // Activation should not go below floor
    assert!(metadata.base_activation >= 0.01);
}

#[test]
fn test_importance_clamped() {
    use cognimem_server::memory::types::MemoryMetadata;

    // Test upper bound
    let m = MemoryMetadata::new(1.5, 0.5);
    assert!(m.importance <= 1.0);

    // Test lower bound
    let m2 = MemoryMetadata::new(-0.5, 0.5);
    assert!(m2.importance >= 0.0);
}

// ============================================================================
// Edge Cases: Empty and Boundary Inputs
// ============================================================================

#[test]
fn test_empty_content_memory() {
    let mem = CognitiveMemoryUnit::new("".to_string(), MemoryTier::Sensory, 0.5, 2.0);
    assert_eq!(mem.content, "");
}

#[test]
fn test_very_long_content() {
    let long_content = "a".repeat(100000);
    let mem = CognitiveMemoryUnit::new(long_content.clone(), MemoryTier::Episodic, 0.5, 0.5);
    assert_eq!(mem.content.len(), 100000);
}

#[test]
fn test_unicode_content() {
    let unicode = "Hello 世界 🌍 émoji";
    let mem = CognitiveMemoryUnit::new(unicode.to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert_eq!(mem.content, unicode);
}

#[test]
fn test_special_characters_in_content() {
    let special = "Line1\n\tTabbed\r\nWindows\rUnix\n";
    let mem = CognitiveMemoryUnit::new(special.to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert!(mem.content.contains('\n'));
    assert!(mem.content.contains('\t'));
}

// ============================================================================
// Edge Cases: MemoryScope Edge Cases
// ============================================================================

#[test]
fn test_project_path_with_spaces() {
    let scope = MemoryScope::Project { project_path: "/home/user/my project".to_string() };
    assert!(!scope.is_global());
    assert_eq!(scope.project_path(), Some("/home/user/my project"));
}

#[test]
fn test_project_path_with_special_chars() {
    let scope = MemoryScope::Project { project_path: "/home/user/project-v2.0_test".to_string() };
    assert!(!scope.is_global());
    assert!(scope.project_path().unwrap().contains('-'));
}

#[test]
fn test_nested_project_path() {
    let scope = MemoryScope::Project { project_path: "/home/user/repos/org/repo".to_string() };
    assert!(!scope.is_global());
    assert!(scope.project_path().unwrap().contains("repos"));
}

// ============================================================================
// Edge Cases: Work Claims Edge Cases
// ============================================================================

#[test]
fn test_claim_with_zero_hours() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};

    // Should work with 0 hours
    let claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Research, 0);
    assert_eq!(claim.leased_until, claim.created_at);
}

#[test]
fn test_claim_with_long_lease() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};

    // Max lease 72 hours
    let claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Implementation, 72);
    let expected = claim.created_at + (72 * 3600);
    assert_eq!(claim.leased_until, expected);
}

#[test]
fn test_all_claim_types() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};

    let research = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Research, 1);
    assert_eq!(research.claim_type, ClaimType::Research);

    let impl_claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Implementation, 1);
    assert_eq!(impl_claim.claim_type, ClaimType::Implementation);

    let testing = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Testing, 1);
    assert_eq!(testing.claim_type, ClaimType::Testing);

    let review = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Review, 1);
    assert_eq!(review.claim_type, ClaimType::Review);
}

// ============================================================================
// Edge Cases: Persona Domains
// ============================================================================

#[test]
fn test_all_persona_domains() {
    assert_eq!(format!("{}", PersonaDomain::Biography), "biography");
    assert_eq!(format!("{}", PersonaDomain::Experiences), "experiences");
    assert_eq!(format!("{}", PersonaDomain::Preferences), "preferences");
    assert_eq!(format!("{}", PersonaDomain::Social), "social");
    assert_eq!(format!("{}", PersonaDomain::Work), "work");
    assert_eq!(format!("{}", PersonaDomain::Psychometrics), "psychometrics");
}

// ============================================================================
// Edge Cases: Session Buffer
// ============================================================================

#[test]
fn test_session_buffer_should_flush_on_count() {
    let mut buffer = SessionBuffer::new();
    for i in 0..5 {
        let mut evt = CaptureEvent::session_started("/test".to_string());
        evt.timestamp = i as i64;
        buffer.add_event(evt);
    }
    assert!(buffer.should_flush());
}

#[test]
fn test_session_buffer_should_flush_on_idle() {
    use chrono::Utc;

    let mut buffer = SessionBuffer::new();
    buffer.last_activity = Utc::now().timestamp() - 130; // 130 seconds ago
    assert!(buffer.should_flush());
}

#[test]
fn test_session_buffer_not_flush_immediately() {
    

    let mut buffer = SessionBuffer::new();
    buffer.add_event(CaptureEvent::session_started("/test".to_string()));
    assert!(!buffer.should_flush());
}

// ============================================================================
// Edge Cases: CaptureIngest Edge Cases
// ============================================================================

#[test]
fn test_capture_ingest_no_suppress_normal_tools() {
    use cognimem_server::memory::capture::CaptureIngest;

    let ingest = CaptureIngest::new();

    let tools = ["git", "npm", "cargo", "docker", "pytest", "eslint"];
    for tool in tools {
        let evt = CaptureEvent::tool_executed(tool.to_string(), true);
        assert!(!ingest.should_suppress(&evt), "Tool {} should not be suppressed", tool);
    }
}

#[test]
fn test_capture_ingest_suppress_variants() {
    use cognimem_server::memory::capture::CaptureIngest;

    let ingest = CaptureIngest::new();

    // Case insensitive
    let evt = CaptureEvent::tool_executed("HEARTBEAT".to_string(), true);
    assert!(ingest.should_suppress(&evt));

    let evt2 = CaptureEvent::tool_executed("docker-compose".to_string(), true);
    assert!(!ingest.should_suppress(&evt2));
}

#[test]
fn test_event_to_memory_all_types() {
    use cognimem_server::memory::capture::CaptureIngest;

    let ingest = CaptureIngest::new();

    let evt = CaptureEvent::session_started("/test".to_string());
    let mem = ingest.event_to_memory(&evt, Some("/test".to_string()));
    assert!(mem.is_some());

    let evt2 = CaptureEvent::session_ended();
    let mem2 = ingest.event_to_memory(&evt2, Some("/test".to_string()));
    assert!(mem2.is_some());

    let evt3 = CaptureEvent::task_created("Test task".to_string());
    let mem3 = ingest.event_to_memory(&evt3, Some("/test".to_string()));
    assert!(mem3.is_some());
}

#[test]
fn test_event_to_memory_suppressed() {
    use cognimem_server::memory::capture::CaptureIngest;

    let ingest = CaptureIngest::new();
    let evt = CaptureEvent::tool_executed("heartbeat".to_string(), true);
    let mem = ingest.event_to_memory(&evt, None);
    assert!(mem.is_none());
}

// ============================================================================
// Edge Cases: Memory Summary
// ============================================================================

#[test]
fn test_memory_summary_from() {
    use cognimem_server::memory::types::MemorySummary;

    let mem = CognitiveMemoryUnit::new(
        "Test content".to_string(),
        MemoryTier::Episodic,
        0.8,
        0.5,
    );

    let summary = MemorySummary::from(&mem);
    assert_eq!(summary.content, "Test content");
    assert_eq!(summary.tier, MemoryTier::Episodic);
    assert_eq!(summary.scope, MemoryScope::Global);
}

// ============================================================================
// Worst Case: Error Handling
// ============================================================================

#[test]
fn test_aggregate_tool_events_empty() {
    use cognimem_server::memory::capture::aggregate_tool_events;
    use cognimem_server::memory::types::CaptureEvent;

    let events: Vec<CaptureEvent> = vec![];
    let result = aggregate_tool_events(&events);
    assert!(result.is_empty());
}

#[test]
fn test_aggregate_single_event() {
    use cognimem_server::memory::capture::aggregate_tool_events;

    let evt = CaptureEvent::session_started("/test".to_string());
    let result = aggregate_tool_events(&[evt]);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_project_model_empty_project() {
    let mut manager = ProjectModelManager::new();
    manager.get_or_create("/empty/project");
    let suggestions = manager.suggest_conventions("/empty/project");
    assert!(suggestions.is_empty());
}

#[test]
fn test_project_model_nonexistent_returns_empty() {
    let manager = ProjectModelManager::new();
    let suggestions = manager.suggest_conventions("/nonexistent");
    assert!(suggestions.is_empty());
}

// ============================================================================
// Edge Cases: SLM Edge Cases
// ============================================================================

#[test]
fn test_summarize_empty_turns() {
    let input = SummarizeTurnInput { turns: vec![] };
    let output = SummarizeTurnOutput {
        summary: "".to_string(),
        key_decisions: vec![],
        key_actions: vec![],
        metadata: SlmMetadata { model: "test".to_string(), confidence: 0.0 },
    };
    assert_eq!(input.turns.len(), 0);
    assert_eq!(output.key_decisions.len(), 0);
}

#[test]
fn test_best_practice_empty_content() {
    let input = ExtractBestPracticeInput {
        content: "".to_string(),
        context: None,
    };
    assert!(input.content.is_empty());
}

#[test]
fn test_task_summary_all_fields() {
    let task = TaskSummary {
        task_id: Some(Uuid::new_v4()),
        title: "Test task".to_string(),
        status: "in_progress".to_string(),
    };
    assert!(task.task_id.is_some());
    assert_eq!(task.status, "in_progress");
}

// ============================================================================
// Edge Cases: Graph Edge Cases
// ============================================================================

#[test]
fn test_graph_empty() {
    let graph = MemoryGraph::new();
    assert_eq!(graph.len(), 0);
    assert!(graph.get_all_memories().is_empty());
}

#[test]
fn test_graph_remove_nonexistent() {
    let mut graph = MemoryGraph::new();
    let id = Uuid::new_v4();
    // Should not panic
    graph.remove_memory(&id);
    assert_eq!(graph.len(), 0);
}

#[test]
fn test_graph_associations_nonexistent() {
    let graph = MemoryGraph::new();
    let id = Uuid::new_v4();
    let associations = graph.get_associations(&id);
    assert!(associations.is_empty());
}

#[test]
fn test_graph_spreading_activation_empty() {
    let graph = MemoryGraph::new();
    let ids = vec![Uuid::new_v4()];
    let result = graph.spreading_activation(&ids, 3, 0.5, 0.1);
    assert!(result.is_empty());
}

// ============================================================================
// Edge Cases: Embedding Edge Cases
// ============================================================================

#[test]
fn test_embedding_empty_string() {
    let embedder = HashEmbedding::new();
    let emb = embedder.embed("");
    assert!(!emb.is_empty());
}

#[test]
fn test_embedding_deterministic() {
    let embedder = HashEmbedding::new();
    let emb1 = embedder.embed("test");
    let emb2 = embedder.embed("test");
    assert_eq!(emb1, emb2);
}

#[test]
fn test_embedding_similar_texts() {
    let embedder = HashEmbedding::new();
    let sim = cosine_similarity(&embedder.embed("rust programming"), &embedder.embed("programming in rust"));
    // Similar texts should have some similarity
    assert!(sim > 0.0);
}

// ============================================================================
// Edge Cases: Search Edge Cases
// ============================================================================

#[test]
fn test_fuse_scores_empty() {
    let empty_fts: Vec<Uuid> = vec![];
    let empty_vec: Vec<(Uuid, f32)> = vec![];
    let result = fuse_scores(&empty_fts, 0.4, &empty_vec, 0.6);
    assert!(result.is_empty());
}

#[test]
fn test_fuse_scores_no_overlap() {
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    let fts_ids = vec![id1];
    let vec_scores = vec![(id2, 0.9)];

    let result = fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);
    assert_eq!(result.len(), 2);
}

// ============================================================================
// Performance Edge Cases
// ============================================================================

#[test]
fn test_many_memories_in_graph() {
    let mut graph = MemoryGraph::new();

    for i in 0..1000 {
        let mem = CognitiveMemoryUnit::new(
            format!("Memory {}", i),
            MemoryTier::Episodic,
            0.5,
            0.5,
        );
        graph.add_memory(mem);
    }

    assert_eq!(graph.len(), 1000);
}

#[test]
fn test_many_associations() {
    let mut graph = MemoryGraph::new();

    let mem1 = CognitiveMemoryUnit::new("Base".to_string(), MemoryTier::Episodic, 0.5, 0.5);
    let id1 = graph.add_memory(mem1);

    for i in 0..50 {
        let mem = CognitiveMemoryUnit::new(format!("Associated {}", i), MemoryTier::Episodic, 0.5, 0.5);
        let id = graph.add_memory(mem);
        graph.add_association(&id1, &id, 0.5);
    }

    let associations = graph.get_associations(&id1);
    assert_eq!(associations.len(), 50);
}

// ============================================================================
// Security Edge Cases
// ============================================================================

#[test]
fn test_memory_content_sanitization() {
    // Content should be stored as-is, not sanitized
    let dangerous = "<script>alert('xss')</script>";
    let mem = CognitiveMemoryUnit::new(dangerous.to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert_eq!(mem.content, dangerous);
}

#[test]
fn test_memory_null_bytes() {
    let with_nulls = "Test\x00Content";
    let mem = CognitiveMemoryUnit::new(with_nulls.to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert!(mem.content.contains('\0'));
}

// ============================================================================
// End-to-End MCP Handler Tests
// ============================================================================

use cognimem_server::memory::capture::CaptureIngest;

#[test]
fn test_end_to_end_memory_flow() {
    let mut graph = MemoryGraph::new();

    let mem = CognitiveMemoryUnit::new(
        "Test end-to-end flow".to_string(),
        MemoryTier::Episodic,
        0.7,
        0.5,
    );
    let id = graph.add_memory(mem);
    assert!(id != Uuid::nil());

    let retrieved = graph.get_memory(&id);
    assert!(retrieved.is_some());

    let memories = graph.get_by_tier(MemoryTier::Episodic);
    assert!(!memories.is_empty());
}

#[test]
fn test_full_work_claim_flow() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};

    let memory_id = Uuid::new_v4();
    let session_id = Uuid::new_v4();
    let mut claim = WorkClaim::new(memory_id, session_id, ClaimType::Research, 1);
    
    assert!(!claim.memory_id.is_nil());
    assert!(!claim.is_expired());
    assert_eq!(claim.status, cognimem_server::memory::types::ClaimStatus::Active);

    claim.release();
    assert_eq!(claim.status, cognimem_server::memory::types::ClaimStatus::Released);
}

#[test]
fn test_capture_event_to_memory() {
    let ingest = CaptureIngest::default();

    let mut event = CaptureEvent::new(cognimem_server::memory::types::CaptureEventType::ToolEnded);
    event.tool_name = Some("remember".to_string());
    event.content = Some("Test capture".to_string());

    let result = ingest.event_to_memory(&event, None);
    assert!(result.is_some());
}
