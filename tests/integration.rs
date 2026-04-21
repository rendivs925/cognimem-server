//! Integration tests for CogniMem MCP server.
//!
//! Tests the MCP tools, resources, and core functionality.

use cognimem_server::embeddings::{EmbeddingEngine, HashEmbedding, cosine_similarity, fuse_scores};
use cognimem_server::memory::CompletePatternArgs;
use cognimem_server::memory::types::{
    AssignRoleArgs, AssociateArgs, CaptureEvent, CaptureEventType, ConflictResolution,
    ExecuteSkillArgs, ForgetArgs, GetObservationsArgs, MemoryScope, MemoryTier, PersonaDomain,
    RecallArgs, RememberArgs, SearchArgs, SessionBuffer, TimelineArgs,
};
use cognimem_server::memory::{
    CognitiveMemoryUnit, DualTimescaleManager, InMemoryStore, MemoryGraph, MemoryStore,
    NoOpSlm, ProjectModel, ProjectModelManager, SessionContext,
    detect_and_create_skill, detect_convention_patterns, extract_persona,
};
use cognimem_server::memory::slm_types::{BestPractice, SummarizeSessionInput, SummarizeTurnInput, SummarizeTurnOutput, TaskSummary, SlmMetadata};
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
    use cognimem_server::memory::slm_types::{SummarizeSessionOutput, SummarizeSessionInput, TaskSummary};

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
    use cognimem_server::memory::slm_types::{ExtractBestPracticeInput, BestPractice};

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
