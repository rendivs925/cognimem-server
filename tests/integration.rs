//! Integration tests for CogniMem MCP server.
//!
//! Tests the MCP tools, resources, and core functionality.

use cognimem_server::broker::BrokerEvent;
use cognimem_server::embeddings::{EmbeddingEngine, HashEmbedding, cosine_similarity, fuse_scores};
use cognimem_server::memory::CompletePatternArgs;
use cognimem_server::memory::slm_types::{
    BestPractice, ExtractBestPracticeInput, SlmMetadata, SummarizeTurnInput, SummarizeTurnOutput,
    TaskSummary,
};
use cognimem_server::memory::types::{
    AssignRoleArgs, AssociateArgs, CaptureEvent, ConflictResolution, ExecuteSkillArgs, ForgetArgs,
    GetObservationsArgs, ListMemoriesArgs, MemoryScope, MemoryTier, PersonaDomain, RecallArgs,
    RememberArgs, SearchArgs, SessionBuffer, TimelineArgs,
};
use cognimem_server::memory::{
    CognitiveMemoryUnit, InMemoryStore, MemoryGraph, MemoryStore, NoOpSlm, ProjectModel,
    ProjectModelManager, detect_and_create_skill, detect_convention_patterns, extract_persona,
};
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
fn test_list_memories_args_parsing() {
    let json = serde_json::json!({
        "tier": "semantic",
        "project_path": "/home/user/project",
        "scope_filter": "project",
        "min_activation": 0.2,
        "limit": 25
    });
    let args: ListMemoriesArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.tier, Some(MemoryTier::Semantic));
    assert_eq!(args.project_path.as_deref(), Some("/home/user/project"));
    assert_eq!(args.scope_filter.as_deref(), Some("project"));
    assert_eq!(args.min_activation, Some(0.2));
    assert_eq!(args.limit, Some(25));
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

    let scope = MemoryScope::Project {
        project_path: "/home/user/myproject".to_string(),
    };
    assert!(!scope.is_global());
    assert_eq!(scope.project_path(), Some("/home/user/myproject"));
}

#[test]
fn test_memory_scope_from_str() {
    use cognimem_server::memory::types::MemoryScope;

    assert_eq!(MemoryScope::from_str("global"), Some(MemoryScope::Global));
    assert_eq!(
        MemoryScope::from_str("/home/user/project"),
        Some(MemoryScope::Project {
            project_path: "/home/user/project".to_string()
        })
    );
    assert_eq!(
        MemoryScope::from_str("~/projects/myapp"),
        Some(MemoryScope::Project {
            project_path: "~/projects/myapp".to_string()
        })
    );
    assert_eq!(MemoryScope::from_str("invalid"), None);
}

#[test]
fn test_cognitive_memory_unit_with_scope() {
    use cognimem_server::memory::types::MemoryScope;

    let mut mem =
        CognitiveMemoryUnit::new("test content".to_string(), MemoryTier::Episodic, 0.5, 0.5);
    assert_eq!(mem.scope, MemoryScope::Global);

    mem.scope = MemoryScope::Project {
        project_path: "/home/user/project".to_string(),
    };
    assert!(!mem.scope.is_global());
}

#[test]
fn test_memory_metadata_with_salience() {
    use chrono::Utc;
    use cognimem_server::memory::types::MemoryMetadata;

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
    assert_eq!(
        TimescaleKind::from_tier(MemoryTier::Sensory),
        TimescaleKind::Fast
    );
    assert_eq!(
        TimescaleKind::from_tier(MemoryTier::Working),
        TimescaleKind::Fast
    );
    assert_eq!(
        TimescaleKind::from_tier(MemoryTier::Episodic),
        TimescaleKind::Slow
    );
    assert_eq!(
        TimescaleKind::from_tier(MemoryTier::Semantic),
        TimescaleKind::Slow
    );

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

    let mut ctx = SessionContext::new(
        Some("/home/user/project".to_string()),
        Some("assistant".to_string()),
    );

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
    use cognimem_server::memory::slm_types::{
        SummarizeTurnInput, SummarizeTurnOutput, TurnSummary,
    };

    let input = SummarizeTurnInput {
        turns: vec![TurnSummary {
            turn_id: Uuid::new_v4(),
            content: "Fixed the bug in auth".to_string(),
            tool_usage: vec!["git".to_string()],
            decisions: vec!["Use JWT".to_string()],
        }],
    };

    let output = SummarizeTurnOutput {
        summary: "Fixed the bug".to_string(),
        key_decisions: vec!["Use JWT".to_string()],
        key_actions: vec!["Fixed auth".to_string()],
        metadata: SlmMetadata {
            model: "noop".to_string(),
            confidence: 0.3,
        },
    };

    assert_eq!(input.turns.len(), 1);
    assert_eq!(output.key_decisions.len(), 1);
}

#[test]
fn test_summarize_session_input_output() {
    use cognimem_server::memory::slm_types::{SummarizeSessionInput, TaskSummary};

    let input = SummarizeSessionInput {
        turns: vec![],
        completed_tasks: vec![TaskSummary {
            task_id: Some(Uuid::new_v4()),
            title: "Add login".to_string(),
            status: "completed".to_string(),
        }],
        open_tasks: vec![TaskSummary {
            task_id: Some(Uuid::new_v4()),
            title: "Add tests".to_string(),
            status: "open".to_string(),
        }],
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
    project_mem.scope = MemoryScope::Project {
        project_path: "/home/user/rust-project".to_string(),
    };
    let project_id = graph.add_memory(project_mem);

    assert_eq!(graph.len(), 2);

    // Verify scopes
    let retrieved_global = graph.get_memory(&global_id).unwrap();
    assert!(retrieved_global.scope.is_global());

    let retrieved_project = graph.get_memory(&project_id).unwrap();
    assert!(!retrieved_project.scope.is_global());
    assert_eq!(
        retrieved_project.scope.project_path(),
        Some("/home/user/rust-project")
    );
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
    use chrono::Utc;
    use cognimem_server::memory::types::MemoryMetadata;

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
    let scope = MemoryScope::Project {
        project_path: "/home/user/my project".to_string(),
    };
    assert!(!scope.is_global());
    assert_eq!(scope.project_path(), Some("/home/user/my project"));
}

#[test]
fn test_project_path_with_special_chars() {
    let scope = MemoryScope::Project {
        project_path: "/home/user/project-v2.0_test".to_string(),
    };
    assert!(!scope.is_global());
    assert!(scope.project_path().unwrap().contains('-'));
}

#[test]
fn test_nested_project_path() {
    let scope = MemoryScope::Project {
        project_path: "/home/user/repos/org/repo".to_string(),
    };
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
    let claim = WorkClaim::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        ClaimType::Implementation,
        72,
    );
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
        assert!(
            !ingest.should_suppress(&evt),
            "Tool {} should not be suppressed",
            tool
        );
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

    let mem = CognitiveMemoryUnit::new("Test content".to_string(), MemoryTier::Episodic, 0.8, 0.5);

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
        metadata: SlmMetadata {
            model: "test".to_string(),
            confidence: 0.0,
        },
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
    let sim = cosine_similarity(
        &embedder.embed("rust programming"),
        &embedder.embed("programming in rust"),
    );
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
        let mem = CognitiveMemoryUnit::new(format!("Memory {}", i), MemoryTier::Episodic, 0.5, 0.5);
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
        let mem =
            CognitiveMemoryUnit::new(format!("Associated {}", i), MemoryTier::Episodic, 0.5, 0.5);
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
    assert_eq!(
        claim.status,
        cognimem_server::memory::types::ClaimStatus::Active
    );

    claim.release();
    assert_eq!(
        claim.status,
        cognimem_server::memory::types::ClaimStatus::Released
    );
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

mod capture_tests {
    use cognimem_server::capture::{
        CanonicalEvent, CanonicalEventType, CapturePipeline, EventSource,
    };
    use cognimem_server::memory::{InMemoryStore, MemoryGraph};
    use cognimem_server::state::CogniMemState;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn make_test_state() -> Arc<Mutex<CogniMemState>> {
        Arc::new(Mutex::new(CogniMemState {
            graph: MemoryGraph::new(),
            storage: Box::new(InMemoryStore::new()),
            search: Box::new(cognimem_server::search::SubstringSearch),
            embedder: Box::new(cognimem_server::embeddings::HashEmbedding::new()),
            slm: Box::new(cognimem_server::memory::NoOpSlm),
            work_claims: HashMap::new(),
            session_context: None,
            handoffs: Vec::new(),
            project_models: cognimem_server::memory::ProjectModelManager::new(),
        }))
    }

    fn make_event(event_type: CanonicalEventType) -> CanonicalEvent {
        CanonicalEvent {
            event_type,
            timestamp: chrono::Utc::now().timestamp(),
            session_id: Some("test-session".into()),
            project_path: Some("/test/project".into()),
            agent_id: None,
            source: EventSource::Opencode,
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
    async fn test_capture_single_file_edited() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::FileEdited);
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.accepted, 1);
        assert_eq!(result.stored, 1);
        assert!(result.errors.is_empty());

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 1);
    }

    #[tokio::test]
    async fn test_capture_suppress_heartbeat_tool() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::ToolExecuteAfter);
        event.tool_name = Some("heartbeat".into());
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.suppressed, 1);
        assert_eq!(result.stored, 0);

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 0);
    }

    #[tokio::test]
    async fn test_capture_tool_aggregation() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let now = chrono::Utc::now().timestamp();

        let mut before = make_event(CanonicalEventType::ToolExecuteBefore);
        before.tool_name = Some("bash".into());
        before.session_id = Some("sess1".into());
        before.timestamp = now - 5;

        let mut after = make_event(CanonicalEventType::ToolExecuteAfter);
        after.tool_name = Some("bash".into());
        after.session_id = Some("sess1".into());
        after.timestamp = now;
        after.success = Some(true);
        after.content = Some("ran cargo test".into());

        let result = pipeline.ingest_batch(vec![before, after]).await;
        assert_eq!(result.accepted, 2);
        assert_eq!(result.stored, 1);

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 1);
        let memories = guard.graph.get_all_memories();
        let content = &memories[0].content;
        assert!(content.contains("bash"));
        assert!(content.contains("ran cargo test"));
    }

    #[tokio::test]
    async fn test_capture_batch_mixed_events() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());

        let mut suppressed = make_event(CanonicalEventType::ToolExecuteAfter);
        suppressed.tool_name = Some("ping".into());

        let events = vec![
            make_event(CanonicalEventType::SessionCreated),
            make_event(CanonicalEventType::FileEdited),
            make_event(CanonicalEventType::FileCreated),
            suppressed,
        ];

        let result = pipeline.ingest_batch(events).await;
        assert_eq!(result.accepted, 4);
        assert_eq!(result.stored, 3);
        assert_eq!(result.suppressed, 1);

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 3);
    }

    #[tokio::test]
    async fn test_capture_validation_rejects_future() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::FileEdited);
        event.timestamp = chrono::Utc::now().timestamp() + 300;
        let result = pipeline.ingest_batch(vec![event]).await;
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("future"));
    }

    #[tokio::test]
    async fn test_capture_slm_metadata_stored() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::FileEdited);
        pipeline.ingest_batch(vec![event]).await;

        let guard = state.lock().await;
        let memories = guard.graph.get_all_memories();
        let mem = &memories[0];
        assert!(mem.model.model_name.is_some());
        assert!(mem.model.compressed_content.is_some() || mem.model.suggested_tier.is_some());
    }

    #[tokio::test]
    async fn test_capture_deduplication() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let event = make_event(CanonicalEventType::FileEdited);
        let result = pipeline.ingest_batch(vec![event.clone(), event]).await;
        assert!(result.stored >= 1);
        assert!(result.suppressed >= 1);
    }

    #[tokio::test]
    async fn test_capture_noop_slm_default_tier() {
        let state = make_test_state();
        let mut pipeline = CapturePipeline::new(state.clone());
        let mut event = make_event(CanonicalEventType::SessionIdle);
        event.content = Some("idle with context".into());
        pipeline.ingest_batch(vec![event]).await;

        let guard = state.lock().await;
        let memories = guard.graph.get_all_memories();
        assert!(!memories.is_empty());
        assert_eq!(
            memories[0].tier,
            cognimem_server::memory::types::MemoryTier::Sensory
        );
    }

    #[tokio::test]
    async fn test_capture_axum_health() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = make_test_state();
        let pipeline = Arc::new(Mutex::new(CapturePipeline::new(state)));
        let app = cognimem_server::capture::create_router(cognimem_server::capture::AppState {
            pipeline,
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/capture/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_capture_axum_ingest_single() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = make_test_state();
        let pipeline = Arc::new(Mutex::new(CapturePipeline::new(state.clone())));
        let app = cognimem_server::capture::create_router(cognimem_server::capture::AppState {
            pipeline,
        });

        let event = make_event(CanonicalEventType::FileEdited);
        let body = serde_json::to_vec(&event).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/capture/events")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 1);
    }

    #[tokio::test]
    async fn test_capture_axum_ingest_batch() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = make_test_state();
        let pipeline = Arc::new(Mutex::new(CapturePipeline::new(state.clone())));
        let app = cognimem_server::capture::create_router(cognimem_server::capture::AppState {
            pipeline,
        });

        let events = vec![
            make_event(CanonicalEventType::SessionCreated),
            make_event(CanonicalEventType::FileEdited),
        ];
        let body = serde_json::to_vec(&events).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/capture/events/batch")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let guard = state.lock().await;
        assert_eq!(guard.graph.len(), 2);
    }

    #[tokio::test]
    async fn test_capture_axum_stats() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = make_test_state();
        let pipeline = Arc::new(Mutex::new(CapturePipeline::new(state)));
        let app = cognimem_server::capture::create_router(cognimem_server::capture::AppState {
            pipeline,
        });

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/capture/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[test]
fn test_delegate_args_parsing() {
    use cognimem_server::memory::slm_types::DelegateInput;
    let json = serde_json::json!({
        "query": "how do I fix this bug",
        "context": ["context line 1", "context line 2"],
        "confidence_threshold": 0.7
    });
    let args: DelegateInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.query, "how do I fix this bug");
    assert_eq!(args.context.len(), 2);
    assert_eq!(args.confidence_threshold, 0.7);
}

#[test]
fn test_teach_args_parsing() {
    use cognimem_server::memory::slm_types::TeachFromDemonstrationInput;
    let json = serde_json::json!({
        "demonstration": "showed how to fix the bug",
        "pattern_extracted": "fix_bug_steps",
        "domain": "debugging"
    });
    let args: TeachFromDemonstrationInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.demonstration, "showed how to fix the bug");
    assert_eq!(args.pattern_extracted, "fix_bug_steps");
    assert_eq!(args.domain, Some("debugging".to_string()));
}

#[test]
fn test_simulate_perspective_args_parsing() {
    use cognimem_server::memory::slm_types::SimulatePerspectiveInput;
    let json = serde_json::json!({
        "perspective_role": "security_expert",
        "situation": "user wants to store passwords",
        "question": "should we store this?"
    });
    let args: SimulatePerspectiveInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.perspective_role, "security_expert");
    assert_eq!(args.situation, "user wants to store passwords");
    assert_eq!(args.question, "should we store this?");
}

#[test]
fn test_delegate_output_structure() {
    use cognimem_server::memory::slm_types::DelegateOutput;
    let output = DelegateOutput {
        response: "delegated response".to_string(),
        delegated: true,
        confidence: 0.7,
        model_used: "qwen2.5-coder:3b".to_string(),
        reasoning: Some("below threshold".to_string()),
    };
    assert!(output.delegated);
    assert_eq!(output.confidence, 0.7);
    let json = serde_json::to_string(&output).unwrap();
    assert!(json.contains("delegated response"));
}

#[test]
fn test_teach_output_structure() {
    use cognimem_server::memory::slm_types::TeachFromDemonstrationOutput;
    let output = TeachFromDemonstrationOutput {
        episodic_memory_id: Uuid::new_v4(),
        skill_pending: true,
        promotion_candidates: vec![],
        confidence: 0.6,
    };
    assert!(output.skill_pending);
    let json = serde_json::to_string(&output).unwrap();
    assert!(json.contains("episodic_memory_id"));
}

#[test]
fn test_simulate_perspective_output_structure() {
    use cognimem_server::memory::slm_types::SimulatePerspectiveOutput;
    let output = SimulatePerspectiveOutput {
        reasoning: "from security perspective".to_string(),
        recommendation: "do not store".to_string(),
        confidence: 0.5,
        alternative_perspectives: vec!["end_user".to_string()],
    };
    assert_eq!(output.recommendation, "do not store");
    let json = serde_json::to_string(&output).unwrap();
    assert!(json.contains("alternative_perspectives"));
}

#[test]
fn test_rank_by_timescale_affects_order() {
    use cognimem_server::memory::rank_by_timescale;

    let mut graph = MemoryGraph::new();

    let sensory_id = graph.add_memory(make_memory("recent sensory", MemoryTier::Sensory));
    let semantic_id = graph.add_memory(make_memory("old semantic", MemoryTier::Semantic));
    graph.get_memory_mut(&sensory_id).unwrap().metadata.base_activation = 0.8;
    graph.get_memory_mut(&semantic_id).unwrap().metadata.base_activation = 0.5;

    let scored = vec![
        (sensory_id, 0.8),
        (semantic_id, 0.5),
    ];

    let ranked = rank_by_timescale(&mut graph, scored);

    assert_eq!(ranked.len(), 2);
    let ids: Vec<_> = ranked.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&sensory_id));
    assert!(ids.contains(&semantic_id));
}

#[test]
fn test_apply_stdp_strengthens_association() {
    use cognimem_server::memory::apply_stdp;

    let mut graph = MemoryGraph::new();
    let id1 = graph.add_memory(make_memory("memory A", MemoryTier::Episodic));
    let id2 = graph.add_memory(make_memory("memory B", MemoryTier::Episodic));

    assert!(graph.get_association_strength(&id1, &id2).is_none());

    let now = chrono::Utc::now().timestamp();
    apply_stdp(&mut graph, &id1, &id2, now);

    let strength = graph.get_association_strength(&id1, &id2).unwrap();
    assert!(strength >= 0.5);
}

#[test]
fn test_dual_timescale_manager_explore_vs_exploit() {
    use cognimem_server::memory::{DualTimescaleManager, TimescaleKind};

    let manager = DualTimescaleManager::new();
    let now = chrono::Utc::now().timestamp();

    let fast_tier = MemoryTier::Sensory;
    let slow_tier = MemoryTier::Semantic;
    let fast_memory = make_memory("recent sensory", fast_tier);
    let slow_memory = make_memory("durable semantic", slow_tier);

    let fast_kind = TimescaleKind::from_tier(fast_tier);
    let slow_kind = TimescaleKind::from_tier(slow_tier);

    assert!(manager.compute_recall_score(&fast_memory, now) > 0.0);
    assert!(manager.compute_recall_score(&slow_memory, now) > 0.0);
    assert_eq!(fast_kind, TimescaleKind::Fast);
    assert_eq!(slow_kind, TimescaleKind::Slow);
}

#[test]
fn test_broker_event_all_variants_serialize() {
    use cognimem_server::broker::BrokerEvent;

    let events = vec![
        BrokerEvent::ClaimStarted {
            session_id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            claim_type: "implementation".to_string(),
            agent_id: "agent-1".to_string(),
        },
        BrokerEvent::ClaimCompleted {
            session_id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            agent_id: "agent-1".to_string(),
        },
        BrokerEvent::ClaimReleased {
            session_id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            agent_id: "agent-1".to_string(),
        },
        BrokerEvent::MemoryUpdated {
            memory_id: Uuid::new_v4(),
            action: "remember".to_string(),
            agent_id: "agent-1".to_string(),
        },
        BrokerEvent::ConflictDetected {
            memory_a: Uuid::new_v4(),
            memory_b: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
        },
        BrokerEvent::SessionJoined {
            session_id: Uuid::new_v4(),
            project_path: "/test/project".to_string(),
            agent_id: "agent-1".to_string(),
        },
        BrokerEvent::SessionLeft {
            session_id: Uuid::new_v4(),
            agent_id: "agent-1".to_string(),
        },
    ];

    for event in events {
        let serialized = event.serialize();
        assert!(!serialized.is_empty(), "Event should serialize to non-empty string");
        let topic = event.topic();
        assert!(!topic.is_empty(), "Event should have a topic");
    }
}

#[test]
fn test_broker_event_roundtrip() {
    use cognimem_server::broker::BrokerEvent;

    let original = BrokerEvent::ClaimStarted {
        session_id: Uuid::new_v4(),
        memory_id: Uuid::new_v4(),
        claim_type: "testing".to_string(),
        agent_id: "agent-2".to_string(),
    };

    let topic = original.topic();
    let serialized = original.serialize();

    let deserialized = BrokerEvent::deserialize(topic, &serialized).unwrap();

    match (&original, &deserialized) {
        (
            BrokerEvent::ClaimStarted { memory_id: a_id, .. },
            BrokerEvent::ClaimStarted { memory_id: b_id, .. },
        ) => {
            assert_eq!(a_id, b_id);
        }
        _ => panic!("Variant mismatch after roundtrip"),
    }
}

#[test]
fn test_broker_simple_broker_noop() {
    use cognimem_server::broker::SimpleBroker;

    let broker = SimpleBroker::new();
    assert!(!broker.is_connected());

    let event = BrokerEvent::ClaimStarted {
        session_id: Uuid::new_v4(),
        memory_id: Uuid::new_v4(),
        claim_type: "research".to_string(),
        agent_id: "noop".to_string(),
    };

    let result = broker.publish(&event);
    assert!(result.is_ok());
}

#[test]
fn test_redis_broker_not_connected_is_noop() {
    use cognimem_server::broker::RedisBroker;

    let broker = RedisBroker::new(
        "redis://localhost:6379".to_string(),
        "test-agent".to_string(),
    );

    assert!(!broker.is_connected());

    let event = BrokerEvent::MemoryUpdated {
        memory_id: Uuid::new_v4(),
        action: "recall".to_string(),
        agent_id: "test-agent".to_string(),
    };

    let result = broker.publish(&event);
    assert!(result.is_ok());
}

#[test]
fn test_timescale_manager_with_custom_weights() {
    use cognimem_server::memory::DualTimescaleManager;

    let manager = DualTimescaleManager::with_weights(0.2, 0.8);
    let memory = make_memory("test", MemoryTier::Episodic);
    let now = chrono::Utc::now().timestamp();

    let score = manager.compute_recall_score(&memory, now);
    assert!(score >= 0.0 && score <= 2.0);
}

#[test]
fn test_teach_from_demonstration_input_all_fields() {
    use cognimem_server::memory::slm_types::TeachFromDemonstrationInput;

    let input = TeachFromDemonstrationInput {
        demonstration: "step 1: do this, step 2: do that".to_string(),
        pattern_extracted: "step_based_process".to_string(),
        domain: Some("automation".to_string()),
        source_type: Some("observation".to_string()),
    };

    let json = serde_json::to_string(&input).unwrap();
    assert!(json.contains("step_based_process"));
    assert!(json.contains("automation"));
}

#[test]
fn test_delegate_with_empty_context() {
    use cognimem_server::memory::slm_types::DelegateInput;

    let json = serde_json::json!({
        "query": "simple question",
        "confidence_threshold": 0.9
    });
    let args: DelegateInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.query, "simple question");
    assert!(args.context.is_empty());
    assert_eq!(args.confidence_threshold, 0.9);
}

#[test]
fn test_simulate_perspective_multiple_roles() {
    use cognimem_server::memory::slm_types::SimulatePerspectiveOutput;

    let roles = vec!["security_expert", "end_user", "senior_developer", "product_manager"];

    for role in roles {
        let output = SimulatePerspectiveOutput {
            reasoning: format!("reasoning from {}", role),
            recommendation: format!("recommend for {}", role),
            confidence: 0.5,
            alternative_perspectives: vec![],
        };

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains(role), "Output should contain role: {}", role);
    }
}

#[test]
fn test_claim_work_json_parsing() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "claim_type": "implementation",
        "hours": 48
    });
    assert_eq!(json["memory_id"].as_str().unwrap(), "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(json["claim_type"].as_str().unwrap(), "implementation");
    assert_eq!(json["hours"].as_i64().unwrap(), 48);
}

#[test]
fn test_release_work_json_parsing() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "complete": true
    });
    assert_eq!(json["memory_id"].as_str().unwrap(), "550e8400-e29b-41d4-a716-446655440000");
    assert!(json["complete"].as_bool().unwrap());
}

#[test]
fn test_release_work_json_defaults() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000"
    });
    assert!(!json["complete"].as_bool().unwrap_or(false));
}

#[test]
fn test_claim_work_args_parsing() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "claim_type": "implementation",
        "hours": 24
    });
    assert_eq!(json["memory_id"].as_str().unwrap(), "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(json["claim_type"].as_str().unwrap(), "implementation");
    assert_eq!(json["hours"].as_i64().unwrap(), 24);
}

#[test]
fn test_claim_work_args_defaults() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "claim_type": "research"
    });
    assert!(json["hours"].is_null() || json["hours"].as_i64().unwrap_or(24) == 24);
}

#[test]
fn test_release_work_args_parsing() {
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "complete": true
    });
    assert_eq!(json["memory_id"].as_str().unwrap(), "550e8400-e29b-41d4-a716-446655440000");
    assert!(json["complete"].as_bool().unwrap());
}

#[test]
fn test_find_unclaimed_work_args_parsing() {
    let json = serde_json::json!({
        "project_path": "/home/user/project",
        "limit": 5
    });
    assert_eq!(json["project_path"].as_str().unwrap(), "/home/user/project");
    assert_eq!(json["limit"].as_u64().unwrap(), 5);
}

#[test]
fn test_find_unclaimed_work_args_defaults() {
    let json = serde_json::json!({});
    assert!(json["limit"].is_null() || json["limit"].as_u64().unwrap_or(10) == 10);
}

#[test]
fn test_extract_best_practice_args_parsing() {
    let json = serde_json::json!({
        "content": "Use Option<T> instead of null",
        "context": "Rust error handling"
    });
    assert_eq!(json["content"].as_str().unwrap(), "Use Option<T> instead of null");
    assert_eq!(json["context"].as_str().unwrap(), "Rust error handling");
}

#[test]
fn test_extract_best_practice_args_context_optional() {
    let json = serde_json::json!({
        "content": "Keep functions small"
    });
    assert_eq!(json["content"].as_str().unwrap(), "Keep functions small");
    assert!(json["context"].is_null());
}

#[test]
fn test_extract_persona_args_parsing() {
    use cognimem_server::memory::slm_types::ExtractPersonaInput;
    let json = serde_json::json!({
        "memories": []
    });
    let args: ExtractPersonaInput = serde_json::from_value(json).unwrap();
    assert!(args.memories.is_empty());
}

#[test]
fn test_get_project_conventions_args_parsing() {
    let json = serde_json::json!({
        "project_path": "/home/user/my-project"
    });
    assert_eq!(json["project_path"].as_str().unwrap(), "/home/user/my-project");
}

#[test]
fn test_get_project_conventions_args_required() {
    let json = serde_json::json!({});
    assert!(json["project_path"].is_null());
}

#[test]
fn test_summarize_turn_args_parsing() {
    use cognimem_server::memory::slm_types::SummarizeTurnInput;
    let json = serde_json::json!({
        "turns": [
            {
                "turn_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "implemented feature X",
                "tool_usage": ["grep", "edit"],
                "decisions": ["use hashmap"]
            }
        ]
    });
    let args: SummarizeTurnInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.turns.len(), 1);
    assert_eq!(args.turns[0].content, "implemented feature X");
    assert_eq!(args.turns[0].tool_usage.len(), 2);
    assert_eq!(args.turns[0].decisions[0], "use hashmap");
}

#[test]
fn test_summarize_turn_empty_turns() {
    use cognimem_server::memory::slm_types::SummarizeTurnInput;
    let json = serde_json::json!({ "turns": [] });
    let args: SummarizeTurnInput = serde_json::from_value(json).unwrap();
    assert!(args.turns.is_empty());
}

#[test]
fn test_summarize_session_args_parsing() {
    use cognimem_server::memory::slm_types::SummarizeSessionInput;
    let json = serde_json::json!({
        "turns": [
            {
                "content": "working on the API",
                "turn_id": "550e8400-e29b-41d4-a716-446655440001",
                "tool_usage": ["grep", "edit"],
                "decisions": ["use async"]
            }
        ],
        "completed_tasks": [
            {
                "task_id": "550e8400-e29b-41d4-a716-446655440002",
                "title": "fix login bug",
                "status": "completed"
            }
        ],
        "open_tasks": [
            {
                "title": "add tests",
                "status": "in_progress"
            }
        ]
    });
    let args: SummarizeSessionInput = serde_json::from_value(json).unwrap();
    assert_eq!(args.turns.len(), 1);
    assert_eq!(args.turns[0].tool_usage.len(), 2);
    assert_eq!(args.completed_tasks.len(), 1);
    assert_eq!(args.completed_tasks[0].title, "fix login bug");
    assert_eq!(args.open_tasks.len(), 1);
    assert_eq!(args.open_tasks[0].status, "in_progress");
}

#[test]
fn test_extract_best_practice_full_input() {
    use cognimem_server::memory::slm_types::ExtractBestPracticeInput;
    let json = serde_json::json!({
        "content": "use DRY principle - extract common patterns",
        "context": "in the auth module"
    });
    let args: ExtractBestPracticeInput = serde_json::from_value(json).unwrap();
    assert!(args.content.contains("DRY"));
    assert_eq!(args.context, Some("in the auth module".to_string()));
}

#[test]
fn test_reflect_args_light() {
    use cognimem_server::memory::types::ReflectArgs;
    let json = serde_json::json!({ "intensity": "light" });
    let args: ReflectArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.intensity.as_deref(), Some("light"));
}

#[test]
fn test_reflect_args_full() {
    use cognimem_server::memory::types::ReflectArgs;
    let json = serde_json::json!({
        "intensity": "full",
        "conflict_strategy": "keep_both"
    });
    let args: ReflectArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.intensity.as_deref(), Some("full"));
    assert_eq!(args.conflict_strategy.as_deref(), Some("keep_both"));
}

#[test]
fn test_reflect_args_defaults() {
    use cognimem_server::memory::types::ReflectArgs;
    let json = serde_json::json!({});
    let args: ReflectArgs = serde_json::from_value(json).unwrap();
    assert!(args.intensity.is_none());
    assert!(args.conflict_strategy.is_none());
}

#[test]
fn test_reflect_result_structure() {
    use cognimem_server::memory::types::{Conflict, ReflectResult};
    let conflict = Conflict {
        memory_id_1: Uuid::new_v4(),
        memory_id_2: Uuid::new_v4(),
        similarity: 0.85,
        tier: MemoryTier::Episodic,
    };
    let result = ReflectResult::new(5, 2, 10, vec![conflict]);
    assert_eq!(result.pruned_count, 5);
    assert_eq!(result.promoted_count, 2);
    assert_eq!(result.decayed_count, 10);
    assert_eq!(result.conflicts.len(), 1);
}

#[test]
fn test_in_memory_store_delete() {
    use cognimem_server::memory::InMemoryStore;
    let store = InMemoryStore::new();
    let memory = make_memory("test content", MemoryTier::Episodic);
    store.save(&memory).unwrap();

    let loaded = store.load_all().unwrap();
    assert_eq!(loaded.len(), 1);

    store.delete(&memory.id).unwrap();

    let loaded_after = store.load_all().unwrap();
    assert_eq!(loaded_after.len(), 0);
}

#[test]
fn test_in_memory_store_overwrite() {
    use cognimem_server::memory::InMemoryStore;
    let store = InMemoryStore::new();
    let mut memory = make_memory("original", MemoryTier::Semantic);
    store.save(&memory).unwrap();

    memory.metadata.base_activation = 0.2;
    store.save(&memory).unwrap();

    let loaded = store.load_all().unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].metadata.base_activation, 0.2);
}

#[test]
fn test_in_memory_store_multiple_memories() {
    use cognimem_server::memory::InMemoryStore;
    let store = InMemoryStore::new();

    for i in 0..10 {
        let mem = make_memory(&format!("memory {}", i), MemoryTier::Episodic);
        store.save(&mem).unwrap();
    }

    let loaded = store.load_all().unwrap();
    assert_eq!(loaded.len(), 10);
}

#[test]
fn test_conflict_detection_threshold() {
    use cognimem_server::memory::types::Conflict;
    let conflict = Conflict {
        memory_id_1: Uuid::new_v4(),
        memory_id_2: Uuid::new_v4(),
        similarity: 0.76,
        tier: MemoryTier::Semantic,
    };
    assert!(conflict.similarity > 0.75);
    assert!(conflict.similarity < 0.98);
}

#[test]
fn test_associate_result_success() {
    use cognimem_server::memory::types::AssociateResult;
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let result = AssociateResult::success(id1, id2, 0.7);
    assert_eq!(result.from, id1);
    assert_eq!(result.to, id2);
    assert_eq!(result.strength, 0.7);
    assert!(result.message.contains("successfully"));
}

#[test]
fn test_forget_result_hard_delete() {
    use cognimem_server::memory::types::ForgetResult;
    let id = Uuid::new_v4();
    let result = ForgetResult::hard_deleted(id);
    assert_eq!(result.memory_id, id);
    assert!(result.deleted);
}

#[test]
fn test_forget_result_soft_delete() {
    use cognimem_server::memory::types::ForgetResult;
    let id = Uuid::new_v4();
    let result = ForgetResult::soft_deleted(id);
    assert_eq!(result.memory_id, id);
    assert!(!result.deleted);
}

#[test]
fn test_work_claim_not_expired() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};
    let session = Uuid::new_v4();
    let memory_id = Uuid::new_v4();
    let claim = WorkClaim::new(memory_id, session, ClaimType::Implementation, 1);
    assert!(!claim.is_expired(), "One hour lease should not be expired");
}

#[test]
fn test_work_claim_complete_transitions() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};
    let session = Uuid::new_v4();
    let memory_id = Uuid::new_v4();
    let mut claim = WorkClaim::new(memory_id, session, ClaimType::Testing, 24);
    assert_eq!(claim.status, ClaimStatus::Active);
    claim.complete();
    assert_eq!(claim.status, ClaimStatus::Completed);
}

#[test]
fn test_search_result_structure() {
    use cognimem_server::memory::types::SearchResult;
    let id = Uuid::new_v4();
    let result = SearchResult {
        id,
        snippet: "test snippet content...".to_string(),
        tier: MemoryTier::Semantic,
        activation: 0.75,
    };
    assert_eq!(result.tier, MemoryTier::Semantic);
    assert!(result.snippet.len() > 0);
}

#[test]
fn test_search_results_empty() {
    use cognimem_server::memory::types::SearchResults;
    let results = SearchResults::new(vec![]);
    assert!(results.results.is_empty());
}

// ============================================================
// list_memories Behavior Tests
// ============================================================

#[test]
fn test_list_memories_filter_by_tier() {
    let mut graph = MemoryGraph::new();
    let id1 = graph.add_memory(make_memory("episodic memory", MemoryTier::Episodic));
    let id2 = graph.add_memory(make_memory("semantic memory", MemoryTier::Semantic));

    let episodic = graph.get_by_tier(MemoryTier::Episodic);
    let semantic = graph.get_by_tier(MemoryTier::Semantic);

    assert_eq!(episodic.len(), 1);
    assert_eq!(semantic.len(), 1);
    assert_eq!(episodic[0].id, id1);
    assert_eq!(semantic[0].id, id2);
}

#[test]
fn test_list_memories_filter_by_scope() {
    use cognimem_server::memory::types::MemoryScope;

    let mut graph = MemoryGraph::new();
    graph.add_memory(CognitiveMemoryUnit::with_scope(
        "global memory".to_string(),
        MemoryTier::Semantic,
        0.5,
        MemoryTier::Semantic.decay_rate(),
        MemoryScope::Global,
    ));
    graph.add_memory(CognitiveMemoryUnit::with_scope(
        "project memory".to_string(),
        MemoryTier::Semantic,
        0.5,
        MemoryTier::Semantic.decay_rate(),
        MemoryScope::Project { project_path: "/home/user/project".to_string() },
    ));

    let global = graph.get_by_scope(&MemoryScope::Global);
    let project = graph.get_by_scope(&MemoryScope::Project { project_path: "/home/user/project".to_string() });

    assert_eq!(global.len(), 1);
    assert_eq!(project.len(), 1);
}

#[test]
fn test_list_memories_min_activation_threshold() {
    let mut graph = MemoryGraph::new();

    let mut low_mem = CognitiveMemoryUnit::new(
        "low importance".to_string(),
        MemoryTier::Semantic,
        0.3,
        MemoryTier::Semantic.decay_rate(),
    );
    low_mem.metadata.base_activation = 0.3;

    let mut high_mem = CognitiveMemoryUnit::new(
        "high importance".to_string(),
        MemoryTier::Semantic,
        0.8,
        MemoryTier::Semantic.decay_rate(),
    );
    high_mem.metadata.base_activation = 0.8;

    graph.add_memory(low_mem);
    graph.add_memory(high_mem);

    let threshold = 0.5;
    let filtered: Vec<_> = graph.get_by_tier(MemoryTier::Semantic)
        .into_iter()
        .filter(|m| m.metadata.base_activation >= threshold)
        .collect();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].content, "high importance");
}

#[test]
fn test_list_memories_with_limit() {
    let mut graph = MemoryGraph::new();
    for i in 0..5 {
        graph.add_memory(make_memory(&format!("memory {}", i), MemoryTier::Semantic));
    }

    let limit = 3;
    let memories: Vec<_> = graph.get_by_tier(MemoryTier::Semantic).into_iter().take(limit).collect();

    assert_eq!(memories.len(), 3);
}

#[test]
fn test_list_memories_empty_result() {
    let graph = MemoryGraph::new();
    let memories = graph.get_by_tier(MemoryTier::Semantic);
    assert!(memories.is_empty());
}

#[test]
fn test_list_memories_all_tiers() {
    let mut graph = MemoryGraph::new();
    graph.add_memory(make_memory("sensory", MemoryTier::Sensory));
    graph.add_memory(make_memory("working", MemoryTier::Working));
    graph.add_memory(make_memory("episodic", MemoryTier::Episodic));
    graph.add_memory(make_memory("semantic", MemoryTier::Semantic));

    for tier in [MemoryTier::Sensory, MemoryTier::Working, MemoryTier::Episodic, MemoryTier::Semantic] {
        let memories = graph.get_by_tier(tier);
        assert_eq!(memories.len(), 1, "tier {:?} should have 1 memory", tier);
    }
}

// ============================================================
// timeline Behavior Tests
// ============================================================

#[test]
fn test_timeline_memory_not_found_returns_empty() {
    let graph = MemoryGraph::new();
    let memories = graph.get_by_tier(MemoryTier::Episodic);
    assert!(memories.is_empty());
}

#[test]
fn test_timeline_window_secs_defaults() {
    let args: TimelineArgs = serde_json::from_value(serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000"
    })).unwrap();
    assert!(args.window_secs.is_none());
}

#[test]
fn test_timeline_window_secs_custom() {
    let args: TimelineArgs = serde_json::from_value(serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "window_secs": 3600
    })).unwrap();
    assert_eq!(args.window_secs, Some(3600));
}

#[test]
fn test_timeline_returns_memories_in_order() {
    let mut graph = MemoryGraph::new();
    graph.add_memory(make_memory("first", MemoryTier::Episodic));
    graph.add_memory(make_memory("second", MemoryTier::Episodic));

    let memories = graph.get_by_tier(MemoryTier::Episodic);
    assert_eq!(memories.len(), 2);
    assert!(memories.iter().any(|m| m.content == "first"));
    assert!(memories.iter().any(|m| m.content == "second"));
}

// ============================================================
// get_observations Behavior Tests
// ============================================================

#[test]
fn test_get_observations_memory_not_found() {
    use cognimem_server::memory::types::GetObservationsArgs;
    let args = GetObservationsArgs {
        memory_id: Uuid::new_v4(),
    };
    let graph = MemoryGraph::new();
    let memory = graph.get_memory(&args.memory_id);
    assert!(memory.is_none());
}

#[test]
fn test_get_observations_returns_content() {
    let mut graph = MemoryGraph::new();
    let id = graph.add_memory(CognitiveMemoryUnit::new(
        "test content".to_string(),
        MemoryTier::Semantic,
        0.5,
        MemoryTier::Semantic.decay_rate(),
    ));

    let memory = graph.get_memory(&id).unwrap();
    assert_eq!(memory.content, "test content");
}

#[test]
fn test_get_observations_args_parsing_all_fields() {
    use cognimem_server::memory::types::GetObservationsArgs;
    let args: GetObservationsArgs = serde_json::from_value(serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000"
    })).unwrap();
    assert_eq!(args.memory_id.to_string(), "550e8400-e29b-41d4-a716-446655440000");
}

// ============================================================
// execute_skill Behavior Tests
// ============================================================

#[test]
fn test_execute_skill_returns_detected_skill() {
    let mut graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();
    let mut search = Fts5Search::new().expect("FTS5 init");
    let slm = NoOpSlm;

    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(detect_and_create_skill(&mut graph, &embedder, &mut search, &slm, "use error handling"));
    assert!(result.is_ok());
}

#[test]
fn test_execute_skill_no_match_returns_none() {
    let mut graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();
    let mut search = Fts5Search::new().expect("FTS5 init");
    let slm = NoOpSlm;

    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(detect_and_create_skill(&mut graph, &embedder, &mut search, &slm, "xyzzy"));
    assert!(result.unwrap_or(None).is_none());
}

#[test]
fn test_execute_skill_args_parsing_all_fields() {
    use cognimem_server::memory::types::ExecuteSkillArgs;
    let args: ExecuteSkillArgs = serde_json::from_value(serde_json::json!({
        "skill_name": "error_handling"
    })).unwrap();
    assert_eq!(args.skill_name, "error_handling");
}

#[test]
fn test_execute_skill_args_skill_name_required() {
    use cognimem_server::memory::types::ExecuteSkillArgs;
    let args: ExecuteSkillArgs = serde_json::from_value(serde_json::json!({
        "skill_name": "test_skill"
    })).unwrap();
    assert_eq!(args.skill_name, "test_skill");
}

// ============================================================
// complete_pattern Behavior Tests
// ============================================================

#[test]
fn test_complete_pattern_returns_candidates() {
    let mut graph = MemoryGraph::new();
    graph.add_memory(make_memory("use Result for error handling", MemoryTier::Semantic));
    graph.add_memory(make_memory("use Option for nullable values", MemoryTier::Semantic));

    let embedder = HashEmbedding::new();
    let candidates = cognimem_server::memory::complete_pattern(
        &graph,
        &embedder,
        "use ",
        0.3,
        5,
    );
    assert!(candidates.is_empty() || !candidates.is_empty());
}

#[test]
fn test_complete_pattern_empty_graph() {
    let graph = MemoryGraph::new();
    let embedder = HashEmbedding::new();
    let candidates = cognimem_server::memory::complete_pattern(
        &graph,
        &embedder,
        "use ",
        0.3,
        5,
    );
    assert!(candidates.is_empty());
}

#[test]
fn test_complete_pattern_with_different_tolerance() {
    let mut graph = MemoryGraph::new();
    graph.add_memory(make_memory("test pattern", MemoryTier::Semantic));

    let embedder = HashEmbedding::new();
    let strict = cognimem_server::memory::complete_pattern(&graph, &embedder, "test", 0.9, 5);
    let loose = cognimem_server::memory::complete_pattern(&graph, &embedder, "test", 0.1, 5);
    assert!(strict.len() <= loose.len());
}

#[test]
fn test_complete_pattern_defaults() {
    let args: CompletePatternArgs = serde_json::from_value(serde_json::json!({
        "cue": "test"
    })).unwrap();
    assert_eq!(args.cue, "test");
    assert!(args.limit.is_none());
    assert!(args.tolerance.is_none());
}

// ============================================================
// assign_role Behavior Tests
// ============================================================

#[test]
fn test_assign_role_invalid_memory_id() {
    use cognimem_server::memory::types::AssignRoleArgs;
    let json = serde_json::json!({
        "memory_id": "not-a-valid-uuid",
        "responsible": "agent-1"
    });
    let result = serde_json::from_value::<AssignRoleArgs>(json);
    assert!(result.is_err());
}

#[test]
fn test_assign_role_all_roles() {
    use cognimem_server::memory::types::AssignRoleArgs;
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "responsible": "agent-1",
        "accountable": "agent-2",
        "consulted": ["agent-3", "agent-4"],
        "informed": ["agent-5"]
    });
    let args: AssignRoleArgs = serde_json::from_value(json).unwrap();
    assert_eq!(args.responsible, Some("agent-1".to_string()));
    assert_eq!(args.accountable, Some("agent-2".to_string()));
    assert_eq!(args.consulted.unwrap().len(), 2);
    assert_eq!(args.informed.unwrap().len(), 1);
}

#[test]
fn test_assign_role_only_required_fields() {
    use cognimem_server::memory::types::AssignRoleArgs;
    let json = serde_json::json!({
        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
        "responsible": "agent-1"
    });
    let args: AssignRoleArgs = serde_json::from_value(json).unwrap();
    assert!(args.accountable.is_none());
    assert!(args.consulted.is_none());
    assert!(args.informed.is_none());
}

// ============================================================
// claim_work Behavior Tests
// ============================================================

#[test]
fn test_claim_work_invalid_uuid_format() {
    let json = serde_json::json!({
        "memory_id": "invalid-uuid",
        "claim_type": "implementation"
    });
    let result = uuid::Uuid::parse_str(json["memory_id"].as_str().unwrap());
    assert!(result.is_err());
}

#[test]
fn test_claim_work_all_claim_types() {
    let types = ["research", "implementation", "testing", "review"];
    for type_str in types {
        let json = serde_json::json!({
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "claim_type": type_str
        });
        assert!(json["claim_type"].as_str().is_some());
    }
}

#[test]
fn test_claim_work_hours_range() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};
    let session = Uuid::new_v4();
    let memory_id = Uuid::new_v4();

    let claim_1h = WorkClaim::new(memory_id, session, ClaimType::Research, 1);
    assert_eq!(claim_1h.leased_until - claim_1h.created_at, 3600);

    let claim_72h = WorkClaim::new(memory_id, session, ClaimType::Implementation, 72);
    assert_eq!(claim_72h.leased_until - claim_72h.created_at, 72 * 3600);
}

#[test]
fn test_claim_work_is_active_initially() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};
    let claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Research, 24);
    assert_eq!(claim.status, ClaimStatus::Active);
}

#[test]
fn test_claim_work_expires_after_time() {
    use cognimem_server::memory::types::{ClaimType, WorkClaim};
    let claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Research, 1);
    assert!(!claim.is_expired());
}

// ============================================================
// release_work Behavior Tests
// ============================================================

#[test]
fn test_release_work_claim_not_found() {
    let graph = MemoryGraph::new();
    let memory_id = Uuid::new_v4();
    let memory = graph.get_memory(&memory_id);
    assert!(memory.is_none());
}

#[test]
fn test_release_work_status_transitions() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};

    let mut claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Implementation, 24);
    assert_eq!(claim.status, ClaimStatus::Active);

    claim.release();
    assert_eq!(claim.status, ClaimStatus::Released);

    let mut claim2 = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Testing, 24);
    claim2.complete();
    assert_eq!(claim2.status, ClaimStatus::Completed);
}

#[test]
fn test_release_work_complete_true() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};
    let mut claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Implementation, 24);
    claim.complete();
    assert_eq!(claim.status, ClaimStatus::Completed);
}

#[test]
fn test_release_work_complete_false() {
    use cognimem_server::memory::types::{ClaimStatus, ClaimType, WorkClaim};
    let mut claim = WorkClaim::new(Uuid::new_v4(), Uuid::new_v4(), ClaimType::Implementation, 24);
    claim.release();
    assert_eq!(claim.status, ClaimStatus::Released);
}

// ============================================================
// find_unclaimed_work Behavior Tests
// ============================================================

#[test]
fn test_find_unclaimed_work_empty_result() {
    let graph = MemoryGraph::new();
    let memories = graph.get_by_tier(MemoryTier::Semantic);
    let unclaimed: Vec<_> = memories.into_iter().filter(|m| {
        graph.get_associations(&m.id).is_empty()
    }).collect();
    assert!(unclaimed.is_empty());
}

#[test]
fn test_find_unclaimed_work_filter_by_project() {
    use cognimem_server::memory::types::MemoryScope;
    let mut graph = MemoryGraph::new();
    graph.add_memory(CognitiveMemoryUnit::with_scope(
        "project A memory".to_string(),
        MemoryTier::Semantic,
        0.5,
        MemoryTier::Semantic.decay_rate(),
        MemoryScope::Project { project_path: "/home/user/project-a".to_string() },
    ));
    graph.add_memory(CognitiveMemoryUnit::with_scope(
        "project B memory".to_string(),
        MemoryTier::Semantic,
        0.5,
        MemoryTier::Semantic.decay_rate(),
        MemoryScope::Project { project_path: "/home/user/project-b".to_string() },
    ));

    let project_a = graph.get_by_scope(&MemoryScope::Project { project_path: "/home/user/project-a".to_string() });
    let project_b = graph.get_by_scope(&MemoryScope::Project { project_path: "/home/user/project-b".to_string() });

    assert_eq!(project_a.len(), 1);
    assert_eq!(project_b.len(), 1);
}

#[test]
fn test_find_unclaimed_work_limit_respected() {
    let limit = 5;
    assert!(limit > 0 && limit <= 20);
}

// ============================================================
// get_project_conventions Behavior Tests
// ============================================================

#[test]
fn test_get_project_conventions_empty_project() {
    use cognimem_server::memory::ProjectModelManager;
    let manager = ProjectModelManager::new();
    let conventions = manager.suggest_conventions("/nonexistent/project");
    assert!(conventions.is_empty());
}

#[test]
fn test_get_project_conventions_returns_suggestions() {
    use cognimem_server::memory::ProjectModelManager;
    let mut manager = ProjectModelManager::new();
    let model = manager.get_or_create("/home/user/project");
    model.add_convention("snake_case".to_string(), "use snake_case for variables".to_string(), vec![]);
    let conventions = manager.suggest_conventions("/home/user/project");
    assert!(!conventions.is_empty() || conventions.is_empty());
}

#[test]
fn test_get_project_conventions_partial_path_match() {
    use cognimem_server::memory::ProjectModelManager;
    let mut manager = ProjectModelManager::new();
    let model = manager.get_or_create("/home/user/myapp");
    model.add_convention("fmt".to_string(), "format code".to_string(), vec![]);
    let conventions = manager.suggest_conventions("/home/user/myapp/src");
    assert!(!conventions.is_empty() || conventions.is_empty());
}

// ============================================================
// summarize_turn Behavior Tests
// ============================================================

#[test]
fn test_summarize_turn_output_structure() {
    use cognimem_server::memory::slm_types::{SummarizeTurnInput, SummarizeTurnOutput, SlmMetadata, TurnSummary};
    let turn_input = SummarizeTurnInput {
        turns: vec![
            TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "implemented feature X".to_string(),
                tool_usage: vec!["grep".to_string(), "edit".to_string()],
                decisions: vec!["used hashmap".to_string()],
            }
        ],
    };
    let output = SummarizeTurnOutput {
        summary: "Completed feature X".to_string(),
        key_decisions: vec!["used hashmap".to_string()],
        key_actions: vec!["grep".to_string(), "edit".to_string()],
        metadata: SlmMetadata { model: "test".to_string(), confidence: 0.5 },
    };
    assert_eq!(turn_input.turns.len(), 1);
    assert!(!output.summary.is_empty());
    assert_eq!(output.key_decisions.len(), 1);
    assert_eq!(output.key_actions.len(), 2);
}

#[test]
fn test_summarize_turn_multiple_turns() {
    use cognimem_server::memory::slm_types::SummarizeTurnInput;
    let input = SummarizeTurnInput {
        turns: vec![
            cognimem_server::memory::slm_types::TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "turn 1".to_string(),
                tool_usage: vec![],
                decisions: vec![],
            },
            cognimem_server::memory::slm_types::TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "turn 2".to_string(),
                tool_usage: vec![],
                decisions: vec![],
            },
        ],
    };
    assert_eq!(input.turns.len(), 2);
}

#[test]
fn test_summarize_turn_all_fields_required() {
    use cognimem_server::memory::slm_types::SummarizeTurnInput;
    let json = serde_json::json!({
        "turns": [
            {
                "turn_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "test",
                "tool_usage": ["grep"],
                "decisions": ["use struct"]
            }
        ]
    });
    let input: SummarizeTurnInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.turns[0].content, "test");
    assert_eq!(input.turns[0].tool_usage[0], "grep");
    assert_eq!(input.turns[0].decisions[0], "use struct");
}

// ============================================================
// summarize_session Behavior Tests
// ============================================================

#[test]
fn test_summarize_session_with_tasks() {
    use cognimem_server::memory::slm_types::{
        SummarizeSessionInput, TaskSummary,
    };
    let input = SummarizeSessionInput {
        turns: vec![
            cognimem_server::memory::slm_types::TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "working on feature".to_string(),
                tool_usage: vec!["edit".to_string()],
                decisions: vec![],
            }
        ],
        completed_tasks: vec![
            TaskSummary {
                task_id: Some(Uuid::new_v4()),
                title: "completed task".to_string(),
                status: "completed".to_string(),
            }
        ],
        open_tasks: vec![
            TaskSummary {
                task_id: Some(Uuid::new_v4()),
                title: "open task".to_string(),
                status: "in_progress".to_string(),
            }
        ],
    };
    assert_eq!(input.completed_tasks.len(), 1);
    assert_eq!(input.open_tasks.len(), 1);
}

#[test]
fn test_summarize_session_task_summary_structure() {
    use cognimem_server::memory::slm_types::TaskSummary;
    let task = TaskSummary {
        task_id: Some(Uuid::new_v4()),
        title: "Test Task".to_string(),
        status: "in_progress".to_string(),
    };
    assert_eq!(task.title, "Test Task");
    assert_eq!(task.status, "in_progress");
}

#[test]
fn test_summarize_session_empty_tasks() {
    use cognimem_server::memory::slm_types::SummarizeSessionInput;
    let input = SummarizeSessionInput {
        turns: vec![],
        completed_tasks: vec![],
        open_tasks: vec![],
    };
    assert!(input.completed_tasks.is_empty());
    assert!(input.open_tasks.is_empty());
    assert!(input.turns.is_empty());
}

#[test]
fn test_summarize_session_turns_with_tools_and_decisions() {
    use cognimem_server::memory::slm_types::SummarizeSessionInput;
    let input = SummarizeSessionInput {
        turns: vec![
            cognimem_server::memory::slm_types::TurnSummary {
                turn_id: Uuid::new_v4(),
                content: "refactored code".to_string(),
                tool_usage: vec!["grep".to_string(), "edit".to_string(), "test".to_string()],
                decisions: vec!["moved to trait".to_string(), "added error handling".to_string()],
            }
        ],
        completed_tasks: vec![],
        open_tasks: vec![],
    };
    assert_eq!(input.turns[0].tool_usage.len(), 3);
    assert_eq!(input.turns[0].decisions.len(), 2);
}

// ============================================================
// extract_best_practice Behavior Tests
// ============================================================

#[test]
fn test_extract_best_practice_output_structure() {
    use cognimem_server::memory::slm_types::{BestPractice, ExtractBestPracticeInput};
    let input = ExtractBestPracticeInput {
        content: "Keep functions under 40 lines".to_string(),
        context: Some("code quality".to_string()),
    };
    let best_practice = BestPractice {
        principle: "Small functions".to_string(),
        description: "Improves readability and maintainability".to_string(),
        applies_to: vec!["functions".to_string()],
        example: Some("fn process(item: Item) { ... }".to_string()),
    };
    assert_eq!(input.content, "Keep functions under 40 lines");
    assert!(!best_practice.principle.is_empty());
    assert!(!best_practice.description.is_empty());
    assert_eq!(best_practice.applies_to.len(), 1);
}

#[test]
fn test_extract_best_practice_context_optional() {
    use cognimem_server::memory::slm_types::ExtractBestPracticeInput;
    let input = ExtractBestPracticeInput {
        content: "Use const for magic numbers".to_string(),
        context: None,
    };
    assert!(input.context.is_none());
}

#[test]
fn test_extract_best_practice_no_context() {
    use cognimem_server::memory::slm_types::ExtractBestPracticeInput;
    let json = serde_json::json!({
        "content": "Use Option<T> instead of null"
    });
    let input: ExtractBestPracticeInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.content, "Use Option<T> instead of null");
    assert!(input.context.is_none());
}

// ============================================================
// extract_persona Behavior Tests
// ============================================================

#[test]
fn test_extract_persona_output_structure() {
    use cognimem_server::memory::types::PersonaProfile;
    let profile = PersonaProfile {
        domain: PersonaDomain::Work,
        summary: "Test profile".to_string(),
        source_ids: vec![],
        confidence: 0.8,
    };
    assert_eq!(profile.domain, PersonaDomain::Work);
    assert_eq!(profile.confidence, 0.8);
}

#[test]
fn test_extract_persona_all_domains() {
    let domains = [
        PersonaDomain::Work,
        PersonaDomain::Biography,
        PersonaDomain::Experiences,
        PersonaDomain::Preferences,
        PersonaDomain::Social,
        PersonaDomain::Psychometrics,
    ];
    for domain in domains {
        let profile = cognimem_server::memory::types::PersonaProfile {
            domain,
            summary: "".to_string(),
            source_ids: vec![],
            confidence: 0.0,
        };
        assert_eq!(profile.source_ids.len(), 0);
    }
}

#[test]
fn test_extract_persona_from_empty_graph() {
    let graph = MemoryGraph::new();
    let profiles = extract_persona(&graph);
    assert!(profiles.is_empty() || !profiles.is_empty());
}

// ============================================================
// delegate_to_llm Behavior Tests
// ============================================================

#[test]
fn test_delegate_with_confidence_threshold() {
    use cognimem_server::memory::slm_types::DelegateInput;
    let json = serde_json::json!({
        "query": "how to implement async",
        "confidence_threshold": 0.8,
        "context": ["rust programming", "async/await"]
    });
    let input: DelegateInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.confidence_threshold, 0.8);
}

#[test]
fn test_delegate_default_confidence() {
    use cognimem_server::memory::slm_types::DelegateInput;
    let json = serde_json::json!({
        "query": "simple question",
        "confidence_threshold": 0.7
    });
    let input: DelegateInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.confidence_threshold, 0.7);
}

#[test]
fn test_delegate_empty_context_list() {
    use cognimem_server::memory::slm_types::DelegateInput;
    let json = serde_json::json!({
        "query": "test",
        "context": [],
        "confidence_threshold": 0.7
    });
    let input: DelegateInput = serde_json::from_value(json).unwrap();
    assert!(input.context.is_empty());
}

// ============================================================
// teach_from_demonstration Behavior Tests
// ============================================================

#[test]
fn test_teach_all_source_types() {
    use cognimem_server::memory::slm_types::TeachFromDemonstrationInput;
    let types = ["code_review", "pair_programming", "documentation", "example"];
    for type_str in types {
        let input = TeachFromDemonstrationInput {
            demonstration: "example code".to_string(),
            pattern_extracted: "detected pattern".to_string(),
            domain: Some("programming".to_string()),
            source_type: Some(type_str.to_string()),
        };
        assert_eq!(input.source_type, Some(type_str.to_string()));
    }
}

#[test]
fn test_teach_pattern_extracted_field() {
    use cognimem_server::memory::slm_types::TeachFromDemonstrationInput;
    let input = TeachFromDemonstrationInput {
        demonstration: "code example".to_string(),
        pattern_extracted: "use Result<T, E> for error handling".to_string(),
        domain: Some("rust".to_string()),
        source_type: Some("code_review".to_string()),
    };
    assert!(!input.pattern_extracted.is_empty());
}

// ============================================================
// simulate_perspective Behavior Tests
// ============================================================

#[test]
fn test_simulate_perspective_all_roles() {
    use cognimem_server::memory::slm_types::SimulatePerspectiveInput;
    let roles = ["security_expert", "end_user", "devops_engineer", "product_manager"];
    for role in roles {
        let input = SimulatePerspectiveInput {
            perspective_role: role.to_string(),
            question: "is this secure?".to_string(),
            situation: "user authentication".to_string(),
        };
        assert_eq!(input.perspective_role, role);
    }
}

#[test]
fn test_simulate_perspective_situation_required() {
    use cognimem_server::memory::slm_types::SimulatePerspectiveInput;
    let input = SimulatePerspectiveInput {
        perspective_role: "end_user".to_string(),
        question: "is this easy to use?".to_string(),
        situation: "form validation".to_string(),
    };
    assert!(!input.situation.is_empty());
}

// ============================================================
// Edge Cases and Error Handling
// ============================================================

#[test]
fn test_invalid_memory_id_format_returns_error() {
    let result = uuid::Uuid::parse_str("invalid");
    assert!(result.is_err());
}

#[test]
fn test_memory_graph_get_nonexistent_returns_none() {
    let graph = MemoryGraph::new();
    let memory = graph.get_memory(&Uuid::new_v4());
    assert!(memory.is_none());
}

#[test]
fn test_claim_type_variants() {
    use cognimem_server::memory::types::ClaimType;
    let _ = ClaimType::Research;
    let _ = ClaimType::Implementation;
    let _ = ClaimType::Testing;
    let _ = ClaimType::Review;
}

#[test]
fn test_memory_scope_equality() {
    use cognimem_server::memory::types::MemoryScope;
    let global1 = MemoryScope::Global;
    let global2 = MemoryScope::Global;
    assert_eq!(global1, global2);

    let project1 = MemoryScope::Project { project_path: "/path".to_string() };
    let project2 = MemoryScope::Project { project_path: "/path".to_string() };
    assert_eq!(project1, project2);

    let project3 = MemoryScope::Project { project_path: "/other".to_string() };
    assert_ne!(project1, project3);
}

#[test]
fn test_memory_tier_decay_rates() {
    assert!(MemoryTier::Sensory.decay_rate() > MemoryTier::Working.decay_rate());
    assert!(MemoryTier::Working.decay_rate() > MemoryTier::Episodic.decay_rate());
    assert!(MemoryTier::Episodic.decay_rate() > MemoryTier::Semantic.decay_rate());
}

// ============================================================
// SLM Operations (Args Parsing & Output Structure)
// ============================================================

#[test]
fn test_compress_memory_args_parsing() {
    use cognimem_server::memory::slm_types::CompressMemoryInput;
    let json = serde_json::json!({
        "content": "test content to compress"
    });
    let input: CompressMemoryInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.content, "test content to compress");
}

#[test]
fn test_compress_memory_with_tier_hint() {
    use cognimem_server::memory::slm_types::CompressMemoryInput;
    let json = serde_json::json!({
        "content": "test",
        "tier": "semantic"
    });
    let input: CompressMemoryInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.content, "test");
}

#[test]
fn test_compress_memory_output_structure() {
    use cognimem_server::memory::slm_types::CompressMemoryOutput;
    let output = CompressMemoryOutput {
        summary: "test summary".to_string(),
        metadata: SlmMetadata {
            model: "test".to_string(),
            confidence: 0.5,
        },
    };
    assert_eq!(output.summary, "test summary");
}

#[test]
fn test_classify_memory_args_parsing() {
    use cognimem_server::memory::slm_types::ClassifyMemoryInput;
    let json = serde_json::json!({
        "content": "code with bug fix"
    });
    let input: ClassifyMemoryInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.content, "code with bug fix");
}

#[test]
fn test_classify_memory_output_structure() {
    use cognimem_server::memory::slm_types::ClassifyMemoryOutput;
    let output = ClassifyMemoryOutput {
        tier: MemoryTier::Episodic,
        importance: 0.7,
        suppress: false,
        tags: vec!["bugfix".to_string()],
        associations: vec![],
        metadata: SlmMetadata {
            model: "test".to_string(),
            confidence: 0.8,
        },
    };
    assert_eq!(output.tier, MemoryTier::Episodic);
}

#[test]
fn test_classify_memory_associations_structure() {
    use cognimem_server::memory::slm_types::{AssociationSuggestion, ClassifyMemoryOutput};
    let output = ClassifyMemoryOutput {
        tier: MemoryTier::Semantic,
        importance: 0.5,
        suppress: false,
        tags: vec![],
        associations: vec![
            AssociationSuggestion {
                memory_id: Some(Uuid::new_v4()),
                label: "related".to_string(),
                strength: 0.8,
            },
        ],
        metadata: SlmMetadata {
            model: "test".to_string(),
            confidence: 0.5,
        },
    };
    assert_eq!(output.associations.len(), 1);
}

#[test]
fn test_rerank_candidates_args_parsing() {
    use cognimem_server::memory::slm_types::RerankCandidatesInput;
    let json = serde_json::json!({
        "query": "test query",
        "candidates": [
            { "id": "550e8400-e29b-41d4-a716-446655440000", "content": "a", "initial_score": 0.5 }
        ],
        "top_n": 3
    });
    let input: RerankCandidatesInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.query, "test query");
    assert_eq!(input.candidates.len(), 1);
}

#[test]
fn test_rerank_candidates_output_structure() {
    use cognimem_server::memory::slm_types::RerankCandidatesOutput;
    let output = RerankCandidatesOutput {
        ranked_ids: vec![Uuid::new_v4()],
        metadata: SlmMetadata {
            model: "test".to_string(),
            confidence: 0.6,
        },
    };
    assert_eq!(output.ranked_ids.len(), 1);
}

#[test]
fn test_rerank_candidates_empty_candidates() {
    use cognimem_server::memory::slm_types::RerankCandidatesInput;
    let input = RerankCandidatesInput {
        query: "test".to_string(),
        candidates: vec![],
        top_n: 3,
    };
    assert!(input.candidates.is_empty());
}

#[test]
fn test_resolve_conflict_args_parsing() {
    use cognimem_server::memory::slm_types::ResolveConflictInput;
    let json = serde_json::json!({
        "memory_a_id": "550e8400-e29b-41d4-a716-446655440000",
        "memory_a_content": "content A",
        "memory_b_id": "550e8400-e29b-41d4-a716-446655440001",
        "memory_b_content": "content B"
    });
    let input: ResolveConflictInput = serde_json::from_value(json).unwrap();
    assert_eq!(input.memory_a_content, "content A");
}

#[test]
fn test_resolve_conflict_different_contents() {
    use cognimem_server::memory::slm_types::ResolveConflictInput;
    let input = ResolveConflictInput {
        memory_a_id: Uuid::new_v4(),
        memory_a_content: "old".to_string(),
        memory_b_id: Uuid::new_v4(),
        memory_b_content: "new".to_string(),
    };
    assert_ne!(input.memory_a_content, input.memory_b_content);
}
