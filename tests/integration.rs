//! Integration tests for CogniMem MCP server.
//!
//! Tests the MCP tools, resources, and core functionality.

use cognimem_server::embeddings::{cosine_similarity, fuse_scores, EmbeddingEngine, HashEmbedding};
use cognimem_server::memory::types::{
    AssignRoleArgs, AssociateArgs, ConflictResolution, ExecuteSkillArgs, ForgetArgs,
    GetObservationsArgs, MemoryTier, PersonaDomain, RecallArgs, RememberArgs, SearchArgs, TimelineArgs,
};
use cognimem_server::memory::CompletePatternArgs;
use cognimem_server::memory::{
    CognitiveMemoryUnit, InMemoryStore, detect_and_create_skill, extract_persona, MemoryGraph,
    MemoryStore,
};
use cognimem_server::search::{Fts5Search, SearchEngine};

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
    assert_eq!(args.consulted, Some(vec!["agent-3".to_string(), "agent-4".to_string()]));
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

    assert!(sim_same > sim_diff, "similar texts should have higher similarity");
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

    // Check that detect_and_create_skill is callable
    _ = detect_and_create_skill(
        &mut graph,
        &embedder,
        &mut search,
        "test query",
    );
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
    assert!(found_ids.contains(&id2) || found_ids.contains(&id3), "should find at least one neighbor");
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
