use super::graph::MemoryGraph;
use super::types::{CognitiveMemoryUnit, MemoryTier, SkillMemory};
use crate::embeddings::EmbeddingEngine;
use crate::search::SearchEngine;
use tracing::error;

const MIN_PATTERN_OCCURRENCES: usize = 3;
const SKILL_SIMILARITY_THRESHOLD: f32 = 0.65;

pub fn detect_and_create_skill(
    graph: &mut MemoryGraph,
    embedder: &dyn EmbeddingEngine,
    search: &mut dyn SearchEngine,
    new_content: &str,
) -> Option<CognitiveMemoryUnit> {
    let candidates = find_similar_memories(graph, embedder, new_content);
    if candidates.len() < MIN_PATTERN_OCCURRENCES - 1 {
        return None;
    }

    let all_contents: Vec<&str> = candidates
        .iter()
        .map(|(m, _)| m.content.as_str())
        .chain(std::iter::once(new_content))
        .collect();

    let skill_name = extract_skill_name(&all_contents);
    let steps = distill_steps(&all_contents);
    let pattern = extract_pattern(&all_contents);

    if is_skill_already_exists(graph, &skill_name) {
        return None;
    }

    let source_ids: Vec<uuid::Uuid> = candidates.iter().map(|(m, _)| m.id).collect();

    let skill = SkillMemory::new(skill_name.clone(), pattern, steps, source_ids);

    let mut memory = CognitiveMemoryUnit::new(
        format!("[skill] {}", skill_name),
        MemoryTier::Procedural,
        0.9,
        MemoryTier::Procedural.decay_rate(),
    );
    memory.metadata.base_activation = 1.0;

    let skill_json = serde_json::to_string(&skill).unwrap_or_else(|e| {
        error!("Failed to serialize skill memory: {e}");
        format!(
            "{{\"name\":\"{}\",\"pattern\":\"\",\"steps\":[],\"source_ids\":[]}}",
            skill.name
        )
    });
    memory.content.push_str(&format!("\n{}", skill_json));

    let id = graph.add_memory(memory.clone());
    let emb = embedder.embed(&memory.content);
    graph.set_embedding(id, emb);
    search.index(id, &memory.content, memory.tier);

    Some(memory)
}

pub fn find_skill(graph: &MemoryGraph, name: &str) -> Option<CognitiveMemoryUnit> {
    let prefix = format!("[skill] {}", name.to_lowercase());
    graph
        .get_all_memories()
        .into_iter()
        .filter(|m| m.tier == MemoryTier::Procedural)
        .find(|m| {
            let lower = m.content.to_lowercase();
            lower.starts_with(&prefix)
                || lower.contains(&format!("[skill] {}", name.to_lowercase()))
        })
        .cloned()
}

fn find_similar_memories<'a>(
    graph: &'a MemoryGraph,
    embedder: &dyn EmbeddingEngine,
    content: &str,
) -> Vec<(&'a CognitiveMemoryUnit, f32)> {
    let query_emb = embedder.embed(content);
    graph
        .vector_search(&query_emb, 20, SKILL_SIMILARITY_THRESHOLD)
        .into_iter()
        .filter_map(|(id, score)| {
            let mem = graph.get_memory(&id)?;
            if mem.tier == MemoryTier::Procedural {
                return None;
            }
            Some((mem, score))
        })
        .collect()
}

fn extract_pattern(contents: &[&str]) -> String {
    if contents.is_empty() {
        return String::new();
    }

    let words: Vec<Vec<&str>> = contents
        .iter()
        .map(|c| c.split_whitespace().collect())
        .collect();
    let common: Vec<&str> = words
        .first()
        .map(|w| {
            w.iter()
                .filter(|word| words.iter().all(|doc| doc.contains(word)))
                .copied()
                .collect()
        })
        .unwrap_or_default();

    if common.is_empty() {
        contents[0].chars().take(100).collect()
    } else {
        common.join(" ")
    }
}

fn extract_skill_name(contents: &[&str]) -> String {
    let words: Vec<Vec<&str>> = contents
        .iter()
        .map(|c| c.split_whitespace().collect())
        .collect();

    let mut common_words: Vec<&str> = words
        .first()
        .map(|w| {
            w.iter()
                .filter(|word| {
                    let lower = word.to_lowercase();
                    word.len() > 3
                        && words
                            .iter()
                            .all(|doc| doc.iter().any(|dw| dw.to_lowercase() == lower))
                })
                .take(4)
                .copied()
                .collect()
        })
        .unwrap_or_default();

    if common_words.is_empty() {
        common_words = contents
            .first()
            .map(|c| c.split_whitespace().take(3).collect())
            .unwrap_or_default();
    }

    common_words.join("_").to_lowercase()
}

fn distill_steps(contents: &[&str]) -> Vec<String> {
    let mut all_sentences: Vec<String> = contents
        .iter()
        .flat_map(|c| {
            c.split(['.', '\n'])
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 10)
        })
        .collect();

    all_sentences.sort_by_key(|b| std::cmp::Reverse(b.len()));
    all_sentences.truncate(5);
    all_sentences
}

fn is_skill_already_exists(graph: &MemoryGraph, name: &str) -> bool {
    find_skill(graph, name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::HashEmbedding;
    use crate::search::Fts5Search;

    fn make_memory(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
        CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
    }

    #[test]
    fn test_extract_skill_name_finds_common_words() {
        let contents = vec![
            "deploy rust application to production",
            "deploy rust library to staging",
            "deploy rust server to production",
        ];
        let name = extract_skill_name(&contents);
        assert!(name.contains("deploy"));
        assert!(name.contains("rust"));
    }

    #[test]
    fn test_extract_pattern_finds_common_words() {
        let contents = vec!["deploy rust to production", "deploy rust to staging"];
        let pattern = extract_pattern(&contents);
        assert!(pattern.contains("deploy"));
        assert!(pattern.contains("rust"));
    }

    #[test]
    fn test_distill_steps_extracts_sentences() {
        let contents = vec!["first step. second step here. third action."];
        let steps = distill_steps(&contents);
        assert!(!steps.is_empty());
    }

    #[test]
    fn test_detect_skill_requires_min_occurrences() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();
        let mut search = Fts5Search::new().unwrap();

        graph.add_memory(make_memory("deploy rust app", MemoryTier::Episodic));

        let result =
            detect_and_create_skill(&mut graph, &embedder, &mut search, "deploy rust server");
        assert!(
            result.is_none(),
            "should not create skill with only 1 similar memory"
        );
    }

    #[test]
    fn test_find_skill_returns_procedural_memory() {
        let mut graph = MemoryGraph::new();
        let embedder = HashEmbedding::new();
        let mut search = Fts5Search::new().unwrap();

        for i in 0..3 {
            let mem = make_memory(
                &format!("deploy rust app version {}", i),
                MemoryTier::Episodic,
            );
            let id = graph.add_memory(mem);
            let emb = embedder.embed(&format!("deploy rust app version {}", i));
            graph.set_embedding(id, emb);
        }

        let skill = detect_and_create_skill(
            &mut graph,
            &embedder,
            &mut search,
            "deploy rust app version 3",
        );
        assert!(skill.is_some());
        let skill = skill.unwrap();
        assert_eq!(skill.tier, MemoryTier::Procedural);
        assert!(skill.content.contains("[skill]"));
    }

    #[test]
    fn test_find_skill_by_name() {
        let mut graph = MemoryGraph::new();
        let mut mem = CognitiveMemoryUnit::new(
            "[skill] deploy_rust\n{}".to_string(),
            MemoryTier::Procedural,
            0.9,
            0.1,
        );
        mem.metadata.base_activation = 1.0;
        graph.add_memory(mem);

        let found = find_skill(&graph, "deploy_rust");
        assert!(found.is_some());
    }

    #[test]
    fn test_find_skill_not_found() {
        let graph = MemoryGraph::new();
        let found = find_skill(&graph, "nonexistent");
        assert!(found.is_none());
    }
}
