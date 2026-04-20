use super::graph::MemoryGraph;
use super::types::{PersonaDomain, PersonaProfile};

const PERSONA_KEYWORDS: &[(PersonaDomain, &[&str])] = &[
    (PersonaDomain::Biography, &["born", "grew up", "hometown", "family", "childhood", "education", "age", "lives in", "from", "born in"]),
    (PersonaDomain::Experiences, &["visited", "traveled", "attended", "happened", "experienced", "went to", "saw", "encountered", "achieved", "completed"]),
    (PersonaDomain::Preferences, &["prefer", "like", "enjoy", "love", "hate", "favorite", "always", "never", "best", "worst", "prefer"]),
    (PersonaDomain::Social, &["friend", "colleague", "team", "group", "community", "partner", "collaborated", "worked with", "met", "mentor"]),
    (PersonaDomain::Work, &["project", "deploy", "code", "build", "fix", "implement", "feature", "bug", "release", "task", "sprint", "deadline", "meeting", "repo"]),
    (PersonaDomain::Psychometrics, &["always", "usually", "tend to", "typically", "style", "approach", "methodology", "workflow", "habit", "consistent"]),
];

pub fn extract_persona(graph: &MemoryGraph) -> Vec<PersonaProfile> {
    let mut profiles: std::collections::HashMap<PersonaDomain, (Vec<uuid::Uuid>, f32)> =
        std::collections::HashMap::new();

    for mem in graph.get_all_memories() {
        let content_lower = mem.content.to_lowercase();

        for (domain, keywords) in PERSONA_KEYWORDS {
            let match_count = keywords.iter().filter(|kw| content_lower.contains(*kw)).count();
            if match_count > 0 {
                let confidence = (match_count as f32 / keywords.len() as f32).min(1.0);
                let entry = profiles.entry(*domain).or_insert((Vec::new(), 0.0));
                entry.0.push(mem.id);
                entry.1 = entry.1.max(confidence);
            }
        }
    }

    profiles
        .into_iter()
        .filter_map(|(domain, (source_ids, confidence))| {
            if source_ids.is_empty() {
                return None;
            }
            let summary = summarize_domain(&domain, &source_ids, graph);
            Some(PersonaProfile {
                domain,
                summary,
                source_ids,
                confidence,
            })
        })
        .collect()
}

fn summarize_domain(domain: &PersonaDomain, source_ids: &[uuid::Uuid], graph: &MemoryGraph) -> String {
    let domain_name = match domain {
        PersonaDomain::Biography => "Biographical",
        PersonaDomain::Experiences => "Experiential",
        PersonaDomain::Preferences => "Preference",
        PersonaDomain::Social => "Social",
        PersonaDomain::Work => "Work",
        PersonaDomain::Psychometrics => "Behavioral",
    };

    let snippets: Vec<String> = source_ids
        .iter()
        .filter_map(|id| {
            let mem = graph.get_memory(id)?;
            let snippet: String = mem.content.chars().take(80).collect();
            Some(format!("- {}", snippet))
        })
        .take(5)
        .collect();

    format!("{} profile ({} memories):", domain_name, source_ids.len())
        + &if snippets.is_empty() { String::new() } else { "\n".to_string() + &snippets.join("\n") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{CognitiveMemoryUnit, MemoryTier};

    fn make_memory(content: &str, tier: MemoryTier) -> CognitiveMemoryUnit {
        CognitiveMemoryUnit::new(content.to_string(), tier, 0.5, tier.decay_rate())
    }

    #[test]
    fn test_extract_persona_detects_work() {
        let mut graph = MemoryGraph::new();
        graph.add_memory(make_memory("Ideployed the rust project to production", MemoryTier::Semantic));
        graph.add_memory(make_memory("Ifixed a bug in the auth module", MemoryTier::Semantic));

        let profiles = extract_persona(&graph);
        let work = profiles.iter().find(|p| p.domain == PersonaDomain::Work);
        assert!(work.is_some());
        assert!(!work.unwrap().source_ids.is_empty());
    }

    #[test]
    fn test_extract_persona_detects_preferences() {
        let mut graph = MemoryGraph::new();
        graph.add_memory(make_memory("Iprefer dark mode for all my editors", MemoryTier::Semantic));
        graph.add_memory(make_memory("Ienjoy using vim over vscode", MemoryTier::Semantic));

        let profiles = extract_persona(&graph);
        let pref = profiles.iter().find(|p| p.domain == PersonaDomain::Preferences);
        assert!(pref.is_some());
        assert!(pref.unwrap().confidence > 0.0);
    }

    #[test]
    fn test_extract_persona_empty_graph() {
        let graph = MemoryGraph::new();
        let profiles = extract_persona(&graph);
        assert!(profiles.is_empty());
    }

    #[test]
    fn test_extract_persona_multiple_domains() {
        let mut graph = MemoryGraph::new();
        graph.add_memory(make_memory("I built a new feature for the project", MemoryTier::Semantic));
        graph.add_memory(make_memory("I prefer oat milk in my coffee", MemoryTier::Semantic));
        graph.add_memory(make_memory("I met with my team yesterday", MemoryTier::Semantic));

        let profiles = extract_persona(&graph);
        assert!(profiles.len() >= 2, "should detect at least 2 domains");
    }
}