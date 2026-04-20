use super::types::{ConflictResolution, PersonaProfile};
use super::slm::{RerankCandidate, SlmEngine};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    pub model: String,
    pub base_url: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5-coder:3b".to_string(),
            base_url: "http://localhost:11434".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

pub struct OllamaSlm {
    client: reqwest::blocking::Client,
    model: String,
    base_url: String,
}

impl OllamaSlm {
    pub fn new(model: Option<String>, base_url: Option<String>) -> Self {
        let config = OllamaConfig {
            model: model.unwrap_or_else(|| "qwen2.5-coder:3b".to_string()),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
        };
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("failed to build reqwest client"),
            model: config.model,
            base_url: config.base_url,
        }
    }

    pub fn check_available(&self) -> bool {
        match self.client.get(&self.base_url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn generate(&self, prompt: &str, max_tokens: usize) -> Option<String> {
        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            options: OllamaOptions {
                temperature: 0.3,
                num_predict: max_tokens,
            },
        };

        let url = format!("{}/api/generate", self.base_url);
        match self.client.post(&url).json(&request).send() {
            Ok(resp) => resp.json::<OllamaResponse>().ok().map(|r| r.response),
            Err(e) => {
                tracing::warn!("Ollama request failed: {e}");
                None
            }
        }
    }
}

impl SlmEngine for OllamaSlm {
    fn compress(&self, content: &str) -> String {
        let prompt = format!(
            "Summarize this memory in 20 words or less: {}",
            content
        );
        self.generate(&prompt, 30).unwrap_or_else(|| {
            content
                .split_whitespace()
                .take(20)
                .collect::<Vec<_>>()
                .join(" ")
        })
    }

    fn rerank(&self, candidates: &[RerankCandidate], query: &str, top_n: usize) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let context = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i, c.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Given the query: '{}'\nRank these candidates by relevance (return indices separated by commas):\n{}\nReturn only the indices, most relevant first.",
            query,
            context
        );

        let result = self.generate(&prompt, 100);

        result
            .map(|r| {
                r.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .take(top_n)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                candidates
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i)
                    .take(top_n)
                    .collect()
            })
    }

    fn resolve_conflict(&self, content_a: &str, content_b: &str) -> ConflictResolution {
        let prompt = format!(
            "Two memories contain potentially conflicting information:\n1. {}\n2. {}\nWhich one is more likely to be correct or should be kept? Reply with '1' or '2'.",
            content_a,
            content_b
        );

        let result = self.generate(&prompt, 10);

        result
            .map(|r| r.trim().to_string())
            .map(|_r| ConflictResolution::LatestWins)
            .unwrap_or(ConflictResolution::LatestWins)
    }

    fn complete_pattern_hint(&self, partial: &str, context: &[&str]) -> String {
        let context_str = context.join("\n");
        let prompt = format!(
            "Given this partial cue: '{}'\nAnd this context:\n{}\nSuggest what comes next or completes this memory:",
            partial,
            context_str
        );

        self.generate(&prompt, 50).unwrap_or_else(|| partial.to_string())
    }

    fn extract_persona(&self, memories: &[String]) -> Vec<PersonaProfile> {
        if memories.is_empty() {
            return Vec::new();
        }

        let memories_text = memories
            .iter()
            .enumerate()
            .map(|(i, m)| format!("{}. {}", i + 1, m))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Analyze these memories and extract a structured persona profile.
Return a JSON array with up to 6 entries, one per domain found. Use this format:
[{{"domain": "work", "summary": "2-3 sentence summary", "confidence": 0.85}}]

Available domains: biography, experiences, preferences, social, work, psychometrics

Memories:
{}

Return only valid JSON array, nothing else:"#,
            memories_text
        );

        let result = self.generate(&prompt, 500);

        result
            .and_then(|r| serde_json::from_str(&r).ok())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_config_defaults() {
        let config = OllamaConfig::default();
        assert_eq!(config.model, "qwen2.5-coder:3b");
        assert_eq!(config.base_url, "http://localhost:11434");
    }
}