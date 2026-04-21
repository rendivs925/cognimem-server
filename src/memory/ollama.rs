use super::slm::SlmEngine;
use super::slm_prompts::{
    classify_memory_prompt, complete_pattern_prompt, compress_memory_prompt,
    distill_skill_prompt, extract_persona_prompt, rerank_candidates_prompt,
    resolve_conflict_prompt,
};
use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, DistillSkillInput, DistillSkillOutput,
    ExtractPersonaInput, ExtractPersonaOutput, RerankCandidatesInput, RerankCandidatesOutput,
    ResolveConflictInput, ResolveConflictOutput, SlmMetadata,
};
use crate::memory::DEFAULT_SLM_MODEL;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    pub model: String,
    pub base_url: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_SLM_MODEL.to_string(),
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
            model: model.unwrap_or_else(|| DEFAULT_SLM_MODEL.to_string()),
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

    fn parse_json<T: serde::de::DeserializeOwned>(&self, response: Option<String>) -> Option<T> {
        response.and_then(|r| serde_json::from_str(&r).ok())
    }

    fn clamp_confidence(confidence: f32) -> f32 {
        confidence.clamp(0.0, 1.0)
    }
}

impl SlmEngine for OllamaSlm {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn compress_memory(&self, input: CompressMemoryInput) -> Option<CompressMemoryOutput> {
        let prompt = compress_memory_prompt(&input);
        let mut output: CompressMemoryOutput = self.parse_json(self.generate(&prompt, 80))?;
        output.summary = output.summary.trim().to_string();
        if output.summary.is_empty() {
            return None;
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn classify_memory(&self, input: ClassifyMemoryInput) -> Option<ClassifyMemoryOutput> {
        let prompt = classify_memory_prompt(&input);
        let mut output: ClassifyMemoryOutput = self.parse_json(self.generate(&prompt, 160))?;
        output.importance = output.importance.clamp(0.0, 1.0);
        output.tags.retain(|tag| !tag.trim().is_empty());
        output.tags.sort();
        output.tags.dedup();
        for assoc in &mut output.associations {
            assoc.label = assoc.label.trim().to_string();
            assoc.strength = assoc.strength.clamp(0.0, 1.0);
        }
        output.associations.retain(|assoc| !assoc.label.is_empty());
        output.associations.truncate(20);
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn rerank_candidates(&self, input: RerankCandidatesInput) -> Option<RerankCandidatesOutput> {
        if input.candidates.is_empty() {
            return Some(RerankCandidatesOutput {
                ranked_ids: Vec::new(),
                metadata: SlmMetadata {
                    model: self.model.clone(),
                    confidence: 1.0,
                },
            });
        }

        let prompt = rerank_candidates_prompt(&input);
        let mut output: RerankCandidatesOutput = self.parse_json(self.generate(&prompt, 160))?;
        output.ranked_ids.truncate(input.top_n);
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn resolve_conflict(&self, input: ResolveConflictInput) -> Option<ResolveConflictOutput> {
        let prompt = resolve_conflict_prompt(&input);
        let mut output: ResolveConflictOutput = self.parse_json(self.generate(&prompt, 120))?;
        output.merged_summary = output.merged_summary.map(|s| s.trim().to_string());
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn extract_persona(&self, input: ExtractPersonaInput) -> Option<ExtractPersonaOutput> {
        if input.memories.is_empty() {
            return Some(ExtractPersonaOutput {
                profiles: Vec::new(),
                metadata: SlmMetadata {
                    model: self.model.clone(),
                    confidence: 1.0,
                },
            });
        }

        let prompt = extract_persona_prompt(&input);
        let mut output: ExtractPersonaOutput = self.parse_json(self.generate(&prompt, 500))?;
        for profile in &mut output.profiles {
            profile.summary = profile.summary.trim().to_string();
            profile.confidence = profile.confidence.clamp(0.0, 1.0);
        }
        output.profiles.retain(|profile| !profile.summary.is_empty());
        output.profiles.truncate(6);
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn distill_skill(&self, input: DistillSkillInput) -> Option<DistillSkillOutput> {
        let prompt = distill_skill_prompt(&input);
        let mut output: DistillSkillOutput = self.parse_json(self.generate(&prompt, 220))?;
        output.name = output.name.trim().to_string();
        output.pattern = output.pattern.trim().to_string();
        output.steps.retain(|step| !step.trim().is_empty());
        if output.name.is_empty() {
            return None;
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }

    fn complete_pattern(&self, input: CompletePatternInput) -> Option<CompletePatternOutput> {
        let prompt = complete_pattern_prompt(&input);
        let mut output: CompletePatternOutput = self.parse_json(self.generate(&prompt, 180))?;
        output.completed_text = output.completed_text.trim().to_string();
        output.evidence.retain(|item| !item.trim().is_empty());
        if output.completed_text.is_empty() {
            return None;
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_config_defaults() {
        let config = OllamaConfig::default();
        assert_eq!(config.model, DEFAULT_SLM_MODEL);
        assert_eq!(config.base_url, "http://localhost:11434");
    }
}
