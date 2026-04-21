use super::slm::{SlmEngine, SlmError};
use super::slm_prompts::{
    classify_memory_prompt, complete_pattern_prompt, compress_memory_prompt,
    distill_skill_prompt, extract_best_practice_prompt, extract_persona_prompt,
    rerank_candidates_prompt, resolve_conflict_prompt, summarize_session_prompt,
    summarize_turn_prompt,
};
use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, DistillSkillInput, DistillSkillOutput,
    ExtractBestPracticeInput, ExtractBestPracticeOutput, ExtractPersonaInput, ExtractPersonaOutput,
    RerankCandidatesInput, RerankCandidatesOutput, ResolveConflictInput, ResolveConflictOutput,
    SlmMetadata, SummarizeSessionInput, SummarizeSessionOutput, SummarizeTurnInput,
    SummarizeTurnOutput,
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

    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, SlmError> {
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
            Ok(resp) => resp
                .json::<OllamaResponse>()
                .map(|r| r.response)
                .map_err(|e| SlmError::InvalidResponse(e.to_string())),
            Err(e) => {
                tracing::warn!("Ollama request failed: {e}");
                Err(SlmError::RequestFailed(e.to_string()))
            }
        }
    }

    fn parse_json<T: serde::de::DeserializeOwned>(
        &self,
        response: Result<String, SlmError>,
    ) -> Result<T, SlmError> {
        let response = response?;
        serde_json::from_str(&response).map_err(|e| SlmError::InvalidResponse(e.to_string()))
    }

    fn clamp_confidence(confidence: f32) -> f32 {
        confidence.clamp(0.0, 1.0)
    }
}

impl SlmEngine for OllamaSlm {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn compress_memory(&self, input: CompressMemoryInput) -> Result<CompressMemoryOutput, SlmError> {
        let prompt = compress_memory_prompt(&input);
        let mut output: CompressMemoryOutput = self.parse_json(self.generate(&prompt, 80))?;
        output.summary = output.summary.trim().to_string();
        if output.summary.is_empty() {
            return Err(SlmError::ValidationFailed(
                "compression summary was empty".to_string(),
            ));
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn classify_memory(&self, input: ClassifyMemoryInput) -> Result<ClassifyMemoryOutput, SlmError> {
        let prompt = classify_memory_prompt(&input);
        let mut output: ClassifyMemoryOutput = self.parse_json(self.generate(&prompt, 160))?;

        output.importance = output.importance.clamp(0.0, 1.0);
        output.tags = output
            .tags
            .into_iter()
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect();
        output.tags.sort();
        output.tags.dedup();
        if output.tags.len() > 20 {
            return Err(SlmError::ValidationFailed(
                "classification returned more than 20 tags".to_string(),
            ));
        }

        let mut seen_association_ids = std::collections::HashSet::new();
        let mut seen_association_labels = std::collections::HashSet::new();
        for assoc in &mut output.associations {
            assoc.label = assoc.label.trim().to_string();
            assoc.strength = assoc.strength.clamp(0.0, 1.0);
        }
        output.associations.retain(|assoc| {
            if assoc.label.is_empty() {
                return false;
            }

            let label_key = assoc.label.to_lowercase();
            if !seen_association_labels.insert(label_key) {
                return false;
            }

            if let Some(id) = assoc.memory_id {
                return seen_association_ids.insert(id);
            }

            true
        });
        if output.associations.len() > 20 {
            return Err(SlmError::ValidationFailed(
                "classification returned more than 20 associations".to_string(),
            ));
        }

        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn rerank_candidates(
        &self,
        input: RerankCandidatesInput,
    ) -> Result<RerankCandidatesOutput, SlmError> {
        if input.candidates.is_empty() {
            return Ok(RerankCandidatesOutput {
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
        Ok(output)
    }

    fn resolve_conflict(
        &self,
        input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError> {
        let prompt = resolve_conflict_prompt(&input);
        let mut output: ResolveConflictOutput = self.parse_json(self.generate(&prompt, 120))?;
        output.merged_summary = output.merged_summary.map(|s| s.trim().to_string());
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn extract_persona(
        &self,
        input: ExtractPersonaInput,
    ) -> Result<ExtractPersonaOutput, SlmError> {
        if input.memories.is_empty() {
            return Ok(ExtractPersonaOutput {
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
        Ok(output)
    }

    fn distill_skill(&self, input: DistillSkillInput) -> Result<DistillSkillOutput, SlmError> {
        let prompt = distill_skill_prompt(&input);
        let mut output: DistillSkillOutput = self.parse_json(self.generate(&prompt, 220))?;
        output.name = output.name.trim().to_string();
        output.pattern = output.pattern.trim().to_string();
        output.steps.retain(|step| !step.trim().is_empty());
        if output.name.is_empty() {
            return Err(SlmError::ValidationFailed(
                "skill name was empty".to_string(),
            ));
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError> {
        let prompt = complete_pattern_prompt(&input);
        let mut output: CompletePatternOutput = self.parse_json(self.generate(&prompt, 180))?;
        output.completed_text = output.completed_text.trim().to_string();
        output.evidence.retain(|item| !item.trim().is_empty());
        if output.completed_text.is_empty() {
            return Err(SlmError::ValidationFailed(
                "completed pattern text was empty".to_string(),
            ));
        }
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn summarize_turn(&self, input: SummarizeTurnInput) -> Result<SummarizeTurnOutput, SlmError> {
        let prompt = summarize_turn_prompt(&input);
        let mut output: SummarizeTurnOutput = self.parse_json(self.generate(&prompt, 180))?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn summarize_session(&self, input: SummarizeSessionInput) -> Result<SummarizeSessionOutput, SlmError> {
        let prompt = summarize_session_prompt(&input);
        let mut output: SummarizeSessionOutput = self.parse_json(self.generate(&prompt, 240))?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    fn extract_best_practice(&self, input: ExtractBestPracticeInput) -> Result<ExtractBestPracticeOutput, SlmError> {
        let prompt = extract_best_practice_prompt(&input);
        let mut output: ExtractBestPracticeOutput = self.parse_json(self.generate(&prompt, 180))?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
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
