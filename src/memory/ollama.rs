use super::slm::{SlmEngine, SlmError};
use super::slm_prompts::{
    classify_memory_prompt, complete_pattern_prompt, compress_memory_prompt, distill_skill_prompt,
    extract_best_practice_prompt, extract_persona_prompt, rerank_candidates_prompt,
    resolve_conflict_prompt, summarize_session_prompt, summarize_turn_prompt,
};
use super::slm_types::{
    ClassifyMemoryInput, ClassifyMemoryOutput, CompletePatternInput, CompletePatternOutput,
    CompressMemoryInput, CompressMemoryOutput, DelegateInput, DelegateOutput,
    DistillSkillInput, DistillSkillOutput, ExtractBestPracticeInput,
    ExtractBestPracticeOutput, ExtractPersonaInput, ExtractPersonaOutput,
    SimulatePerspectiveInput, SimulatePerspectiveOutput,
    TeachFromDemonstrationInput, TeachFromDemonstrationOutput, RerankCandidatesInput,
    RerankCandidatesOutput, ResolveConflictInput, ResolveConflictOutput,
    SlmMetadata, SummarizeSessionInput, SummarizeSessionOutput, SummarizeTurnInput,
    SummarizeTurnOutput,
};
use crate::memory::DEFAULT_SLM_MODEL;
use async_trait::async_trait;
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
    client: reqwest::Client,
    check_client: reqwest::blocking::Client,
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
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("failed to build async reqwest client"),
            check_client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("failed to build blocking reqwest client"),
            model: config.model,
            base_url: config.base_url,
        }
    }

    pub fn check_available(&self) -> bool {
        match self.check_client.get(&self.base_url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    const MAX_RETRIES: usize = 3;
    const RETRY_BACKOFF_MS: u64 = 500;

    async fn generate_with_retry(&self, prompt: &str, max_tokens: usize) -> Result<String, SlmError> {
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < Self::MAX_RETRIES {
            attempt += 1;
            
            match self.generate(prompt, max_tokens).await {
                Ok(response) => {
                    let cleaned = Self::extract_json(&response);
                    if Self::validate_json(&cleaned) {
                        return Ok(cleaned);
                    }
                    
                    if attempt < Self::MAX_RETRIES {
                        tracing::warn!("Invalid JSON attempt {}, trying repair prompt", attempt);
                        let repair_prompt = Self::repair_prompt(&cleaned, prompt);
                        tokio::time::sleep(std::time::Duration::from_millis(Self::RETRY_BACKOFF_MS)).await;
                        match self.generate(&repair_prompt, max_tokens).await {
                            Ok(repair_response) => {
                                let repaired = Self::extract_json(&repair_response);
                                if Self::validate_json(&repaired) {
                                    return Ok(repaired);
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Repair prompt failed: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    if attempt < 3 {
                        tracing::warn!(" Ollama request attempt {} failed: {}", attempt, e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(500 * attempt as u64)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| 
            SlmError::InvalidResponse(format!("Failed after {} attempts", 3))))
    }

    fn repair_prompt(invalid_json: &str, original_prompt: &str) -> String {
        format!(
            "Fix the following invalid JSON so it matches the required schema.\n\
            Return ONLY the corrected JSON, no explanation.\n\n\
            Invalid JSON:\n{}\n\n\
            Original instruction:\n{}\n\n\
            Corrected JSON:",
            invalid_json, original_prompt
        )
    }

    fn validate_json(json_str: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(json_str).is_ok()
    }

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, SlmError> {
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
        match self.client.post(&url).json(&request).send().await {
            Ok(resp) => resp
                .json::<OllamaResponse>()
                .await
                .map(|r| r.response)
                .map_err(|e| SlmError::InvalidResponse(e.to_string())),
            Err(e) => {
                tracing::warn!("Ollama request failed: {e}");
                Err(SlmError::RequestFailed(e.to_string()))
            }
        }
    }

    fn extract_json(raw: &str) -> String {
        let mut s = raw.to_string();
        loop {
            if let Some(start) = s.find("<think>") {
                let _tag_len = 7;
                if let Some(end) = s.find("</think>") {
                    s.replace_range(start..end + 8, "");
                    continue;
                } else {
                    s.replace_range(start.., "");
                    break;
                }
            }
            break;
        }
        if let Some(start) = s.find("```json") {
            let after = start + 7;
            if let Some(end) = s[after..].find("```") {
                return s[after..after + end].trim().to_string();
            }
        }
        if let Some(start) = s.find("```") {
            let after = start + 3;
            if let Some(end) = s[after..].find("```") {
                let block = s[after..after + end].trim().to_string();
                if block.starts_with('{') || block.starts_with('[') {
                    return block;
                }
            }
        }
        if let Some(first) = s.find('{') {
            if let Some(last) = s.rfind('}') {
                if last > first {
                    return s[first..=last].to_string();
                }
            }
        }
        if let Some(first) = s.find('[') {
            if let Some(last) = s.rfind(']') {
                if last > first {
                    return s[first..=last].to_string();
                }
            }
        }
        s
    }

    fn normalize_array_fields(json_str: &str) -> String {
        let mut val: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => return json_str.to_string(),
        };
        if let serde_json::Value::Object(ref mut map) = val {
            let array_fields = [
                "ranked_ids",
                "tags",
                "steps",
                "evidence",
                "source_ids",
                "associations",
                "practices",
                "applies_to",
                "key_decisions",
                "key_actions",
                "completed",
                "unresolved",
                "next_steps",
                "consulted",
                "informed",
            ];
            for field in array_fields {
                if let Some(serde_json::Value::Object(obj)) = map.get(field) {
                    let items: Vec<serde_json::Value> = obj
                        .keys()
                        .filter_map(|k| k.parse::<usize>().ok())
                        .filter_map(|i| obj.get(&i.to_string()).cloned())
                        .collect();
                    if !items.is_empty() {
                        map.insert(field.to_string(), serde_json::Value::Array(items));
                    }
                }
            }
        }
        serde_json::to_string(&val).unwrap_or_else(|_| json_str.to_string())
    }

    async fn parse_json<T: serde::de::DeserializeOwned>(
        &self,
        response: Result<String, SlmError>,
    ) -> Result<T, SlmError> {
        let response = response?;
        let cleaned = Self::extract_json(&response);
        let normalized = Self::normalize_array_fields(&cleaned);
        serde_json::from_str(&normalized).map_err(|e| {
            tracing::warn!(
                "Ollama JSON parse failed: {}. Raw (first 300): {:?}",
                e,
                &response[..response.len().min(300)]
            );
            SlmError::InvalidResponse(e.to_string())
        })
    }

    fn clamp_confidence(confidence: f32) -> f32 {
        confidence.clamp(0.0, 1.0)
    }
}

#[async_trait]
impl SlmEngine for OllamaSlm {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn compress_memory(
        &self,
        input: CompressMemoryInput,
    ) -> Result<CompressMemoryOutput, SlmError> {
        let prompt = compress_memory_prompt(&input);
        let output: CompressMemoryOutput =
            self.parse_json(self.generate_with_retry(&prompt, 80).await).await?;
        if output.summary.is_empty() {
            return Err(SlmError::ValidationFailed(
                "compression summary was empty".to_string(),
            ));
        }
        Ok(CompressMemoryOutput {
            summary: output.summary.trim().to_string(),
            metadata: SlmMetadata {
                model: self.model.clone(),
                confidence: output.metadata.confidence.clamp(0.0, 1.0),
            },
        })
    }

    async fn classify_memory(
        &self,
        input: ClassifyMemoryInput,
    ) -> Result<ClassifyMemoryOutput, SlmError> {
        let prompt = classify_memory_prompt(&input);
        let mut output: ClassifyMemoryOutput =
            self.parse_json(self.generate_with_retry(&prompt, 160).await).await?;

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

        for assoc in &mut output.associations {
            assoc.label = assoc.label.trim().to_string();
            assoc.strength = assoc.strength.clamp(0.0, 1.0);
        }
        let mut seen_labels = std::collections::HashSet::new();
        let mut seen_ids = std::collections::HashSet::new();
        output.associations.retain(|assoc| {
            if assoc.label.is_empty() {
                return false;
            }
            let key = assoc.label.to_lowercase();
            if !seen_labels.insert(key) {
                return false;
            }
            if let Some(id) = assoc.memory_id {
                seen_ids.insert(id)
            } else {
                true
            }
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

    async fn rerank_candidates(
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
        let mut output: RerankCandidatesOutput =
            self.parse_json(self.generate_with_retry(&prompt, 160).await).await?;
        output.ranked_ids.truncate(input.top_n);
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn resolve_conflict(
        &self,
        input: ResolveConflictInput,
    ) -> Result<ResolveConflictOutput, SlmError> {
        let prompt = resolve_conflict_prompt(&input);
        let mut output: ResolveConflictOutput =
            self.parse_json(self.generate_with_retry(&prompt, 120).await).await?;
        output.merged_summary = output.merged_summary.map(|s| s.trim().to_string());
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn extract_persona(
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
        let mut output: ExtractPersonaOutput =
            self.parse_json(self.generate_with_retry(&prompt, 500).await).await?;
        for profile in &mut output.profiles {
            profile.summary = profile.summary.trim().to_string();
            profile.confidence = profile.confidence.clamp(0.0, 1.0);
        }
        output
            .profiles
            .retain(|profile| !profile.summary.is_empty());
        output.profiles.truncate(6);
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn distill_skill(
        &self,
        input: DistillSkillInput,
    ) -> Result<DistillSkillOutput, SlmError> {
        let prompt = distill_skill_prompt(&input);
        let mut output: DistillSkillOutput =
            self.parse_json(self.generate_with_retry(&prompt, 220).await).await?;
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

    async fn complete_pattern(
        &self,
        input: CompletePatternInput,
    ) -> Result<CompletePatternOutput, SlmError> {
        let prompt = complete_pattern_prompt(&input);
        let mut output: CompletePatternOutput =
            self.parse_json(self.generate_with_retry(&prompt, 180).await).await?;
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

    async fn summarize_turn(
        &self,
        input: SummarizeTurnInput,
    ) -> Result<SummarizeTurnOutput, SlmError> {
        let prompt = summarize_turn_prompt(&input);
        let mut output: SummarizeTurnOutput =
            self.parse_json(self.generate_with_retry(&prompt, 180).await).await?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn summarize_session(
        &self,
        input: SummarizeSessionInput,
    ) -> Result<SummarizeSessionOutput, SlmError> {
        let prompt = summarize_session_prompt(&input);
        let mut output: SummarizeSessionOutput =
            self.parse_json(self.generate_with_retry(&prompt, 240).await).await?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn extract_best_practice(
        &self,
        input: ExtractBestPracticeInput,
    ) -> Result<ExtractBestPracticeOutput, SlmError> {
        let prompt = extract_best_practice_prompt(&input);
        let mut output: ExtractBestPracticeOutput =
            self.parse_json(self.generate_with_retry(&prompt, 180).await).await?;
        output.metadata.model = self.model.clone();
        output.metadata.confidence = Self::clamp_confidence(output.metadata.confidence);
        Ok(output)
    }

    async fn delegate_to_llm(
        &self,
        input: DelegateInput,
    ) -> Result<DelegateOutput, SlmError> {
        let context = input.context.join("\n- ");
        let prompt = format!(
            "Query: {}\nContext:\n- {}\n\nProvide a detailed response to the query above. Include your confidence level (0-1) in the response.",
            input.query, context
        );
        let response = self.generate_with_retry(&prompt, 300).await?;
        Ok(DelegateOutput {
            response: response.clone(),
            delegated: input.confidence_threshold < 0.85,
            confidence: 0.7,
            model_used: self.model.clone(),
            reasoning: Some("Response from SLM".to_string()),
        })
    }

    async fn teach_from_demonstration(
        &self,
        input: TeachFromDemonstrationInput,
    ) -> Result<TeachFromDemonstrationOutput, SlmError> {
        let prompt = format!(
            "Extract the core pattern from this demonstration:\n\nDemonstration: {}\nExtracted pattern: {}\n\nAnalyze if this pattern should be promoted to a procedural skill.",
            input.demonstration, input.pattern_extracted
        );
        let _ = self.generate_with_retry(&prompt, 150).await?;
        Ok(TeachFromDemonstrationOutput {
            episodic_memory_id: uuid::Uuid::new_v4(),
            skill_pending: false,
            promotion_candidates: Vec::new(),
            confidence: 0.5,
        })
    }

    async fn simulate_perspective(
        &self,
        input: SimulatePerspectiveInput,
    ) -> Result<SimulatePerspectiveOutput, SlmError> {
        let prompt = format!(
            "Answer from the perspective of a {}.\n\nSituation: {}\nQuestion: {}\n\nProvide reasoning, recommendation, and alternative perspectives.",
            input.perspective_role, input.situation, input.question
        );
        let mut output: SimulatePerspectiveOutput =
            self.parse_json(self.generate_with_retry(&prompt, 250).await).await?;
        output.confidence = Self::clamp_confidence(output.confidence);
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

    #[test]
    fn test_extract_json_pure() {
        let raw = r#"{"summary":"test","metadata":{"model":"qwen2.5-coder:3b","confidence":0.5}}"#;
        assert_eq!(OllamaSlm::extract_json(raw), raw);
    }

    #[test]
    fn test_extract_json_with_think_tags() {
        let raw = "<think>Let me analyze</think>{\"summary\":\"test\",\"metadata\":{\"model\":\"qwen2.5-coder:3b\",\"confidence\":0.5}}";
        let extracted = OllamaSlm::extract_json(raw);
        assert!(extracted.starts_with('{'));
        assert!(extracted.contains("summary"));
    }

    #[test]
    fn test_extract_json_with_markdown_fence() {
        let raw = "```json\n{\"summary\":\"test\",\"metadata\":{\"model\":\"qwen2.5-coder:3b\",\"confidence\":0.5}}\n```";
        let extracted = OllamaSlm::extract_json(raw);
        assert!(extracted.starts_with('{'));
        assert!(extracted.contains("summary"));
    }

    #[test]
    fn test_extract_json_with_prefix_text() {
        let raw = "Here is the result:\n{\"summary\":\"test\",\"metadata\":{\"model\":\"qwen2.5-coder:3b\",\"confidence\":0.5}}\nDone.";
        let extracted = OllamaSlm::extract_json(raw);
        assert!(extracted.starts_with('{'));
        assert!(extracted.ends_with('}'));
    }

    #[test]
    fn test_extract_json_nested_braces() {
        let raw = "Some text {\"a\":{\"b\":1},\"c\":2} more text";
        let extracted = OllamaSlm::extract_json(raw);
        assert_eq!(extracted, "{\"a\":{\"b\":1},\"c\":2}");
    }

    #[test]
    fn test_normalize_array_fields_object_as_array() {
        let json = r#"{"ranked_ids":{"0":"aaa","1":"bbb"}}"#;
        let normalized = OllamaSlm::normalize_array_fields(json);
        let parsed: serde_json::Value = serde_json::from_str(&normalized).unwrap();
        assert!(parsed["ranked_ids"].is_array());
    }
}
