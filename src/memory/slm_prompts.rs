use super::slm_types::{
    ClassifyMemoryInput, CompletePatternInput, CompressMemoryInput, DistillSkillInput, DreamInput,
    ExtractBestPracticeInput, ExtractPersonaInput, RerankCandidatesInput, ResolveConflictInput,
    ScoreRelevanceInput, SummarizeSessionInput, SummarizeTurnInput, TagEmotionInput,
};

const JSON_FUNCTION_HEADER: &str = r#"You are a JSON function.
Output EXACTLY one JSON object. No markdown. No explanation. No <think> tags.
If unsure, return a valid object with conservative defaults.
Schema:"#;

const OUTPUT_RULES: &str = r#"
Rules:
- Every required key must be present.
- Arrays must always be arrays, never objects.
- UUID fields must be valid UUIDs copied exactly from input.
- Do NOT invent new UUIDs.
- Confidence must be a number 0-1.
- If no items, return [].
- Strings must be plain text."#;

pub fn compress_memory_prompt(input: &CompressMemoryInput) -> String {
    format!(
        "{} {{\"summary\":\"string <= 20 words\",\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Content: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.content
    )
}

pub fn classify_memory_prompt(input: &ClassifyMemoryInput) -> String {
    format!(
        "{} {{\"tier\":\"sensory|working|episodic|semantic|procedural\",\"importance\":0.0,\"suppress\":false,\"tags\":[\"string\"],\"associations\":[{{\"memory_id\":null,\"label\":\"string\",\"strength\":0.0}}],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Content: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.content
    )
}

pub fn rerank_candidates_prompt(input: &RerankCandidatesInput) -> String {
    let candidates = input
        .candidates
        .iter()
        .map(|c| format!("- id={} score={} content={}", c.id, c.initial_score, c.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{} {{\"ranked_ids\":[\"uuid\"],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Top_n: {}. Query: {}. Candidates:\n{}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.top_n, input.query, candidates
    )
}

pub fn resolve_conflict_prompt(input: &ResolveConflictInput) -> String {
    format!(
        "{} {{\"kind\":\"duplicate|contradiction|complement|unrelated\",\"action\":\"latest_wins|keep_both|human_decide\",\"merged_summary\":null,\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Memory A: {}. Memory B: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.memory_a_content, input.memory_b_content
    )
}

pub fn extract_persona_prompt(input: &ExtractPersonaInput) -> String {
    let memories = input
        .memories
        .iter()
        .map(|m| format!("- id={} content={}", m.id, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{} {{\"profiles\":[{{\"domain\":\"biography|experiences|preferences|social|work|psychometrics\",\"summary\":\"string\",\"source_ids\":[\"uuid\"],\"confidence\":0.0}}],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Memories:\n{}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, memories
    )
}

pub fn distill_skill_prompt(input: &DistillSkillInput) -> String {
    let examples = input.examples.join("\n");
    format!(
        "{} {{\"name\":\"string\",\"pattern\":\"string\",\"steps\":[\"string\"],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Examples:\n{}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, examples
    )
}

pub fn complete_pattern_prompt(input: &CompletePatternInput) -> String {
    let context = input.context.join("\n");
    format!(
        "{} {{\"completed_text\":\"string\",\"evidence\":[\"string\"],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Cue: {}. Context:\n{}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.cue, context
    )
}

pub fn summarize_turn_prompt(input: &SummarizeTurnInput) -> String {
    let turns = input
        .turns
        .iter()
        .map(|t| format!("Turn {}: {} [tools: {:?}]", t.turn_id, t.content, t.tool_usage))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{} {{\"summary\":\"string\",\"key_decisions\":[\"string\"],\"key_actions\":[\"string\"],\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Turns:\n{}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, turns
    )
}

pub fn summarize_session_prompt(input: &SummarizeSessionInput) -> String {
    let turns = input
        .turns
        .iter()
        .map(|t| format!("Turn {}: {}", t.turn_id, t.content))
        .collect::<Vec<_>>()
        .join("\n");
    let completed = input
        .completed_tasks
        .iter()
        .map(|t| t.title.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let open = input
        .open_tasks
        .iter()
        .map(|t| t.title.clone())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{} {{\"summary\":\"string\",\"completed\":[\"string\"],\"unresolved\":[\"string\"],\"next_steps\":[\"string\"],\"handoff_context\":null,\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Turns:\n{}. Completed: {}. Open: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, turns, completed, open
    )
}

pub fn extract_best_practice_prompt(input: &ExtractBestPracticeInput) -> String {
    format!(
        "{} {{\"practices\":[{{\"principle\":\"DRY|KISS|SOLID|YAGNI|GuardClauses|DesignPatterns\",\"description\":\"string\",\"applies_to\":[\"string\"],\"example\":null}}],\"confidence\":0.0,\"should_persist\":false,\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Content: {}. Context: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.content, input.context.as_deref().unwrap_or("none")
    )
}

pub fn score_relevance_prompt(input: &ScoreRelevanceInput) -> String {
    format!(
        "{} {{\"relevance\":0.0,\"reasoning\":null,\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Score relevance 0-1 of candidate to query. Query: {}. Candidate: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.query, input.candidate_content
    )
}

pub fn dream_prompt(input: &DreamInput) -> String {
    format!(
        "{} {{\"dreamt_content\":\"string\",\"insight\":\"string\",\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Creatively merge these two unrelated memories into one novel dream-like memory. Memory A: {}. Memory B: {}. Extract a surprising insight from the combination.",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.memory_a, input.memory_b
    )
}

pub fn tag_emotion_prompt(input: &TagEmotionInput) -> String {
    format!(
        "{} {{\"valence\":0.0,\"arousal\":0.0,\"metadata\":{{\"model\":\"\",\"confidence\":0.0}}}} {} Analyze the emotional tone. valence: -1.0 to 1.0 (negative to positive). arousal: 0.0 to 1.0 (calm to excited). Content: {}",
        JSON_FUNCTION_HEADER, OUTPUT_RULES, input.content
    )
}
