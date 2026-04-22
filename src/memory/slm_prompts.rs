use super::slm::DEFAULT_SLM_MODEL;
use super::slm_types::{
    ClassifyMemoryInput, CompletePatternInput, CompressMemoryInput, DistillSkillInput,
    ExtractBestPracticeInput, ExtractPersonaInput, RerankCandidatesInput, ResolveConflictInput,
    SummarizeSessionInput, SummarizeTurnInput,
};

pub fn compress_memory_prompt(input: &CompressMemoryInput) -> String {
    format!(
        "You are compressing memory content using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"summary\":\"string <= 20 words\",\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Content: {}",
        input.content
    )
}

pub fn classify_memory_prompt(input: &ClassifyMemoryInput) -> String {
    format!(
        "You are classifying memory content using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"tier\":\"sensory|working|episodic|semantic|procedural\",\"importance\":0.0,\"suppress\":false,\"tags\":[\"string\"],\"associations\":[{{\"memory_id\":null,\"label\":\"string\",\"strength\":0.0}}],\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Content: {}",
        input.content
    )
}

pub fn rerank_candidates_prompt(input: &RerankCandidatesInput) -> String {
    let candidates = input
        .candidates
        .iter()
        .map(|c| {
            format!(
                "- id={} score={} content={}",
                c.id, c.initial_score, c.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are reranking memory candidates using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"ranked_ids\":[\"uuid\"],\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Top_n: {}. Query: {}. Candidates:\n{}",
        input.top_n, input.query, candidates
    )
}

pub fn resolve_conflict_prompt(input: &ResolveConflictInput) -> String {
    format!(
        "You are resolving a memory conflict using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"kind\":\"duplicate|contradiction|complement|unrelated\",\"action\":\"latest_wins|keep_both|human_decide\",\"merged_summary\":\"string or null\",\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Memory A: {}. Memory B: {}",
        input.memory_a_content, input.memory_b_content
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
        "You are extracting persona signals using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"profiles\":[{{\"domain\":\"biography|experiences|preferences|social|work|psychometrics\",\"summary\":\"string\",\"source_ids\":[\"uuid\"],\"confidence\":0.0}}],\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Memories:\n{}",
        memories
    )
}

pub fn distill_skill_prompt(input: &DistillSkillInput) -> String {
    let examples = input.examples.join("\n");
    format!(
        "You are distilling a procedural skill using model {DEFAULT_SLM_MODEL}. Return only valid JSON. Schema: {{\"name\":\"string\",\"pattern\":\"string\",\"steps\":[\"string\"],\"metadata\":{{\"model\":\"{DEFAULT_SLM_MODEL}\",\"confidence\":0.0}}}}. Examples:\n{}",
        examples
    )
}

pub fn complete_pattern_prompt(input: &CompletePatternInput) -> String {
    let context = input.context.join("\n");
    format!(
        "You are completing a memory pattern using model {}. Return only valid JSON. Schema: {{\"completed_text\":\"string\",\"evidence\":[\"string\"],\"metadata\":{{\"model\":\"{}\",\"confidence\":0.0}}}}. Cue: {}. Context:\n{}",
        DEFAULT_SLM_MODEL, DEFAULT_SLM_MODEL, input.cue, context
    )
}

pub fn summarize_turn_prompt(input: &SummarizeTurnInput) -> String {
    let turns = input
        .turns
        .iter()
        .map(|t| {
            format!(
                "Turn {}: {} [tools: {:?}]",
                t.turn_id, t.content, t.tool_usage
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are summarizing AI turns using model {}. Return only valid JSON. Schema: {{\"summary\":\"string\",\"key_decisions\":[\"string\"],\"key_actions\":[\"string\"],\"metadata\":{{\"model\":\"{}\",\"confidence\":0.0}}}}. Turns:\n{}",
        DEFAULT_SLM_MODEL, DEFAULT_SLM_MODEL, turns
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
        "You are summarizing a coding session using model {}. Return only valid JSON. Schema: {{\"summary\":\"string\",\"completed\":[\"string\"],\"unresolved\":[\"string\"],\"next_steps\":[\"string\"],\"handoff_context\":\"string or null\",\"metadata\":{{\"model\":\"{}\",\"confidence\":0.0}}}}. Turns:\n{}. Completed tasks: {}. Open tasks: {}",
        DEFAULT_SLM_MODEL, DEFAULT_SLM_MODEL, turns, completed, open
    )
}

pub fn extract_best_practice_prompt(input: &ExtractBestPracticeInput) -> String {
    format!(
        "You are extracting coding best practices using model {}. Return only valid JSON. Schema: {{\"practices\":[{{\"principle\":\"DRY|KISS|SOLID|YAGNI|GuardClauses|DesignPatterns\",\"description\":\"string\",\"applies_to\":[\"string\"],\"example\":\"string or null\"}}],\"confidence\":0.0,\"should_persist\":false,\"metadata\":{{\"model\":\"{}\",\"confidence\":0.0}}}}. Content: {}. Context: {}",
        DEFAULT_SLM_MODEL,
        DEFAULT_SLM_MODEL,
        input.content,
        input.context.as_deref().unwrap_or("none")
    )
}
