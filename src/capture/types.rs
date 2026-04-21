use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    Opencode,
    ClaudeCode,
    Codex,
    Manual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalEventType {
    SessionCreated,
    SessionDeleted,
    SessionIdle,
    ToolExecuteBefore,
    ToolExecuteAfter,
    ToolExecuteFailure,
    FileEdited,
    FileCreated,
    FileDeleted,
    MessageUpdated,
    UserPromptSubmitted,
    PermissionAsked,
    TaskCreated,
    TaskCompleted,
    SubagentStarted,
    SubagentStopped,
    CwdChanged,
    InstructionsLoaded,
    ConfigChanged,
    WorktreeCreated,
    WorktreeRemoved,
    PreCompact,
    PostCompact,
    Elicitation,
    ElicitationResult,
    Stop,
    StopFailure,
    Notification,
    TeammateIdle,
}

impl CanonicalEventType {
    pub fn is_tool_before(&self) -> bool {
        matches!(self, Self::ToolExecuteBefore)
    }

    pub fn is_tool_after(&self) -> bool {
        matches!(self, Self::ToolExecuteAfter | Self::ToolExecuteFailure)
    }

    pub fn is_noisy(&self) -> bool {
        matches!(
            self,
            Self::SessionIdle | Self::Notification | Self::TeammateIdle | Self::PreCompact
        )
    }

    pub fn requires_content(&self) -> bool {
        matches!(
            self,
            Self::ToolExecuteAfter
                | Self::ToolExecuteFailure
                | Self::FileEdited
                | Self::FileCreated
                | Self::FileDeleted
                | Self::MessageUpdated
                | Self::UserPromptSubmitted
                | Self::TaskCreated
                | Self::TaskCompleted
                | Self::Elicitation
                | Self::ElicitationResult
                | Self::InstructionsLoaded
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::SessionCreated => "session created",
            Self::SessionDeleted => "session deleted",
            Self::SessionIdle => "session idle",
            Self::ToolExecuteBefore => "tool execute before",
            Self::ToolExecuteAfter => "tool execute after",
            Self::ToolExecuteFailure => "tool execute failure",
            Self::FileEdited => "file edited",
            Self::FileCreated => "file created",
            Self::FileDeleted => "file deleted",
            Self::MessageUpdated => "message updated",
            Self::UserPromptSubmitted => "user prompt submitted",
            Self::PermissionAsked => "permission asked",
            Self::TaskCreated => "task created",
            Self::TaskCompleted => "task completed",
            Self::SubagentStarted => "subagent started",
            Self::SubagentStopped => "subagent stopped",
            Self::CwdChanged => "cwd changed",
            Self::InstructionsLoaded => "instructions loaded",
            Self::ConfigChanged => "config changed",
            Self::WorktreeCreated => "worktree created",
            Self::WorktreeRemoved => "worktree removed",
            Self::PreCompact => "pre compact",
            Self::PostCompact => "post compact",
            Self::Elicitation => "elicitation",
            Self::ElicitationResult => "elicitation result",
            Self::Stop => "stop",
            Self::StopFailure => "stop failure",
            Self::Notification => "notification",
            Self::TeammateIdle => "teammate idle",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalEvent {
    pub event_type: CanonicalEventType,
    pub timestamp: i64,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub project_path: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    pub source: EventSource,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_input: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_output: Option<serde_json::Value>,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub success: Option<bool>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CanonicalEvent {
    pub fn compose_content(&self) -> String {
        let mut parts = Vec::new();
        parts.push(self.event_type.label().to_string());

        if let Some(ref tool) = self.tool_name {
            parts.push(format!("tool={}", tool));
        }
        if let Some(ref path) = self.file_path {
            parts.push(format!("file={}", path));
        }
        if let Some(ref content) = self.content {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
        if let Some(success) = self.success {
            parts.push(if success {
                "succeeded".to_string()
            } else {
                "failed".to_string()
            });
        }
        if let Some(dur) = self.duration_ms {
            parts.push(format!("duration={}ms", dur));
        }

        parts.join(" | ")
    }

    pub fn aggregation_key(&self) -> Option<(String, String)> {
        if self.event_type.is_tool_before() || self.event_type.is_tool_after() {
            let session = self.session_id.clone().unwrap_or_default();
            let tool = self.tool_name.clone().unwrap_or_default();
            Some((session, tool))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub accepted: usize,
    pub suppressed: usize,
    pub stored: usize,
    pub errors: Vec<String>,
}

impl Default for IngestResult {
    fn default() -> Self {
        Self {
            accepted: 0,
            suppressed: 0,
            stored: 0,
            errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStats {
    pub total_accepted: u64,
    pub total_suppressed: u64,
    pub total_stored: u64,
    pub total_errors: u64,
    pub pending_aggregations: usize,
    pub uptime_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureError {
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_serde_roundtrip() {
        for variant in [
            CanonicalEventType::SessionCreated,
            CanonicalEventType::ToolExecuteAfter,
            CanonicalEventType::FileEdited,
            CanonicalEventType::TeammateIdle,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let parsed: CanonicalEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn test_source_serde_roundtrip() {
        for variant in [
            EventSource::Opencode,
            EventSource::ClaudeCode,
            EventSource::Codex,
            EventSource::Manual,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let parsed: EventSource = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn test_compose_content() {
        let event = CanonicalEvent {
            event_type: CanonicalEventType::ToolExecuteAfter,
            timestamp: 1000,
            session_id: Some("s1".into()),
            project_path: None,
            agent_id: None,
            source: EventSource::Opencode,
            tool_name: Some("bash".into()),
            tool_input: None,
            tool_output: None,
            file_path: None,
            content: Some("ran tests".into()),
            success: Some(true),
            duration_ms: Some(150),
            metadata: HashMap::new(),
        };
        let content = event.compose_content();
        assert!(content.contains("tool execute after"));
        assert!(content.contains("tool=bash"));
        assert!(content.contains("ran tests"));
        assert!(content.contains("succeeded"));
        assert!(content.contains("duration=150ms"));
    }

    #[test]
    fn test_aggregation_key() {
        let before = CanonicalEvent {
            event_type: CanonicalEventType::ToolExecuteBefore,
            timestamp: 0,
            session_id: Some("s1".into()),
            project_path: None,
            agent_id: None,
            source: EventSource::Opencode,
            tool_name: Some("bash".into()),
            tool_input: None,
            tool_output: None,
            file_path: None,
            content: None,
            success: None,
            duration_ms: None,
            metadata: HashMap::new(),
        };
        assert_eq!(before.aggregation_key(), Some(("s1".into(), "bash".into())));

        let non_tool = CanonicalEvent {
            event_type: CanonicalEventType::FileEdited,
            timestamp: 0,
            session_id: Some("s1".into()),
            project_path: None,
            agent_id: None,
            source: EventSource::Opencode,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            file_path: None,
            content: None,
            success: None,
            duration_ms: None,
            metadata: HashMap::new(),
        };
        assert_eq!(non_tool.aggregation_key(), None);
    }

    #[test]
    fn test_is_noisy() {
        assert!(CanonicalEventType::SessionIdle.is_noisy());
        assert!(CanonicalEventType::Notification.is_noisy());
        assert!(!CanonicalEventType::ToolExecuteAfter.is_noisy());
        assert!(!CanonicalEventType::FileEdited.is_noisy());
    }

    #[test]
    fn test_requires_content() {
        assert!(CanonicalEventType::ToolExecuteAfter.requires_content());
        assert!(CanonicalEventType::FileEdited.requires_content());
        assert!(!CanonicalEventType::SessionCreated.requires_content());
        assert!(!CanonicalEventType::CwdChanged.requires_content());
    }

    #[test]
    fn test_full_event_serde_roundtrip() {
        let event = CanonicalEvent {
            event_type: CanonicalEventType::FileEdited,
            timestamp: 1700000000,
            session_id: Some("abc".into()),
            project_path: Some("/home/user/project".into()),
            agent_id: None,
            source: EventSource::ClaudeCode,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            file_path: Some("src/main.rs".into()),
            content: Some("added new function".into()),
            success: None,
            duration_ms: None,
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CanonicalEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.event_type, event.event_type);
        assert_eq!(parsed.source, event.source);
        assert_eq!(parsed.session_id, event.session_id);
        assert_eq!(parsed.file_path, event.file_path);
    }

    #[test]
    fn test_ingest_result_default() {
        let result = IngestResult::default();
        assert_eq!(result.accepted, 0);
        assert_eq!(result.suppressed, 0);
        assert_eq!(result.stored, 0);
        assert!(result.errors.is_empty());
    }
}
