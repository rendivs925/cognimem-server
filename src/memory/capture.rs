use super::types::{CaptureEvent, CaptureEventType, MemoryScope, MemoryTier, CognitiveMemoryUnit};
use uuid::Uuid;

pub struct CaptureIngest {
    suppress_patterns: Vec<String>,
}

impl Default for CaptureIngest {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureIngest {
    pub fn new() -> Self {
        Self {
            suppress_patterns: vec![
                "heartbeat".to_string(),
                "ping".to_string(),
                "pong".to_string(),
            ],
        }
    }

    pub fn should_suppress(&self, event: &CaptureEvent) -> bool {
        if let Some(ref tool) = event.tool_name {
            for pattern in &self.suppress_patterns {
                if tool.to_lowercase().contains(&pattern.to_lowercase()) {
                    return true;
                }
            }
        }
        false
    }

    pub fn event_to_memory(
        &self,
        event: &CaptureEvent,
        project_path: Option<String>,
    ) -> Option<CognitiveMemoryUnit> {
        if self.should_suppress(event) {
            return None;
        }

        let content = match event.event_type {
            CaptureEventType::SessionStarted => {
                format!("Session started for project: {}", event.project_path.as_deref().unwrap_or("unknown"))
            }
            CaptureEventType::SessionEnded => {
                "Session ended".to_string()
            }
            CaptureEventType::TurnStarted => {
                "Turn started".to_string()
            }
            CaptureEventType::TurnEnded => {
                event.content.clone().unwrap_or_else(|| "Turn completed".to_string())
            }
            CaptureEventType::ToolStarted => {
                format!("Started tool: {}", event.tool_name.as_deref().unwrap_or("unknown"))
            }
            CaptureEventType::ToolEnded => {
                let status = if event.success.unwrap_or(false) {
                    "succeeded"
                } else {
                    "failed"
                };
                format!("Tool {} {}", event.tool_name.as_deref().unwrap_or("unknown"), status)
            }
            CaptureEventType::TaskCreated => {
                format!("Task created: {}", event.task_name.as_deref().unwrap_or("unknown"))
            }
            CaptureEventType::TaskCompleted => {
                format!("Task completed: {}", event.task_name.as_deref().unwrap_or("unknown"))
            }
            CaptureEventType::SessionIdle => {
                "Session became idle".to_string()
            }
        };

        let scope = if let Some(ref path) = project_path {
            MemoryScope::Project { project_path: path.clone() }
        } else {
            MemoryScope::Global
        };

        Some(CognitiveMemoryUnit {
            id: Uuid::new_v4(),
            tier: MemoryTier::Sensory,
            content,
            metadata: super::types::MemoryMetadata::new(0.3, 2.0),
            associations: Vec::new(),
            scope,
            scope_override: None,
            persona: None,
            raci: super::types::RaciRoles::default(),
            model: super::types::ModelMemoryMetadata::default(),
        })
    }
}

pub fn aggregate_tool_events(events: &[CaptureEvent]) -> Vec<CaptureEvent> {
    let mut aggregated: Vec<CaptureEvent> = Vec::new();
    let mut pending_tool: Option<&CaptureEvent> = None;

    for event in events {
        match event.event_type {
            CaptureEventType::ToolStarted => {
                pending_tool = Some(event);
            }
            CaptureEventType::ToolEnded => {
                if let Some(start) = pending_tool {
                    if start.tool_name == event.tool_name {
                        let status = if event.success.unwrap_or(false) {
                            "completed successfully"
                        } else {
                            "failed"
                        };
                        let mut combined = event.clone();
                        combined.content = Some(format!(
                            "Tool '{}' {}",
                            start.tool_name.as_deref().unwrap_or("unknown"),
                            status
                        ));
                        aggregated.push(combined);
                        pending_tool = None;
                    } else {
                        aggregated.push(event.clone());
                    }
                } else {
                    aggregated.push(event.clone());
                }
            }
            _ => {
                if pending_tool.is_some() {
                    aggregated.push(pending_tool.unwrap().clone());
                    pending_tool = None;
                }
                aggregated.push(event.clone());
            }
        }
    }

    if let Some(start) = pending_tool {
        aggregated.push(start.clone());
    }

    aggregated
}