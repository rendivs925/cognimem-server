use redis::{Client, Commands};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use uuid::Uuid;

pub const DEFAULT_REDIS_URL: &str = "redis://localhost:6379";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrokerEvent {
    ClaimStarted {
        session_id: Uuid,
        memory_id: Uuid,
        claim_type: String,
        agent_id: String,
    },
    ClaimCompleted {
        session_id: Uuid,
        memory_id: Uuid,
        agent_id: String,
    },
    ClaimReleased {
        session_id: Uuid,
        memory_id: Uuid,
        agent_id: String,
    },
    MemoryUpdated {
        memory_id: Uuid,
        action: String,
        agent_id: String,
    },
    ConflictDetected {
        memory_a: Uuid,
        memory_b: Uuid,
        session_id: Uuid,
    },
    SessionJoined {
        session_id: Uuid,
        project_path: String,
        agent_id: String,
    },
    SessionLeft {
        session_id: Uuid,
        agent_id: String,
    },
}

impl BrokerEvent {
    pub fn topic(&self) -> &str {
        match self {
            BrokerEvent::ClaimStarted { .. } => "claim:started",
            BrokerEvent::ClaimCompleted { .. } => "claim:completed",
            BrokerEvent::ClaimReleased { .. } => "claim:released",
            BrokerEvent::MemoryUpdated { .. } => "memory:updated",
            BrokerEvent::ConflictDetected { .. } => "conflict:detected",
            BrokerEvent::SessionJoined { .. } => "session:joined",
            BrokerEvent::SessionLeft { .. } => "session:left",
        }
    }

    pub fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn deserialize(topic: &str, payload: &str) -> Option<Self> {
        match topic {
            "claim:started" => serde_json::from_str(payload).ok(),
            "claim:completed" => serde_json::from_str(payload).ok(),
            "claim:released" => serde_json::from_str(payload).ok(),
            "memory:updated" => serde_json::from_str(payload).ok(),
            "conflict:detected" => serde_json::from_str(payload).ok(),
            "session:joined" => serde_json::from_str(payload).ok(),
            "session:left" => serde_json::from_str(payload).ok(),
            _ => None,
        }
    }
}

pub struct RedisBroker {
    conn: RefCell<Option<redis::Connection>>,
    url: String,
    agent_id: String,
}

impl RedisBroker {
    pub fn new(url: String, agent_id: String) -> Self {
        Self {
            conn: RefCell::new(None),
            url,
            agent_id,
        }
    }

    pub fn connect(&self) -> Result<(), redis::RedisError> {
        let client = Client::open(self.url.as_str())?;
        let conn = client.get_connection()?;
        tracing::info!("Redis broker connected to {}", self.url);
        *self.conn.borrow_mut() = Some(conn);
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        self.conn.borrow().is_some()
    }

    pub fn publish(&self, event: &BrokerEvent) -> Result<(), redis::RedisError> {
        let mut binding = self.conn.borrow_mut();
        let conn = match binding.as_mut() {
            Some(c) => c,
            None => return Ok(()),
        };
        let topic = event.topic();
        let payload = event.serialize();
        let _: () = conn.publish(topic, &payload)?;
        tracing::debug!("Published event to {}: {}", topic, payload);
        Ok(())
    }

    pub fn get_agent_id(&self) -> &str {
        &self.agent_id
    }
}

pub struct SimpleBroker;

impl SimpleBroker {
    pub fn new() -> Self {
        Self
    }

    pub fn publish(&self, _event: &BrokerEvent) -> Result<(), redis::RedisError> {
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        false
    }
}

impl Default for SimpleBroker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broker_event_serialize() {
        let event = BrokerEvent::ClaimStarted {
            session_id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            claim_type: "implementation".to_string(),
            agent_id: "agent-1".to_string(),
        };
        let serialized = event.serialize();
        assert!(serialized.contains("claim:started"));
    }

    #[test]
    fn test_broker_event_deserialize() {
        let json = r#"{"ClaimStarted":{"session_id":"00000000-0000-0000-0000-000000000001","memory_id":"00000000-0000-0000-0000-000000000002","claim_type":"testing","agent_id":"agent-1"}}"#;
        let event = BrokerEvent::deserialize("claim:started", json);
        assert!(event.is_some());
    }
}