use crate::memory::MemoryTier;
use rusqlite::{Connection, params};
use tracing::warn;
use uuid::Uuid;

/// A full-text search engine interface for indexing and querying memory content.
pub trait SearchEngine: Send {
    /// Indexes a memory for search. Replaces any existing entry with the same ID.
    fn index(&mut self, id: Uuid, content: &str, tier: MemoryTier);
    /// Removes a memory from the search index.
    fn remove(&mut self, id: &Uuid);
    /// Searches indexed memories matching the query. Optionally filters by tier.
    /// Returns up to `limit` matching memory IDs, ranked by relevance.
    fn search(&self, query: &str, tier: Option<MemoryTier>, limit: usize) -> Vec<Uuid>;
}

/// An FTS5-based search engine using an in-memory SQLite database.
///
/// Supports prefix-matching queries with optional tier filtering and rank ordering.
pub struct Fts5Search {
    conn: Connection,
}

impl Fts5Search {
    /// Creates a new FTS5 search engine with an in-memory SQLite database.
    pub fn new() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_index USING fts5(id, content, tier, tokenize='unicode61');"
        )?;
        Ok(Self { conn })
    }

    fn escape_query(&self, query: &str) -> String {
        let words: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| {
                let cleaned: String = w
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                cleaned
            })
            .filter(|w| !w.is_empty())
            .map(|w| format!("{w}*"))
            .collect();

        if words.is_empty() {
            String::new()
        } else {
            words.join(" OR ")
        }
    }
}

impl SearchEngine for Fts5Search {
    fn index(&mut self, id: Uuid, content: &str, tier: MemoryTier) {
        let tier_str: &str = tier.into();
        if let Err(e) = self.conn.execute(
            "INSERT OR REPLACE INTO memory_index (id, content, tier) VALUES (?1, ?2, ?3)",
            params![id.to_string(), content, tier_str],
        ) {
            warn!("Failed to index memory {id} in FTS5: {e}");
        }
    }

    fn remove(&mut self, id: &Uuid) {
        if let Err(e) = self.conn.execute(
            "DELETE FROM memory_index WHERE id = ?1",
            params![id.to_string()],
        ) {
            warn!("Failed to remove memory {id} from FTS5 index: {e}");
        }
    }

    fn search(&self, query: &str, tier: Option<MemoryTier>, limit: usize) -> Vec<Uuid> {
        let fts_query = self.escape_query(query);

        let result = match tier {
            Some(t) => {
                let tier_str: &str = t.into();
                self.conn
                    .prepare("SELECT id FROM memory_index WHERE memory_index MATCH ?1 AND tier = ?2 ORDER BY rank LIMIT ?3")
                    .and_then(|mut stmt| {
                        let rows = stmt.query_map(params![fts_query, tier_str, limit], |row| row.get::<_, String>(0))?;
                        Ok(rows.flatten().filter_map(|s| Uuid::parse_str(&s).ok()).collect())
                    })
            }
            None => {
                self.conn
                    .prepare("SELECT id FROM memory_index WHERE memory_index MATCH ?1 ORDER BY rank LIMIT ?2")
                    .and_then(|mut stmt| {
                        let rows = stmt.query_map(params![fts_query, limit], |row| row.get::<_, String>(0))?;
                        Ok(rows.flatten().filter_map(|s| Uuid::parse_str(&s).ok()).collect())
                    })
            }
        };

        result.unwrap_or_default()
    }
}

/// A no-op search engine that indexes nothing and returns empty results.
///
/// Useful as a fallback when FTS5 is unavailable.
pub struct SubstringSearch;

impl SearchEngine for SubstringSearch {
    fn index(&mut self, _id: Uuid, _content: &str, _tier: MemoryTier) {}

    fn remove(&mut self, _id: &Uuid) {}

    fn search(&self, _query: &str, _tier: Option<MemoryTier>, _limit: usize) -> Vec<Uuid> {
        Vec::new()
    }
}

/// Returns `true` if `content` contains the query string (case-insensitive),
/// or if any whitespace-delimited query word appears in the content.
pub fn matches_query(content: &str, query: &str) -> bool {
    let lower = content.to_lowercase();
    lower.contains(query) || query.split_whitespace().any(|w| lower.contains(w))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fts() -> Fts5Search {
        Fts5Search::new().unwrap()
    }

    #[test]
    fn fts5_index_and_search() {
        let mut engine = make_fts();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        engine.index(
            id1,
            "The quick brown fox jumps over lazy dog",
            MemoryTier::Episodic,
        );
        engine.index(
            id2,
            "A brown bear walked through the forest",
            MemoryTier::Semantic,
        );
        engine.index(
            id3,
            "The fox and the hound are friends",
            MemoryTier::Episodic,
        );

        let results = engine.search("fox", None, 10);
        assert!(results.contains(&id1));
        assert!(results.contains(&id3));
        assert!(!results.contains(&id2));
    }

    #[test]
    fn fts5_search_with_tier_filter() {
        let mut engine = make_fts();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        engine.index(id1, "fox in the forest", MemoryTier::Episodic);
        engine.index(id2, "fox news channel", MemoryTier::Semantic);

        let results = engine.search("fox", Some(MemoryTier::Episodic), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id1);
    }

    #[test]
    fn fts5_search_no_results() {
        let mut engine = make_fts();
        engine.index(Uuid::new_v4(), "hello world", MemoryTier::Episodic);

        let results = engine.search("xyz_nonexistent", None, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn fts5_remove_from_index() {
        let mut engine = make_fts();
        let id = Uuid::new_v4();
        engine.index(id, "removable content", MemoryTier::Working);

        let results = engine.search("removable", None, 10);
        assert_eq!(results.len(), 1);

        engine.remove(&id);
        let results = engine.search("removable", None, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn fts5_rank_ordering() {
        let mut engine = make_fts();
        let id_high = Uuid::new_v4();
        let id_low = Uuid::new_v4();

        engine.index(
            id_low,
            "rust programming occasionally",
            MemoryTier::Episodic,
        );
        engine.index(id_high, "rust rust rust programming", MemoryTier::Episodic);

        let results = engine.search("rust", None, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], id_high);
    }

    #[test]
    fn fts5_multi_word_query() {
        let mut engine = make_fts();
        let id = Uuid::new_v4();
        engine.index(id, "the quick brown fox", MemoryTier::Episodic);

        let results = engine.search("quick fox", None, 10);
        assert!(results.contains(&id));
    }

    #[test]
    fn substring_query_match() {
        assert!(matches_query("Hello World Rust Programming", "rust"));
        assert!(matches_query("Hello World Rust Programming", "world"));
        assert!(matches_query(
            "Hello World Rust Programming",
            "rust program"
        ));
    }

    #[test]
    fn substring_query_no_match() {
        assert!(!matches_query("Hello World", "python"));
    }
}
