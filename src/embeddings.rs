use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Default dimensionality for embedding vectors.
pub const EMBEDDING_DIM: usize = 256;

/// An embedding engine that converts text into dense vector representations.
pub trait EmbeddingEngine: Send {
    /// Produces a normalized embedding vector for the given text.
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// A deterministic, hash-based embedding engine for testing and lightweight use.
///
/// Generates `EMBEDDING_DIM`-dimensional vectors by hashing words and character n-grams
/// into vector positions, then normalizing to unit length.
pub struct HashEmbedding {
    dim: usize,
}

impl HashEmbedding {
    /// Creates a new `HashEmbedding` with the default dimensionality.
    pub fn new() -> Self {
        Self { dim: EMBEDDING_DIM }
    }
}

impl Default for HashEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingEngine for HashEmbedding {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let lower = text.to_lowercase();
        let bigrams = extract_ngrams(&lower, 3);
        let words: Vec<&str> = lower.split_whitespace().collect();

        for word in &words {
            let mut h = DefaultHasher::new();
            word.hash(&mut h);
            let idx = (h.finish() as usize) % self.dim;
            vec[idx] += 1.0;

            if word.len() >= 3 {
                let mut h2 = DefaultHasher::new();
                word[..word.len() / 2].hash(&mut h2);
                let idx2 = (h2.finish() as usize) % self.dim;
                vec[idx2] += 0.5;
            }
        }

        for ngram in &bigrams {
            let mut h = DefaultHasher::new();
            ngram.hash(&mut h);
            let idx = (h.finish() as usize) % self.dim;
            vec[idx] += 0.3;
        }

        normalize(&mut vec);
        vec
    }
}

fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return vec![text.to_string()];
    }
    (0..=chars.len().saturating_sub(n))
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Computes the cosine similarity between two vectors.
///
/// Returns 0.0 if vectors have different lengths, are empty, or have zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Combines full-text search and vector similarity scores using a weighted fusion strategy.
///
/// FTS results contribute `fts_weight * (1 - rank/total)`, while vector results
/// contribute `vec_weight * similarity`. Results are sorted by combined score descending.
pub fn fuse_scores(
    fts_ids: &[uuid::Uuid],
    fts_weight: f32,
    vec_scores: &[(uuid::Uuid, f32)],
    vec_weight: f32,
) -> Vec<(uuid::Uuid, f32)> {
    let mut scores: std::collections::HashMap<uuid::Uuid, f32> = std::collections::HashMap::new();

    let fts_count = fts_ids.len().max(1) as f32;
    for (rank, id) in fts_ids.iter().enumerate() {
        let rank_score = 1.0 - (rank as f32 / fts_count);
        let entry = scores.entry(*id).or_insert(0.0);
        *entry += fts_weight * rank_score;
    }

    for (id, sim) in vec_scores {
        let entry = scores.entry(*id).or_insert(0.0);
        *entry += vec_weight * sim;
    }

    let mut result: Vec<(uuid::Uuid, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedding_produces_normalized_vectors() {
        let engine = HashEmbedding::new();
        let vec = engine.embed("hello world rust programming");
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm = {norm}");
    }

    #[test]
    fn hash_embedding_similar_texts() {
        let engine = HashEmbedding::new();
        let v1 = engine.embed("rust programming language");
        let v2 = engine.embed("rust programming");
        let v3 = engine.embed("unrelated cooking recipe");

        let sim_same = cosine_similarity(&v1, &v2);
        let sim_diff = cosine_similarity(&v1, &v3);

        assert!(
            sim_same > sim_diff,
            "similar texts should have higher similarity: {sim_same} vs {sim_diff}"
        );
    }

    #[test]
    fn hash_embedding_deterministic() {
        let engine = HashEmbedding::new();
        let v1 = engine.embed("hello world");
        let v2 = engine.embed("hello world");
        assert_eq!(v1, v2);
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn fuse_scores_combines_fts_and_vector() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();

        let fts_ids = vec![id1, id2];
        let vec_scores = vec![(id2, 0.9), (id3, 0.8)];

        let result = fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);
        assert!(!result.is_empty());
        assert_eq!(result[0].0, id2);
    }

    #[test]
    fn fuse_scores_empty_fts() {
        let id1 = uuid::Uuid::new_v4();
        let vec_scores = vec![(id1, 0.9)];
        let fts_ids: Vec<uuid::Uuid> = vec![];

        let result = fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, id1);
    }

    #[test]
    fn extract_ngrams_basic() {
        let ngrams = extract_ngrams("hello", 3);
        assert!(ngrams.contains(&"hel".to_string()));
        assert!(ngrams.contains(&"ell".to_string()));
        assert!(ngrams.contains(&"llo".to_string()));
    }
}
