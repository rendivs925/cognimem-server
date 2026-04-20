use cognimem_server::embeddings::{EmbeddingEngine, HashEmbedding};
use cognimem_server::memory::types::CognitiveMemoryUnit;
use cognimem_server::memory::{MemoryGraph, MemoryTier};
use cognimem_server::search::{Fts5Search, SearchEngine};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_recall_substring(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_substring");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            for i in 0..size {
                let mem = CognitiveMemoryUnit::new(
                    format!("memory content number {i} about rust programming"),
                    MemoryTier::Episodic,
                    0.5,
                    0.5,
                );
                graph.add_memory(mem);
            }
            b.iter(|| {
                let by_tier = graph.get_by_tier(MemoryTier::Episodic);
                let results: Vec<&&CognitiveMemoryUnit> = by_tier
                    .iter()
                    .filter(|m| m.content.to_lowercase().contains("rust"))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

fn bench_recall_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_hybrid");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            let mut search = Fts5Search::new().expect("FTS5 init");
            let embedder = HashEmbedding::new();
            for i in 0..size {
                let mem = CognitiveMemoryUnit::new(
                    format!("memory content number {i} about rust programming"),
                    MemoryTier::Episodic,
                    0.5,
                    0.5,
                );
                let id = mem.id;
                search.index(id, &mem.content, mem.tier);
                let emb = embedder.embed(&mem.content);
                graph.add_memory(mem);
                graph.set_embedding(id, emb);
            }
            b.iter(|| {
                let fts_ids = search.search("rust programming", None, 20);
                let query_emb = embedder.embed("rust programming");
                let vec_scores = graph.vector_search(&query_emb, 20, 0.1);
                let fused =
                    cognimem_server::embeddings::fuse_scores(&fts_ids, 0.4, &vec_scores, 0.6);
                black_box(fused);
            });
        });
    }
    group.finish();
}

fn bench_add_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_memory");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            for i in 0..size {
                let mem = CognitiveMemoryUnit::new(
                    format!("prefill {i}"),
                    MemoryTier::Episodic,
                    0.5,
                    0.5,
                );
                graph.add_memory(mem);
            }
            b.iter(|| {
                let mem = CognitiveMemoryUnit::new(
                    "new memory about rust".to_string(),
                    MemoryTier::Episodic,
                    0.5,
                    0.5,
                );
                black_box(graph.add_memory(mem));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_recall_substring,
    bench_recall_hybrid,
    bench_add_memory
);
criterion_main!(benches);
