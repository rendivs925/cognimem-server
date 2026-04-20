use cognimem_server::memory::{
    CognitiveMemoryUnit, MemoryGraph, MemoryTier, apply_decay_to_all, promote_memories,
    prune_below_threshold,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_all");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            for i in 0..size {
                let mem =
                    CognitiveMemoryUnit::new(format!("memory {i}"), MemoryTier::Episodic, 0.5, 0.5);
                graph.add_memory(mem);
            }
            b.iter(|| {
                apply_decay_to_all(&mut graph);
                black_box(());
            });
        });
    }
    group.finish();
}

fn bench_prune(c: &mut Criterion) {
    let mut group = c.benchmark_group("prune");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            for i in 0..size {
                let mut mem =
                    CognitiveMemoryUnit::new(format!("memory {i}"), MemoryTier::Episodic, 0.5, 0.5);
                mem.metadata.base_activation = if i % 10 == 0 { 0.005 } else { 0.5 };
                graph.add_memory(mem);
            }
            b.iter(|| {
                let pruned = prune_below_threshold(&mut graph, 0.01);
                black_box(pruned.len());
            });
        });
    }
    group.finish();
}

fn bench_promote(c: &mut Criterion) {
    let mut group = c.benchmark_group("promote");
    for size in [1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut graph = MemoryGraph::new();
            for i in 0..size {
                let mut mem = CognitiveMemoryUnit::new(
                    format!("memory {i}"),
                    if i % 2 == 0 {
                        MemoryTier::Episodic
                    } else {
                        MemoryTier::Semantic
                    },
                    0.5,
                    0.5,
                );
                mem.metadata.base_activation = if i % 5 == 0 { 0.95 } else { 0.4 };
                graph.add_memory(mem);
            }
            b.iter(|| {
                let promoted = promote_memories(&mut graph);
                black_box(promoted);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decay, bench_prune, bench_promote);
criterion_main!(benches);
