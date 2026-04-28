use std::sync::atomic::{AtomicU64, Ordering};

/// Total number of memories currently stored.
pub static MEMORY_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total number of `remember` operations performed.
pub static REMEMBER_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total number of `recall` operations performed.
pub static RECALL_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total number of `forget` operations performed.
pub static FORGET_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total number of `reflect` operations performed.
pub static REFLECT_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total number of pruned (removed) memories.
pub static PRUNE_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total number of `associate` operations performed.
pub static ASSOCIATE_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total code nodes discovered.
pub static CODE_NODE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Total injection events.
pub static INJECT_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total SLM inference calls.
pub static SLM_INFER_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total dreams generated.
pub static DREAM_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Total imagination calls.
pub static IMAGINE_TOTAL: AtomicU64 = AtomicU64::new(0);

#[inline]
/// Initializes the metrics subsystem. Currently a no-op placeholder.
pub fn init() {}

#[inline]
/// Encodes all metrics in Prometheus exposition format.
pub fn encode() -> String {
    format!(
        "# HELP cognimem_memory_count Total number of memories\n\
         # TYPE cognimem_memory_count gauge\n\
         cognimem_memory_count {}\n\
         # HELP cognimem_remember_total Total remember operations\n\
         # TYPE cognimem_remember_total counter\n\
         cognimem_remember_total {}\n\
         # HELP cognimem_recall_total Total recall operations\n\
         # TYPE cognimem_recall_total counter\n\
         cognimem_recall_total {}\n\
         # HELP cognimem_forget_total Total forget operations\n\
         # TYPE cognimem_forget_total counter\n\
         cognimem_forget_total {}\n\
         # HELP cognimem_reflect_total Total reflect operations\n\
         # TYPE cognimem_reflect_total counter\n\
         cognimem_reflect_total {}\n\
         # HELP cognimem_prune_total Total pruned memories\n\
         # TYPE cognimem_prune_total counter\n\
         cognimem_prune_total {}\n\
         # HELP cognimem_associate_total Total associate operations\n\
         # TYPE cognimem_associate_total counter\n\
         cognimem_associate_total {}\n\
         # HELP cognimem_code_node_count Total code nodes discovered\n\
         # TYPE cognimem_code_node_count gauge\n\
         cognimem_code_node_count {}\n\
         # HELP cognimem_inject_total Total injection events\n\
         # TYPE cognimem_inject_total counter\n\
         cognimem_inject_total {}\n\
         # HELP cognimem_slm_infer_total Total SLM inference calls\n\
         # TYPE cognimem_slm_infer_total counter\n\
         cognimem_slm_infer_total {}\n\
         # HELP cognimem_dream_total Total dreams generated\n\
         # TYPE cognimem_dream_total counter\n\
         cognimem_dream_total {}\n\
         # HELP cognimem_imagine_total Total imagination calls\n\
         # TYPE cognimem_imagine_total counter\n\
         cognimem_imagine_total {}\n",
        MEMORY_COUNT.load(Ordering::Relaxed),
        REMEMBER_TOTAL.load(Ordering::Relaxed),
        RECALL_TOTAL.load(Ordering::Relaxed),
        FORGET_TOTAL.load(Ordering::Relaxed),
        REFLECT_TOTAL.load(Ordering::Relaxed),
        PRUNE_TOTAL.load(Ordering::Relaxed),
        ASSOCIATE_TOTAL.load(Ordering::Relaxed),
        CODE_NODE_COUNT.load(Ordering::Relaxed),
        INJECT_TOTAL.load(Ordering::Relaxed),
        SLM_INFER_TOTAL.load(Ordering::Relaxed),
        DREAM_TOTAL.load(Ordering::Relaxed),
        IMAGINE_TOTAL.load(Ordering::Relaxed),
    )
}

#[inline]
/// Sets the total memory count gauge to the given value.
pub fn set_memory_count(v: u64) {
    MEMORY_COUNT.store(v, Ordering::Relaxed);
}
#[inline]
/// Increments the `remember` operation counter by 1.
pub fn inc_remember() {
    REMEMBER_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
/// Increments the `recall` operation counter by 1.
pub fn inc_recall() {
    RECALL_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
/// Increments the `forget` operation counter by 1.
pub fn inc_forget() {
    FORGET_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
/// Increments the `reflect` operation counter by 1.
pub fn inc_reflect() {
    REFLECT_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
/// Increments the prune counter by `v`.
pub fn inc_prune(v: u64) {
    PRUNE_TOTAL.fetch_add(v, Ordering::Relaxed);
}
#[inline]
/// Increments the `associate` operation counter by 1.
pub fn inc_associate() {
    ASSOCIATE_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn set_code_node_count(v: u64) {
    CODE_NODE_COUNT.store(v, Ordering::Relaxed);
}
#[inline]
pub fn inc_inject() {
    INJECT_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn inc_slm_infer() {
    SLM_INFER_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn inc_dream() {
    DREAM_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn inc_imagine() {
    IMAGINE_TOTAL.fetch_add(1, Ordering::Relaxed);
}
