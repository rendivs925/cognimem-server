use std::sync::atomic::{AtomicU64, Ordering};

pub static MEMORY_COUNT: AtomicU64 = AtomicU64::new(0);
pub static REMEMBER_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static RECALL_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static FORGET_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static REFLECT_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static PRUNE_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static ASSOCIATE_TOTAL: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn init() {}

#[inline]
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
         cognimem_associate_total {}\n",
        MEMORY_COUNT.load(Ordering::Relaxed),
        REMEMBER_TOTAL.load(Ordering::Relaxed),
        RECALL_TOTAL.load(Ordering::Relaxed),
        FORGET_TOTAL.load(Ordering::Relaxed),
        REFLECT_TOTAL.load(Ordering::Relaxed),
        PRUNE_TOTAL.load(Ordering::Relaxed),
        ASSOCIATE_TOTAL.load(Ordering::Relaxed),
    )
}

#[inline]
pub fn set_memory_count(v: u64) { MEMORY_COUNT.store(v, Ordering::Relaxed); }
#[inline]
pub fn inc_remember() { REMEMBER_TOTAL.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_recall() { RECALL_TOTAL.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_forget() { FORGET_TOTAL.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_reflect() { REFLECT_TOTAL.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_prune(v: u64) { PRUNE_TOTAL.fetch_add(v, Ordering::Relaxed); }
#[inline]
pub fn inc_associate() { ASSOCIATE_TOTAL.fetch_add(1, Ordering::Relaxed); }