use std::sync::Mutex;
use std::time::Instant;

pub struct RateLimiter {
    state: Mutex<RateLimitState>,
    max_requests: u32,
    window_secs: u64,
}

struct RateLimitState {
    count: u32,
    window_start: Instant,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            state: Mutex::new(RateLimitState {
                count: 0,
                window_start: Instant::now(),
            }),
            max_requests,
            window_secs,
        }
    }

    pub fn allow(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let elapsed = state.window_start.elapsed().as_secs();
        if elapsed >= self.window_secs {
            state.count = 0;
            state.window_start = Instant::now();
        }
        if state.count < self.max_requests {
            state.count += 1;
            true
        } else {
            false
        }
    }
}
