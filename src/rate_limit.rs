use std::sync::Mutex;
use std::time::Instant;

/// A sliding-window rate limiter that tracks request counts per time window.
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
    /// Creates a new rate limiter allowing `max_requests` per `window_secs` sliding window.
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

    /// Checks whether a request is allowed under the current rate limit.
    ///
    /// Resets the window counter if the window has expired. Returns `true` if
    /// the request is permitted (and increments the count), `false` if the limit is exceeded.
    pub fn allow(&self) -> bool {
        let mut state = self.state.lock().expect("rate limiter mutex not poisoned");
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
