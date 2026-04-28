use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher, Event, EventKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::state::CogniMemState;

pub async fn start_file_watcher(
    state: Arc<Mutex<CogniMemState>>,
    project_path: Option<PathBuf>,
    debounce_ms: u64,
) {
    let path = match project_path {
        Some(p) => p,
        None => return,
    };

    if !path.exists() {
        return;
    }

    let debounce = Duration::from_millis(debounce_ms);

    tokio::spawn(async move {
        let watcher_result = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) => {
                            for path in &event.paths {
                                reprocess_file(&state, path);
                            }
                        }
                        EventKind::Remove(_) => {
                            for path in &event.paths {
                                tracing::debug!("File removed: {:?}", path);
                            }
                        }
                        _ => {}
                    }
                }
            },
            Config::default().with_poll_interval(debounce),
        );

        if let Ok(mut watcher) = watcher_result {
            if let Err(e) = watcher.watch(&path, RecursiveMode::Recursive) {
                tracing::error!("Failed to watch project path: {}", e);
                return;
            }
            tracing::info!("Watching {} for file changes", path.display());
            
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        } else {
            tracing::error!("Failed to create file watcher");
        }
    });
}

fn reprocess_file(state: &Arc<Mutex<CogniMemState>>, path: &Path) {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !matches!(ext, "rs" | "py" | "js" | "jsx" | "ts" | "tsx") {
        return;
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let path_owned = path.to_path_buf();
    let nodes = crate::memory::codegraph::parse_file(&path_owned, &content);
    let count = nodes.len();
    
    if count > 0 {
        let mut guard = state.blocking_lock();
        for node in nodes {
            guard.code_graph.add_node(node);
        }
        tracing::debug!("Updated {} code nodes from {:?}", count, path);
    }
}