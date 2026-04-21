use super::types::{CanonicalEvent, IngestResult, IngestStats};
use super::pipeline::{CapturePipeline, get_ingest_stats};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Mutex<CapturePipeline>>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/capture/events", post(ingest_single))
        .route("/capture/events/batch", post(ingest_batch))
        .route("/capture/health", get(health_check))
        .route("/capture/stats", get(ingest_stats))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn start_capture_server(pipeline: Arc<Mutex<CapturePipeline>>, port: u16) {
    let state = AppState { pipeline };
    let app = create_router(state);
    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind capture server on port {port}: {e}");
            return;
        }
    };
    tracing::info!("Capture server listening on port {port}");
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("Capture server error: {e}");
    }
}

async fn ingest_single(
    State(state): State<AppState>,
    Json(event): Json<CanonicalEvent>,
) -> (StatusCode, Json<IngestResult>) {
    let mut pipeline = state.pipeline.lock().await;
    let result = pipeline.ingest_batch(vec![event]).await;
    let status = if result.errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (status, Json(result))
}

async fn ingest_batch(
    State(state): State<AppState>,
    Json(events): Json<Vec<CanonicalEvent>>,
) -> (StatusCode, Json<IngestResult>) {
    let mut pipeline = state.pipeline.lock().await;
    let result = pipeline.ingest_batch(events).await;
    let status = if result.errors.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };
    (status, Json(result))
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok"
    }))
}

async fn ingest_stats(State(state): State<AppState>) -> Json<IngestStats> {
    let pipeline = state.pipeline.lock().await;
    let pending = pipeline.pending_count();
    let uptime = pipeline.uptime_secs();
    Json(get_ingest_stats(pending, uptime))
}
