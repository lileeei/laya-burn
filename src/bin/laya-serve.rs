//! HTTP inference server for laya-burn (the counterpart of the official
//! laya[serve] extra): a Jev-compatible POST /v1/decide endpoint backed by
//! the three-checkpoint Router, plus health and model introspection.
//!
//! Run: cargo run --release --bin laya-serve [-- --port 8080]
//! Probe: curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": {...}}'

use std::sync::Mutex;

use axum::{Json, Router as AxumRouter, extract::State, http::StatusCode, routing::{get, post}};
use laya_burn::inference::QuestionInput;
use laya_burn::router::Router;
use serde_json::{Value, json};

pub struct BatchRequest {
    pub state: Value,
    pub questions: Vec<QuestionInput>,
    pub reply: tokio::sync::oneshot::Sender<Result<(String, Vec<laya_burn::inference::Answer>), String>>,
}

struct AppState {
    router: Mutex<Router>,
    /// Some(..) when LAYA_BATCH_MS is set: requests go through the micro-batcher.
    /// Held for the process lifetime (the flush loop owns the receiver);
    /// never read after construction by design.
    #[allow(dead_code)]
    batch_tx: Option<tokio::sync::mpsc::UnboundedSender<BatchRequest>>,
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("PORT").ok())
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let device = burn::tensor::Device::default();
    let router = Router::new(&device);
    if !router.available() {
        eprintln!("laya-serve: no checkpoint directories found; serving will fail per request");
    }
    let forced = std::env::var("LAYA_MODEL").unwrap_or_else(|_| "auto".into());
    eprintln!("laya-serve: listening on 0.0.0.0:{port} (LAYA_MODEL={forced})");

    let (batch_tx, batch_rx) = tokio::sync::mpsc::unbounded_channel::<BatchRequest>();
    let batch_ms: u64 = std::env::var("LAYA_BATCH_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let batch_max: usize = std::env::var("LAYA_BATCH_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let state = std::sync::Arc::new(AppState {
        router: Mutex::new(router),
        batch_tx: if batch_ms > 0 { Some(batch_tx.clone()) } else { None },
    });
    if batch_ms > 0 {
        eprintln!("laya-serve: micro-batching on (window {batch_ms}ms, max {batch_max})");
        let st = state.clone();
        tokio::spawn(batch_flush_loop(st, batch_rx, batch_ms, batch_max));
    }

    let app = AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/decide", post(decide))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<std::sync::Arc<AppState>>) -> Json<Value> {
    let _r = state.router.lock().unwrap();
    Json(json!({
        "routes": ["english", "typed-decisions", "multilingual"],
    }))
}

async fn decide(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let req = match laya_burn::jev::parse_request(&body) {
        Ok(r) => r,
        Err(msg) => return Err((StatusCode::BAD_REQUEST, msg)),
    };

    // Model inference blocks the thread for ~0.1-3 s; run it on the blocking
    // pool under the router lock to keep the async runtime healthy.
    let st = state.clone();
    let inputs = req.questions.clone();
    let state_string = req.state.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let r = st.router.lock().unwrap();
        r.predict(&Value::String(state_string), &inputs)
    })
    .await
    .expect("join");

    match outcome {
        Ok((route, answers)) => {
            let answer_map = laya_burn::jev::answers_json(&req.ids, &req.questions, &answers);
            Ok(Json(json!({
                "model": route.name(),
                "answers": Value::Object(answer_map),
                "usage": {"output_tokens": 0},
            })))
        }
        Err(e) => Err((StatusCode::SERVICE_UNAVAILABLE, format!("{e:?}"))),
    }
}

/// Collect requests for up to window_ms (or until max queued) and run them
/// as one router.predict_many call per flush.
async fn batch_flush_loop(
    st: std::sync::Arc<AppState>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<BatchRequest>,
    window_ms: u64,
    max: usize,
) {
    loop {
        let first = match rx.recv().await {
            Some(r) => r,
            None => break,
        };
        let mut pending = vec![first];
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(window_ms);
        while pending.len() < max {
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(r)) => pending.push(r),
                Ok(None) => break,
                Err(_) => break,
            }
        }
        let items: Vec<(Value, Vec<QuestionInput>)> = pending
            .iter()
            .map(|r| (r.state.clone(), r.questions.clone()))
            .collect();
        let st2 = st.clone();
        let results = tokio::task::spawn_blocking(move || {
            let r = st2.router.lock().unwrap();
            r.predict_many(&items)
        })
        .await
        .expect("batch join");
        for (req, res) in pending.into_iter().zip(results) {
            let out = res.map(|(route, answers)| (route.name().to_string(), answers)).map_err(|e| format!("{e:?}"));
            let _ = req.reply.send(out);
        }
    }
}

