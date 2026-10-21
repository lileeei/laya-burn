//! HTTP inference server for laya-burn (the counterpart of the official
//! laya[serve] extra): a Jev-compatible POST /v1/decide endpoint backed by
//! the three-checkpoint Router, plus health and model introspection.
//!
//! Run: cargo run --release --bin laya-serve [-- --port 8080]
//! Probe: curl -s localhost:8080/v1/decide -d '{"state": "...", "questions": {...}}'

use std::sync::Mutex;

use axum::{Json, Router as AxumRouter, extract::State, http::StatusCode, routing::{get, post}};
use laya_burn::inference::QuestionInput;
use laya_burn::router::{Route, Router};
use serde_json::{Value, json};

struct AppState {
    router: Mutex<Router>,
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

    let state = std::sync::Arc::new(AppState {
        router: Mutex::new(router),
    });

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
    let r = state.router.lock().unwrap();
    Json(json!({
        "routes": ["english", "typed-decisions", "multilingual"],
    }))
}

fn render_options(qtype: &str, criteria: &Value) -> Vec<String> {
    match qtype {
        "choice" => criteria
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(label, desc)| {
                        let blank = matches!(desc, Value::Null)
                            || matches!(desc, Value::String(s) if s.is_empty());
                        if blank {
                            label.clone()
                        } else {
                            format!("{label}: {desc}")
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
        "score" => criteria
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, d)| format!("level {i}: {d}"))
                    .collect()
            })
            .unwrap_or_default(),
        _ => vec![
            "false: no, the statement does not hold".into(),
            "true: yes, the statement holds".into(),
        ],
    }
}

async fn decide(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let state_string = match body.get("state") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => return Err((StatusCode::BAD_REQUEST, "missing state".into())),
    };
    let questions_body = body
        .get("questions")
        .and_then(|q| q.as_object())
        .cloned()
        .unwrap_or_default();

    let mut qids: Vec<String> = questions_body.keys().cloned().collect();
    qids.sort();
    let questions: Vec<(String, QuestionInput)> = qids
        .iter()
        .filter_map(|qid| {
            let q = &questions_body[qid];
            let qtype = q.get("type")?.as_str()?;
            let idx = match qtype {
                "choice" => 0,
                "score" => 1,
                "noul" => 2,
                _ => return None,
            };
            let instructions = match q.get("instructions") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
            Some((
                qid.clone(),
                QuestionInput {
                    qtype: idx,
                    instructions,
                    options: render_options(qtype, &criteria),
                },
            ))
        })
        .collect();

    if questions.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "no valid questions (need type in choice|score|noul)".into(),
        ));
    }
    if questions.iter().any(|q| q.1.options.len() < 2) {
        return Err((
            StatusCode::BAD_REQUEST,
            "choice/score questions need >= 2 options in criteria".into(),
        ));
    }

    // Model inference blocks the thread for ~0.1-3 s; run it on the blocking
    // pool under the router lock to keep the async runtime healthy.
    let st = state.clone();
    let inputs: Vec<QuestionInput> = questions.iter().map(|q| q.1.clone()).collect();
    let outcome = tokio::task::spawn_blocking(move || {
        let r = st.router.lock().unwrap();
        r.predict(&Value::String(state_string), &inputs)
    })
    .await
    .expect("join");

    match outcome {
        Ok((route, answers)) => {
            let mut answer_map = serde_json::Map::new();
            for (i, (qid, _)) in questions.iter().enumerate() {
                answer_map.insert(qid.clone(), answer_json(&questions[i].1, &answers[i]));
            }
            Ok(Json(json!({
                "model": route.name(),
                "answers": Value::Object(answer_map),
                "usage": {"output_tokens": 0},
            })))
        }
        Err(e) => Err((StatusCode::SERVICE_UNAVAILABLE, format!("{e:?}"))),
    }
}

fn answer_json(q: &QuestionInput, a: &laya_burn::inference::Answer) -> Value {
    use laya_burn::inference::Answer;
    match a {
        Answer::Choice { index, probabilities, confidence } => json!({
            "type": "choice",
            "choice": q.options.get(*index).cloned().unwrap_or_default(),
            "probabilities": q.options.iter().zip(probabilities.iter())
                .map(|(o, p)| (o.clone(), json!(p)))
                .collect::<serde_json::Map<String, Value>>(),
            "confidence": confidence,
        }),
        Answer::Score { score, probabilities, confidence } => json!({
            "type": "score",
            "score": score,
            "probabilities": probabilities,
            "confidence": confidence,
        }),
        Answer::Noul { prob } => json!({
            "type": "noul",
            "noul": prob,
        }),
    }
}
