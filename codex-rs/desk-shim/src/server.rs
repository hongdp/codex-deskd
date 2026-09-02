//! Loopback HTTP surface: `/v1/responses` (SSE), `/v1/models`, `/health`.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::Value;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;
use tracing::warn;

use crate::ShimConfig;
use crate::translate::Translator;
use crate::translate::build_gemini_request;

const GEMINI_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

struct Shared {
    config: ShimConfig,
    http: reqwest::Client,
}

pub(crate) fn router(config: ShimConfig) -> Router {
    let state = Arc::new(Shared {
        config,
        http: reqwest::Client::new(),
    });
    Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/responses", post(responses))
        .route("/v1/responses", post(responses))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

/// Codex fetches the provider's model catalog at startup; an empty list makes
/// it keep its bundled catalog instead of erroring on a 404.
async fn models() -> Json<Value> {
    Json(json!({ "models": [] }))
}

fn api_key(config: &ShimConfig) -> anyhow::Result<String> {
    if let Ok(k) = std::env::var("GEMINI_API_KEY")
        && !k.trim().is_empty()
    {
        return Ok(k.trim().to_string());
    }
    let Some(path) = &config.gemini_key_file else {
        anyhow::bail!("no GEMINI_API_KEY and no [desk_shim].gemini_key_file configured");
    };
    let k = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let k = k.trim().to_string();
    if k.is_empty() {
        anyhow::bail!("{} is empty", path.display());
    }
    Ok(k)
}

fn sse_frame(event: &Value) -> Bytes {
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    Bytes::from(format!("event: {kind}\ndata: {event}\n\n"))
}

async fn responses(State(st): State<Arc<Shared>>, Json(req): Json<Value>) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::convert::Infallible>>(64);
    tokio::spawn(run_turn(st, req, tx));
    let body = Body::from_stream(ReceiverStream::new(rx));
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

enum Attempt {
    Done,
    RetrySanitized,
}

async fn run_turn(
    st: Arc<Shared>,
    req: Value,
    tx: mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
) {
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let mut tr = Translator::new(response_id.clone());
    let send = |ev: Value| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(sse_frame(&ev))).await;
        }
    };
    send(tr.created()).await;

    if let Some(dir) = &st.config.dump_dir
        && let Ok(()) = tokio::fs::create_dir_all(dir).await
    {
        let path = dir.join(format!("{}.request.json", &response_id));
        let _ = tokio::fs::write(&path, req.to_string()).await;
    }

    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("gemini-3.7-flash")
        .to_string();
    let items = req
        .get("input")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let tools = req
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    info!("desk-shim: -> {model} input_items={items} tools={tools}");

    let key = match api_key(&st.config) {
        Ok(k) => k,
        Err(e) => {
            warn!("desk-shim: {e}");
            send(tr.failed(&e.to_string())).await;
            return;
        }
    };

    for sanitized in [false, true] {
        let body = build_gemini_request(&req, sanitized);
        match stream_gemini(&st, &model, &key, &body, &mut tr, &tx).await {
            Ok(Attempt::Done) => return,
            Ok(Attempt::RetrySanitized) => {
                info!("desk-shim: schema rejected, retrying with sanitized parameters");
                continue;
            }
            Err(e) => {
                warn!("desk-shim: {e}");
                send(tr.failed(&e.to_string())).await;
                return;
            }
        }
    }
    send(tr.failed("gemini rejected the tool schema twice")).await;
}

async fn stream_gemini(
    st: &Shared,
    model: &str,
    key: &str,
    body: &Value,
    tr: &mut Translator,
    tx: &mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
) -> anyhow::Result<Attempt> {
    let url = format!("{GEMINI_BASE}/models/{model}:streamGenerateContent?alt=sse");
    let resp = st
        .http
        .post(&url)
        .header("x-goog-api-key", key)
        .json(body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("gemini request: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let detail = resp.text().await.unwrap_or_default();
        if status == StatusCode::BAD_REQUEST
            && detail.contains("parameters")
            && tr.items_emitted == 0
            && body
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|t| t.iter().any(|x| x.get("functionDeclarations").is_some()))
            && body.to_string().contains("parametersJsonSchema")
        {
            return Ok(Attempt::RetrySanitized);
        }
        let detail: String = detail.chars().take(800).collect();
        anyhow::bail!("gemini {status}: {detail}");
    }

    let mut usage_meta = Value::Null;
    let mut stream = resp.bytes_stream().eventsource();
    while let Some(ev) = stream.next().await {
        let ev = ev.map_err(|e| anyhow::anyhow!("gemini stream: {e}"))?;
        if ev.data.trim().is_empty() {
            continue;
        }
        let chunk: Value = serde_json::from_str(&ev.data)
            .map_err(|e| anyhow::anyhow!("gemini chunk parse: {e}"))?;
        if let Some(err) = chunk.get("error") {
            anyhow::bail!("gemini error: {err}");
        }
        let cands = chunk.get("candidates").and_then(Value::as_array);
        for cand in cands.into_iter().flatten() {
            let parts = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array);
            for part in parts.into_iter().flatten() {
                for out in tr.on_part(part) {
                    if tx.send(Ok(sse_frame(&out))).await.is_err() {
                        return Ok(Attempt::Done);
                    }
                }
            }
            if let Some(reason) = cand.get("finishReason").and_then(Value::as_str)
                && !matches!(reason, "STOP" | "MAX_TOKENS")
            {
                warn!("desk-shim: finishReason={reason}");
            }
        }
        if let Some(u) = chunk.get("usageMetadata") {
            usage_meta = u.clone();
        }
    }
    for out in tr.finish(&usage_meta) {
        let _ = tx.send(Ok(sse_frame(&out))).await;
    }
    let total_tokens = usage_meta
        .get("totalTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let items = tr.items_emitted;
    info!("desk-shim: <- done items={items} total_tokens={total_tokens}");
    Ok(Attempt::Done)
}
