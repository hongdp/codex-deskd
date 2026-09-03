//! Loopback HTTP surface.
//!
//! The provider's base_url is `http://127.0.0.1:<port>/backend-api/codex`, so
//! Codex treats the shim as the ChatGPT backend and sends its own auth headers.
//! `POST …/responses` for `gemini-*` models is translated to the Gemini API;
//! every other request (other models, `/models`, rate limits, files, …) is
//! forwarded verbatim to the upstream origin with the caller's headers, and the
//! `/models` catalog gets the desk's extra entries appended.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::HeaderName;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::http::header;
use axum::http::request::Parts;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
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
const MAX_BODY: usize = 64 * 1024 * 1024;

struct Shared {
    config: ShimConfig,
    http: reqwest::Client,
}

pub(crate) fn router(config: ShimConfig) -> Router {
    let http = reqwest::Client::builder()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .build()
        .unwrap_or_default();
    let state = Arc::new(Shared { config, http });
    Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .fallback(dispatch)
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

/// Route by path suffix so both `/v1/...` and `/backend-api/codex/...` work.
async fn dispatch(State(st): State<Arc<Shared>>, req: Request) -> Response {
    let path = req.uri().path().trim_end_matches('/').to_string();
    let last = path.rsplit('/').next().unwrap_or("");
    match (req.method(), last) {
        (&Method::POST, "responses") => responses(st, req).await,
        (&Method::GET, "models") => models(st, req).await,
        _ => proxy(st, req).await,
    }
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "host"
            | "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
            | "accept-encoding"
    )
}

fn forward_headers(src: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in src {
        if !is_hop_by_hop(name) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

async fn read_body(req: Request) -> Result<(Parts, Bytes), Response> {
    let (parts, body) = req.into_parts();
    match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(bytes) => Ok((parts, bytes)),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("desk-shim: body: {e}")).into_response()),
    }
}

/// Send the request upstream unchanged (path preserved) and return the
/// upstream response for streaming.
async fn forward(st: &Shared, parts: &Parts, body: Bytes) -> Result<reqwest::Response, Response> {
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(axum::http::uri::PathAndQuery::as_str)
        .unwrap_or("/");
    let url = format!("{}{}", st.config.upstream, path_and_query);
    st.http
        .request(parts.method.clone(), &url)
        .headers(forward_headers(&parts.headers))
        .body(body)
        .send()
        .await
        .map_err(|e| {
            warn!("desk-shim: upstream {url}: {e}");
            (StatusCode::BAD_GATEWAY, format!("desk-shim: upstream: {e}")).into_response()
        })
}

fn relay(upstream: reqwest::Response) -> Response {
    let status = upstream.status();
    let mut headers = HeaderMap::new();
    for (name, value) in upstream.headers() {
        let hop = matches!(
            name.as_str(),
            "transfer-encoding" | "connection" | "content-length" | "content-encoding"
        );
        if !hop {
            headers.append(name.clone(), value.clone());
        }
    }
    let body = Body::from_stream(upstream.bytes_stream());
    (status, headers, body).into_response()
}

async fn proxy(st: Arc<Shared>, req: Request) -> Response {
    let (parts, body) = match read_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    match forward(&st, &parts, body).await {
        Ok(upstream) => relay(upstream),
        Err(resp) => resp,
    }
}

/// Upstream model catalog plus the desk's extra entries. If upstream fails
/// (no auth, offline) the extra entries are served alone.
async fn models(st: Arc<Shared>, req: Request) -> Response {
    let (parts, body) = match read_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let extra = extra_models(&st.config).await;
    let mut headers = HeaderMap::new();
    let mut catalog = json!({ "models": [] });
    match forward(&st, &parts, body).await {
        Ok(resp) if resp.status().is_success() => {
            for (name, value) in resp.headers() {
                if name.as_str().starts_with("x-") {
                    headers.append(name.clone(), value.clone());
                }
            }
            if let Ok(v) = resp.json::<Value>().await {
                catalog = v;
            }
        }
        Ok(resp) => warn!("desk-shim: upstream /models {}", resp.status()),
        Err(_) => {}
    }
    if let Some(list) = catalog.get_mut("models").and_then(Value::as_array_mut) {
        for m in extra {
            let slug = m.get("slug").cloned();
            if !list.iter().any(|x| x.get("slug") == slug.as_ref()) {
                list.push(m);
            }
        }
    }
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    (StatusCode::OK, headers, catalog.to_string()).into_response()
}

async fn extra_models(config: &ShimConfig) -> Vec<Value> {
    let Some(path) = &config.models_json else {
        return Vec::new();
    };
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v.get("models").and_then(Value::as_array).cloned())
            .unwrap_or_default(),
        Err(e) => {
            warn!("desk-shim: read {}: {e}", path.display());
            Vec::new()
        }
    }
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

/// Codex compresses request bodies to the codex backend; decode for
/// translation while the raw bytes stay untouched for forwarding.
fn decode_body(headers: &HeaderMap, body: &Bytes) -> anyhow::Result<Bytes> {
    use std::io::Read;
    let encoding = headers
        .get(header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("identity")
        .trim()
        .to_ascii_lowercase();
    let mut out = Vec::new();
    match encoding.as_str() {
        "" | "identity" => return Ok(body.clone()),
        "gzip" | "x-gzip" => flate2::read::GzDecoder::new(body.as_ref()).read_to_end(&mut out)?,
        "deflate" => flate2::read::ZlibDecoder::new(body.as_ref()).read_to_end(&mut out)?,
        "zstd" => zstd::stream::read::Decoder::new(body.as_ref())?.read_to_end(&mut out)?,
        other => anyhow::bail!("unsupported content-encoding {other:?}"),
    };
    Ok(Bytes::from(out))
}

fn sse_frame(event: &Value) -> Bytes {
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    Bytes::from(format!("event: {kind}\ndata: {event}\n\n"))
}

async fn responses(st: Arc<Shared>, req: Request) -> Response {
    let (parts, body) = match read_body(req).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let decoded = match decode_body(&parts.headers, &body) {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("desk-shim: {e}")).into_response();
        }
    };
    let Ok(payload) = serde_json::from_slice::<Value>(&decoded) else {
        return (StatusCode::BAD_REQUEST, "desk-shim: request is not JSON").into_response();
    };
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !model.starts_with(&st.config.gemini_prefix) {
        return match forward(&st, &parts, body).await {
            Ok(upstream) => relay(upstream),
            Err(resp) => resp,
        };
    }
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::convert::Infallible>>(64);
    tokio::spawn(run_turn(st, payload, tx));
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
