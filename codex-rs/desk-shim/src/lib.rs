//! In-process wire adapter: OpenAI Responses API (what Codex speaks) -> Gemini.
//!
//! Codex has exactly one wire protocol, `/v1/responses` streamed as SSE. This
//! crate serves that protocol on loopback and translates each request into a
//! Gemini `streamGenerateContent` call, so a `[model_providers.<id>]` entry
//! with `wire_api = "responses"` and `base_url = "http://127.0.0.1:<port>/v1"`
//! drives Gemini without touching the core request/SSE path.
//!
//! Configured by an optional `[desk_shim]` table in `$CODEX_HOME/config.toml`:
//!
//! ```toml
//! [desk_shim]
//! port = 8397
//! gemini_key_file = "/abs/path/to/key"   # or the GEMINI_API_KEY env var
//! dump_dir = "/abs/path"                 # optional: record every request
//! ```
//!
//! The listener is started by the CLI before any subcommand runs. If the port
//! is already bound, a sibling codex process is assumed to be serving it.

mod server;
pub mod translate;

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use tokio::net::TcpListener;
use tracing::info;
use tracing::warn;

pub const DEFAULT_PORT: u16 = 8397;

#[derive(Debug, Clone)]
pub struct ShimConfig {
    pub port: u16,
    pub gemini_key_file: Option<PathBuf>,
    pub dump_dir: Option<PathBuf>,
}

impl ShimConfig {
    /// Parse `[desk_shim]` from `$CODEX_HOME/config.toml`. Returns `None` when
    /// the table is absent or the file is unreadable/unparseable (the shim is
    /// opt-in and must never block startup).
    pub fn load(codex_home: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(codex_home.join("config.toml")).ok()?;
        let value: toml::Value = toml::from_str(&raw).ok()?;
        let table = value.get("desk_shim")?.as_table()?;
        let port = table
            .get("port")
            .and_then(toml::Value::as_integer)
            .and_then(|p| u16::try_from(p).ok())
            .unwrap_or(DEFAULT_PORT);
        let resolve = |key: &str| -> Option<PathBuf> {
            let s = table.get(key)?.as_str()?;
            let p = PathBuf::from(s);
            Some(if p.is_absolute() {
                p
            } else {
                codex_home.join(p)
            })
        };
        Some(Self {
            port,
            gemini_key_file: resolve("gemini_key_file"),
            dump_dir: resolve("dump_dir"),
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }
}

/// Bind the loopback listener and serve in a background task. Returns after
/// the socket is bound, so callers can rely on the endpoint being reachable.
pub async fn start(config: ShimConfig) -> Result<()> {
    let addr = format!("127.0.0.1:{}", config.port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
            info!("desk-shim: {addr} already bound; assuming a sibling process serves it");
            return Ok(());
        }
        Err(err) => return Err(err).with_context(|| format!("desk-shim: bind {addr}")),
    };
    info!("desk-shim: serving /v1/responses on http://{addr}");
    let app = server::router(config);
    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            warn!("desk-shim: server exited: {err}");
        }
    });
    Ok(())
}
