//! Pure translation between the Responses API and Gemini `generateContent`.
//!
//! Request side: `build_gemini_request` turns a Responses request (JSON) into a
//! Gemini request body. Response side: [`Translator`] consumes Gemini stream
//! parts and yields Responses SSE events (as JSON values).
//!
//! Replay fidelity: Gemini 3 attaches `thoughtSignature`s to model parts and
//! expects them back verbatim, and its server-side tool invocations
//! (`toolCall`/`toolResponse`, e.g. Google Search) must be replayed too. Codex
//! replays `reasoning` items verbatim, so those raw parts ride inside a
//! reasoning item's `encrypted_content` as an opaque blob that *follows* the
//! visible item(s) it stands for (`covers_prev`); on replay the blob replaces
//! the plain parts the visible items produced.

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

const BLOB_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct ReplayBlob {
    v: u32,
    covers_prev: usize,
    parts: Vec<Value>,
}

fn encode_blob(covers_prev: usize, parts: Vec<Value>) -> String {
    let blob = ReplayBlob {
        v: BLOB_VERSION,
        covers_prev,
        parts,
    };
    let bytes = serde_json::to_vec(&blob).unwrap_or_default();
    BASE64_STANDARD.encode(bytes)
}

fn decode_blob(s: &str) -> Option<ReplayBlob> {
    let bytes = BASE64_STANDARD.decode(s).ok()?;
    let blob: ReplayBlob = serde_json::from_slice(&bytes).ok()?;
    (blob.v == BLOB_VERSION).then_some(blob)
}

fn text_of_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn function_output_text(item: &Value) -> String {
    match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(_)) => text_of_content(item.get("output")),
        Some(Value::Object(o)) => {
            let t = text_of_content(o.get("content"));
            if t.is_empty() {
                Value::Object(o.clone()).to_string()
            } else {
                t
            }
        }
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Accumulates Gemini `contents`, merging adjacent same-role turns and
/// tracking which parts each visible Responses item produced so a replay blob
/// can replace them.
#[derive(Default)]
struct Contents {
    contents: Vec<Value>,
    /// Number of model parts each visible model-side item contributed.
    visible: Vec<usize>,
}

impl Contents {
    fn push(&mut self, role: &str, part: Value) {
        if let Some(last) = self.contents.last_mut()
            && last.get("role").and_then(Value::as_str) == Some(role)
            && let Some(parts) = last.get_mut("parts").and_then(Value::as_array_mut)
        {
            parts.push(part);
            return;
        }
        self.contents.push(json!({ "role": role, "parts": [part] }));
    }

    fn push_visible_model(&mut self, part: Value) {
        self.push("model", part);
        self.visible.push(1);
    }

    fn apply_blob(&mut self, blob: ReplayBlob) {
        let mut to_remove = 0;
        for _ in 0..blob.covers_prev {
            to_remove += self.visible.pop().unwrap_or(0);
        }
        if to_remove > 0
            && let Some(last) = self.contents.last_mut()
            && last.get("role").and_then(Value::as_str) == Some("model")
            && let Some(parts) = last.get_mut("parts").and_then(Value::as_array_mut)
        {
            let keep = parts.len().saturating_sub(to_remove);
            parts.truncate(keep);
        }
        for part in blob.parts {
            self.push("model", part);
        }
        self.contents.retain(|c| {
            c.get("parts")
                .and_then(Value::as_array)
                .is_some_and(|p| !p.is_empty())
        });
    }
}

fn build_contents(req: &Value) -> (Vec<String>, Vec<Value>) {
    let mut system: Vec<String> = Vec::new();
    if let Some(instr) = req.get("instructions").and_then(Value::as_str)
        && !instr.is_empty()
    {
        system.push(instr.to_string());
    }
    let mut acc = Contents::default();
    let mut call_names: std::collections::HashMap<String, String> = Default::default();

    let items = req.get("input").and_then(Value::as_array);
    for item in items.into_iter().flatten() {
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match kind {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = text_of_content(item.get("content"));
                match role {
                    "system" | "developer" => system.push(text),
                    "assistant" => acc.push_visible_model(json!({ "text": text })),
                    _ => acc.push("user", json!({ "text": text })),
                }
            }
            "reasoning" => {
                if let Some(blob) = item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .and_then(decode_blob)
                {
                    acc.apply_blob(blob);
                }
            }
            "function_call" => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                call_names.insert(call_id.to_string(), name.to_string());
                let raw = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let args = match serde_json::from_str::<Value>(raw) {
                    Ok(Value::Object(o)) => Value::Object(o),
                    Ok(other) => json!({ "value": other }),
                    Err(_) => json!({ "_raw": raw }),
                };
                acc.push_visible_model(json!({ "functionCall": { "name": name, "args": args } }));
            }
            "function_call_output" => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let name = call_names
                    .get(call_id)
                    .cloned()
                    .unwrap_or_else(|| "tool".to_string());
                acc.push(
                    "user",
                    json!({ "functionResponse": { "name": name, "response": { "output": function_output_text(item) } } }),
                );
            }
            "web_search_call" => acc.visible.push(0),
            _ => {}
        }
    }
    if acc.contents.is_empty() {
        acc.contents
            .push(json!({ "role": "user", "parts": [{ "text": "" }] }));
    }
    (system, acc.contents)
}

/// Reduce a JSON schema to the OpenAPI subset Gemini `parameters` accepts.
fn sanitize_schema(s: &Value) -> Value {
    match s {
        Value::Array(items) => Value::Array(items.iter().map(sanitize_schema).collect()),
        Value::Object(o) => {
            let mut out = serde_json::Map::new();
            for (k, v) in o {
                match k.as_str() {
                    "additionalProperties"
                    | "$schema"
                    | "strict"
                    | "default"
                    | "examples"
                    | "title"
                    | "$id" => continue,
                    "type" => {
                        if let Value::Array(types) = v {
                            let non_null: Vec<&Value> = types
                                .iter()
                                .filter(|t| t.as_str() != Some("null"))
                                .collect();
                            out.insert(
                                "type".into(),
                                non_null
                                    .first()
                                    .map(|t| (*t).clone())
                                    .unwrap_or(json!("string")),
                            );
                            if types.iter().any(|t| t.as_str() == Some("null")) {
                                out.insert("nullable".into(), json!(true));
                            }
                            continue;
                        }
                        out.insert(k.clone(), v.clone());
                    }
                    _ => {
                        out.insert(k.clone(), sanitize_schema(v));
                    }
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

struct Tools {
    tools: Vec<Value>,
    uses_search: bool,
}

fn build_tools(req: &Value, sanitized: bool) -> Tools {
    let mut decls: Vec<Value> = Vec::new();
    let mut uses_search = false;
    let list = req.get("tools").and_then(Value::as_array);
    for tool in list.into_iter().flatten() {
        let kind = tool.get("type").and_then(Value::as_str).unwrap_or("");
        let inner: Vec<&Value> = if kind == "namespace" {
            tool.get("tools")
                .and_then(Value::as_array)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        } else {
            vec![tool]
        };
        for t in inner {
            let kind = t.get("type").and_then(Value::as_str).unwrap_or("");
            if kind.starts_with("web_search") {
                uses_search = true;
                continue;
            }
            if kind != "function" {
                tracing::debug!("desk-shim: dropping unsupported tool type {kind:?}");
                continue;
            }
            let params = t
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            let mut d = json!({
                "name": t.get("name").cloned().unwrap_or(Value::Null),
                "description": t.get("description").cloned().unwrap_or(json!("")),
            });
            if let Some(obj) = d.as_object_mut() {
                if sanitized {
                    obj.insert("parameters".into(), sanitize_schema(&params));
                } else {
                    obj.insert("parametersJsonSchema".into(), params);
                }
            }
            decls.push(d);
        }
    }
    let mut tools = Vec::new();
    if uses_search {
        tools.push(json!({ "googleSearch": {} }));
    }
    if !decls.is_empty() {
        tools.push(json!({ "functionDeclarations": decls }));
    }
    Tools { tools, uses_search }
}

/// Build the Gemini request body. `sanitized` selects the fallback schema
/// encoding used when Gemini rejects `parametersJsonSchema`.
pub fn build_gemini_request(req: &Value, sanitized: bool) -> Value {
    let (system, contents) = build_contents(req);
    let mut body = json!({ "contents": contents });
    let Some(obj) = body.as_object_mut() else {
        return body;
    };
    if !system.is_empty() {
        obj.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": system.join("\n\n") }] }),
        );
    }
    let tools = build_tools(req, sanitized);
    if !tools.tools.is_empty() {
        obj.insert("tools".into(), Value::Array(tools.tools));
        let mut tool_config = serde_json::Map::new();
        if tools.uses_search {
            tool_config.insert("includeServerSideToolInvocations".into(), json!(true));
        }
        let mode = match req.get("tool_choice").and_then(Value::as_str) {
            Some("required") => Some("ANY"),
            Some("none") => Some("NONE"),
            _ => None,
        };
        if let Some(mode) = mode {
            tool_config.insert("functionCallingConfig".into(), json!({ "mode": mode }));
        }
        if !tool_config.is_empty() {
            obj.insert("toolConfig".into(), Value::Object(tool_config));
        }
    }
    let model = req.get("model").and_then(Value::as_str).unwrap_or("");
    let effort = req
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str);
    let mut thinking = json!({ "includeThoughts": true });
    if model.starts_with("gemini-3") {
        let level = match effort {
            Some("minimal" | "low") => Some("low"),
            Some("medium") => Some("medium"),
            Some("high" | "xhigh") => Some("high"),
            _ => None,
        };
        if let Some(level) = level
            && let Some(t) = thinking.as_object_mut()
        {
            t.insert("thinkingLevel".into(), json!(level));
        }
    } else if effort == Some("minimal")
        && let Some(t) = thinking.as_object_mut()
    {
        t.insert("thinkingBudget".into(), json!(0));
    }
    obj.insert(
        "generationConfig".into(),
        json!({ "thinkingConfig": thinking }),
    );
    body
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    Reasoning,
    Message,
}

/// Streaming state machine: Gemini parts in, Responses events out.
pub struct Translator {
    response_id: String,
    seq: u64,
    output_index: u64,
    open: Option<Open>,
    item_id: String,
    buf: String,
    /// Signature seen on the text parts of the currently open message.
    sig: Option<String>,
    /// Number of visible output items emitted (for the caller's retry logic).
    pub items_emitted: u64,
}

impl Translator {
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            response_id: response_id.into(),
            seq: 0,
            output_index: 0,
            open: None,
            item_id: String::new(),
            buf: String::new(),
            sig: None,
            items_emitted: 0,
        }
    }

    fn event(&mut self, kind: &str, mut fields: Value) -> Value {
        self.seq += 1;
        if let Some(o) = fields.as_object_mut() {
            o.insert("type".into(), json!(kind));
            o.insert("sequence_number".into(), json!(self.seq));
        }
        fields
    }

    pub fn created(&mut self) -> Value {
        self.event(
            "response.created",
            json!({ "response": { "id": self.response_id, "object": "response", "status": "in_progress" } }),
        )
    }

    fn item_added_done(&mut self, item: Value, out: &mut Vec<Value>) {
        let idx = self.output_index;
        out.push(self.event(
            "response.output_item.added",
            json!({ "output_index": idx, "item": item }),
        ));
        out.push(self.event(
            "response.output_item.done",
            json!({ "output_index": idx, "item": item }),
        ));
        self.output_index += 1;
        self.items_emitted += 1;
    }

    fn blob_item(&mut self, covers_prev: usize, parts: Vec<Value>, out: &mut Vec<Value>) {
        let item = json!({
            "type": "reasoning",
            "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
            "summary": [],
            "encrypted_content": encode_blob(covers_prev, parts),
        });
        self.item_added_done(item, out);
    }

    fn open(&mut self, kind: Open, out: &mut Vec<Value>) {
        if self.open == Some(kind) {
            return;
        }
        self.close(out);
        self.open = Some(kind);
        self.buf.clear();
        self.sig = None;
        let idx = self.output_index;
        match kind {
            Open::Reasoning => {
                self.item_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
                let item = json!({ "type": "reasoning", "id": self.item_id, "summary": [] });
                out.push(self.event(
                    "response.output_item.added",
                    json!({ "output_index": idx, "item": item }),
                ));
                let item_id = self.item_id.clone();
                out.push(self.event(
                    "response.reasoning_summary_part.added",
                    json!({ "item_id": item_id, "output_index": idx, "summary_index": 0,
                            "part": { "type": "summary_text", "text": "" } }),
                ));
            }
            Open::Message => {
                self.item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                let item = json!({ "type": "message", "id": self.item_id, "role": "assistant",
                                   "status": "in_progress", "content": [] });
                out.push(self.event(
                    "response.output_item.added",
                    json!({ "output_index": idx, "item": item }),
                ));
            }
        }
    }

    fn close(&mut self, out: &mut Vec<Value>) {
        let Some(kind) = self.open.take() else {
            return;
        };
        let text = std::mem::take(&mut self.buf);
        let idx = self.output_index;
        let item_id = self.item_id.clone();
        let item = match kind {
            Open::Reasoning => {
                out.push(self.event(
                    "response.reasoning_summary_text.done",
                    json!({ "item_id": item_id, "output_index": idx, "summary_index": 0, "text": text }),
                ));
                json!({ "type": "reasoning", "id": item_id,
                        "summary": [{ "type": "summary_text", "text": text }] })
            }
            Open::Message => json!({ "type": "message", "id": item_id, "role": "assistant",
                                     "status": "completed",
                                     "content": [{ "type": "output_text", "text": text, "annotations": [] }] }),
        };
        out.push(self.event(
            "response.output_item.done",
            json!({ "output_index": idx, "item": item }),
        ));
        self.output_index += 1;
        self.items_emitted += 1;
        if kind == Open::Message
            && let Some(sig) = self.sig.take()
        {
            self.blob_item(
                1,
                vec![json!({ "text": text, "thoughtSignature": sig })],
                out,
            );
        }
    }

    /// Feed one Gemini content part; returns the Responses events to emit.
    pub fn on_part(&mut self, part: &Value) -> Vec<Value> {
        let mut out = Vec::new();
        let sig = part
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(fc) = part.get("functionCall") {
            self.close(&mut out);
            let args = fc.get("args").cloned().unwrap_or(json!({}));
            let item = json!({
                "type": "function_call",
                "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                "call_id": format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]),
                "name": fc.get("name").cloned().unwrap_or(json!("")),
                "arguments": args.to_string(),
                "status": "completed",
            });
            self.item_added_done(item, &mut out);
            if sig.is_some() {
                self.blob_item(1, vec![part.clone()], &mut out);
            }
            return out;
        }
        if let Some(tc) = part.get("toolCall") {
            self.close(&mut out);
            let query = tc
                .get("args")
                .and_then(|a| a.get("queries"))
                .and_then(Value::as_array)
                .map(|qs| {
                    qs.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            let item = json!({
                "type": "web_search_call",
                "id": format!("ws_{}", uuid::Uuid::new_v4().simple()),
                "status": "completed",
                "action": { "type": "search", "query": query },
            });
            self.item_added_done(item, &mut out);
            self.blob_item(1, vec![part.clone()], &mut out);
            return out;
        }
        if part.get("toolResponse").is_some() {
            self.close(&mut out);
            self.blob_item(0, vec![part.clone()], &mut out);
            return out;
        }
        let text = part.get("text").and_then(Value::as_str).unwrap_or("");
        if text.is_empty() {
            if sig.is_some() {
                // Signature on a part we cannot represent: replay it verbatim.
                self.close(&mut out);
                self.blob_item(0, vec![part.clone()], &mut out);
            }
            return out;
        }
        let is_thought = part
            .get("thought")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let kind = if is_thought {
            Open::Reasoning
        } else {
            Open::Message
        };
        self.open(kind, &mut out);
        self.buf.push_str(text);
        let item_id = self.item_id.clone();
        let idx = self.output_index;
        let ev = match kind {
            Open::Reasoning => self.event(
                "response.reasoning_summary_text.delta",
                json!({ "item_id": item_id, "output_index": idx, "summary_index": 0, "delta": text }),
            ),
            Open::Message => self.event(
                "response.output_text.delta",
                json!({ "item_id": item_id, "output_index": idx, "content_index": 0, "delta": text }),
            ),
        };
        out.push(ev);
        if let Some(sig) = sig {
            if kind == Open::Message {
                self.sig = Some(sig);
            } else {
                // Thoughts are not replayed; keep the signature on its own part.
                self.close(&mut out);
                self.blob_item(
                    0,
                    vec![json!({ "text": "", "thoughtSignature": sig })],
                    &mut out,
                );
            }
        }
        out
    }

    /// Close any open item and emit `response.completed` with usage mapped
    /// from Gemini's `usageMetadata`.
    pub fn finish(&mut self, usage_meta: &Value) -> Vec<Value> {
        let mut out = Vec::new();
        self.close(&mut out);
        let n = |k: &str| usage_meta.get(k).and_then(Value::as_u64).unwrap_or(0);
        let prompt = n("promptTokenCount");
        let cand = n("candidatesTokenCount");
        let thoughts = n("thoughtsTokenCount");
        let total = usage_meta
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(prompt + cand + thoughts);
        let usage = json!({
            "input_tokens": prompt,
            "output_tokens": cand + thoughts,
            "total_tokens": total,
            "input_tokens_details": { "cached_tokens": n("cachedContentTokenCount") },
            "output_tokens_details": { "reasoning_tokens": thoughts },
        });
        let rid = self.response_id.clone();
        out.push(self.event(
            "response.completed",
            json!({ "response": { "id": rid, "object": "response", "status": "completed", "usage": usage } }),
        ));
        out
    }

    pub fn failed(&mut self, message: &str) -> Value {
        let rid = self.response_id.clone();
        let message: String = message.chars().take(2000).collect();
        self.event(
            "response.failed",
            json!({ "response": { "id": rid, "status": "failed",
                                  "error": { "code": "desk_shim_error", "message": message } } }),
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn kinds(events: &[Value]) -> Vec<String> {
        events
            .iter()
            .map(|e| {
                let t = e["type"].as_str().unwrap().to_string();
                match e
                    .get("item")
                    .and_then(|i| i.get("type"))
                    .and_then(Value::as_str)
                {
                    Some(it) if t == "response.output_item.done" => format!("{t}:{it}"),
                    _ => t,
                }
            })
            .collect()
    }

    #[test]
    fn stream_function_call_with_signature_round_trips() {
        let mut tr = Translator::new("resp_1");
        let mut events = vec![tr.created()];
        events.extend(tr.on_part(&json!({ "text": "hmm", "thought": true })));
        events.extend(tr.on_part(&json!({
            "functionCall": { "name": "shell", "args": { "cmd": "ls" } },
            "thoughtSignature": "SIG1"
        })));
        events.extend(tr.finish(
            &json!({ "promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3 }),
        ));
        let k = kinds(&events);
        assert!(k.contains(&"response.output_item.done:reasoning".to_string()));
        assert!(k.contains(&"response.output_item.done:function_call".to_string()));
        assert_eq!(k.last().unwrap(), "response.completed");
        let usage = &events.last().unwrap()["response"]["usage"];
        assert_eq!(usage["output_tokens"], 8);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 3);

        // Replay: feed the done items back as input and check the signature
        // lands on the functionCall part exactly once.
        let items: Vec<Value> = events
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .map(|e| e["item"].clone())
            .collect();
        let fc = items.iter().find(|i| i["type"] == "function_call").unwrap();
        let mut input = items.clone();
        input.push(json!({ "type": "function_call_output", "call_id": fc["call_id"], "output": "file.txt" }));
        let req = json!({ "model": "gemini-3.7-flash", "instructions": "sys", "input": input,
                          "tools": [{ "type": "function", "name": "shell", "parameters": {"type":"object"} },
                                    { "type": "web_search" }] });
        let body = build_gemini_request(&req, false);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2, "{contents:?}");
        let model_parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(model_parts.len(), 1, "{model_parts:?}");
        assert_eq!(model_parts[0]["thoughtSignature"], "SIG1");
        assert_eq!(model_parts[0]["functionCall"]["name"], "shell");
        assert_eq!(contents[1]["parts"][0]["functionResponse"]["name"], "shell");
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
        assert_eq!(body["tools"][0]["googleSearch"], json!({}));
        assert_eq!(body["toolConfig"]["includeServerSideToolInvocations"], true);
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );
    }

    #[test]
    fn server_side_search_becomes_web_search_call_and_replays_raw() {
        let mut tr = Translator::new("resp_2");
        let call = json!({ "toolCall": { "toolType": "GOOGLE_SEARCH_WEB", "args": { "queries": ["a", "b"] }, "id": "c1" }, "thoughtSignature": "S1" });
        let resp = json!({ "toolResponse": { "toolType": "GOOGLE_SEARCH_WEB", "response": {}, "id": "c1" }, "thoughtSignature": "S2" });
        let mut events = tr.on_part(&call);
        events.extend(tr.on_part(&resp));
        events.extend(tr.on_part(&json!({ "text": "answer", "thoughtSignature": "S3" })));
        events.extend(tr.finish(&json!({})));
        let k = kinds(&events);
        assert_eq!(
            k.iter().filter(|s| s.ends_with(":web_search_call")).count(),
            1
        );
        let ws = events
            .iter()
            .find(|e| e["item"]["type"] == "web_search_call")
            .unwrap();
        assert_eq!(ws["item"]["action"]["query"], "a; b");

        let input: Vec<Value> = events
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .map(|e| e["item"].clone())
            .collect();
        let body = build_gemini_request(&json!({ "input": input }), false);
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(body["contents"][0]["role"], "model");
        assert_eq!(parts.len(), 3, "{parts:?}");
        assert_eq!(parts[0]["toolCall"]["id"], "c1");
        assert_eq!(parts[1]["toolResponse"]["id"], "c1");
        assert_eq!(parts[2]["text"], "answer");
        assert_eq!(parts[2]["thoughtSignature"], "S3");
    }

    #[test]
    fn sanitized_schema_drops_unsupported_keys() {
        let s = json!({ "type": "object", "additionalProperties": false,
                        "properties": { "x": { "type": ["string", "null"], "default": 1 } } });
        let out = sanitize_schema(&s);
        assert!(out.get("additionalProperties").is_none());
        assert_eq!(out["properties"]["x"]["type"], "string");
        assert_eq!(out["properties"]["x"]["nullable"], true);
        assert!(out["properties"]["x"].get("default").is_none());
    }

    #[test]
    fn foreign_encrypted_content_is_ignored() {
        let req = json!({ "input": [
            { "type": "reasoning", "encrypted_content": "not-base64-json" },
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }
        ]});
        let body = build_gemini_request(&req, true);
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
    }
}
