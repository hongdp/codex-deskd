#!/usr/bin/env python3
"""Local wire adapter: OpenAI Responses API (what Codex speaks) -> Gemini API.

Codex only knows one wire protocol, `/v1/responses` streamed as SSE. This shim
listens on loopback, translates each request into a Gemini
`streamGenerateContent` call and streams back the Responses events Codex
consumes (`response.created`, `output_item.added/done`, `output_text.delta`,
`reasoning_summary_*`, `response.completed|failed`).

Fidelity notes:
  * Gemini 3 thought signatures ride inside a Responses `reasoning` item's
    `encrypted_content` (Codex replays reasoning items verbatim), so the
    signature returns to Gemini on the exact model part it belongs to.
  * Function tools are sent via `parametersJsonSchema`; if Gemini rejects the
    schema the request is retried with a sanitized `parameters` block.
  * Non-function tool types (web_search, namespaces, ...) are dropped.

Stdlib only. Run:  python3 desk/shim/responses_gemini.py --port 8397
Key: $GEMINI_API_KEY, else the file <repo>/keys/gemini.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DUMP_DIR = os.environ.get("SHIM_DUMP_DIR")  # set to a dir to record traffic
LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOG_LOCK:
        sys.stderr.write(f"[shim {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        p = REPO / "keys" / "gemini"
        if not p.exists():
            raise RuntimeError("no GEMINI_API_KEY and no keys/gemini file")
        k = p.read_text().strip()
    return k


# ---------------------------------------------------------------- request side

def _sig_encode(sig: str, attach: str) -> str:
    return base64.b64encode(json.dumps({"sig": sig, "attach": attach}).encode()).decode()


def _sig_decode(blob: str | None) -> tuple[str, str] | None:
    if not blob:
        return None
    try:
        d = json.loads(base64.b64decode(blob))
        return d["sig"], d.get("attach", "next")
    except Exception:  # noqa: BLE001 — foreign/opaque blob, ignore
        return None


def _text_of_content(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            out.append(c["text"])
    return "\n".join(out)


def _function_output_text(item: dict) -> str:
    out = item.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, list):
        return _text_of_content(out)
    if isinstance(out, dict):
        return _text_of_content(out.get("content", "")) or json.dumps(out)
    return "" if out is None else str(out)


def build_contents(req: dict) -> tuple[list[str], list[dict]]:
    """Return (system_texts, gemini_contents)."""
    system: list[str] = []
    if isinstance(req.get("instructions"), str) and req["instructions"]:
        system.append(req["instructions"])
    contents: list[dict] = []
    call_names: dict[str, str] = {}
    pending_sig: str | None = None  # signature to attach to the NEXT model part
    last_model_part: dict | None = None

    def push(role: str, part: dict) -> None:
        nonlocal last_model_part
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})
        if role == "model":
            last_model_part = part

    for item in req.get("input") or []:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "message")
        if t == "message":
            role = item.get("role", "user")
            text = _text_of_content(item.get("content"))
            if role in ("system", "developer"):
                system.append(text)
                continue
            if role == "assistant":
                part = {"text": text}
                if pending_sig:
                    part["thoughtSignature"] = pending_sig
                    pending_sig = None
                push("model", part)
            else:
                push("user", {"text": text})
        elif t == "reasoning":
            dec = _sig_decode(item.get("encrypted_content"))
            if dec:
                sig, attach = dec
                if attach == "prev" and last_model_part is not None:
                    last_model_part["thoughtSignature"] = sig
                else:
                    pending_sig = sig
        elif t == "function_call":
            name = item.get("name", "")
            call_id = item.get("call_id") or item.get("id") or ""
            call_names[call_id] = name
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": item.get("arguments")}
            part = {"functionCall": {"name": name, "args": args if isinstance(args, dict) else {"value": args}}}
            if pending_sig:
                part["thoughtSignature"] = pending_sig
                pending_sig = None
            push("model", part)
        elif t == "function_call_output":
            call_id = item.get("call_id") or ""
            name = call_names.get(call_id, "tool")
            push("user", {"functionResponse": {"name": name, "response": {"output": _function_output_text(item)}}})
        # local_shell_call, custom tool calls, etc. are not produced by this shim
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})
    return system, contents


def _sanitize_schema(s):
    """Reduce a JSON schema to the OpenAPI subset Gemini `parameters` accepts."""
    if isinstance(s, list):
        return [_sanitize_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k in ("additionalProperties", "$schema", "strict", "default", "examples", "title", "$id"):
            continue
        if k == "type" and isinstance(v, list):
            non_null = [x for x in v if x != "null"]
            out["type"] = non_null[0] if non_null else "string"
            if "null" in v:
                out["nullable"] = True
            continue
        out[k] = _sanitize_schema(v)
    return out


def build_tools(req: dict, sanitized: bool) -> list[dict]:
    decls = []
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            inner = tool.get("tools") or []
        else:
            inner = [tool]
        for t in inner:
            if t.get("type") != "function":
                log(f"dropping unsupported tool type {t.get('type')!r}")
                continue
            d = {"name": t["name"], "description": t.get("description", "")}
            params = t.get("parameters") or {"type": "object", "properties": {}}
            if sanitized:
                d["parameters"] = _sanitize_schema(params)
            else:
                d["parametersJsonSchema"] = params
            decls.append(d)
    return [{"functionDeclarations": decls}] if decls else []


def build_gemini_request(req: dict, sanitized: bool) -> dict:
    system, contents = build_contents(req)
    body: dict = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
    tools = build_tools(req, sanitized)
    if tools:
        body["tools"] = tools
        choice = req.get("tool_choice")
        mode = None
        if choice == "required":
            mode = "ANY"
        elif choice == "none":
            mode = "NONE"
        if mode:
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
    gen: dict = {}
    reasoning = req.get("reasoning") or {}
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    model = req.get("model", "")
    thinking: dict = {"includeThoughts": True}
    if model.startswith("gemini-3"):
        level = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}.get(effort or "", None)
        if level:
            thinking["thinkingLevel"] = level
    elif effort == "minimal":
        thinking["thinkingBudget"] = 0
    gen["thinkingConfig"] = thinking
    body["generationConfig"] = gen
    return body


# --------------------------------------------------------------- response side

class SseWriter:
    def __init__(self, wfile):
        self.w = wfile
        self.seq = 0

    def emit(self, kind: str, **fields) -> None:
        self.seq += 1
        payload = {"type": kind, "sequence_number": self.seq, **fields}
        data = json.dumps(payload, ensure_ascii=False)
        self.w.write(f"event: {kind}\ndata: {data}\n\n".encode())
        self.w.flush()


class Translator:
    """Turn Gemini stream chunks into Responses items on the fly."""

    def __init__(self, sse: SseWriter, response_id: str):
        self.sse = sse
        self.rid = response_id
        self.output_index = 0
        self.open: str | None = None  # "reasoning" | "message"
        self.item_id = ""
        self.buf: list[str] = []
        self.pending_sig: str | None = None  # signature seen on a text part
        self.usage = {}

    def _open(self, kind: str) -> None:
        if self.open == kind:
            return
        self._close()
        self.open = kind
        self.buf = []
        if kind == "reasoning":
            self.item_id = f"rs_{uuid.uuid4().hex}"
            item = {"type": "reasoning", "id": self.item_id, "summary": []}
            self.sse.emit("response.output_item.added", output_index=self.output_index, item=item)
            self.sse.emit("response.reasoning_summary_part.added", item_id=self.item_id,
                          output_index=self.output_index, summary_index=0,
                          part={"type": "summary_text", "text": ""})
        else:
            self.item_id = f"msg_{uuid.uuid4().hex}"
            item = {"type": "message", "id": self.item_id, "role": "assistant", "status": "in_progress", "content": []}
            self.sse.emit("response.output_item.added", output_index=self.output_index, item=item)

    def _close(self) -> None:
        if not self.open:
            return
        text = "".join(self.buf)
        if self.open == "reasoning":
            item = {"type": "reasoning", "id": self.item_id,
                    "summary": [{"type": "summary_text", "text": text}]}
            self.sse.emit("response.reasoning_summary_text.done", item_id=self.item_id,
                          output_index=self.output_index, summary_index=0, text=text)
        else:
            item = {"type": "message", "id": self.item_id, "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}
        self.sse.emit("response.output_item.done", output_index=self.output_index, item=item)
        self.output_index += 1
        self.open = None
        if self.pending_sig and self.open is None:
            # signature belonged to the text part just closed -> attach "prev"
            self._emit_sig_item(self.pending_sig, "prev")
            self.pending_sig = None

    def _emit_sig_item(self, sig: str, attach: str) -> None:
        item = {"type": "reasoning", "id": f"rs_{uuid.uuid4().hex}", "summary": [],
                "encrypted_content": _sig_encode(sig, attach)}
        self.sse.emit("response.output_item.added", output_index=self.output_index, item=item)
        self.sse.emit("response.output_item.done", output_index=self.output_index, item=item)
        self.output_index += 1

    def part(self, p: dict) -> None:
        sig = p.get("thoughtSignature")
        if "functionCall" in p:
            self._close()
            if sig:
                self._emit_sig_item(sig, "next")
            fc = p["functionCall"]
            item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": f"call_{uuid.uuid4().hex[:24]}", "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False), "status": "completed"}
            self.sse.emit("response.output_item.added", output_index=self.output_index, item=item)
            self.sse.emit("response.output_item.done", output_index=self.output_index, item=item)
            self.output_index += 1
            return
        text = p.get("text")
        if text is None or text == "":
            if sig:
                self.pending_sig = sig
            return
        if p.get("thought"):
            self._open("reasoning")
            self.buf.append(text)
            self.sse.emit("response.reasoning_summary_text.delta", item_id=self.item_id,
                          output_index=self.output_index, summary_index=0, delta=text)
        else:
            self._open("message")
            self.buf.append(text)
            self.sse.emit("response.output_text.delta", item_id=self.item_id,
                          output_index=self.output_index, content_index=0, delta=text)
        if sig:
            self.pending_sig = sig

    def finish(self, usage_meta: dict) -> None:
        self._close()
        prompt = usage_meta.get("promptTokenCount", 0)
        cand = usage_meta.get("candidatesTokenCount", 0)
        thoughts = usage_meta.get("thoughtsTokenCount", 0)
        cached = usage_meta.get("cachedContentTokenCount", 0)
        usage = {"input_tokens": prompt, "output_tokens": cand + thoughts,
                 "total_tokens": usage_meta.get("totalTokenCount", prompt + cand + thoughts),
                 "input_tokens_details": {"cached_tokens": cached},
                 "output_tokens_details": {"reasoning_tokens": thoughts}}
        self.sse.emit("response.completed", response={"id": self.rid, "object": "response",
                                                      "status": "completed", "usage": usage})


def gemini_stream(model: str, body: dict, key: str):
    url = f"{GEMINI_BASE}/models/{model}:streamGenerateContent?alt=sse"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    resp = urllib.request.urlopen(req, timeout=600)
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                yield json.loads(payload)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default log
        log(f"{self.address_string()} {fmt % args}")

    def do_GET(self):  # health + the model catalog Codex fetches at startup
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/health", "/v1/health"):
            body = b'{"ok":true}'
        elif path in ("/models", "/v1/models"):
            body = b'{"models":[]}'  # empty => Codex keeps its bundled catalog
        else:
            body = None
        if body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") not in ("/v1/responses", "/responses"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        rid = f"resp_{uuid.uuid4().hex}"
        if DUMP_DIR:
            Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
            (Path(DUMP_DIR) / f"{time.strftime('%Y%m%d-%H%M%S')}-{rid[-8:]}.request.json").write_text(
                json.dumps(req, ensure_ascii=False, indent=1))
        model = req.get("model", "gemini-3.1-pro-preview")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        sse = SseWriter(self.wfile)
        sse.emit("response.created", response={"id": rid, "object": "response", "status": "in_progress"})
        tr = Translator(sse, rid)
        n_items = len(req.get("input") or [])
        n_tools = len(req.get("tools") or [])
        log(f"-> {model} input_items={n_items} tools={n_tools}")
        usage_meta: dict = {}
        try:
            key = api_key()
            for attempt, sanitized in enumerate((False, True)):
                body = build_gemini_request(req, sanitized)
                try:
                    for chunk in gemini_stream(model, body, key):
                        if "error" in chunk:
                            raise RuntimeError(json.dumps(chunk["error"]))
                        for cand in chunk.get("candidates") or []:
                            for p in (cand.get("content") or {}).get("parts") or []:
                                tr.part(p)
                            fr = cand.get("finishReason")
                            if fr and fr not in ("STOP", "MAX_TOKENS"):
                                log(f"finishReason={fr}")
                        if chunk.get("usageMetadata"):
                            usage_meta = chunk["usageMetadata"]
                    break
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", "replace")
                    if e.code == 400 and attempt == 0 and "parameters" in detail and tr.output_index == 0:
                        log("400 on schema, retrying with sanitized parameters")
                        continue
                    raise RuntimeError(f"gemini {e.code}: {detail[:800]}") from None
            tr.finish(usage_meta)
            log(f"<- done items={tr.output_index} usage={usage_meta.get('totalTokenCount')}")
        except Exception as e:  # noqa: BLE001 — surface to Codex as response.failed
            log(f"!! {e}")
            sse.emit("response.failed", response={"id": rid, "status": "failed",
                                                  "error": {"code": "shim_error", "message": str(e)[:2000]}})
        finally:
            try:
                self.wfile.flush()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8397)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log(f"responses->gemini shim on http://{a.host}:{a.port}/v1/responses")
    srv.serve_forever()


if __name__ == "__main__":
    main()
