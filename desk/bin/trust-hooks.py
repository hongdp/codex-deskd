#!/usr/bin/env python3
"""Pin trust for every command hook in $CODEX_HOME/hooks.json.

Codex runs an unmanaged hook only if config.toml carries a `hooks.state`
entry whose `trusted_hash` equals the hook's normalized identity hash — and in
headless mode there is nobody to click "trust" in the TUI, so an untrusted
wall is silently skipped (fail-open). This reproduces codex-rs's
`hook_hash`: sha256 over the canonical (key-sorted, compact) JSON of
{event_name, matcher?, hooks:[normalized handler]}; the state key is
`<abs path of hooks.json>:<event_label>:<group>:<handler>`.
Verified 2026-09-02 against the running binary (PreToolUse wall enforced).
"""
import hashlib, json, pathlib, re, sys

home = pathlib.Path(__import__("os").environ.get("CODEX_HOME") or pathlib.Path(__file__).resolve().parents[1])
hooks_path = home / "hooks.json"; cfg_path = home / "config.toml"
hooks = json.loads(hooks_path.read_text())["hooks"]

def label(event):  # HookEventName -> snake_case key label
    return re.sub(r"(?<!^)(?=[A-Z])", "_", event).lower()

def canon(v):
    if isinstance(v, dict): return {k: canon(v[k]) for k in sorted(v)}
    if isinstance(v, list): return [canon(x) for x in v]
    return v

blocks = []
for event, groups in hooks.items():
    for gi, grp in enumerate(groups):
        for hi, h in enumerate(grp.get("hooks", [])):
            if h.get("type", "command") != "command":
                continue
            norm = {"type": "command", "command": h["command"],
                    "timeout": h.get("timeout", 600), "async": bool(h.get("async", False))}
            if h.get("statusMessage"): norm["statusMessage"] = h["statusMessage"]
            if h.get("additionalContextLimit") not in (None, 2500): norm["additionalContextLimit"] = h["additionalContextLimit"]
            identity = {"event_name": label(event), "hooks": [norm]}
            if grp.get("matcher"): identity["matcher"] = grp["matcher"]
            ser = json.dumps(canon(identity), separators=(",", ":"), ensure_ascii=False).encode()
            digest = "sha256:" + hashlib.sha256(ser).hexdigest()
            key = f"{hooks_path}:{label(event)}:{gi}:{hi}"
            blocks.append(f'[hooks.state."{key}"]\ntrusted_hash = "{digest}"\nenabled = true\n')
            print(f"{key}\n  {digest}")

s = cfg_path.read_text()
s = re.sub(r'\n\[hooks\.state\."[^"]+"\]\ntrusted_hash = "[^"]+"\nenabled = true\n', "\n", s).rstrip() + "\n"
cfg_path.write_text(s + "\n" + "\n".join(blocks))
print(f"wrote {len(blocks)} trust block(s) to {cfg_path}")
