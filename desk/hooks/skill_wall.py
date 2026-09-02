#!/usr/bin/env python3
"""PreToolUse wall for codex-deskd (ported from parlay's skill_guard_hook.py).

Blocks (a) any role editing the desk contract, and (b) brokerage WRITE tools
from any role other than the trader. Codex feeds the payload on stdin with
`tool_name`, `tool_input`, `agent_type` (the spawned role, absent for the root
thread) and `cwd`; exit 2 + stderr reason is a hard block on both harnesses.
"""
import json, os, re, sys

CONTRACT = re.compile(r"(^|/)desk/(AGENTS\.md|config\.toml|agents/|rules/|hooks/)")
ORDER_TOOLS = {"place_equity_order", "place_option_order", "place_crypto_order",
               "cancel_equity_order", "cancel_option_order", "cancel_crypto_order",
               "exercise_option", "review_equity_order", "review_option_order"}

def deny(reason):
    sys.stderr.write(f"skill_wall: {reason}\n")
    sys.exit(2)

payload = json.load(sys.stdin)
tool = payload.get("tool_name", "")
inp = payload.get("tool_input") or {}
role = payload.get("agent_type") or "root"

# (a) contract edits — check every string-valued input for a protected path
blob = json.dumps(inp)
if tool.lower() in {"apply_patch", "write", "edit", "shell", "bash", "exec_command"} \
        and CONTRACT.search(blob):
    deny(f"{role} may not edit the desk contract; propose a diff instead")

# (b) brokerage writes — trader only, and never from the root thread
short = tool.rsplit("__", 1)[-1]
if short in ORDER_TOOLS and role != "trader":
    deny(f"{role} has no order surface ({short} is trader-only)")

print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                         "permissionDecision": "allow"}}))
