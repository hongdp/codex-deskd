#!/usr/bin/env python3
"""PreToolUse wall for codex-deskd (ported from parlay's skill_guard_hook.py).

Blocks (a) any role editing the desk contract, and (b) brokerage WRITE tools
from any role other than the trader. Codex feeds the payload on stdin with
`tool_name`, `tool_input`, `agent_type` (the spawned role, absent for the root
thread) and `cwd`; exit 2 + stderr reason is a hard block on both harnesses.
"""
import json, os, re, sys

# Any mention of a contract path, whether absolute, relative, quoted or after a
# shell operator: the char before `desk/` must not be part of another name.
CONTRACT = re.compile(r"(^|[^\w.-])desk/(AGENTS\.md|config\.toml|agents/|rules/|hooks/)")
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

# Read-only shell verbs (supervisor ruling 2026-09-06: the contract is readable,
# not writable). A command mentioning a contract path passes only if EVERY
# pipeline segment starts with one of these and nothing in it can write.
READ_VERBS = {"cat", "head", "tail", "less", "more", "grep", "egrep", "rg", "wc",
              "ls", "stat", "file", "diff", "find", "awk", "sort", "uniq", "cut", "tr"}
GIT_READ = {"show", "diff", "log", "status", "blame", "grep", "cat-file", "ls-files"}
WRITE_HINTS = re.compile(r"(>|\btee\b|\bmv\b|\bcp\b|\brm\b|\btouch\b|\bchmod\b|\bchown\b|"
                         r"\btruncate\b|\bdd\b|\binstall\b|\bln\b|\bpatch\b|\bxargs\b|\beval\b|"
                         r"\bexec\b|\bpython3?\b|\bperl\b|\bsh\b|\bbash\b|\bzsh\b|\benv\b|\bsudo\b)")

def command_text(inp):
    c = inp.get("command", inp.get("cmd", ""))
    if isinstance(c, list):
        # ["/bin/bash", "-lc", "<script>"] -> the script itself
        if len(c) >= 3 and c[1] in ("-lc", "-c"):
            return str(c[-1])
        return " ".join(str(x) for x in c)
    return str(c)

def read_only(cmd):
    if WRITE_HINTS.search(cmd):
        return False
    for seg in re.split(r"\|\||&&|[;|\n]", cmd):
        words = seg.strip().split()
        if not words:
            continue
        verb = words[0].rsplit("/", 1)[-1]
        if verb == "git":
            if len(words) < 2 or words[1] not in GIT_READ:
                return False
        elif verb == "sed":
            if "-n" not in words or any(w.startswith("-i") for w in words):
                return False
        elif verb not in READ_VERBS:
            return False
    return True

# (a) contract edits — any edit tool touching a protected path, or any shell
# command that mentions one and is not provably read-only
blob = json.dumps(inp)
if CONTRACT.search(blob):
    t = tool.lower()
    if t in {"apply_patch", "write", "edit"}:
        deny(f"{role} may not edit the desk contract; propose a diff instead")
    if t in {"shell", "bash", "exec_command"} and not read_only(command_text(inp)):
        deny(f"{role} may not edit the desk contract (read-only access is allowed); propose a diff instead")

# (b) brokerage writes — trader only, and never from the root thread
short = tool.rsplit("__", 1)[-1]
if short in ORDER_TOOLS and role != "trader":
    deny(f"{role} has no order surface ({short} is trader-only)")

# Allow = say nothing. Codex accepts permissionDecision:allow only together with
# updatedInput; a bare allow is reported as "unsupported" and fails open.
sys.exit(0)
