#!/usr/bin/env python3
"""Price watch for the codex-deskd desk: the program keeps watch, the agent is woken.

Ported from parlay's watcher/wake pattern: a plain process polls quotes, decides
when a guarded level is breached (parlay's 1%-step sentinel cadence), and then
wakes a persistent TRADER thread through the Codex app-server (Python SDK).
The trader answers with a PROPOSAL; nothing here can reach a broker — the
Robinhood MCP stays disabled and the wake turn runs in a read-only sandbox.

Usage (from the repo root, with .venv):
  .venv/bin/python desk/watch/watch.py --once            # poll every level once
  .venv/bin/python desk/watch/watch.py --loop            # poll forever (default 300s)
  .venv/bin/python desk/watch/watch.py --fire nvda-stop  # simulate a breach now
  add --dry-run to print the wake message without waking anyone.

State (thread ids, fired keys) lives in desk/watch/state.json; the ledger of
wakes and proposals in desk/watch/log/. Watch a woken thread live with
`desk/bin/codex-desk resume <thread id>` or /agents in a daemon-attached TUI.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESK = HERE.parent
REPO = DESK.parent
BIN = DESK / "bin" / "codex"
STATE = HERE / "state.json"
LOG_DIR = HERE / "log"
LEVELS = HERE / "levels.toml"
STEP = 0.01  # dedup step: distance past the level, in 1% of the level

STANDING_RULES = """Standing rules for this turn:
- Paper desk. There is no broker connection; do not attempt to place, preview, cancel or modify any order, and do not look for a way to.
- Answer with a PROPOSAL only: hold / reduce / exit / add, size versus the NAV cap, and the written review you would file before any order. If nothing should change, say NO ACTION and why.
- Do not spawn agents. Keep it under 200 words. End with one line: `PROPOSAL: <verb> <ticker> <size>` or `PROPOSAL: NO ACTION`."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with (LOG_DIR / "watch.log").open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"threads": {}, "fired": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True))


def load_levels() -> list[dict]:
    return tomllib.loads(LEVELS.read_text()).get("level", [])


def load_role(role: str) -> dict:
    return tomllib.loads((DESK / "agents" / f"{role}.toml").read_text())


# ------------------------------------------------------------------ quotes

def last_price(ticker: str) -> float | None:
    try:
        import yfinance as yf  # noqa: PLC0415 — optional at import time

        return float(yf.Ticker(ticker).fast_info.last_price)
    except Exception as e:  # noqa: BLE001 — a quote failure is a log line, not a crash
        log(f"quote {ticker}: {e}")
        return None


def breach(level: dict, px: float) -> str | None:
    """Return the dedup key when `px` is past the level, else None."""
    lv = float(level["price"])
    past = (px < lv) if level["side"] == "below" else (px > lv)
    if not past:
        return None
    step = int(abs(px - lv) / lv / STEP)
    return f"{level['id']}:{level['side']}:{step}"


# ------------------------------------------------------------------- wake

def wake_message(level: dict, px: float, simulated: bool) -> str:
    verb = "BELOW" if level["side"] == "below" else "ABOVE"
    sim = " (SIMULATED — watch self-test, treat the price as hypothetical)" if simulated else ""
    return (
        "Message Type: WAKE\n"
        "Source: price-watch\n"
        f"Time: {now()}\n"
        f"Ticker: {level['ticker']}\n"
        f"Event: last {px:.2f} is {verb} the {level['kind']} level {float(level['price']):.2f}{sim}\n"
        f"Level note: {level.get('note', '')}\n\n"
        f"{STANDING_RULES}\n"
    )


def wake(role: str, message: str, state: dict) -> tuple[str, str]:
    """Run one turn on the role's persistent thread; returns (thread_id, reply)."""
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox  # noqa: PLC0415

    role_cfg = load_role(role)
    config = CodexConfig(
        codex_bin=str(BIN),
        cwd=str(REPO),
        env={"CODEX_HOME": str(DESK)},
        client_name="desk_watch",
        client_title="codex-deskd price watch",
    )
    thread_id = state["threads"].get(role)
    with Codex(config=config) as codex:
        if thread_id:
            thread = codex.thread_resume(thread_id, approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only)
        else:
            thread = codex.thread_start(
                cwd=str(REPO),
                model=role_cfg.get("model"),
                developer_instructions=role_cfg.get("developer_instructions", ""),
                sandbox=Sandbox.read_only,
                approval_mode=ApprovalMode.deny_all,  # never blocks on a human; denies escalation
                config={"model_reasoning_effort": "medium", "approval_policy": "never"},
            )
            thread_id = getattr(thread, "id", None) or getattr(thread, "thread_id")
            state["threads"][role] = thread_id
            save_state(state)
            log(f"{role}: started persistent thread {thread_id}")
        result = thread.run(message)
    reply = result.final_response or ""
    usage = getattr(result, "usage", None)
    log(f"{role}: turn done on {thread_id} usage={usage}")
    return thread_id, reply


def record(level: dict, px: float, thread_id: str, reply: str, simulated: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "SIMULATED " if simulated else ""
    with (LOG_DIR / "proposals.md").open("a") as f:
        f.write(
            f"\n## {now()} — {tag}{level['ticker']} {level['kind']} {level['side']} "
            f"{float(level['price']):.2f} (last {px:.2f})\n\n"
            f"thread: `{thread_id}`\n\n{reply.strip()}\n"
        )


def handle(level: dict, px: float, state: dict, *, simulated: bool, dry_run: bool) -> None:
    key = breach(level, px)
    if key is None:
        return
    if not simulated and key in state["fired"]:
        return
    msg = wake_message(level, px, simulated)
    if dry_run:
        print("--- would wake", level.get("role", "trader"), "with:\n" + msg)
        return
    log(f"BREACH {key}: {level['ticker']} last {px:.2f} {level['side']} {level['price']}")
    thread_id, reply = wake(level.get("role", "trader"), msg, state)
    record(level, px, thread_id, reply, simulated)
    if not simulated:
        state["fired"][key] = now()
        save_state(state)
    print(f"\n=== {level['ticker']} proposal (thread {thread_id}):\n{reply.strip()}\n")
    print(f"watch it: desk/bin/codex-desk resume {thread_id}")


def poll(state: dict, *, dry_run: bool) -> None:
    levels = load_levels()
    quotes: dict[str, float | None] = {}
    for lv in levels:
        t = lv["ticker"]
        if t not in quotes:
            quotes[t] = last_price(t)
    for lv in levels:
        px = quotes.get(lv["ticker"])
        if px is None:
            continue
        handle(lv, px, state, simulated=False, dry_run=dry_run)
    summary = ", ".join(f"{t} {px:.2f}" if px else f"{t} n/a" for t, px in quotes.items())
    log(f"poll: {summary}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="poll every level once")
    ap.add_argument("--loop", action="store_true", help="poll forever")
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls in --loop")
    ap.add_argument("--fire", metavar="LEVEL_ID", help="simulate a breach of this level now")
    ap.add_argument("--px", type=float, help="price to use with --fire (default: 1%% past the level)")
    ap.add_argument("--dry-run", action="store_true", help="print the wake message; wake nobody")
    a = ap.parse_args()
    state = load_state()
    if a.fire:
        lv = next((x for x in load_levels() if x["id"] == a.fire), None)
        if lv is None:
            print(f"no level with id {a.fire!r}", file=sys.stderr)
            return 2
        px = a.px if a.px is not None else float(lv["price"]) * (0.99 if lv["side"] == "below" else 1.01)
        handle(lv, px, state, simulated=True, dry_run=a.dry_run)
        return 0
    if a.once:
        poll(state, dry_run=a.dry_run)
        return 0
    if a.loop:
        log(f"watch loop every {a.interval}s over {LEVELS}")
        while True:
            poll(state, dry_run=a.dry_run)
            time.sleep(a.interval)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
