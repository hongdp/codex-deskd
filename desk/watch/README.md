# desk/watch — the program keeps watch, the agent is woken

Paper-only port of parlay's watcher → wake loop onto codex-deskd.

- `levels.toml` — guarded price levels (stop / entry / target, side, note).
- `watch.py` — polls quotes (yfinance), detects breaches with parlay's 1%-step
  sentinel cadence, and wakes the persistent **trader** thread through the
  Codex app-server (Python SDK, `desk/bin/codex` with `CODEX_HOME=desk`).
- `state.json` — thread ids per role + fired keys (git-ignored).
- `log/watch.log`, `log/proposals.md` — the wake ledger (git-ignored).

Safety: the wake turn runs `sandbox=read-only`, `approval_policy=never`, the
Robinhood MCP is disabled desk-wide, and the standing rules in every wake say
"proposal only". Nothing in this directory can reach a broker.

```bash
.venv/bin/python desk/watch/watch.py --fire nvda-stop --dry-run   # message only
.venv/bin/python desk/watch/watch.py --fire nvda-stop             # real wake, simulated price
.venv/bin/python desk/watch/watch.py --loop --interval 300        # keep watch
desk/bin/codex-desk resume <thread id>                            # watch the trader live
```
