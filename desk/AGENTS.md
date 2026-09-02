# codex-deskd — the desk contract

Three seats collaborate on a signal-driven, paper-first stock desk.

1. **Programs only propose; the trader reviews every order.** No code path,
   tool, or agent reaches the broker without the trader's per-order review.
2. **The supervisor is a human, not an agent.** Nobody claims that role, reads
   control credentials, or fakes supervisor decisions.
3. **Risk policy is monotonic.** Tighten freely with a trace; loosenings go to
   the supervisor BEFORE any change lands.
4. **Verify against the governing artifact**, not a status row; an empty
   result must prove its emptiness; a number contradicting both book and
   broker is the wrong number.
5. **Journal every turn**; lessons that generalize go to KNOWLEDGE.md via the
   knowledge transaction, never by direct edit under line-cap pressure.

Roles are defined in `agents/*.toml`; shell gates in `rules/`; the tool wall
in `hooks/`. These files are the contract — edit them only through the
supervisor.
