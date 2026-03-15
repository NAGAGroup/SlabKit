# Subtask 08 — Cleanup and Close

## Delegation
- **Agent:** HeadWrench directly
- **Model tier:** N/A — git operations; no subagent needed
- **Reason:** File deletion and final session commit are HeadWrench-owned operations (git ops). No writing or analysis needed.

---

## Objective

Delete `docs/design/slabkit-v0.1-design.md` and the now-empty `docs/design/` directory. Write the session closing note summarizing what was accomplished. Make the final session commit. Update `index.md` and `spec.json` to reflect session completion.

---

## Todolist

### 1. Delete design doc and empty directory
- [ ] `git rm docs/design/slabkit-v0.1-design.md`
- [ ] Remove `docs/design/` directory if now empty: `rmdir docs/design`

### 2. Verify docs/ state
- [ ] Confirm `docs/` contains exactly: `getting-started.md`, `concepts.md`, `api-reference.md`, `patterns.md`, `anti-patterns.md`
- [ ] Confirm `docs/design/` is gone

### 3. Write closing session note
- [ ] Write `.opencode/sessions/rewrite-user-docs/notes/session-close.md` covering:
  - What was accomplished: 5 new docs/ pages written, README condensed, design doc deleted
  - Any significant findings from the session (stale content found in design doc, scope changes, etc.)
  - Follow-up recommendations for future sessions (v0.2 changes to document, etc.)

### 4. Update index.md and spec.json
- [ ] Mark all subtasks as `completed` in `index.md`
- [ ] Update `index.md` session status to `completed`
- [ ] Update `spec.json` status to `"complete"`, all subtask statuses to `"completed"`

### 5. Final session commit
- [ ] `git add -A && git commit -m "docs: complete session rewrite-user-docs — add docs/ pages, condense README, delete design doc"`

---

## Scope
- **Edit:** `index.md`, `spec.json`, `docs/design/slabkit-v0.1-design.md` (delete)
- **Read:** `docs/` directory listing (verify state)
- **Write:** `.opencode/sessions/rewrite-user-docs/notes/session-close.md`
- **Excluded:** all content pages (already committed in prior subtasks)

---

## Patterns
```
✅ GOOD — Verify docs/ state before committing; confirm no unexpected files or missing files
❌ BAD  — Delete design doc without confirming that concepts.md absorbed the relevant content
✅ GOOD — Final commit message summarizes the full session outcome, not just this subtask
```

---

## Constraints
- **Design doc content has already been absorbed** into `docs/concepts.md` (staleness-checked in subtask 04) — safe to delete
- **No review gate** — this is cleanup only; no user approval needed beyond what was given for each doc page
- **Final commit is not a WIP commit** — use `docs:` prefix per session convention

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
