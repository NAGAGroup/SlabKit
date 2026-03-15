# Subtask 02 — Rewrite README

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — requires judgment about what to keep vs. cut, tone calibration for a lean entry point, and accurate link strategy; explanatory/conceptual style
- **Reason:** Documentation writing with judgment calls about content scope and tone. Needs to write a README that reads as a genuine overview while being genuinely lean.

---

## Objective

Rewrite `README.md` from its current 619-line state into a lean entry point of approximately 100–150 lines. The new README should tell a new user what SlabKit is, how to get it, and how to start using it — then direct them to the `docs/` pages for everything else. No inline API reference. No anti-pattern catalogue. Just: what, why, how to start, where to learn more.

---

## Todolist

### 1. Draft README.md
- [ ] Write the new README.md using the content mapping from subtask 01 (provided in session notes)
- [ ] Include: project tagline + 1-paragraph description, quick install (CMake FetchContent), minimal quick-start snippet, links to all 5 docs/ pages with 1-line descriptions
- [ ] Tone: direct and confident; explain the "why SlabKit" briefly without over-selling

### 2. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 3. Commit
- [ ] `git add README.md && git commit -m "docs: rewrite README.md as lean entry point"`

### 4. [🚫 GATE] README approval
- [ ] User has explicitly approved the README before this gate is marked complete

---

## Scope
- **Edit:** `README.md` — rewrite in place
- **Read:** subtask-01 session notes (content mapping), existing `README.md` for reference
- **Write:** nothing new
- **Excluded:** all `docs/` pages (not written yet), all `include/` headers (read-only reference only)

---

## Patterns
```
✅ GOOD — "SlabKit is a header-only C++20 library for SYCL USM memory management. It gives you pooled device allocation, RAII device arrays, and a safe view system — without sycl::buffer."
❌ BAD  — Re-including the full API reference or anti-pattern catalogue in the README
✅ GOOD — Link to docs/ pages: "See [Ownership Model](docs/concepts.md) for how DeviceArray and views interact."
❌ BAD  — Inline deep explanations of ownership semantics in the README itself
```

---

## Constraints
- **No AllocatorRegistry** — not mentioned anywhere; it's gone
- **No device-backed overloads** — the `make_device_array(range, dev)` convenience form is not documented
- **Lean**: target 100–150 lines. If it's growing past 200, something is wrong — content belongs in a docs/ page
- **Links must use relative paths** to `docs/` pages (e.g., `docs/getting-started.md`)
- **Audience:** Experienced C++/SYCL developers — no SYCL background explanation needed
- **Tone:** Explanatory and conceptual — the "why" matters alongside the "what"

---

## [🚫 GATE] README Approval
HeadWrench shows the full revised README.md to the user. User must explicitly approve (or provide feedback for revision) before subtask 03 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
