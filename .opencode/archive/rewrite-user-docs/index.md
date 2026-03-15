# Session: rewrite-user-docs

**Goal:** Rewrite README.md as a lean entry point and write five user-facing documentation pages under `docs/` covering getting started, concepts, API reference, patterns, and anti-patterns.

---

## Done Criteria

- [ ] `README.md` is condensed to a lean entry point (~100–150 lines): what SlabKit is, quick install, quick start snippet, links to `docs/` pages
- [ ] `docs/getting-started.md` is written and user-approved
- [ ] `docs/concepts.md` is written and user-approved (ownership model, builder pattern, view safety, RAII semantics)
- [ ] `docs/api-reference.md` is written and user-approved (all public types and free functions)
- [ ] `docs/patterns.md` is written and user-approved (best-practice usage patterns with code examples)
- [ ] `docs/anti-patterns.md` is written and user-approved (footguns + diagnosis + correct fix)
- [ ] `docs/design/slabkit-v0.1-design.md` is deleted (after useful content absorbed into user docs)
- [ ] All pages committed individually after user approval

---

## Subtask Table

| # | Status | Description |
|---|--------|-------------|
| 01 | ✅ done | Read existing README + design doc; produce content mapping and staleness check — **@ContextScout / fast** |
| 02 | ✅ done | Rewrite README.md as lean entry point — **@DocWriter / standard** |
| 03 | ✅ done | Write docs/getting-started.md — **@DocWriter / standard** |
| 04 | ✅ done | Write docs/concepts.md (ownership model, RAII, builder, view safety) — **@DocWriter / standard** |
| 05 | ✅ done | Write docs/api-reference.md (all public types + free functions) — **@DocWriter / standard** |
| 06 | ✅ done | Write docs/patterns.md (best-practice patterns with code examples) — **@DocWriter / standard** |
| 07 | ✅ done | Write docs/anti-patterns.md (footguns + diagnosis + correct fix) — **@DocWriter / standard** |
| 08 | ✅ done | Delete design doc, final commit, session close — **HeadWrench** |

---

## Gates

Each documentation subtask (02–07) ends with a review gate. HeadWrench shows the draft to the user; the user provides written feedback; DocWriter revises; user approves. The gate fires when the user explicitly approves the page. No subtask transitions until approval is given.

### G1 — Content Plan Approval (after subtask 01)
Stop after subtask 01. Show the user the content mapping: what README sections are kept/cut, and which design doc content maps to which new docs/ page. Wait for explicit approval of the plan before writing anything.

### G2 — README Approval (after subtask 02)
Show the full revised README.md draft. Collect feedback, revise if needed. Wait for explicit approval before proceeding to getting-started.md.

### G3 — Getting Started Approval (after subtask 03)
Show the full docs/getting-started.md draft. Collect feedback, revise if needed. Wait for explicit approval.

### G4 — Concepts Approval (after subtask 04)
Show the full docs/concepts.md draft. Collect feedback, revise if needed. Wait for explicit approval.

### G5 — API Reference Approval (after subtask 05)
Show the full docs/api-reference.md draft. Collect feedback, revise if needed. Wait for explicit approval.

### G6 — Patterns Approval (after subtask 06)
Show the full docs/patterns.md draft. Collect feedback, revise if needed. Wait for explicit approval.

### G7 — Anti-Patterns Approval (after subtask 07)
Show the full docs/anti-patterns.md draft. Collect feedback, revise if needed. Wait for explicit approval.

---

## Current Focus

All subtasks complete. Session closed.

---

## Scope

**In scope:**
- `README.md` — rewrite as lean entry point
- `docs/getting-started.md` — new page
- `docs/concepts.md` — new page
- `docs/api-reference.md` — new page
- `docs/patterns.md` — new page
- `docs/anti-patterns.md` — new page
- `docs/design/slabkit-v0.1-design.md` — delete after content absorbed

**Out of scope:**
- `include/` headers — read-only reference, no changes
- `tests/` — read-only reference, no changes
- Any build, CMake, or CI configuration
- AllocatorRegistry — not mentioned anywhere in new docs
- Device-backed convenience overloads (`make_device_array(range, dev)`) — not documented
- `docs/` subdirectory structure beyond what's planned

---

## Patterns & Constraints

- **Target audience:** Experienced C++/SYCL developers who are new to SlabKit
- **Tone:** Explanatory and conceptual — explain the *why* alongside the *how*
- **API scope:** Context-based API only (no AllocatorRegistry, no device-backed overloads)
- **Legacy:** AllocatorRegistry is gone — not mentioned anywhere in new docs
- **Design doc staleness:** Verify design doc's architecture/invariants/thread-safety sections against current headers before absorbing into concepts.md
- **DocWriter step limit risk:** api-reference.md may be large — HW splits into two DocWriter delegations if needed
- **Review loop:** Every doc page goes through draft → show → feedback → revise before commit
- **Commit cadence:** One commit per finalized page (after user approval)
- **Git workflow:** Work on `main` directly; no feature branches
- **Checkpoint protocol:** Global protocol at `~/.config/opencode/protocols/checkpoint.md` (no session override)
