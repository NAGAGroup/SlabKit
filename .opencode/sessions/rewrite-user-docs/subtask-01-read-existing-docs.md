# Subtask 01 — Read Existing Docs

## Delegation
- **Agent:** @ContextScout
- **Model tier:** fast — pure read-only exploration, clear inputs and outputs
- **Reason:** This is an unambiguous analysis task: read two existing files, compare sections against current headers, produce a structured mapping. No judgment or multi-system reasoning needed.

---

## Objective

Read `README.md` and `docs/design/slabkit-v0.1-design.md` in full. Produce:
1. A **content mapping** — for each section of the existing README, note whether it should be: kept in the new lean README, moved to a specific `docs/` page, or dropped entirely.
2. A **staleness check** — compare the design doc's architecture diagram (§2), invariants list (§2), key design decisions (§3), and thread-safety section (§3 "Thread safety") against the current state of `include/slabkit/context.hpp`, `include/slabkit/array.hpp`, and `include/slabkit/memory.hpp`. Flag anything that is outdated or no longer accurate.
3. A **content absorption plan** — for each relevant piece of the design doc, identify which new `docs/` page it belongs in (e.g., the architecture diagram → `docs/concepts.md`; the thread-safety note → `docs/concepts.md`).

---

## Todolist

### 1. Read existing README.md
- [ ] Read `README.md` in full (all 619 lines)
- [ ] For each major section, classify as: keep in README / move to docs page / drop

### 2. Read design doc
- [ ] Read `docs/design/slabkit-v0.1-design.md` in full
- [ ] Identify sections with user-relevant content (architecture diagram, invariants, thread safety, v0.2 roadmap)
- [ ] Note sections that are purely internal (file state tracking, fold-expression details, allocator internals)

### 3. Staleness check against current headers
- [ ] Read `include/slabkit/context.hpp` — check: does ContextBuilder still work as described? Does the lazy pool init / enable_shared_from_this pattern still hold?
- [ ] Read `include/slabkit/array.hpp` — check: are DeviceArray/DeviceView/DeviceSubView/MirroredArray still as described?
- [ ] Read `include/slabkit/memory.hpp` — check: DeviceUSMPoolDeleter and malloc_device still as described?
- [ ] Flag any stale or incorrect content in the design doc

### 4. Produce structured output
- [ ] Write a content mapping table: README section → disposition (keep/move-to-page/drop)
- [ ] Write a design doc absorption table: design doc section → target docs/ page (or drop)
- [ ] List any stale content found and what the correct current state is

### 5. [🚫 GATE] Content plan approval
- [ ] HeadWrench surfaces the content mapping and absorption plan to the user
- [ ] User explicitly approves the plan before any writing begins

---

## Scope
- **Edit:** nothing — read-only subtask
- **Read:** `README.md`, `docs/design/slabkit-v0.1-design.md`, `include/slabkit/context.hpp`, `include/slabkit/array.hpp`, `include/slabkit/memory.hpp`
- **Write:** nothing (output is returned to HeadWrench as a report)
- **Excluded:** test files, CMakeLists, pixi.toml, scratch.hpp

---

## Patterns
```
✅ GOOD — Read the actual header files to verify design doc claims, not just the design doc itself
❌ BAD  — Assume design doc is accurate without cross-checking against current headers
```

---

## Constraints
- This is a read-only subtask — no files should be modified or created
- The staleness check is mandatory: content from the design doc may be stale after the registry-removal refactor
- The gate at step 5 must fire: do not proceed to subtask 02 until the user explicitly approves the content plan

---

## [🚫 GATE] Content Plan Approval
HeadWrench shows the user:
- The README content mapping (section → disposition)
- The design doc absorption plan (section → target page or drop)
- Any stale content found and the correct current state

User must explicitly approve the plan before subtask 02 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
