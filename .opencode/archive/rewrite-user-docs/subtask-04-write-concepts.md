# Subtask 04 — Write docs/concepts.md

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — requires accurate conceptual description of ownership model, RAII semantics, builder pattern, and view safety; must accurately reflect current code state after staleness verification
- **Reason:** Complex conceptual material with multiple interacting systems (Context, pools, DeviceArray, views, weak_ptr lifetime anchors). Needs judgment about depth and accuracy.

---

## Objective

Write `docs/concepts.md` — the deep conceptual guide to SlabKit's design. This page is for users who want to understand *why* the API is shaped the way it is, not just how to use it. It covers: the ownership hierarchy, why Context is heap-allocated, how DeviceArray is move-only and why, what DeviceView's trivial-copyability means for kernel safety, the weak_ptr lifetime anchor pattern, MirroredArray semantics, and thread-safety guarantees.

Before delegating to DocWriter, HeadWrench must verify the design doc's staleness by reading the current headers (context.hpp, array.hpp, memory.hpp) and cross-checking the invariants from `docs/design/slabkit-v0.1-design.md §2 and §3`. Any stale content is corrected before passing to DocWriter.

---

## Todolist

### 1. Staleness check (HeadWrench — before delegating to DocWriter)
- [ ] Read `include/slabkit/context.hpp` in full
- [ ] Read `include/slabkit/array.hpp` — focus on DeviceArray, DeviceView, DeviceSubView, MirroredArray declarations
- [ ] Read `include/slabkit/memory.hpp` — focus on DeviceUSMPoolDeleter
- [ ] Compare against design doc §2 (architecture diagram + invariants) and §3 (ContextBuilder decision, USM deleter decision, DeviceArray move-only decision, thread safety section)
- [ ] Note any stale content; prepare corrected facts to pass to DocWriter

### 2. Draft docs/concepts.md
- [ ] Write the page with these sections:
  - **The ownership hierarchy** — architecture diagram (adapted from design doc §2), prose explanation
  - **Context** — why it must be a shared_ptr; ContextBuilder auto-groups devices by platform; lazy pool initialization; no global state
  - **DeviceArray** — RAII, move-only, unique_ptr backing; what happens when it's destroyed (pool return via weak_ptr deleter)
  - **Views: DeviceView and DeviceSubView** — trivially copyable; kernel-safe capture by value; contiguous vs. strided; why they don't extend array lifetime (and what that means for user responsibility)
  - **MirroredArray** — paired device+host; move-only; clone() semantics; when to use it vs. manual arrays
  - **Scratch memory** — DeviceScratchBuffer as temporary workspace; allocate once, reuse via DeviceScratchView
  - **Thread safety** — current v0.1 contract: single-threaded allocation per device; Context construction not thread-safe; what is safe once constructed
- [ ] Absorb: architecture diagram from design doc §2; invariants list from design doc §2; thread-safety section from design doc §3
- [ ] Tone: conceptual and explanatory — explain the design rationale, not just the facts

### 3. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 4. Commit
- [ ] `git add docs/concepts.md && git commit -m "docs: add concepts.md"`

### 5. [🚫 GATE] Concepts approval
- [ ] User has explicitly approved this page before the gate is marked complete

---

## Scope
- **Edit:** nothing existing
- **Read:** `include/slabkit/context.hpp`, `include/slabkit/array.hpp`, `include/slabkit/memory.hpp`, `include/slabkit/scratch.hpp`, `docs/design/slabkit-v0.1-design.md` (for §2 and §3 content only)
- **Write:** `docs/concepts.md` (new file)
- **Excluded:** README (already done), getting-started.md (already done), test internals, CMake config, device-backed overloads, AllocatorRegistry

---

## Patterns
```
✅ GOOD — "DeviceArray is move-only. This is intentional: USM pointers represent exclusive ownership of a device allocation. Copying a DeviceArray would silently duplicate a pointer, not the data — which is almost never what you want."
❌ BAD  — "DeviceArray cannot be copied." (states fact without explaining why)
✅ GOOD — "The deleter holds a weak_ptr<Context>, not a shared_ptr. This means the array doesn't keep the Context alive — which is correct: if the Context is destroyed, its slab is freed and the weak_ptr lock fails silently."
❌ BAD  — Omitting the rationale for the weak_ptr choice (it's the key insight about lifetime safety)
✅ GOOD — Use the architecture diagram from the design doc (adapted) as a visual anchor for the ownership hierarchy discussion
❌ BAD  — Describe the ownership hierarchy in prose only without a diagram
```

---

## Constraints
- **Staleness check is mandatory** — HeadWrench reads current headers BEFORE delegating to DocWriter; any stale design doc content is corrected in the brief passed to DocWriter
- **No AllocatorRegistry** — not mentioned
- **No device-backed overloads**
- **Audience:** Experienced C++/SYCL developers; can assume familiarity with RAII, unique_ptr, weak_ptr semantics
- **Accuracy first** — if the design doc says X and the code says Y, use Y
- **Tone:** Explanatory and conceptual — users reading this page want to *understand* the design

---

## [🚫 GATE] Concepts Approval
HeadWrench shows the full `docs/concepts.md` draft to the user. User must explicitly approve (or provide feedback for revision) before subtask 05 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
