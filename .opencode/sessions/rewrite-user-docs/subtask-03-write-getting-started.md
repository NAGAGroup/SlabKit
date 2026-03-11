# Subtask 03 — Write docs/getting-started.md

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — first page a new user reads; must balance practical quick-start content with just enough conceptual grounding; tone calibration matters
- **Reason:** The getting-started page sets the tone for all subsequent docs. It needs judgment about what to include to get a user productive fast vs. what to defer to concepts/API pages.

---

## Objective

Write `docs/getting-started.md` — the first page a new user lands on after the README. This page should take a C++/SYCL developer from zero SlabKit knowledge to their first working code in ~10 minutes. It covers: adding SlabKit to a project, constructing a Context, allocating a DeviceArray, using views in a kernel, and doing a basic host↔device transfer. Conceptual explanations are brief but present — this page explains *enough* to make the code make sense, with links to `concepts.md` for deeper dives.

---

## Todolist

### 1. Draft docs/getting-started.md
- [ ] Write the page with these sections:
  - **Installation** — CMake FetchContent snippet, include paths
  - **Your first Context** — `ContextBuilder` usage with explanation of what it does
  - **Allocating a DeviceArray** — `make_device_array<T,N>` with context and device index
  - **Using views in a kernel** — capturing `DeviceView` by value in a SYCL kernel; why views not arrays
  - **Host↔device transfers** — `copy_to_device` / `copy_to_host` (async + event dependency); brief mention of `MirroredArray` for the paired-storage pattern
  - **What's next** — brief paragraph linking to concepts.md, api-reference.md, patterns.md
- [ ] Tone: conceptual and explanatory — don't just show code, explain why each step is the right approach

### 2. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 3. Commit
- [ ] `git add docs/getting-started.md && git commit -m "docs: add getting-started.md"`

### 4. [🚫 GATE] Getting started approval
- [ ] User has explicitly approved this page before the gate is marked complete

---

## Scope
- **Edit:** nothing existing
- **Read:** `include/slabkit/context.hpp`, `include/slabkit/array.hpp`, `include/slabkit/scratch.hpp`, `tests/test_context.cpp`, `tests/test_array.cpp` (for realistic examples)
- **Write:** `docs/getting-started.md` (new file)
- **Excluded:** README (already done), design doc internals, AllocatorRegistry, device-backed overloads

---

## Patterns
```
✅ GOOD — Show a minimal but complete working snippet; explain why ContextBuilder does what it does
❌ BAD  — Just paste code with no explanation of the "why"
✅ GOOD — "Views are trivially copyable so you can capture them by value in a SYCL kernel. The DeviceArray must outlive any queued work that uses the view."
❌ BAD  — "Use DeviceView to access data in kernels." (no explanation of the ownership implication)
✅ GOOD — Link to concepts.md for ownership model depth: "See [Ownership Model](concepts.md) for the full picture."
❌ BAD  — Inline the full ownership model explanation in the getting-started page
```

---

## Constraints
- **No AllocatorRegistry** — not mentioned
- **No device-backed overloads** — only context + device-index forms
- **Audience:** Experienced C++/SYCL developers — no SYCL background explanation; they know USM
- **Length:** Aim for 200–300 lines. If growing past 350, something is drifting into concepts territory — link out instead
- **Tone:** Explanatory and conceptual — explain the *why* alongside the *how*
- **Code examples must compile** — all snippets should use real SlabKit API signatures; check against current headers

---

## [🚫 GATE] Getting Started Approval
HeadWrench shows the full `docs/getting-started.md` draft to the user. User must explicitly approve (or provide feedback for revision) before subtask 04 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
