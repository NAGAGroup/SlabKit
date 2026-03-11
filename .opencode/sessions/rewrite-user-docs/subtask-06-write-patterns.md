# Subtask 06 — Write docs/patterns.md

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — requires curating which patterns to include, writing clear rationale for each, and producing accurate, idiomatic code examples; judgment about depth per pattern
- **Reason:** Documentation writing with multiple interacting concepts and code examples. Needs standard-tier judgment to produce good rationale prose alongside accurate code.

---

## Objective

Write `docs/patterns.md` — a guide to best-practice usage patterns for SlabKit. This page answers: "I have SlabKit — what does good code look like?" Each pattern should include: what to do, why it's the right approach, and a realistic code example. Patterns are drawn from the existing README "DO THIS" section (6 patterns), extended and re-explained with the conceptual tone appropriate for this docs set.

---

## Todolist

### 1. Draft docs/patterns.md
- [ ] Write the page with these patterns (minimum set — DocWriter may add more if clearly appropriate):
  1. **Pass views to kernels** — never pass DeviceArray; capture DeviceView by value; explain why (trivially copyable, kernel-safe)
  2. **Use subviews for tiling** — DeviceSubView for strided access; avoid manual pointer arithmetic
  3. **Prefer const views for read-only access** — `DeviceView<const T, N>` for read-only kernel parameters
  4. **Allocate scratch once, reuse** — `DeviceScratchBuffer` allocated before the loop; passed as `DeviceScratchView` to functions
  5. **Use MirroredArray for paired device+host storage** — instead of maintaining a separate `std::vector` and `DeviceArray` manually
  6. **Use event dependencies for async pipelines** — pass `sycl::event` deps to transfer functions; never `.wait()` in hot paths
- [ ] Each pattern: name, 1-sentence summary, code example (before/after or just correct form), explanation of *why* this is right
- [ ] Tone: conceptual and explanatory — the rationale is as important as the pattern itself

### 2. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 3. Commit
- [ ] `git add docs/patterns.md && git commit -m "docs: add patterns.md"`

### 4. [🚫 GATE] Patterns approval
- [ ] User has explicitly approved this page before the gate is marked complete

---

## Scope
- **Edit:** nothing existing
- **Read:** `README.md` (existing "DO THIS" section for reference), `include/slabkit/array.hpp`, `include/slabkit/scratch.hpp`, `tests/test_array.cpp`, `tests/test_transfers.cpp`
- **Write:** `docs/patterns.md` (new file)
- **Excluded:** AllocatorRegistry, device-backed overloads, internal detail headers

---

## Patterns
```
✅ GOOD — Each pattern has: name / why it matters / code example / explanation
❌ BAD  — Just a list of code snippets with no rationale
✅ GOOD — "Capturing a DeviceArray by reference in a lambda is undefined behaviour — the array lives on the host stack; the lambda may execute after it's been destroyed."
❌ BAD  — "Don't capture DeviceArray by reference." (no explanation of why or what goes wrong)
✅ GOOD — Cross-reference anti-patterns.md for the failure mode of each pattern
❌ BAD  — Repeat the anti-pattern description inline — link out instead
```

---

## Constraints
- **No AllocatorRegistry**
- **No device-backed overloads**
- **Code examples must be accurate** — check signatures against current headers
- **Cross-reference anti-patterns.md** — each pattern should link or reference its corresponding anti-pattern in `docs/anti-patterns.md`
- **Tone:** Explanatory and conceptual — the *why* is as important as the *what*
- **Length:** Aim for 200–300 lines. If patterns are growing too long, some explanation may belong in concepts.md — link out

---

## [🚫 GATE] Patterns Approval
HeadWrench shows the full `docs/patterns.md` draft to the user. User must explicitly approve (or provide feedback for revision) before subtask 07 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
