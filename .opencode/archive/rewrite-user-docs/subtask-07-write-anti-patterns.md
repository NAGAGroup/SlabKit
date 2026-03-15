# Subtask 07 — Write docs/anti-patterns.md

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — must accurately describe what goes wrong in each anti-pattern, produce realistic incorrect examples, and show the correct fix; judgment about which footguns to prioritise
- **Reason:** Anti-patterns require the hardest kind of documentation writing: the incorrect example must be obviously broken once diagnosed, the diagnosis must be accurate, and the fix must be correct. Standard tier for judgment and accuracy.

---

## Objective

Write `docs/anti-patterns.md` — a catalogue of footguns, incorrect usage patterns, and things that look right but aren't. For each anti-pattern: show what the bad code looks like, explain exactly what goes wrong (including the failure mode — UB, dangling pointer, wasted allocation, etc.), and show the correct alternative. The existing README has 9 documented anti-patterns; this page expands and explains them properly, with the conceptual tone this docs set uses.

---

## Todolist

### 1. Draft docs/anti-patterns.md
- [ ] Write the page covering these anti-patterns (minimum set — draw from existing README "NEVER DO THIS" section):
  1. **Capturing DeviceArray by reference in a kernel** — lambda captures by ref; array may be destroyed before kernel executes; UB
  2. **Holding a view after the owning array is destroyed** — dangling view; UB when kernel executes
  3. **Using data() for manual pointer arithmetic in kernels** — defeats the view system; error-prone indexing
  4. **Creating a view but using data() anyway** — contradicts the purpose of the view; loses bounds safety
  5. **Allocating scratch memory in a hot loop** — defeats the pool; each `make_scratch` call hits the allocator
  6. **Manually maintaining a host vector + device array pair instead of MirroredArray** — sync discipline breaks down; use MirroredArray
  7. **Manual stride arithmetic instead of DeviceSubView** — error-prone; use subviews
  8. **Copying a DeviceArray** — copy is deleted for good reason; shows what the intent should be (clone data explicitly, or share via view)
  9. **Using sycl::malloc_device / sycl::free directly for temporaries** — allocate once from pool, pass as view; don't bypass SlabKit
- [ ] Each anti-pattern: name / bad code example / what goes wrong (diagnosis) / correct alternative code
- [ ] Tone: non-judgmental but precise; explain *why* the bad code is bad, not just that it is

### 2. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 3. Commit
- [ ] `git add docs/anti-patterns.md && git commit -m "docs: add anti-patterns.md"`

### 4. [🚫 GATE] Anti-patterns approval
- [ ] User has explicitly approved this page before the gate is marked complete

---

## Scope
- **Edit:** nothing existing
- **Read:** `README.md` (existing "NEVER DO THIS" section for reference), `include/slabkit/array.hpp`, `include/slabkit/scratch.hpp`, `tests/test_array.cpp`
- **Write:** `docs/anti-patterns.md` (new file)
- **Excluded:** AllocatorRegistry, device-backed overloads, internal detail headers

---

## Patterns
```
✅ GOOD — Show the bad code, label it clearly, then diagnose: "This compiles. The problem is timing: the kernel may execute after the arr local variable is destroyed. The view's pointer is now dangling."
❌ BAD  — "Don't do this" without showing the actual failure mode
✅ GOOD — Show the fix immediately after the diagnosis with a clear before/after structure
❌ BAD  — Describe the fix in prose without a code example
✅ GOOD — Cross-reference patterns.md for the correct approach: "See [Pass views to kernels](patterns.md#pass-views-to-kernels)"
```

---

## Constraints
- **No AllocatorRegistry**
- **No device-backed overloads**
- **Bad code examples must actually be wrong** — don't show subtly-wrong code that's hard to tell apart from correct
- **Diagnosis must be accurate** — explain the exact failure mode (UB, dangling pointer, performance regression, etc.)
- **Correct alternatives must be accurate** — check all code against current headers
- **Tone:** Explanatory and conceptual — users should understand *why* each pattern is a footgun

---

## [🚫 GATE] Anti-Patterns Approval
HeadWrench shows the full `docs/anti-patterns.md` draft to the user. User must explicitly approve (or provide feedback for revision) before subtask 08 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
