# Subtask 05 — Write docs/api-reference.md

## Delegation
- **Agent:** @DocWriter
- **Model tier:** standard (claude-sonnet) — requires precise, accurate coverage of all public types and free functions; large output scope; needs judgment about depth vs. brevity for each symbol
- **Reason:** Comprehensive API reference covering multiple interacting systems. Accuracy is critical (all signatures must match current headers). Standard tier for judgment about documentation depth.

---

## Objective

Write `docs/api-reference.md` — a complete reference for every public type and free function in SlabKit. This is a reference page, not a tutorial. Users come here to look up a specific type or function. Each entry should include: the type/function signature, a brief description, key parameters, and any important notes (e.g., thread safety, preconditions, exception behaviour).

**Risk note:** This page may be large enough to hit DocWriter step limits. If DocWriter returns an incomplete draft, HeadWrench splits into two delegations: Part 1 (Context, ContextBuilder, memory.hpp functions) and Part 2 (array types, view types, scratch types, transfer functions). Combine the two parts before the review loop.

---

## Todolist

### 1. Draft docs/api-reference.md (first attempt — single DocWriter delegation)
- [ ] DocWriter writes the full page with sections:
  - **Context management** — `Context`, `ContextBuilder` (all builder methods + `build()`)
  - **Memory allocation** — `malloc_device<T>`, `DeviceUSMPoolDeleter`, `unique_device_ptr<T>`
  - **Arrays** — `DeviceArray<T,N>` (all methods), `MirroredArray<T,N>` (all methods)
  - **Views** — `DeviceView<T,N>` (all methods), `DeviceSubView<T,N>` (all methods), `DeviceViewLike` concept
  - **Scratch memory** — `DeviceScratchBuffer`, `DeviceScratchView<T,N>`, `make_scratch()`
  - **Factory functions** — `make_device_array<T,N>()`, `make_in_order_queue()`, `make_out_of_order_queue()`, `make_queue_with_props()`
  - **Transfer functions** — `copy_to_host()`, `copy_to_device()`, `copy()`, `sync_to_host()`, `sync_to_device()`
  - **Error types** — `AllocationError`

### 2. If step limit hit — split and retry
- [ ] (HeadWrench only, if needed) Re-delegate Part 1: Context, ContextBuilder, malloc_device, DeviceUSMPoolDeleter, unique_device_ptr
- [ ] (HeadWrench only, if needed) Re-delegate Part 2: DeviceArray, MirroredArray, DeviceView, DeviceSubView, DeviceScratchBuffer, DeviceScratchView, all factory + transfer functions, AllocationError
- [ ] Combine Part 1 + Part 2 into a single docs/api-reference.md

### 3. Review loop
- [ ] HeadWrench reads the draft and shows it to the user
- [ ] User provides written feedback
- [ ] DocWriter revises if needed
- [ ] HeadWrench shows revision and gets explicit approval

### 4. Commit
- [ ] `git add docs/api-reference.md && git commit -m "docs: add api-reference.md"`

### 5. [🚫 GATE] API reference approval
- [ ] User has explicitly approved this page before the gate is marked complete

---

## Scope
- **Edit:** nothing existing
- **Read:** `include/slabkit/context.hpp`, `include/slabkit/array.hpp`, `include/slabkit/memory.hpp`, `include/slabkit/scratch.hpp`, `include/slabkit/error.hpp`, `include/slabkit/slabkit.hpp`
- **Write:** `docs/api-reference.md` (new file)
- **Excluded:** `include/slabkit/detail/` (internal — not documented), device-backed overloads, AllocatorRegistry

---

## Patterns
```
✅ GOOD — Include the actual template signature: template<typename T, int N> class DeviceArray
✅ GOOD — Note preconditions and postconditions: "device_index must be < ctx->device_count()"
✅ GOOD — Note exception behaviour: "Throws AllocationError if the pool is exhausted"
❌ BAD  — Document internal detail headers (device_pool.hpp, device_stack.hpp) — those are private
❌ BAD  — Include the sycl::device overloads (make_device_array(range, dev)) — not in scope
✅ GOOD — Cross-reference: "See [Concepts](concepts.md) for ownership model details"
```

---

## Constraints
- **Accuracy is paramount** — all signatures must match current headers exactly; read headers before writing
- **No device-backed overloads** — only Context + device_index forms
- **No AllocatorRegistry**
- **No internal types** — `DeviceSlabAllocator`, `DeviceMemoryPool` etc. are not documented (they are private implementation details)
- **Step limit mitigation** — if DocWriter times out or returns incomplete output, HeadWrench splits into two delegations (see step 2 above)
- **Audience:** Experienced C++/SYCL developers — concise, precise, no hand-holding

---

## [🚫 GATE] API Reference Approval
HeadWrench shows the full `docs/api-reference.md` draft to the user. User must explicitly approve (or provide feedback for revision) before subtask 06 begins.

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
