# Subtask 07 — Update Design Doc

## Delegation
- **Agent:** @DocWriter
- **Model tier:** fast (haiku) — documentation update, no code writing; structural changes are clear and bounded
- **Reason:** Pure documentation work; content changes are well-specified below; no code reasoning required

---

## Objective

Update `docs/design/slabkit-v0.1-design.md` to reflect the new architecture: `AllocatorRegistry` is gone, `SlabKitContext` and `SlabKitContextBuilder` are the entry points, sycl::context is auto-created per platform group, and all factory free functions require explicit context. The document must be accurate as a reference for the post-refactor codebase.

---

## Todolist

### 1. Update the Architecture Overview (§2)
- [ ] Replace the ASCII diagram: remove `AllocatorRegistry` from the diagram and replace with `SlabKitContext / SlabKitContextBuilder`:
  ```
  User code
      │
      ├── DeviceArray<T,N>          (RAII owner, move-only)
      ├── MirroredArray<T,N>        (paired device+host)
      ├── DeviceView<T,N>           (non-owning, contiguous, trivially copyable)
      └── DeviceSubView<T,N>        (non-owning, strided, trivially copyable)
              │
              └── make_device_array(range, ctx, device_index|sycl::device)
                      │
                      └── malloc_device<T>(count, ctx, device_index|sycl::device)
                              │
                              └── SlabKitContext  (explicit, user-owned, no singleton)
                                          │
                              shared_ptr<DeviceMemoryPool>  (one per device, lazy)
                                          │
                              DeviceSlabAllocator  (BlockAllocator, owned by pool internally)
                                          │
                              sycl::malloc_device slab (fraction of device global_mem, allocated once)
  ```
- [ ] Update the **Invariants** list in §2:
  - Remove: "Queues are user-only; registry and allocator internals never construct or store queues"
  - Replace with: "Queues are user-only; `SlabKitContext` and allocator internals never construct or store queues"
  - Update any other invariant that mentions `AllocatorRegistry`

### 2. Update §3 Key Design Decisions
- [ ] **Delete** the section: `### Registry — pools only, no allocator storage` — this no longer applies
- [ ] **Delete** the section: `### AllocatorRegistry::Config — runtime configuration` — the registry is gone
- [ ] **Replace** the section `### sycl::context — one per platform, no queues in registry` with a new section describing `SlabKitContextBuilder` auto-context creation:

  **New section title:** `### SlabKitContextBuilder — auto-context creation per platform group`

  **New content (summarise):**
  - `SlabKitContextBuilder` takes only a `devices(vector<sycl::device>)` input — no explicit `sycl::context` required
  - `build()` groups devices by `(backend, platform_name)` composite key and creates one `sycl::context` per platform group
  - Result: `SlabKitContext::platform_contexts` is auto-populated; `device_to_platform` maps each device index to its context index
  - No global state, no singleton — user creates and owns `shared_ptr<SlabKitContext>`

- [ ] **Update** `### USM pointer ownership — DeviceUSMFromPoolDeleter<T>`: change "holds a weak_ptr — it does not keep the pool alive" section to reference `SlabKitContext` instead of `AllocatorRegistry`. The `weak_ptr<SlabKitContext>` lifetime anchor note should be added: the deleter holds `weak_ptr<void>` anchored to the context's `shared_ptr` — if the context is destroyed before the pointer, the deleter skips deallocation silently.

- [ ] **Update** thread safety section (currently under §3 bottom): remove `AllocatorRegistry::config()` thread safety note; update pool init note to reference `SlabKitContext::pool_at()` instead of registry

### 3. Update §4 Current File State
- [ ] Update `include/slabkit/memory.hpp` entry: remove `AllocatorRegistry` from the "Contains" list; add `DeviceUSMPoolDeleter<T>` (updated name); update free functions list to show `(ctx, device_index)` and `(ctx, dev)` overloads
- [ ] Update `include/slabkit/array.hpp` entry: note `make_device_array` now has `(range, ctx, device_index)` and `(range, ctx, dev)` overloads
- [ ] Add a note about `include/slabkit/context.hpp`: `SlabKitContext` and `SlabKitContextBuilder` (renamed from `SlabContext`)
- [ ] Update `include/slabkit/scratch.hpp`: remove "registry-backed" mention; note two overloads: `(ctx, device_index, size)` and `(ctx, dev, size)`
- [ ] Update `tests/` entry: remove `test_memory.cpp`/`test_array.cpp` stub note; update to "6 test files — all context-backed, no registry tests"

### 4. Update §5 Remaining Work
- [ ] Remove the "Phase 6 — Tests" section (tests are now written and should pass)
- [ ] Remove "Final gate" section (replaced by the session's build-test subtask)
- [ ] Update or remove the Phase 4.5 block-alignment note if it was resolved during the refactor (leave it if still an open item)
- [ ] Add a brief note: "AllocatorRegistry removed in session `remove-allocator-registry-singleton` (2026-03-10)"

### 5. Update §6 Out of Scope
- [ ] Remove the `AllocatorRegistry::config()` note from out of scope (the whole registry is gone)
- [ ] Add: "Global default context / convenience singleton — by design, users must construct `SlabKitContext` explicitly"

### 6. Final check
- [ ] No mention of `AllocatorRegistry` remains anywhere in the doc
- [ ] No mention of `SlabContext` (old name) or `SlabContextBuilder` (old name) remains
- [ ] Architecture diagram reflects the new ownership chain
- [ ] Document reads coherently — remove any orphaned cross-references

---

## Scope
- **Edit:** `docs/design/slabkit-v0.1-design.md`
- **Read:** `include/slabkit/context.hpp` (post-01), `include/slabkit/memory.hpp` (post-02), `include/slabkit/array.hpp` (post-03), `include/slabkit/scratch.hpp` (post-04) — to verify the descriptions match the actual post-refactor code
- **Write:** nothing new
- **Excluded:** all `.hpp` headers, all test files, build system files

---

## Patterns

```
✅ GOOD — Accurate description of builder:
    "SlabKitContextBuilder groups devices by (backend, platform_name) and constructs one
    sycl::context per platform group — no explicit sycl::context input required."

✅ GOOD — Accurate factory signature description:
    "make_device_array<T,N>(range, ctx, device_index) and (range, ctx, dev) overloads"

❌ BAD  — Leaving AllocatorRegistry in any section:
    "AllocatorRegistry (Meyer's singleton, lazy per-device)"

❌ BAD  — Leaving old type names:
    "SlabContext", "SlabContextBuilder"
```

---

## Constraints
- Do not rewrite the document from scratch — update in place; preserve sections that are still accurate (§1 Problem Statement, view operators fold-expression section, DeviceArray copyable via shared_ptr section, etc.)
- Maintain the existing section numbering and markdown heading style
- Do not add new sections beyond what is specified above
- The document does not need to be a perfect API reference — it is a design decision log; preserve the rationale-focused tone

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
