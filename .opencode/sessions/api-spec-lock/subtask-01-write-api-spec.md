# Subtask 01 — write-api-spec

## Objective
Write a formal API surface specification document at `docs/design/api-surface-spec.md` that captures the complete public API of SlabKit v0.1 with stability annotations, design decision rationale (as ADRs), thread-safety model, known limitations, and migration notes. This document should serve as the authoritative reference for stabilizing the public API.

## Scope

### Write
- `docs/design/api-surface-spec.md` (new file — create `docs/design/` directory if it doesn't exist)

### Read (for reference)
- `include/slabkit/error.hpp`
- `include/slabkit/context.hpp`
- `include/slabkit/memory.hpp`
- `include/slabkit/scratch.hpp`
- `include/slabkit/array.hpp`
- `include/slabkit/detail/device_pool.hpp`
- `include/slabkit/detail/device_stack.hpp`
- `docs/concepts.md`
- `docs/api-reference.md`
- `docs/patterns.md`
- `docs/anti-patterns.md`

### Excluded
- `tests/` (no test changes in this subtask)
- All headers except reading (no header modifications)

## Constraints
- Do NOT modify any existing files — write only the new spec file
- The spec must cover ALL public symbols (types, free functions, type aliases, concepts) in the 5 public headers: error.hpp, context.hpp, memory.hpp, scratch.hpp, array.hpp
- Do NOT document `detail/` internals as public API — they can be mentioned under "internal helpers" section only
- Use stability annotations: `[STABLE]` for locked-down public API, `[UNSTABLE]` for things still in flux
- `DeviceScratchBuffer` alias must be documented as `[DEPRECATED — remove before v1.0]` with migration note: "use `DeviceScratchMemory` directly"
- Context move semantics note: move ctor is enabled but move assignment is deleted — this is an asymmetry that should be flagged as `[DECISION PENDING]` in the spec
- The spec must include a Design Decisions section with ADR-style entries for: no-registry explicit-context architecture, shared_ptr management for Context (enable_shared_from_this), move-only DeviceArray, trivially copyable views, 4-tier transfer function design, element strides in DeviceSubView (not byte strides), weak_ptr lifetime anchoring for DeleterS, thread-safety model
- Include a Known Limitations section: no fine-grained pool locking, no automatic fragmentation compaction, no 1D operator[](size_t), no runtime bounds checking in release builds, scratch memory not thread-safe
- Include a Future Additions section: planned `operator[](size_t)` for 1D views, fine-grained pool locking (v0.2 roadmap)
- Transfer functions: document the tier structure (Tier 1–4) explicitly; note all async Tier 1–3 functions return `sycl::event` which MUST be awaited or chained

## Todolist
- [ ] Create `docs/design/` directory and write `api-surface-spec.md`
- [ ] Section 1: Overview — library purpose, namespace, include path, C++ standard requirement
- [ ] Section 2: Public Types — each type with stability annotation, brief description, all public members
- [ ] Section 3: Free Functions — all public free functions with full signatures
- [ ] Section 4: Concepts — `DeviceViewLike<V,T,N>` requirements
- [ ] Section 5: Design Decisions (ADRs) — 8 entries listed above
- [ ] Section 6: Thread-Safety Model — per-component summary
- [ ] Section 7: Migration Notes — DeviceScratchBuffer → DeviceScratchMemory
- [ ] Section 8: Known Limitations
- [ ] Section 9: Future Additions / Roadmap

## Delegation
- **Agent:** SpecWriter
- **Model:** PLACEHOLDER_MODEL_ID
- **Type:** Documentation write (new file)
