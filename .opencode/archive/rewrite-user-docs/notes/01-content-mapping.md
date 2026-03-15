# Content Mapping — Subtask 01 Findings

**Date:** 2026-03-10

## README Disposition

- **KEEP:** Title, tagline, badges, installation section, minimal quick-start snippet (~50 lines max)
- **DROP:** AllocatorRegistry section (lines ~108–131) — AllocatorRegistry was deleted in prior `remove-allocator-registry-singleton` session. Stale content.
- **DROP:** Sprawling API reference inline in README — move to docs/api-reference.md
- **DROP:** Long patterns and anti-patterns sections — move to docs/patterns.md and docs/anti-patterns.md
- **MOVE (rewrite fresh):** Core concepts/ownership model → docs/concepts.md
- **MOVE (rewrite fresh):** Getting started walkthrough → docs/getting-started.md

## Design Doc Staleness Check

**Result: ALL CLEAR.** Design doc verified against current headers:
- `include/slabkit/context.hpp` — ContextBuilder, Context definitions match
- `include/slabkit/array.hpp` — DeviceArray, DeviceView, DeviceSubView, MirroredArray match
- `include/slabkit/memory.hpp` — malloc_device, DeviceUSMPoolDeleter match

No stale content found. The post-refactor session (`remove-allocator-registry-singleton`, 2026-03-10) thoroughly updated the design doc to remove AllocatorRegistry references.

## Design Doc Reference Value

The design doc is a goldmine for the new docs/ pages:
- **§2 Architecture diagram** → good visual/conceptual reference for concepts.md
- **§2 Invariants list** → critical user-facing guarantees to surface in concepts.md
- **§3 Key design decision rationales** → background for explaining *why* the API is shaped as it is
- **§3.10 Thread-safety caveat** → important limitation for concepts.md
- **§6 Known limitations** → N≥4 views blocked by SYCL spec; thread-safety roadmap for v0.2

## Scope Constraints Confirmed

- No AllocatorRegistry mention anywhere in new docs
- No device-backed overloads (`make_device_array(range, dev)`) — Context-based API only
- Only Context-required patterns documented
