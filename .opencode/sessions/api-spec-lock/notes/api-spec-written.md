# Note: api-surface-spec.md Written

**Date:** 2026-03-15
**Session:** api-spec-lock, Subtask 01

## What Was Written

`docs/design/api-surface-spec.md` (~900 lines) is now the authoritative API surface record for SlabKit v0.1.

## Key Decisions Recorded in the Spec

- `Context` move semantics flagged as `[DECISION PENDING]` — move ctor is enabled but move assignment is deleted (asymmetric); agreed resolution is to delete both (make fully non-movable) since `std::mutex` is not movable
- `DeviceScratchBuffer` alias flagged as `[DEPRECATED — remove before v1.0]`
- All stability annotations applied: `[STABLE]`, `[UNSTABLE]`, `[DEPRECATED]`
- 8 ADRs captured covering the registry removal, shared_ptr mandate, move-only arrays, trivially copyable views, 4-tier transfers, element strides, weak_ptr anchoring, pool mutex

## Pending Fixes Documented

The spec records two pending items that will be addressed in Subtask 02:
1. Context move semantics asymmetry
2. Missing `[[nodiscard]]` on factories and event-returning functions
