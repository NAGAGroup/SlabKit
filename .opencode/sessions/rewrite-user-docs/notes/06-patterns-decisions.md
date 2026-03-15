# Subtask 06 — patterns.md Decisions

## Committed
`6927c1d` — docs: add patterns.md — best-practice usage patterns

## Patterns Covered (10 total)
1. Build Context Once, Share via shared_ptr
2. Use .view() to Pass Arrays into Kernels
3. Accept DeviceView in Functions, Not DeviceArray
4. Use DeviceViewLike for Generic Kernels
5. Scratch Memory: Save/Unwind for Loop Temporaries
6. Use MirroredArray for Host-Initialized Data
7. Chain Async Transfers with In-Order Queue
8. Subviews and Strided Transfers
9. Catching AllocationError
10. Use make_scratch with Device Overload

## HW Correction
- Pattern 10 kernel example: `view[id.get(0)]` → `view[id]`
  - `DeviceView::operator[](sycl::id<N>)` takes the full id, not an integer extracted from it
