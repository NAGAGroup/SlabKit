# Subtask 07 — anti-patterns.md Decisions

## Committed
`917b653` — docs: add anti-patterns.md — common footguns with diagnosis and fixes

## 10 Anti-Patterns Covered
1. Capturing DeviceArray by value in kernel lambda (Compile Error)
2. Using DeviceView after backing array destroyed (Runtime UB)
3. shared_ptr<Context> shorter-lived than its arrays (Silent Bug)
4. Rewinding scratch while GPU work is in flight (Runtime UB)
5. q.submit(handler) for USM kernels (Unnecessary)
6. Indexing DeviceView with wrong type (Compile Error)
7. Constructing Context from multiple threads (Runtime UB)
8. Using deprecated DeviceScratchBuffer alias (Deprecated)
9. Assuming device memory is zero-initialized (Silent Bug)
10. Ignoring AllocationError on pool exhaustion (Silent Bug)

## HW Corrections
- Anti-pattern 6: Fixed "The Problem" example — was calling .view() on a temporary DeviceArray (UB)
- Anti-pattern 9: Fixed lambda capture — removed confusing [=, view = arr.view()] init-capture
