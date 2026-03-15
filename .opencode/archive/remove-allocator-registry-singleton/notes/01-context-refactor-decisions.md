# Subtask 01 — context.hpp Refactor Decisions

## What was done
- Renamed `SlabContext` → `SlabKitContext` throughout `context.hpp`
- Renamed `SlabContextBuilder` → `SlabKitContextBuilder`
- Removed `.contexts(std::vector<sycl::context>)` setter and `contexts_` private member
- Removed Pass 1 (duplicate-platform check via `ctx.get_devices()`) — no longer needed
- Removed Pass 2 (coverage check matching devices against user contexts) — replaced by grouping
- Rewrote `build()` to auto-create one `sycl::context(dev_group)` per platform group

## Platform-grouping logic
Ported directly from `AllocatorRegistry::init()` (memory.hpp lines 225–290).
Groups devices by `(backend, platform_name)` composite key using `std::find_if` on `platform_keys`.
Creates one `sycl::context(dev_group)` per group, then maps each device to its platform index.

## Preserved invariants
- `enable_shared_from_this<SlabKitContext>` base — pools use `shared_from_this()` as `weak_ptr<void>` lifetime anchor
- `std::shared_ptr<SlabKitContext>(new SlabKitContext(std::move(result)))` — not `make_shared` — required for `enable_shared_from_this` correctness
- Double-checked locking in `pool_at()` — unchanged
- `mutable` on `pools` and `pool_mutex_` — unchanged
- `const_cast<SlabKitContext*>(this)->pool_at(idx)` const overload pattern — unchanged

## Key insight from memory.hpp comment (line 231–236)
The old `SlabContextBuilder` used `ctx.get_devices()` in its duplicate-platform check, which returns
ALL platform devices (not just the subset given to the context constructor). This caused false
duplicate-platform errors in multi-platform systems. The new approach avoids this entirely by
never using `ctx.get_devices()` — we build contexts ourselves from the device list.

## File size
Old: 282 lines. New: 255 lines.
