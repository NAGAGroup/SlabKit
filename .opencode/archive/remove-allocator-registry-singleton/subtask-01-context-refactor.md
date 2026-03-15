# Subtask 01 — Refactor `context.hpp`

## Delegation
**Agent:** @CodeWriter
**Model:** standard (claude-sonnet) — multiple interacting systems: rename ripples through base class template, builder validation logic overhaul, and platform-grouping logic migration from AllocatorRegistry::init()

---

## Objective

Rename `SlabContext` → `SlabKitContext` and `SlabContextBuilder` → `SlabKitContextBuilder` throughout `include/slabkit/context.hpp`. Simultaneously refactor `SlabKitContextBuilder` so that:

1. The `.contexts(...)` setter is **removed** — users no longer pass explicit `sycl::context` objects.
2. `.devices(std::vector<sycl::device>)` becomes the **only** required input.
3. `build()` auto-creates one `sycl::context` per platform group from the provided devices, using the platform-grouping logic currently in `AllocatorRegistry::init()`.

The public API of `SlabKitContext` (all methods, data members, naming, and semantics) is **unchanged** except for the type name itself.

---

## Todolist

### 1. Rename types
- [ ] `struct SlabContext` → `struct SlabKitContext` (class definition, all internal references within context.hpp, the `enable_shared_from_this` base)
- [ ] `class SlabContextBuilder` → `class SlabKitContextBuilder` (class definition)
- [ ] Update all `SlabContext` references inside context.hpp to `SlabKitContext` (error messages, comments, method bodies)
- [ ] Update the file-level comment block at the top (the `=====` banner and description)

### 2. Remove `.contexts(...)` setter from builder
- [ ] Delete the `SlabKitContextBuilder& contexts(std::vector<sycl::context> ctxs)` setter method
- [ ] Delete the private `std::vector<sycl::context> contexts_` member
- [ ] Remove all uses of `contexts_` from `build()`

### 3. Rewrite `build()` to auto-create contexts
- [ ] Remove the "contexts empty" guard (`if (contexts_.empty())` throw)
- [ ] Remove Pass 1 (duplicate platform check based on `ctx.get_devices()`) — no longer needed because we create one context per platform
- [ ] Port the platform-grouping loop from `AllocatorRegistry::init()`: iterate `devices_`, group by `(backend, platform_name)` key, create one `sycl::context(dev_group)` per group
- [ ] Build `platform_contexts` from these groups; build `platform_keys` vector for the device→platform mapping step
- [ ] Port Pass 2 from `AllocatorRegistry::init()`: for each device in `devices_`, find its platform key in `platform_keys`, set `device_to_platform[i]`
- [ ] Remove the old Pass 2 from `SlabContextBuilder::build()` (the loop that matched devices against `contexts_` via `ctx.get_devices()`)
- [ ] Keep: `devices_.empty()` guard, `slab_size_per_device` logic, `pools.resize(...)`, `node_size_` assignment, `return std::shared_ptr<SlabKitContext>(new SlabKitContext(std::move(result)))`

### 4. Update error messages
- [ ] Change all `"slabkit: device not found in SlabContext"` → `"slabkit: device not found in SlabKitContext"`
- [ ] Change `"slabkit: device index out of range"` and `"slabkit: device not covered by any provided context"` messages to use `SlabKitContext` if referenced

### 5. Verify
- [ ] No reference to `SlabContext` (old name) remains anywhere in context.hpp
- [ ] No reference to `SlabContextBuilder` (old name) remains anywhere in context.hpp
- [ ] No `contexts_` private member or `.contexts()` setter remains
- [ ] `build()` no longer calls `ctx.get_devices()` on user-supplied contexts (those no longer exist)

---

## Scope
- **Edit:** `include/slabkit/context.hpp`
- **Read:** `include/slabkit/memory.hpp` (for the platform-grouping logic to port — see `AllocatorRegistry::init()` lines 225–290)
- **Write:** nothing new
- **Excluded:** `detail/device_pool.hpp`, `detail/device_stack.hpp`, `error.hpp`, all test files, all other headers

---

## Patterns

```
✅ GOOD — Auto-create context from devices:
    // In build():
    std::vector<std::vector<sycl::device>> devices_by_platform;
    std::vector<std::pair<sycl::backend, std::string>> platform_keys;
    for (const auto& dev : devices_) {
        auto backend = dev.get_platform().get_backend();
        auto name    = dev.get_platform().get_info<sycl::info::platform::name>();
        auto it = std::find_if(platform_keys.begin(), platform_keys.end(),
                               [backend, &name](const auto& k){ return k.first == backend && k.second == name; });
        if (it == platform_keys.end()) {
            platform_keys.push_back({backend, name});
            devices_by_platform.push_back({dev});
        } else {
            devices_by_platform[std::distance(platform_keys.begin(), it)].push_back(dev);
        }
    }
    for (const auto& grp : devices_by_platform)
        result.platform_contexts.push_back(sycl::context(grp));

✅ GOOD — device_to_platform mapping after platform_contexts are built:
    for (size_t i = 0; i < devices_.size(); ++i) {
        auto backend = devices_[i].get_platform().get_backend();
        auto name    = devices_[i].get_platform().get_info<sycl::info::platform::name>();
        auto it = std::find_if(platform_keys.begin(), platform_keys.end(), ...);
        result.device_to_platform[i] = std::distance(platform_keys.begin(), it);
    }

❌ BAD  — Retaining the old Pass 1 duplicate-platform check:
    for (const auto& ctx : contexts_) {  // contexts_ no longer exists
        auto ctx_devices = ctx.get_devices();  // don't call get_devices() on user contexts
        ...
    }

❌ BAD  — Keeping the .contexts() setter:
    SlabKitContextBuilder& contexts(std::vector<sycl::context> ctxs) { ... }
```

---

## Constraints
- `enable_shared_from_this<SlabKitContext>` base must remain — pools use `shared_from_this()` for the `weak_ptr<void>` lifetime anchor
- `std::shared_ptr<SlabKitContext>(new SlabKitContext(std::move(result)))` construction pattern must be preserved (not `make_shared`) — required for `enable_shared_from_this` correctness
- Pool lazy-init (double-checked locking in `pool_at()`) must remain unchanged
- `DeviceMemoryPool` constructor call in `pool_at()` must remain unchanged — off-limits detail
- Do not add `sycl::device::get_devices()` call to the builder — no auto-discovery
- The `mutable` qualifier on `pools` and `pool_mutex_` must remain
- `const_cast<SlabKitContext*>(this)->pool_at(idx)` const overload pattern must remain

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
