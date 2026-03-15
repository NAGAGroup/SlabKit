# Subtask 02 — Refactor `memory.hpp`

## Delegation
**Agent:** @CodeWriter
**Model:** standard (claude-sonnet) — deleting an entire class with inline implementations, updating type references across the file, and ensuring the registry-free path is the only remaining path

---

## Objective

Refactor `include/slabkit/memory.hpp` to:

1. **Delete `AllocatorRegistry`** entirely — class declaration, all member definitions, `Config` struct, `get()`, `init()`, `configure()`, `reset_for_testing()`, and the `context()` static method.
2. **Update `DeviceUSMPoolDeleter`** — change `weak_ptr<SlabContext>` → `weak_ptr<SlabKitContext>` in the struct.
3. **Remove registry-backed free functions** — delete `malloc_device<T>(count, dev)`, `make_scratch(dev, size)`, `make_in_order_queue(dev)`, `make_out_of_order_queue(dev)`, `make_queue_with_props(dev, props)`.
4. **Keep the registry-free `malloc_device<T>(count, ctx, device_index)`** overload, updated to use `SlabKitContext`.
5. **Add a `sycl::device` overload** to `malloc_device` that accepts `(count, ctx, device)` and resolves `device_index` via `ctx->device_index(dev)`.
6. Update the file-level comments and all internal `SlabContext` references to `SlabKitContext`.

After this subtask, `memory.hpp` will contain only: `DeviceUSMPoolDeleter`, `unique_device_ptr<T>`, and two `malloc_device` free function templates.

---

## Todolist

### 1. Remove AllocatorRegistry class
- [ ] Delete the entire `class AllocatorRegistry { ... };` block (lines 63–137 in the original)
- [ ] Delete all inline method definitions below the class: `AllocatorRegistry::get()`, `AllocatorRegistry::context()`, `AllocatorRegistry::init()`, `AllocatorRegistry::pool()`, `AllocatorRegistry::device_index()`, `AllocatorRegistry::device_by_index()`, `AllocatorRegistry::make_in_order_queue()`, `AllocatorRegistry::make_out_of_order_queue()`, `AllocatorRegistry::make_queue_with_props()`, `AllocatorRegistry::config()`, `AllocatorRegistry::configure()`, `AllocatorRegistry::reset_for_testing()` (the `#ifdef SLABKIT_TESTING` block)
- [ ] Delete the `// ====` Implementation section heading comment

### 2. Remove registry-backed free functions
- [ ] Delete `malloc_device<T>(count, const sycl::device& dev)` template (the one that calls `AllocatorRegistry::context()`)
- [ ] Delete `make_scratch(const sycl::device& dev, std::size_t scratch_size)` inline function
- [ ] Delete `make_in_order_queue(const sycl::device& dev)` inline function
- [ ] Delete `make_out_of_order_queue(const sycl::device& dev)` inline function
- [ ] Delete `make_queue_with_props(const sycl::device& dev, const sycl::property_list& props)` inline function
- [ ] Delete the `// Queue factory free functions` section heading comment

### 3. Update DeviceUSMPoolDeleter type reference
- [ ] Change `std::weak_ptr<SlabContext> ctx` → `std::weak_ptr<SlabKitContext> ctx` in the struct definition
- [ ] Update the struct comment block: `"weak_ptr<SlabContext>"` → `"weak_ptr<SlabKitContext>"`

### 4. Update registry-free malloc_device
- [ ] Change `std::shared_ptr<SlabContext> ctx` → `std::shared_ptr<SlabKitContext> ctx` in the function signature
- [ ] Change `DeviceUSMPoolDeleter{std::weak_ptr<SlabContext>(ctx), ...}` → `DeviceUSMPoolDeleter{std::weak_ptr<SlabKitContext>(ctx), ...}` in the return statement

### 5. Add sycl::device overload to malloc_device
- [ ] Add a new overload:
    ```cpp
    template <typename T>
    unique_device_ptr<T> malloc_device(size_t count,
                                       std::shared_ptr<SlabKitContext> ctx,
                                       const sycl::device& dev) {
        return malloc_device<T>(count, ctx, ctx->device_index(dev));
    }
    ```
- [ ] Place it immediately after the existing `malloc_device(count, ctx, device_index)` overload

### 6. Update includes and comments
- [ ] Remove `#include "slabkit/detail/device_stack.hpp"` — scratch types are no longer needed in memory.hpp (the registry-backed `make_scratch` is gone)
- [ ] Update the file-level `=====` banner comment to describe the new contents: `DeviceUSMPoolDeleter`, `unique_device_ptr<T>`, `malloc_device` free functions
- [ ] Remove any remaining `SlabContext` references in comments → `SlabKitContext`
- [ ] Remove the `// Note: registry-free make_scratch(ctx, device_index, size) is defined in scratch.hpp.` comment (no longer needed without the registry)

### 7. Verify
- [ ] No `AllocatorRegistry` symbol anywhere in the file
- [ ] No `SLABKIT_TESTING` guard remaining
- [ ] No `SlabContext` (old name) remaining — only `SlabKitContext`
- [ ] `#include "slabkit/detail/device_stack.hpp"` is gone
- [ ] `DeviceUSMPoolDeleter` still has `weak_ptr<SlabKitContext>`, `device_index`, `node_count`
- [ ] Two `malloc_device` overloads remain: `(count, ctx, device_index)` and `(count, ctx, dev)`

---

## Scope
- **Edit:** `include/slabkit/memory.hpp`
- **Read:** `include/slabkit/context.hpp` (post-subtask-01 state — to see `SlabKitContext` definition)
- **Write:** nothing new
- **Excluded:** `detail/device_pool.hpp`, `detail/device_stack.hpp`, `error.hpp`, `context.hpp`, `array.hpp`, `scratch.hpp`, all test files

---

## Patterns

```
✅ GOOD — Final shape of DeviceUSMPoolDeleter:
    struct DeviceUSMPoolDeleter {
        std::weak_ptr<SlabKitContext> ctx;
        size_t device_index;
        size_t node_count;
        void operator()(void* ptr) const noexcept {
            if (auto c = ctx.lock())
                c->pool_at(device_index).deallocate_array(ptr, node_count);
        }
    };

✅ GOOD — Both malloc_device overloads:
    template <typename T>
    unique_device_ptr<T> malloc_device(size_t count,
                                       std::shared_ptr<SlabKitContext> ctx,
                                       size_t device_index) { ... }

    template <typename T>
    unique_device_ptr<T> malloc_device(size_t count,
                                       std::shared_ptr<SlabKitContext> ctx,
                                       const sycl::device& dev) {
        return malloc_device<T>(count, ctx, ctx->device_index(dev));
    }

❌ BAD  — Leaving any AllocatorRegistry symbol:
    class AllocatorRegistry { ... };
    static AllocatorRegistry& get() { ... }

❌ BAD  — Keeping the registry-backed overload:
    template <typename T>
    unique_device_ptr<T> malloc_device(size_t count, const sycl::device& dev) {
        auto ctx = AllocatorRegistry::context();  // REMOVE THIS
        ...
    }
```

---

## Constraints
- `DeviceUSMPoolDeleter::operator()` must remain `noexcept` — called from unique_ptr destructor
- The `if (auto c = ctx.lock())` weak_ptr lock pattern must be preserved — context may be destroyed before deleter fires
- `unique_device_ptr<T>` alias must remain: `using unique_device_ptr = std::unique_ptr<T, DeviceUSMPoolDeleter>`
- Do not add any auto-discovery logic (no `sycl::device::get_devices()`)
- `#include "slabkit/context.hpp"` must remain (needed for `SlabKitContext`)
- `#include "slabkit/detail/device_pool.hpp"` must remain (needed for `DeviceMemoryPool` in pool access)

---

*At the end of this subtask, follow the checkpoint protocol in `~/.config/opencode/protocols/checkpoint.md`.*
