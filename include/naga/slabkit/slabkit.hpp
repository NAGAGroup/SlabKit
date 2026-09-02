#pragma once

// =============================================================================
// naga/slabkit/slabkit.hpp — umbrella
// =============================================================================
//
// slabkit is USM device memory with owning handles: two families of free
// functions -- malloc_* for uninitialized storage and make_* for constructed
// objects -- over four placements: a queue, a device with its context, a
// DeviceArena, a DevicePool. There are no array or view types; shapes belong
// to consumers.

#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/make.hpp"
#include "naga/slabkit/make_shared.hpp"
#include "naga/slabkit/malloc.hpp"
#include "naga/slabkit/pool.hpp"
