#pragma once

#include <stdint.h>
#include "capture_plan.h"

namespace capture {
constexpr unsigned kMinInkPixels = 20;
constexpr uint64_t kAutosaveUs = 2000000;

inline bool autosave_due(unsigned ink, bool touching, uint64_t last_input,
                         uint64_t now) {
    return ink >= kMinInkPixels && !touching && now >= last_input &&
           now - last_input >= kAutosaveUs;
}

// Only the UI core accesses this state. A reserved image is not yet durable.
struct Progress {
    bool complete[kTotal] = {};
    unsigned saved = 0;
    unsigned pending_slot = kTotal;
    bool failed = false;

    bool pending() const { return pending_slot < kTotal; }
    bool done() const { return saved == kTotal && !pending(); }

    unsigned next() const {
        for (unsigned slot = 0; slot < kTotal; ++slot) {
            if (!complete[slot] && slot != pending_slot) return slot;
        }
        return kTotal;
    }

    bool reserve(unsigned slot) {
        if (pending() || slot >= kTotal || slot != next()) return false;
        pending_slot = slot;
        failed = false;
        return true;
    }

    bool finish(unsigned slot, bool ok) {
        if (!pending() || slot != pending_slot) return false;
        failed = !ok;
        if (ok) {
            complete[slot] = true;
            ++saved;
            pending_slot = kTotal;
        }
        return true;
    }
};
}  // namespace capture
