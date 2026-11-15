#ifndef RUN_ON_CORE1_HPP
#define RUN_ON_CORE1_HPP

#include <cstdint>

#include "pico/multicore.h"
#include "pico/stdlib.h"

// Runs `app` on core 1 with a 16 KB stack (TFLite Micro needs more than core
// 0's 4 KB stack). Core 0 services USB stdio and otherwise idles.
inline void run_on_core1(void (*app)()) {
    static uint32_t core1_stack[4096] __attribute__((aligned(8)));
    multicore_launch_core1_with_stack(app, core1_stack, sizeof(core1_stack));
    while (true) {
        tight_loop_contents();
    }
}

#endif  // RUN_ON_CORE1_HPP
