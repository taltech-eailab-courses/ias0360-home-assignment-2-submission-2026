#ifndef STATUS_LED_H_
#define STATUS_LED_H_

// WS2812 status LED on the eval board (GP4).
// Requires pico_generate_pio_header(... common/ws2812.pio).

#include <stdint.h>

#include "hardware/pio.h"
#include "ws2812.pio.h"

#define STATUS_LED_PIN 4

static inline void status_led_init(void) {
    uint offset = pio_add_program(pio0, &ws2812_program);
    ws2812_program_init(pio0, 0, offset, STATUS_LED_PIN, 800000, true);
}

static inline void status_led_rgb(uint8_t r, uint8_t g, uint8_t b) {
    pio_sm_put_blocking(pio0, 0, (((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b) << 8u);
}

#endif  // STATUS_LED_H_
