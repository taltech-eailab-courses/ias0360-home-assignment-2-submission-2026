#ifndef GESTURE_MODEL_SETTINGS_H_
#define GESTURE_MODEL_SETTINGS_H_

#define CLASS_COUNT 4
#define CLASS_LABELS {"negative", "ring", "slope", "wing"}

// Tensor arena for TFLite Micro. Measured with arena_used_bytes() on the Pico
// (printed at start-up), plus a small margin.
constexpr int kArenaSize = 6 * 1024;   // 5076 bytes used on the Pico

#endif  // GESTURE_MODEL_SETTINGS_H_
