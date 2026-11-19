#pragma once

#include <stddef.h>
#include <stdio.h>

// Consecutive rounds of 0..9, with separate acquisition phases for each split.
namespace capture {
constexpr unsigned kTrainRounds = 20;
constexpr unsigned kValRounds = 5;
constexpr unsigned kTestRounds = 5;
constexpr unsigned kTotal = (kTrainRounds + kValRounds + kTestRounds) * 10;
constexpr size_t kPixels = 280 * 240;
constexpr char kPgmHeader[] = "P5\n280 240\n255\n";
constexpr size_t kFileSize = sizeof(kPgmHeader) - 1 + kPixels;

struct Item {
    const char *split;
    unsigned digit;
    unsigned round;
    unsigned rounds;
};

inline Item item(unsigned slot) {
    if (slot < kTrainRounds * 10) {
        return {"TRAIN", slot % 10, slot / 10 + 1, kTrainRounds};
    }
    slot -= kTrainRounds * 10;
    if (slot < kValRounds * 10) {
        return {"VAL", slot % 10, slot / 10 + 1, kValRounds};
    }
    slot -= kValRounds * 10;
    return {"TEST", slot % 10, slot / 10 + 1, kTestRounds};
}

inline void path(char *out, size_t capacity, const char *drive,
                 unsigned slot, const char *extension = "PGM") {
    const Item current = item(slot);
    // Short 8.3 filenames also work when long filenames are disabled in FatFs.
    snprintf(out, capacity, "%s/H2DIGITS/%s/D%u_%03u.%s", drive,
             current.split, current.digit, current.round, extension);
}
}  // namespace capture
