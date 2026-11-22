// Runs the 20 own recordings (stored in flash) through the same preprocessing
// and model as main.cpp, and prints the prediction, the input checksum and the
// timing for each. Compare the output with own_data_reference.txt (the PC result).

#include <cstdio>

#include "pico/stdlib.h"

#include "data_preprocessor.hpp"
#include "model.h"
#include "model_settings.h"
#include "gesture_model_data.h"
#include "own_gestures.h"

static const char* const kLabels[CLASS_COUNT] = CLASS_LABELS;

static Model model;
static DataPreprocessor preprocessor;

int main()
{
    stdio_init_all();
    sleep_ms(3000);   // time to open the serial port

    if (!model.setup()) {
        for (;;) { printf("Model setup failed\n"); sleep_ms(1000); }
    }

    for (;;) {
        printf("Own data test\n");
        printf("Model: %d bytes, tensor arena used: %u of %d bytes\n",
               gesture_model_data_len, (unsigned)model.arena_used_bytes(), kArenaSize);

        int correct = 0;
        uint64_t pre_us = 0, inf_us = 0;
        for (int i = 0; i < own_gesture_count; i++) {
            uint8_t* input = model.input_data();
            uint64_t t0 = time_us_64();
            preprocessor.preprocess(&own_gestures[i][0][0], input,
                                    model.input->params.scale, model.input->params.zero_point);
            uint64_t t1 = time_us_64();

            // Checksum before Invoke(): TFLite Micro reuses the input memory during inference.
            unsigned sum = 0;
            for (int k = 0; k < kTargetLength * kNumAxes; k++) sum += input[k];

            uint64_t t2_start = time_us_64();
            int pred = model.predict();
            uint64_t t2 = time_us_64();
            pre_us += t1 - t0;
            inf_us += t2 - t2_start;
            int label = own_gesture_labels[i];
            correct += (pred == label);
            printf("%s true=%s pred=%s input_sum=%u\n", own_gesture_names[i],
                   kLabels[label], pred >= 0 ? kLabels[pred] : "error", sum);
        }
        printf("Pico accuracy: %d/%d\n", correct, own_gesture_count);
        printf("Average per gesture: preprocessing %.2f ms, inference %.3f ms\n\n",
               pre_us / 1000.0f / own_gesture_count, inf_us / 1000.0f / own_gesture_count);
        sleep_ms(10000);
    }
}
