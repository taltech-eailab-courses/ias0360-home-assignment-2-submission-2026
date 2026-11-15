// Live gesture recognition: records a gesture (2 kHz for 2.1 s) and classifies
// it with every model in the firmware. Results go to the USB serial console.
//
// LED: blue = get ready, green = perform the gesture now, yellow = classifying,
//      then the predicted class of model #1: white = negative, cyan = ring,
//      magenta = slope, orange = wing.

#include <cstdio>

#include "pico/stdlib.h"

#include "data_preprocessor.hpp"
#include "model_registry.h"
#include "model_runner.hpp"
#include "preprocessing_settings.hpp"
#include "run_on_core1.hpp"
#include "status_led.h"

// Included last: this header defines `true`/`false` as macros.
extern "C" {
#include "icm20948.h"
}

namespace {

// 2 kHz for 2.1 s = 4200 samples.
constexpr uint32_t kSamplePeriodUs = 500;
constexpr uint64_t kCaptureTimeUs = 2100 * 1000;
constexpr uint32_t kGetReadyMs = 2000;

int16_t g_raw[kInputWindowLength * kNumAxes];
DataPreprocessor g_preprocessor;
ModelRunner g_runner;

const uint8_t kClassColor[4][3] = {{120, 120, 120}, {0, 180, 180}, {180, 0, 180}, {200, 70, 0}};
constexpr uint32_t kShowResultMs = 2000;

void wait_for_usb_console() {
    for (int i = 0; i < 100 && !stdio_usb_connected(); ++i) {
        sleep_ms(100);
    }
    sleep_ms(500);
}

// Returns the number of samples read; the first kInputWindowLength are kept in g_raw.
size_t capture_gesture(float* rate_hz) {
    IMU_ST_SENSOR_DATA gyro, accel;
    size_t n = 0;
    const uint64_t start = time_us_64();
    uint64_t next = start;
    while (next < start + kCaptureTimeUs) {
        while (time_us_64() < next) {
            tight_loop_contents();
        }
        next += kSamplePeriodUs;
        imuDataOnlyGet(&gyro, &accel);
        if (n < static_cast<size_t>(kInputWindowLength)) {
            g_raw[n * kNumAxes + 0] = accel.s16X;
            g_raw[n * kNumAxes + 1] = accel.s16Y;
            g_raw[n * kNumAxes + 2] = accel.s16Z;
        }
        ++n;
    }
    *rate_hz = n / (kCaptureTimeUs / 1e6f);
    return n;
}

void app_main() {
    status_led_init();
    status_led_rgb(64, 0, 0);
    wait_for_usb_console();
    printf("\n==== HA2 live gesture recognition: %d model(s) ====\n", kModelCount);

    IMU_EN_SENSOR_TYPE sensor_type;
    imuInit(&sensor_type);
    if (sensor_type == IMU_EN_SENSOR_TYPE_NULL) {
        printf("IMU not found\n");
        while (1) {
            tight_loop_contents();
        }
    }

    // Load every model once and print its memory use.
    for (int m = 0; m < kModelCount; ++m) {
        if (!g_runner.Load(kModels[m].data)) {
            printf("model %s failed to load\n", kModels[m].name);
            while (1) {
                tight_loop_contents();
            }
        }
        printf("model %d: %-22s %6u bytes, arena used %u bytes\n", m + 1, kModels[m].name, kModels[m].size,
               static_cast<unsigned>(g_runner.arena_used_bytes()));
    }

    for (int gesture = 1;; ++gesture) {
        printf("\n--- gesture #%d: get ready (hold the board like in the training data) ---\n", gesture);
        status_led_rgb(0, 0, 64);
        sleep_ms(kGetReadyMs);

        printf("GO - perform the gesture now (%.1f s)\n", kCaptureTimeUs / 1e6f);
        status_led_rgb(0, 64, 0);
        float rate_hz = 0;
        const size_t samples = capture_gesture(&rate_hz);
        status_led_rgb(48, 48, 0);
        printf("captured %u samples (%.0f Hz), using the first %u\n", static_cast<unsigned>(samples), rate_hz,
               static_cast<unsigned>(samples < static_cast<size_t>(kInputWindowLength) ? samples
                                                                                        : kInputWindowLength));
        g_preprocessor.Prepare(g_raw, samples);

        int first_prediction = -1;
        for (int m = 0; m < kModelCount; ++m) {
            if (!g_runner.Load(kModels[m].data)) {
                continue;
            }
            g_preprocessor.Quantize(g_runner.input_data(), g_runner.input_scale(), g_runner.input_zero_point());
            uint32_t us = 0;
            const int pred = g_runner.Predict(&us);
            if (pred < 0) {
                continue;
            }
            if (first_prediction < 0) {
                first_prediction = pred;
            }
            const TfLiteTensor* out = g_runner.output();
            printf("[%s] -> %s (%.3f ms)\n", kModels[m].name, kClassNames[pred], us / 1000.0f);
            for (int c = 0; c < kClassCount; ++c) {
                printf("    %-9s %.3f\n", kClassNames[c], out->params.scale * (out->data.uint8[c] - out->params.zero_point));
            }
        }
        if (first_prediction >= 0) {
            const uint8_t* rgb = kClassColor[first_prediction % 4];
            status_led_rgb(rgb[0], rgb[1], rgb[2]);
            printf("Gesture: %s\n", kClassNames[first_prediction]);
        }
        sleep_ms(kShowResultMs);
    }
}

}  // namespace

int main() {
    stdio_init_all();
    run_on_core1(app_main);
}
