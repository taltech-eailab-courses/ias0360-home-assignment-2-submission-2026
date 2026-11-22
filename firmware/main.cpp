// Live gesture recognition on the Pico.
//
// Loop: LED blue 1 s (get ready), green 2.1 s (do the gesture, recorded at
// 2000 Hz), yellow (preprocessing + inference), then the result is printed
// over USB serial and the LED shows the class colour for 2 s.
// Hold the board the same way as for the own recordings (upright).

#include <cstdio>

#include "pico/stdlib.h"
#include "hardware/pio.h"

#include "data_preprocessor.hpp"
#include "model.h"
#include "model_settings.h"
#include "gesture_model_data.h"

// Included last: this header defines `true`/`false` as macros.
#include "icm20948.h"
#include "ws2812.pio.h"

static const uint WS2812_PIN = 4;
static const uint64_t SAMPLE_PERIOD_US = 500;   // 2000 Hz, as in the recordings

static const char* const kLabels[CLASS_COUNT] = CLASS_LABELS;
static const uint8_t kClassColour[CLASS_COUNT][3] = {
    {16, 16, 16},    // negative: dim white
    {255, 0, 255},   // ring: magenta
    {0, 255, 255},   // slope: cyan
    {255, 255, 255}, // wing: white
};

static int16_t raw[kRecordLength * kNumAxes];
static Model model;
static DataPreprocessor preprocessor;

static void led(uint8_t r, uint8_t g, uint8_t b)
{
    pio_sm_put_blocking(pio0, 0, (((uint32_t)g << 16) | ((uint32_t)r << 8) | b) << 8u);
}

static void halt(const char* msg)
{
    for (;;) {
        printf("%s\n", msg);
        led(255, 0, 0);
        sleep_ms(1000);
    }
}

static void record_gesture()
{
    IMU_ST_SENSOR_DATA gyro, accel;
    uint64_t next = time_us_64();
    for (int t = 0; t < kRecordLength; t++) {
        while (time_us_64() < next) tight_loop_contents();
        next += SAMPLE_PERIOD_US;
        imuDataAccGyrGet(&gyro, &accel);
        raw[t * 3 + 0] = accel.s16X;
        raw[t * 3 + 1] = accel.s16Y;
        raw[t * 3 + 2] = accel.s16Z;
    }
}

int main()
{
    stdio_init_all();
    ws2812_program_init(pio0, 0, pio_add_program(pio0, &ws2812_program), WS2812_PIN, 800000, true);
    led(255, 0, 0);
    sleep_ms(3000);   // time to open the serial port

    IMU_EN_SENSOR_TYPE sensor;
    imuInit(&sensor);
    if (sensor != IMU_EN_SENSOR_TYPE_ICM20948) halt("IMU not found");
    if (!model.setup()) halt("Model setup failed");

    printf("Gesture recognition\n");
    printf("Model: %d bytes, tensor arena used: %u of %d bytes\n",
           gesture_model_data_len, (unsigned)model.arena_used_bytes(), kArenaSize);

    const TfLiteTensor* output = model.output_tensor();
    for (;;) {
        led(0, 0, 255);
        sleep_ms(1000);
        led(0, 255, 0);
        record_gesture();
        led(255, 160, 0);

        uint64_t t0 = time_us_64();
        preprocessor.preprocess(raw, model.input_data(),
                                model.input->params.scale, model.input->params.zero_point);
        uint64_t t1 = time_us_64();
        int result = model.predict();
        uint64_t t2 = time_us_64();
        if (result < 0) halt("Inference failed");

        printf("Predicted: %s  (", kLabels[result]);
        for (int c = 0; c < CLASS_COUNT; c++) {
            printf("%s %.2f%s", kLabels[c], output->data.uint8[c] / 255.0f,
                   c < CLASS_COUNT - 1 ? ", " : ")\n");
        }
        printf("Preprocessing %.1f ms, inference %.2f ms\n\n", (t1 - t0) / 1000.0f, (t2 - t1) / 1000.0f);

        led(kClassColour[result][0], kClassColour[result][1], kClassColour[result][2]);
        sleep_ms(2000);
    }
}
