// IMU gesture recorder that sends the data over USB serial instead of an SD card.
//
// Same data as lab_2_2/imu/data_collection (accelerometer, raw int16 counts,
// CSV "ax,ay,az"), sampled at a fixed 2000 Hz. Each recording is buffered in
// RAM and dumped over USB after it ends, so printing never disturbs the timing.
//
// Protocol (host: hw2/tools/record_serial.py):
//   host sends 'r'  -> LED blue for COUNTDOWN_MS (get ready)
//                   -> LED green while recording RECORD_TIME_US
//                   -> LED yellow while sending
//   device prints:  BEGIN <n_samples> <duration_us>
//                   ax,ay,az
//                   <n_samples lines>
//                   END
//   host sends 'p'  -> device prints PONG (connection check)
//   LED dim white = idle, waiting for 'r'.

#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"

#include "icm20948.h"
#include "ws2812.pio.h"

static const uint WS2812_PIN = 4;

// 2.1 s matches the shared dataset (~4200 rows at ~2 kHz) and src/preprocessor.py.
static const uint64_t RECORD_TIME_US = 2100 * 1000;
static const uint32_t COUNTDOWN_MS = 1000;

// Fixed 2000 Hz (500 us period): the free-running loop gives ~2160 Hz on this
// board, which puts files >5% off the 4200 rows that src/preprocessor.py expects.
static const uint64_t SAMPLE_PERIOD_US = 500;

// 8000 samples * 6 B = 48 KB; enough for 2.1 s up to ~3.8 kHz.
#define MAX_SAMPLES 8000
static int16_t buf[MAX_SAMPLES][3];

static inline void show_color_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    uint32_t grb = ((uint32_t)g << 16) | ((uint32_t)r << 8) | (uint32_t)b;
    pio_sm_put_blocking(pio0, 0, grb << 8u);
}

static void init_pio_for_ws2812()
{
    uint offset = pio_add_program(pio0, &ws2812_program);
    ws2812_program_init(pio0, 0, offset, WS2812_PIN, 800000, true);
}

static void record_and_send()
{
    IMU_ST_SENSOR_DATA gyro, accel;

    show_color_rgb(0, 0, 255);
    sleep_ms(COUNTDOWN_MS);

    show_color_rgb(0, 255, 0);
    uint32_t n = 0;
    uint64_t start = time_us_64();
    uint64_t end = start + RECORD_TIME_US;
    uint64_t next = start;
    uint64_t now = start;
    while (now < end && n < MAX_SAMPLES) {
        while (time_us_64() < next) { tight_loop_contents(); }
        next += SAMPLE_PERIOD_US;
        imuDataOnlyGet(&gyro, &accel);
        buf[n][0] = accel.s16X;
        buf[n][1] = accel.s16Y;
        buf[n][2] = accel.s16Z;
        n++;
        now = time_us_64();
    }
    uint64_t duration = now - start;

    show_color_rgb(255, 160, 0);
    printf("BEGIN %lu %llu\n", (unsigned long)n, (unsigned long long)duration);
    printf("ax,ay,az\n");
    for (uint32_t i = 0; i < n; i++) {
        printf("%d,%d,%d\n", buf[i][0], buf[i][1], buf[i][2]);
    }
    printf("END\n");
    stdio_flush();
}

int main()
{
    stdio_init_all();
    init_pio_for_ws2812();
    show_color_rgb(255, 0, 0);
    sleep_ms(1500);

    IMU_EN_SENSOR_TYPE sensor;
    imuInit(&sensor);
    if (sensor != IMU_EN_SENSOR_TYPE_ICM20948) {
        for (;;) {
            printf("ERROR no ICM20948\n");
            show_color_rgb(255, 0, 0); sleep_ms(250);
            show_color_rgb(0, 0, 0);   sleep_ms(250);
        }
    }

    show_color_rgb(8, 8, 8);
    printf("READY\n");

    for (;;) {
        int c = getchar_timeout_us(100 * 1000);
        if (c == 'r') {
            record_and_send();
            show_color_rgb(8, 8, 8);
        } else if (c == 'p') {
            printf("PONG\n");
        }
    }
}
