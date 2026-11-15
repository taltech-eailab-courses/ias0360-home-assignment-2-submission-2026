// IMU gesture data collection (Step 2).
//
// Core 0 samples the accelerometer at 2 kHz for 2.1 s into a queue, core 1
// writes it as CSV to /<class>/<class>_NNN.txt on the SD card. Controlled from
// the USB serial console: choose the class and the number of recordings.
//
// LED: red = starting, blue = get ready, green = recording, white = saved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hardware/pio.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "pico/util/queue.h"

extern "C" {
#include "f_util.h"
#include "ff.h"
#include "hw_config.h"
#include "sd_card.h"
}

#include "status_led.h"

// Included last: this header defines `true`/`false` as macros.
extern "C" {
#include "icm20948.h"
}

// 2 kHz for 2.1 s = 4200 samples, same as the provided dataset.
static const uint32_t kSamplePeriodUs = 500;
static const uint32_t kRecordTimeUs = 2100 * 1000;
static const uint32_t kDefaultRecordings = 25;
static const uint32_t kCountdownMs = 1500;              // get-ready time before each recording
static const char* const kClassNames[] = {"negative", "ring", "slope", "wing"};
static const int kClassCount = 4;

// core 0 <-> core 1 shared state
typedef struct __attribute__((packed)) {
    int16_t ax, ay, az;
    uint8_t _pad[2];  // 8 bytes per sample
} imu_sample_t;

#define QUEUE_DEPTH (8192)  // 64 KB
static queue_t sample_q;

static volatile int sd_state = 0;        // 0 = mounting, 1 = ready, -1 = failed
static volatile int request_class = -1;  // class of the next file to open
static volatile uint32_t open_seq = 0;   // core0 increments to request a new file
static volatile uint32_t opened_seq = 0; // core1: file for open_seq is open (or failed)
static volatile bool open_ok = false;
static volatile bool recording = false;  // core0 is sampling
static volatile uint32_t done_seq = 0;   // core1: file for this seq is closed
static volatile uint32_t lines_written = 0;
static char current_path[96];

static FATFS fs;

// core 1: SD card
static bool mount_sd(const char** drive_out) {
    if (!sd_init_driver()) {
        printf("sd_init_driver() failed\n");
        return false;
    }
    sd_card_t* sd = sd_get_by_num(0);
    if (!sd) {
        printf("no SD card configuration\n");
        return false;
    }
    const char* drive = sd_get_drive_prefix(sd);
    FRESULT fr = f_mount(&fs, drive, 1);
    if (fr != FR_OK) {
        printf("f_mount -> %s (%d)\n", FRESULT_str(fr), fr);
        return false;
    }
    *drive_out = drive;
    return true;
}

// Next free NNN in /<class>/<class>_NNN.txt.
static int next_index(const char* dir_path, const char* cls) {
    DIR dir;
    FILINFO info;
    int next = 0;
    const size_t prefix_len = strlen(cls) + 1;  // "<class>_"
    if (f_opendir(&dir, dir_path) != FR_OK) return 0;
    while (f_readdir(&dir, &info) == FR_OK && info.fname[0] != '\0') {
        if (strncmp(info.fname, cls, prefix_len - 1) == 0 && info.fname[prefix_len - 1] == '_') {
            const int n = atoi(info.fname + prefix_len);
            if (n + 1 > next) next = n + 1;
        }
    }
    f_closedir(&dir);
    return next;
}

static bool open_next_file(FIL* f, const char* drive, int cls) {
    char dir_path[48];
    snprintf(dir_path, sizeof dir_path, "%s/%s", drive, kClassNames[cls]);
    FRESULT fr = f_mkdir(dir_path);
    if (fr != FR_OK && fr != FR_EXIST) {
        printf("f_mkdir(%s) -> %s (%d)\n", dir_path, FRESULT_str(fr), fr);
        return false;
    }
    snprintf(current_path, sizeof current_path, "%s/%s_%03d.txt", dir_path, kClassNames[cls],
             next_index(dir_path, kClassNames[cls]));
    fr = f_open(f, current_path, FA_WRITE | FA_CREATE_ALWAYS);
    if (fr != FR_OK) {
        printf("f_open(%s) -> %s (%d)\n", current_path, FRESULT_str(fr), fr);
        return false;
    }
    UINT bw = 0;
    const char* header = "ax,ay,az\n";
    fr = f_write(f, header, (UINT)strlen(header), &bw);
    return fr == FR_OK && bw == strlen(header);
}

static void core1_entry(void) {
    const char* drive = NULL;
    sd_state = mount_sd(&drive) ? 1 : -1;
    if (sd_state < 0) {
        while (1) tight_loop_contents();
    }

    uint32_t handled = 0;
    for (;;) {
        while (open_seq == handled) tight_loop_contents();
        const uint32_t seq = open_seq;
        FIL f;
        open_ok = open_next_file(&f, drive, request_class);
        opened_seq = seq;
        handled = seq;
        if (!open_ok) continue;

        while (!recording) tight_loop_contents();  // core0 is counting down

        // Drain until sampling has stopped and the queue is empty.
        uint32_t lines = 0, since_sync = 0;
        imu_sample_t s;
        bool ok = true;
        for (;;) {
            if (!queue_try_remove(&sample_q, &s)) {
                if (!recording) break;
                continue;
            }
            char line[32];
            const int n = snprintf(line, sizeof line, "%d,%d,%d\n", s.ax, s.ay, s.az);
            UINT bw = 0;
            if (ok && (f_write(&f, line, (UINT)n, &bw) != FR_OK || bw != (UINT)n)) {
                printf("f_write failed on %s\n", current_path);
                ok = false;
            }
            ++lines;
            if (++since_sync >= 128) {
                f_sync(&f);
                since_sync = 0;
            }
        }
        f_close(&f);
        lines_written = lines;
        done_seq = seq;
    }
}

// core 0: console + IMU
static int read_char_blocking(void) {
    int ch;
    do {
        ch = getchar_timeout_us(100 * 1000);
    } while (ch == PICO_ERROR_TIMEOUT);
    return ch;
}

// Reads a number terminated by Enter; empty input returns `fallback`.
static uint32_t read_number(uint32_t fallback) {
    char buf[8];
    int len = 0;
    for (;;) {
        const int ch = read_char_blocking();
        if (ch == '\r' || ch == '\n') break;
        if (ch >= '0' && ch <= '9' && len < 7) {
            buf[len++] = (char)ch;
            putchar(ch);
        }
    }
    printf("\n");
    buf[len] = '\0';
    return len ? (uint32_t)atoi(buf) : fallback;
}

// Deletes every /<class>/*.txt recording (menu option 'x'). Core 1 is idle
// meanwhile, so FatFs is not used concurrently.
static void delete_class_recordings(int cls) {
    char dir_path[48];
    snprintf(dir_path, sizeof dir_path, "0:/%s", kClassNames[cls]);
    DIR dir;
    FILINFO info;
    if (f_opendir(&dir, dir_path) != FR_OK) {
        printf("no folder %s\n", dir_path);
        return;
    }
    int deleted = 0;
    while (f_readdir(&dir, &info) == FR_OK && info.fname[0] != '\0') {
        const size_t len = strlen(info.fname);
        if ((info.fattrib & AM_DIR) || len < 5 || strcmp(info.fname + len - 4, ".txt") != 0) continue;
        char path[96];
        snprintf(path, sizeof path, "%s/%s", dir_path, info.fname);
        if (f_unlink(path) == FR_OK) ++deleted;
    }
    f_closedir(&dir);
    printf("deleted %d recordings from %s\n", deleted, dir_path);
}

static bool user_wants_to_stop(void) {
    const int ch = getchar_timeout_us(0);
    return ch == 'q' || ch == 'Q';
}

static void record_session(int cls, uint32_t count, IMU_ST_SENSOR_DATA* gyro, IMU_ST_SENSOR_DATA* accel) {
    for (uint32_t i = 0; i < count; ++i) {
        // 1. Ask core 1 to open the next file.
        request_class = cls;
        const uint32_t seq = open_seq + 1;
        open_seq = seq;
        while (opened_seq != seq) tight_loop_contents();
        if (!open_ok) {
            printf("cannot create a file on the SD card, stopping\n");
            status_led_rgb(64, 0, 0);
            return;
        }

        // 2. Countdown.
        printf("[%s %lu/%lu] get ready...", kClassNames[cls], (unsigned long)(i + 1), (unsigned long)count);
        status_led_rgb(0, 0, 64);
        sleep_ms(kCountdownMs);
        printf(" GO!\n");

        // 3. Record at 2 kHz.
        uint32_t dropped = 0;
        status_led_rgb(0, 64, 0);
        recording = true;
        const uint64_t start = time_us_64();
        uint64_t next = start;
        while (next < start + kRecordTimeUs) {
            while (time_us_64() < next) tight_loop_contents();
            next += kSamplePeriodUs;
            imuDataOnlyGet(gyro, accel);
            imu_sample_t s = {accel->s16X, accel->s16Y, accel->s16Z, {0, 0}};
            if (!queue_try_add(&sample_q, &s)) {
                ++dropped;
                status_led_rgb(0, 0, 255);  // overflow
            }
        }
        recording = false;

        // 4. Wait for core 1 to flush and close the file.
        while (done_seq != seq) tight_loop_contents();
        printf("    saved %s: %lu samples (%.0f Hz)%s\n", current_path, (unsigned long)lines_written,
               lines_written / (kRecordTimeUs / 1e6f), dropped ? " - WARNING: samples dropped" : "");
        if (dropped) printf("    %lu samples were dropped (queue overflow)\n", (unsigned long)dropped);
        status_led_rgb(40, 40, 40);
        sleep_ms(300);

        if (user_wants_to_stop()) {
            printf("stopped by user after %lu recordings\n", (unsigned long)(i + 1));
            return;
        }
    }
}

int main(void) {
    stdio_init_all();
    status_led_init();
    status_led_rgb(64, 0, 0);
    for (int i = 0; i < 100 && !stdio_usb_connected(); ++i) sleep_ms(100);  // wait up to 10 s for the console
    sleep_ms(500);

    queue_init(&sample_q, sizeof(imu_sample_t), QUEUE_DEPTH);
    multicore_launch_core1(core1_entry);

    IMU_EN_SENSOR_TYPE sensor_type;
    IMU_ST_SENSOR_DATA gyro, accel;
    imuInit(&sensor_type);
    if (sensor_type == IMU_EN_SENSOR_TYPE_NULL) {
        printf("IMU not found\n");
        while (1) tight_loop_contents();
    }
    while (sd_state == 0) tight_loop_contents();
    if (sd_state < 0) {
        printf("SD card not available - check the card and reset the board\n");
        while (1) tight_loop_contents();
    }

    for (;;) {
        status_led_rgb(0, 0, 0);
        printf("\n=== HA2 IMU gesture data collection ===\n"
               "Hold the board like the provided dataset: screen facing DOWN\n"
               "(IMU x = right, y = backward, z = up) - otherwise set the axis map, see README.\n"
               "Each recording is %.1f s; start the gesture right after GO.\n\n",
               kRecordTimeUs / 1e6f);
        for (int c = 0; c < kClassCount; ++c) printf("  %d  %s\n", c, kClassNames[c]);
        printf("  x  delete all recordings of a class\n");
        printf("Select gesture class [0-%d] or x: ", kClassCount - 1);
        int ch;
        do {
            ch = read_char_blocking();
        } while ((ch < '0' || ch >= '0' + kClassCount) && ch != 'x');
        if (ch == 'x') {
            printf("x\nDelete ALL recordings of class [0-%d]: ", kClassCount - 1);
            do {
                ch = read_char_blocking();
            } while (ch < '0' || ch >= '0' + kClassCount);
            const int del = ch - '0';
            printf("%s - press y to confirm, anything else cancels: ", kClassNames[del]);
            if (read_char_blocking() == 'y') {
                printf("y\n");
                delete_class_recordings(del);
            } else {
                printf("cancelled\n");
            }
            continue;
        }
        const int cls = ch - '0';
        printf("%s\nNumber of recordings [Enter = %lu]: ", kClassNames[cls], (unsigned long)kDefaultRecordings);
        const uint32_t count = read_number(kDefaultRecordings);
        printf("Recording %lu x '%s' (press q between recordings to stop)\n", (unsigned long)count,
               kClassNames[cls]);
        record_session(cls, count, &gyro, &accel);
    }
}
