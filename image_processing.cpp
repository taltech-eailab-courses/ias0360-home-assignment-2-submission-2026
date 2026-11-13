#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/sync.h"

#include "LCD_Driver.h"
#include "LCD_Touch.h"
#include "LCD_GUI.h"
#include "DEV_Config.h"

#include "ff.h"
#include "sd_card.h"
#include "f_util.h"
#include "hw_config.h"

#define PATH_MAX_LEN 256
#define IMAGE_WIDTH BOX_W
#define IMAGE_HEIGHT BOX_H
#define DOWNSAMPLED_WIDTH 28
#define DOWNSAMPLED_HEIGHT 28

#ifndef DIGIT_LABEL
#define DIGIT_LABEL 1
#endif

#if DIGIT_LABEL < 0 || DIGIT_LABEL > 9
#error "DIGIT_LABEL must be between 0 and 9"
#endif

static mutex_t touch_mutex;
static TP_DATA tp_data = {0, NULL};

static FATFS fs;
static sd_card_t *g_sd = NULL;
static const char *g_drive = NULL;


static void downsample_to_mnist(const uint8_t *input, uint8_t *output,
                                size_t width, size_t height) {
    for (size_t output_y = 0; output_y < DOWNSAMPLED_HEIGHT; ++output_y) {
        const size_t y_start = output_y * height / DOWNSAMPLED_HEIGHT;
        const size_t y_end = (output_y + 1) * height / DOWNSAMPLED_HEIGHT;

        for (size_t output_x = 0; output_x < DOWNSAMPLED_WIDTH; ++output_x) {
            const size_t x_start = output_x * width / DOWNSAMPLED_WIDTH;
            const size_t x_end = (output_x + 1) * width / DOWNSAMPLED_WIDTH;
            int sum = 0;

            for (size_t y = y_start; y < y_end; ++y) {
                for (size_t x = x_start; x < x_end; ++x) {
                    sum += input[y * width + x];
                }
            }

            const size_t block_area = (y_end - y_start) * (x_end - x_start);
            output[output_y * DOWNSAMPLED_WIDTH + output_x] =
                sum * 2 > (int)block_area ? 1 : 0;
        }
    }
}

static void join_path(char *out, size_t out_size, const char *drive, const char *relative_path) {
    if (relative_path && relative_path[0] == '/') {
        ++relative_path;
    }

    if (drive && drive[strlen(drive) - 1] == '/') {
        snprintf(out, out_size, "%s%s", drive,
                 relative_path ? relative_path : "");
    } else {
        snprintf(out, out_size, "%s/%s", drive,
                 relative_path ? relative_path : "");
    }
}

static bool sd_init_and_mount(void) {
    if (!sd_init_driver()) {
        printf("sd_init_driver() failed\n");
        return false;
    }

    g_sd = sd_get_by_num(0);
    if (!g_sd) {
        return false;
    }

    g_drive = sd_get_drive_prefix(g_sd);
    if (!g_drive) {
        return false;
    }

    FRESULT result = f_mount(&fs, g_drive, 1);
    if (result == FR_NO_FILESYSTEM) {
        BYTE work[4096];
        MKFS_PARM options = {FM_FAT | FM_SFD, 0, 0, 0, 0};
        result = f_mkfs(g_drive, &options, work, sizeof(work));
        if (result == FR_OK) {
            result = f_mount(&fs, g_drive, 1);
        }
    }

    return result == FR_OK;
}

static bool write_bitmap_file(const char *name, const uint8_t *image, size_t width, size_t height) {
    char path[PATH_MAX_LEN];
    join_path(path, sizeof(path), g_drive, name);

    FIL file;
    FRESULT result = f_open(&file, path, FA_WRITE | FA_CREATE_ALWAYS);
    if (result != FR_OK) {
        printf("Could not open %s: %s (%d)\n", path, FRESULT_str(result), result);
        return false;
    }

    char row[IMAGE_WIDTH + 1];
    bool success = true;
    for (size_t y = 0; y < height && success; ++y) {
        for (size_t x = 0; x < width; ++x) {
            row[x] = image[y * width + x] ? '1' : '0';
        }
        row[width] = '\n';

        UINT written = 0;
        result = f_write(&file, row, (UINT)(width + 1), &written);
        success = result == FR_OK && written == width + 1;
    }

    if (success) {
        success = f_sync(&file) == FR_OK;
    }
    f_close(&file);

    if (success) {
        printf("Saved %s\n", path);
    } else {
        printf("Could not write %s\n", path);
    }
    return success;
}

static bool save_capture(uint32_t capture_number) {
    if (!tp_data.data || tp_data.data_len != IMAGE_WIDTH * IMAGE_HEIGHT) {
        printf("No complete drawing is available\n");
        return false;
    }

    char name[64];
    uint8_t *processed = (uint8_t *)malloc(
        DOWNSAMPLED_WIDTH * DOWNSAMPLED_HEIGHT);

    if (!processed) {
        printf("Could not allocate preprocessing buffer\n");
        return false;
    }

    /* ORIGINAL */
    snprintf(name, sizeof(name), "%d_%lu_original.txt",
             DIGIT_LABEL, (unsigned long)(capture_number + 1));
    bool success = write_bitmap_file(name, tp_data.data, IMAGE_WIDTH, IMAGE_HEIGHT);


    /* DOWNSAMPLE TO 28x28 */
    downsample_to_mnist(tp_data.data, processed, IMAGE_WIDTH, IMAGE_HEIGHT);
    snprintf(name, sizeof(name), "%d_%lu_downsampled.txt",
             DIGIT_LABEL, (unsigned long)(capture_number + 1));
    success = write_bitmap_file(name, processed, DOWNSAMPLED_WIDTH, DOWNSAMPLED_HEIGHT) && success;

    free(processed);

    return success;
}

static void core1_entry(void) {
    printf("Core 1 initialized: waiting for drawing captures\n");

    if (!sd_init_and_mount()) {
        printf("Core 1: SD mount failed\n");
        multicore_fifo_push_blocking(WRITE_FAILED_FLAG);
        while (true) {
            tight_loop_contents();
        }
    }

    uint32_t capture_number = 0;
    while (true) {
        uint32_t message = multicore_fifo_pop_blocking();
        if (message != DATA_READY_FLAG) {
            continue;
        }

        mutex_enter_blocking(&touch_mutex);
        bool success = save_capture(capture_number);
        mutex_exit(&touch_mutex);

        if (success) {
            ++capture_number;
            if (capture_number == 10) {
                multicore_fifo_push_blocking(TASK_COMPLETE_FLAG);
                printf("Collected 10 samples for digit %d\n", DIGIT_LABEL);
                break;
            }
            multicore_fifo_push_blocking(TASK_COMPLETE_FLAG);
        } else {
            multicore_fifo_push_blocking(WRITE_FAILED_FLAG);
        }
    }
}

int main(void) {
    stdio_init_all();
    System_Init();
    mutex_init(&touch_mutex);

    LCD_SCAN_DIR lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir, 1000);
    TP_Init(lcd_scan_dir, &tp_data, &touch_mutex);
    TP_GetAdFac();
    TP_Dialog();

    multicore_launch_core1(core1_entry);

    while (true) {
        LCD_SetBackLight(1000);
        TP_DrawBoard();

        if (multicore_fifo_rvalid()) {
            uint32_t message = multicore_fifo_pop_blocking();
            if (message == TASK_COMPLETE_FLAG) {
                printf("Capture saved: original + 28x28 image\n");
            } else if (message == WRITE_FAILED_FLAG) {
                printf("Capture failed: check the SD card and serial output\n");
            }
        }
    }
}
