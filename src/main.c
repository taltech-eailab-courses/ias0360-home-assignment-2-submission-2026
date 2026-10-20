#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/sync.h"

#include "ff.h"
#include "sd_card.h"
#include "f_util.h"
#include "hw_config.h"

#include "DEV_Config.h"
#include "LCD_Driver.h"
#include "LCD_GUI.h"
#include "LCD_Touch.h"

#define PATH_MAX_LEN 256

static uint8_t s_tp_save_buf[BOX_W * BOX_H];

static uint8_t s_downsampled[DOWNSAMPLED_SIZE * DOWNSAMPLED_SIZE];
static uint32_t s_last_sample_id = 0;
static uint16_t s_last_nonzero_px = 0;
static uint8_t s_last_max_px = 0;

static char s_chunk_buf[4096];

static FATFS fs;
static FIL s_fil;
static char s_path[PATH_MAX_LEN];
static char s_name[64];
static sd_card_t *g_sd = NULL;
static const char *g_drive = NULL;

static mutex_t mutex;
static TP_DATA tp_data;
static uint32_t s_max_sample_id = 0;

static void join_path(char *out, size_t out_sz,
                      const char *drive, const char *rel) {
    if (rel && rel[0] == '/') rel++;
    if (drive && drive[strlen(drive) - 1] == '/')
        snprintf(out, out_sz, "%s%s", drive, rel ? rel : "");
    else
        snprintf(out, out_sz, "%s/%s", drive, rel ? rel : "");
}

static void sd_unmount_and_deinit(void) {
    if (g_drive) {
        f_unmount(g_drive);
    }
    if (g_sd && g_sd->deinit) {
        g_sd->deinit(g_sd);
    }
    gpio_pull_up(SD_CS_PIN);
}

static bool sd_init_and_mount(void) {
    if (!sd_init_driver()) {
        printf("[CORE1] [ERR] [SD] sd_init_driver() failed\n");
        return false;
    }

    g_sd = sd_get_by_num(0);
    if (!g_sd) {
        printf("[CORE1] [ERR] [SD] No SD config found (sd_get_by_num(0) == NULL)\n");
        return false;
    }

    g_drive = sd_get_drive_prefix(g_sd);
    if (!g_drive) {
        printf("[CORE1] [ERR] [SD] sd_get_drive_prefix() returned NULL\n");
        return false;
    }

    // Force driver state to uninitialized so f_mount performs a fresh probe.
    sd_unmount_and_deinit();

    FRESULT fr = f_mount(&fs, g_drive, 1);
    printf("[CORE1] [SD] f_mount -> %s (%d)\n", FRESULT_str(fr), fr);

    if (fr != FR_OK) {
        printf("[CORE1] [ERR] [SD] Mount failed: %s (%d)\n", FRESULT_str(fr), fr);
        sd_unmount_and_deinit();
        return false;
    }

    return true;
}

static FRESULT check_and_list_files(const char *root_drive) {
    static char root[PATH_MAX_LEN];
    join_path(root, sizeof(root), root_drive, "");

    static DIR dir;
    static FILINFO fno;
    FRESULT fr = f_opendir(&dir, root);
    if (fr != FR_OK) {
        printf("[CORE1] [ERR] [SD] f_opendir('%s') -> %s (%d)\n",
               root, FRESULT_str(fr), fr);
        return fr;
    }

    for (;;) {
        fr = f_readdir(&dir, &fno);
        if (fr != FR_OK || fno.fname[0] == '\0') break;
        if (!(fno.fattrib & AM_DIR)) {
            unsigned long sid = 0;
            if (sscanf(fno.fname, "sample_%lu.txt", &sid) == 1 &&
                (uint32_t)sid > s_max_sample_id) {
                s_max_sample_id = (uint32_t)sid;
            }
        }
    }
    f_closedir(&dir);

    printf("[CORE1] [SD] Mounted '%s'. Existing dataset up to sample %lu\n\n",
           root_drive, (unsigned long)s_max_sample_id);
    return FR_OK;
}

static void downsample_canvas(const uint8_t *src, uint8_t *dst) {
    for (int r = 0; r < DOWNSAMPLED_SIZE; r++) {
        int y0 = (r * BOX_H) / DOWNSAMPLED_SIZE;
        int y1 = ((r + 1) * BOX_H) / DOWNSAMPLED_SIZE;
        for (int c = 0; c < DOWNSAMPLED_SIZE; c++) {
            int x0 = (c * BOX_W) / DOWNSAMPLED_SIZE;
            int x1 = ((c + 1) * BOX_W) / DOWNSAMPLED_SIZE;
            uint32_t count = 0;
            uint32_t area = (uint32_t)((y1 - y0) * (x1 - x0));
            for (int y = y0; y < y1; y++) {
                const uint8_t *row = &src[y * BOX_W];
                for (int x = x0; x < x1; x++) {
                    count += row[x];
                }
            }
            dst[r * DOWNSAMPLED_SIZE + c] =
                (area > 0) ? (uint8_t)((count * 255) / area) : 0;
        }
    }
}

static void print_ascii_preview(const uint8_t *img) {
    static const char ramp[] = " .:-=+*#%@";
    const int ramp_len = (int)(sizeof(ramp) - 1);
    for (int r = 0; r < DOWNSAMPLED_SIZE; r++) {
        putchar(' ');
        for (int c = 0; c < DOWNSAMPLED_SIZE; c++) {
            uint8_t px = img[r * DOWNSAMPLED_SIZE + c];
            int idx = (px * (ramp_len - 1)) / 255;
            putchar(ramp[idx]);
        }
        putchar('\n');
    }
}

void core1_entry(void) {
    printf("[CORE1] Worker started (28x28 downsampling & SD logging)\n");

    bool mounted = sd_init_and_mount();
    if (mounted) {
        check_and_list_files(g_drive);
    } else {
        printf("[CORE1] [WARN] [SD] Card not detected at boot (standby)\n");
    }

    uint32_t gesture_counter = 0;

    while (true) {
        uint32_t msg = multicore_fifo_pop_blocking();
        if (msg != DATA_READY_FLAG) {
            continue;
        }

        mutex_enter_blocking(&mutex);
        downsample_canvas(s_tp_save_buf, s_downsampled);
        mutex_exit(&mutex);

        gesture_counter++;

        uint16_t nonzero_px = 0;
        uint8_t max_px = 0;
        for (int i = 0; i < DOWNSAMPLED_SIZE * DOWNSAMPLED_SIZE; i++) {
            uint8_t v = s_downsampled[i];
            if (v > 0) {
                nonzero_px++;
                if (v > max_px) {
                    max_px = v;
                }
            }
        }
        s_last_nonzero_px = nonzero_px;
        s_last_max_px = max_px;

        if (!mounted) {
            printf("[CORE1] [SD] Mounting on demand...\n");
            mounted = sd_init_and_mount();
            if (mounted) {
                check_and_list_files(g_drive);
            }
        }

        uint32_t target_sd_id = mounted ? (s_max_sample_id + 1) : gesture_counter;
        s_last_sample_id = target_sd_id;

        if (mounted) {
            printf("\n[CORE1] [SAMPLE] %lu (active_px=%u, peak=%u)\n",
                   (unsigned long)target_sd_id, nonzero_px, max_px);
        } else {
            printf("\n[CORE1] [GESTURE] %lu (RAM-only, active_px=%u, peak=%u)\n",
                   (unsigned long)gesture_counter, nonzero_px, max_px);
        }

        printf("[CORE1] [PREVIEW] Downsampled 28x28 grayscale matrix:\n");
        print_ascii_preview(s_downsampled);

        bool write_ok = false;
        if (mounted) {
            snprintf(s_name, sizeof(s_name), "sample_%lu.txt",
                     (unsigned long)target_sd_id);
            join_path(s_path, sizeof(s_path), g_drive, s_name);

            FRESULT fr = f_open(&s_fil, s_path, FA_WRITE | FA_CREATE_ALWAYS);
            if (fr == FR_OK) {
                int hlen = snprintf(s_chunk_buf, sizeof(s_chunk_buf),
                                    "# Sample: %lu\n",
                                    (unsigned long)target_sd_id);

                UINT bw = 0;
                fr = f_write(&s_fil, s_chunk_buf, (UINT)hlen, &bw);

                if (fr == FR_OK) {
                    for (int r = 0; r < DOWNSAMPLED_SIZE; r++) {
                        int pos = 0;
                        for (int c = 0; c < DOWNSAMPLED_SIZE; c++) {
                            uint8_t px = s_downsampled[r * DOWNSAMPLED_SIZE + c];
                            pos += snprintf(&s_chunk_buf[pos],
                                            sizeof(s_chunk_buf) - pos,
                                            (c == DOWNSAMPLED_SIZE - 1) ? "%u\n" : "%u,",
                                            px);
                        }
                        fr = f_write(&s_fil, s_chunk_buf, (UINT)pos, &bw);
                        if (fr != FR_OK) break;
                    }
                }
                if (fr == FR_OK) {
                    fr = f_sync(&s_fil);
                }
                f_close(&s_fil);
            }

            if (fr != FR_OK) {
                printf("[CORE1] [ERR] [SD] Write %s: %s (%d)\n",
                       s_name, FRESULT_str(fr), fr);
                sd_unmount_and_deinit();
                mounted = false;
            } else {
                join_path(s_path, sizeof(s_path), g_drive, "dataset.csv");
                fr = f_open(&s_fil, s_path, FA_WRITE | FA_OPEN_APPEND);
                if (fr == FR_OK) {
                    int pos = snprintf(s_chunk_buf, sizeof(s_chunk_buf),
                                       "%lu", (unsigned long)target_sd_id);
                    for (int i = 0; i < DOWNSAMPLED_SIZE * DOWNSAMPLED_SIZE; i++) {
                        pos += snprintf(&s_chunk_buf[pos], sizeof(s_chunk_buf) - pos,
                                        ",%u", s_downsampled[i]);
                        if (pos >= (int)(sizeof(s_chunk_buf) - 32)) {
                            UINT bw = 0;
                            fr = f_write(&s_fil, s_chunk_buf, (UINT)pos, &bw);
                            pos = 0;
                            if (fr != FR_OK) break;
                        }
                    }
                    if (fr == FR_OK && pos > 0) {
                        if (pos < (int)sizeof(s_chunk_buf)) {
                            s_chunk_buf[pos++] = '\n';
                        }
                        UINT bw = 0;
                        fr = f_write(&s_fil, s_chunk_buf, (UINT)pos, &bw);
                    }
                    if (fr == FR_OK) {
                        fr = f_sync(&s_fil);
                    }
                    f_close(&s_fil);
                }

                if (fr != FR_OK) {
                    printf("[CORE1] [ERR] [SD] Append dataset.csv: %s (%d)\n",
                           FRESULT_str(fr), fr);
                    sd_unmount_and_deinit();
                    mounted = false;
                } else {
                    write_ok = true;
                }
            }

            if (mounted && g_sd && g_sd->sync) {
                g_sd->sync(g_sd);
            }
        }

        if (write_ok) {
            s_max_sample_id = target_sd_id;
            printf("[CORE1] [SD] Saved %s (dataset.csv appended)\n", s_name);
            multicore_fifo_push_blocking(WRITE_SUCCESS_FLAG);
        } else {
            printf("[CORE1] [WARN] [SD] Storage unavailable (RAM-only)\n");
            multicore_fifo_push_blocking(SD_UNAVAILABLE_FLAG);
        }
    }
}

int main(void) {
    stdio_init_all();
    System_Init();

    mutex_init(&mutex);

    tp_data.data = s_tp_save_buf;
    tp_data.data_len = sizeof(s_tp_save_buf);

    LCD_SCAN_DIR lcd_scan_dir = SCAN_DIR_DFT;
    LCD_Init(lcd_scan_dir, 1000);
    TP_Init(lcd_scan_dir, &tp_data, &mutex);
    TP_GetAdFac();
    TP_Dialog();

    multicore_launch_core1(core1_entry);
    sleep_ms(2000);

    printf("[SYS] Ready for interactive touchscreen gestures (0-9)...\n\n");

    char disp_buf[80];

    while (true) {
        if (multicore_fifo_rvalid()) {
            uint32_t msg = multicore_fifo_pop_blocking();

            if (msg == WRITE_SUCCESS_FLAG || msg == SD_UNAVAILABLE_FLAG) {
                TP_ShowStatus((msg == WRITE_SUCCESS_FLAG) ?
                              "Status: done (sd)" : "Status: done (no sd)", BLACK);

                snprintf(disp_buf, sizeof(disp_buf),
                         "%s sample_%lu (28x28 grayscale, active=%u, max=%u)",
                         (msg == WRITE_SUCCESS_FLAG) ? "Saved" : "Captured",
                         (unsigned long)s_last_sample_id,
                         (unsigned int)s_last_nonzero_px,
                         (unsigned int)s_last_max_px);
                TP_ShowTelemetry(disp_buf);
                TP_SetSaveBusy(false);
            }
        }

        TP_DrawBoard();
    }

    return 0;
}
