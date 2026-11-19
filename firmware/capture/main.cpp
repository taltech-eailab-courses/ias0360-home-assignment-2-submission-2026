#include <stdio.h>
#include <string.h>

#include "capture_runtime.h"
#include "ff.h"
#include "hw_config.h"
#include "sd_card.h"
#include "LCD_Touch.h"
#include "pico/mutex.h"
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"

static_assert(BOX_W == 280 && BOX_H == 240, "Update PGM dimensions for a new canvas");

static FATFS g_fs;                    // All SD operations run on Core 1.
static uint8_t g_row[BOX_W];           // Owned by the SD worker only.
static uint8_t g_snapshot[capture::kPixels];
static capture::Progress g_progress;   // UI-owned after the startup handshake.
static queue_t g_jobs;
static queue_t g_results;
static queue_t g_boot;
static const char *g_drive = nullptr;
static bool g_manual_save = false;

struct SaveResult { unsigned slot; bool ok; };

static void status(const char *message, COLOR color = BLACK) {
    GUI_DrawRectangle(0, 300, 479, 319, WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(4, 303, message, &Font12, WHITE, color);
}

static void prompt() {
    GUI_DrawRectangle(0, 0, 479, 44, WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    char line[72];
    unsigned next = g_progress.next();
    if (next >= capture::kTotal) {
        GUI_DisString_EN(4, 2, g_progress.done() ? "DONE - 300 images" : "Finishing SD save...",
                         &Font20, WHITE, BLUE);
    } else {
        const auto current = capture::item(next);
        snprintf(line, sizeof(line), "%s  Draw: %u", current.split, current.digit);
        GUI_DisString_EN(4, 2, line, &Font20, WHITE, BLACK);
    }
    if (next < capture::kTotal) {
        const auto current = capture::item(next);
        snprintf(line, sizeof(line), "Round %u/%u  Saved %u/300  %s",
                 current.round, current.rounds, g_progress.saved,
                 g_progress.failed ? "SD ERROR" : (g_progress.pending() ? "Saving..." : "Auto: 2s"));
    } else {
        snprintf(line, sizeof(line), "Saved %u/300  %s", g_progress.saved,
                 g_progress.done() ? "Safe to power off" : "Wait for SD / SAVE retries error");
    }
    GUI_DisString_EN(4, 26, line, &Font12, WHITE, BLACK);
}

static void print_state() {
    unsigned next = g_progress.next();
    if (next < capture::kTotal) {
        const auto current = capture::item(next);
        printf("H2 READY v=2 split=%s digit=%u round=%u/%u saved=%u/%u pending=%u error=%u ink=%u autosave=2000ms idle_ms=%llu\n",
               current.split, current.digit, current.round, current.rounds,
               g_progress.saved, capture::kTotal, g_progress.pending(), g_progress.failed,
               TP_InkPixels(), (unsigned long long)((time_us_64() - TP_LastInputUs()) / 1000));
    } else {
        printf("H2 %s v=2 saved=%u/%u pending=%u error=%u\n",
               g_progress.done() ? "DONE" : "FINISHING", g_progress.saved,
               capture::kTotal, g_progress.pending(), g_progress.failed);
    }
}

static bool mount_and_prepare(const char **drive) {
    if (!sd_init_driver()) return false;
    sd_card_t *card = sd_get_by_num(0);
    if (card == nullptr) return false;
    *drive = sd_get_drive_prefix(card);
    if (*drive == nullptr || f_mount(&g_fs, *drive, 1) != FR_OK) return false;

    const char *folders[] = {"H2DIGITS", "H2DIGITS/TRAIN", "H2DIGITS/VAL", "H2DIGITS/TEST"};
    for (const char *folder : folders) {
        char path[64];
        snprintf(path, sizeof(path), "%s/%s", *drive, folder);
        FRESULT result = f_mkdir(path);
        if (result != FR_OK && result != FR_EXIST) return false;
    }
    return true;
}

// Called by the storage core before publishing the startup handshake.
static bool restore_progress(const char *drive) {
    for (unsigned slot = 0; slot < capture::kTotal; ++slot) {
        char path[64];
        capture::path(path, sizeof(path), drive, slot);
        FILINFO info;
        FRESULT result = f_stat(path, &info);
        if (result == FR_OK) {
            if ((info.fattrib & AM_DIR) || info.fsize != capture::kFileSize) {
                printf("Invalid existing image: %s\n", path);
                return false;
            }
            g_progress.complete[slot] = true;
            ++g_progress.saved;
        } else if (result != FR_NO_FILE) {
            return false;
        }
    }
    return true;
}

static bool write_all(FIL *file, const void *data, UINT size) {
    UINT written = 0;
    return f_write(file, data, size, &written) == FR_OK && written == size;
}

static bool save_image(const char *drive, unsigned slot, const uint8_t *drawing) {
    char path[64], temporary[64];
    capture::path(path, sizeof(path), drive, slot);
    capture::path(temporary, sizeof(temporary), drive, slot, "TMP");
    FILINFO info;
    // Never overwrite a completed training, validation or test image.
    if (f_stat(path, &info) != FR_NO_FILE) return false;
    FIL file;
    if (f_open(&file, temporary, FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) return false;
    bool ok = write_all(&file, capture::kPgmHeader, sizeof(capture::kPgmHeader) - 1);
    for (unsigned y = 0; ok && y < BOX_H; ++y) {
        for (unsigned x = 0; x < BOX_W; ++x) {
            g_row[x] = drawing[y * BOX_W + x] ? 0 : 255;
        }
        ok = write_all(&file, g_row, sizeof(g_row));
    }
    if (ok) ok = f_sync(&file) == FR_OK;
    if (f_close(&file) != FR_OK) ok = false;
    if (ok) ok = f_rename(temporary, path) == FR_OK;
    printf("%s: %s\n", path, ok ? "saved" : "FAILED");
    return ok;
}

// Queue handoff protects the snapshot until the UI receives the result.
// The worker never touches LCD, touch state, or the live drawing buffer.
static void sd_worker() {
    unsigned boot_error = 0;
    if (!mount_and_prepare(&g_drive)) boot_error = 1;
    else if (!restore_progress(g_drive)) boot_error = 2;
    queue_add_blocking(&g_boot, &boot_error);
    if (boot_error) {
        while (true) sleep_ms(100);
    }
    while (true) {
        unsigned slot;
        queue_remove_blocking(&g_jobs, &slot);
        SaveResult result = {slot, save_image(g_drive, slot, g_snapshot)};
        queue_add_blocking(&g_results, &result);
    }
}

static bool enqueue_current() {
    const unsigned slot = g_progress.next();
    if (g_progress.pending() || slot >= capture::kTotal ||
        TP_InkPixels() < capture::kMinInkPixels) return false;
    memcpy(g_snapshot, TP_SavedData(), sizeof(g_snapshot));
    if (!g_progress.reserve(slot)) return false;
    if (!queue_try_add(&g_jobs, &slot)) {
        g_progress.pending_slot = capture::kTotal;
        status("Save queue busy. Drawing kept; try again.", RED);
        return false;
    }
    // A snapshot now owns this image. The user can draw the next one immediately.
    TP_ClearCanvas();
    g_manual_save = false;
    prompt();
    status("Saving in background. You can draw the next digit now.", BLUE);
    print_state();
    return true;
}

static void retry_previous() {
    if (!g_progress.pending() || !g_progress.failed) return;
    const unsigned slot = g_progress.pending_slot;
    if (queue_try_add(&g_jobs, &slot)) {
        g_progress.failed = false;
        prompt();
        status("Retrying previous save. Current drawing is kept.", BLUE);
    }
}

static void poll_results() {
    SaveResult result;
    if (!queue_try_remove(&g_results, &result)) return;
    if (!g_progress.finish(result.slot, result.ok)) return;
    prompt();
    print_state();
    if (!result.ok) {
        status("SD ERROR: SAVE retries previous; current drawing is kept.", RED);
    } else if (g_progress.done()) {
        status("ALL DONE. All 300 images saved. Safe to power off.", BLUE);
    } else {
        status("Saved. Draw next digit; SAVE or 2s pause to continue.", BLUE);
    }
}

int main() {
    stdio_init_all();
    sleep_ms(1500);
    System_Init();
    mutex_t unused_mutex;
    TP_DATA unused_data = {};
    mutex_init(&unused_mutex);
    LCD_Init(SCAN_DIR_DFT, 1000);
    TP_Init(SCAN_DIR_DFT, &unused_data, &unused_mutex);
    TP_GetAdFac();
    TP_Dialog();
    TP_ConsumeClearRequest();
    status("Opening SD card...");
    queue_init(&g_jobs, sizeof(unsigned), 1);
    queue_init(&g_results, sizeof(SaveResult), 1);
    queue_init(&g_boot, sizeof(unsigned), 1);
    multicore_launch_core1(sd_worker);
    unsigned boot_error;
    queue_remove_blocking(&g_boot, &boot_error);
    if (boot_error == 1) {
        printf("H2 ERROR: SD mount/folder setup failed\n");
        status("SD ERROR: power off and check card", RED);
        while (true) sleep_ms(100);
    }
    if (boot_error == 2) {
        printf("H2 ERROR: existing dataset cannot be read\n");
        status("FILE ERROR: check H2DIGITS on computer", RED);
        while (true) sleep_ms(100);
    }
    prompt();
    print_state();
    status("CLEAR left. SAVE right. Auto-save after 2s pen-up (20 pixels).");

    while (true) {
        if (getchar_timeout_us(0) == '?') print_state();
        poll_results();
        TP_DrawBoard();
        if (TP_ConsumeClearRequest()) {
            g_manual_save = false;
            // CLEAR acts on the live canvas only; queued snapshots remain intact.
            if (g_progress.failed) {
                status("SD ERROR: SAVE retries previous; current canvas cleared.", RED);
            } else {
                status("Canvas cleared. Draw the same requested digit.");
            }
        }
        if (TP_ConsumeSaveRequest()) {
            if (g_progress.failed) {
                retry_previous();
            } else if (g_progress.next() >= capture::kTotal) {
                status(g_progress.done() ? "All images saved. Safe to power off." : "Finishing last save. Please wait.");
            } else if (TP_InkPixels() < capture::kMinInkPixels) {
                status("At least 20 drawn pixels needed before SAVE.", RED);
            } else {
                g_manual_save = true;
                if (g_progress.pending()) status("Previous save running. Current digit queued; keep drawing.");
            }
        }
        if (!g_progress.pending() && !g_progress.done()) {
            const bool automatic = capture::autosave_due(TP_InkPixels(), TP_IsTouchDown(),
                                                        TP_LastInputUs(), time_us_64());
            // A queued manual save waits for pen-up if SD was still busy.
            if ((g_manual_save && !TP_IsTouchDown()) || automatic) enqueue_current();
        }
    }
}
