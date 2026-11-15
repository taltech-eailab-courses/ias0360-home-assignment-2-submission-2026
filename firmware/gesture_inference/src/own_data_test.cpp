// Own-data evaluation (Step 5 on the device).
//
// Classifies every recording on the SD card (/<class>/<class>_NNN.txt) with
// every model in the firmware and prints per-gesture predictions and, per
// model: accuracy, confusion matrix, average inference time, tensor arena use
// and model size. "RESULT," and "SUMMARY," lines are CSV.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

extern "C" {
#include "diskio.h"
#include "f_util.h"
#include "ff.h"
#include "hw_config.h"
#include "sd_card.h"
}

#include "data_preprocessor.hpp"
#include "model_registry.h"
#include "model_runner.hpp"
#include "preprocessing_settings.hpp"
#include "run_on_core1.hpp"
#include "status_led.h"

namespace {

constexpr int kMaxFilesPerClass = 64;
constexpr int kMaxNameLen = 48;

FATFS g_fs;
const char* g_drive = nullptr;

char g_files[kClassCount][kMaxFilesPerClass][kMaxNameLen];
int g_file_count[kClassCount];

int16_t g_raw[kInputWindowLength * kNumAxes];
DataPreprocessor g_preprocessor;
ModelRunner g_runner;

void wait_for_usb_console() {
    for (int i = 0; i < 100 && !stdio_usb_connected(); ++i) {
        sleep_ms(100);
    }
    sleep_ms(500);
}

// Diagnostic for a card that does not mount: prints sector 0 and the first
// partition's boot sector.
void dump_sector(const char* what, const uint8_t* s) {
    printf("%s: signature %02x%02x, OEM/FS id '", what, s[510], s[511]);
    for (int i = 3; i < 11; ++i) putchar(s[i] >= 32 && s[i] < 127 ? s[i] : '.');
    printf("', FAT32 type '");
    for (int i = 82; i < 90; ++i) putchar(s[i] >= 32 && s[i] < 127 ? s[i] : '.');
    printf("'\n  first 64 bytes:");
    for (int i = 0; i < 64; ++i) printf("%s%02x", i % 16 ? " " : "\n   ", s[i]);
    printf("\n");
}

void dump_card_layout() {
    static uint8_t sector[512];
    sd_card_t* sd = sd_get_by_num(0);
    printf("card size: %llu sectors (%.1f GB)\n", (unsigned long long)sd->get_num_sectors(sd),
           sd->get_num_sectors(sd) * 512.0 / 1e9);
    if (disk_read(0, sector, 0, 1) != RES_OK) {
        printf("disk_read(sector 0) failed\n");
        return;
    }
    dump_sector("sector 0", sector);
    uint32_t first_lba = 0;
    for (int p = 0; p < 4; ++p) {
        const uint8_t* e = sector + 446 + 16 * p;
        const uint32_t lba = e[8] | e[9] << 8 | e[10] << 16 | (uint32_t)e[11] << 24;
        const uint32_t count = e[12] | e[13] << 8 | e[14] << 16 | (uint32_t)e[15] << 24;
        printf("  MBR entry %d: type 0x%02x, start %lu, sectors %lu\n", p, e[4], (unsigned long)lba,
               (unsigned long)count);
        if (p == 0) first_lba = lba;
    }
    if (first_lba != 0 && disk_read(0, sector, first_lba, 1) == RES_OK) {
        dump_sector("partition 1 boot sector", sector);
    }
}

bool mount_sd() {
    if (!sd_init_driver()) {
        printf("sd_init_driver() failed\n");
        return false;
    }
    sd_card_t* sd = sd_get_by_num(0);
    if (sd == nullptr) {
        printf("No SD card configuration\n");
        return false;
    }
    g_drive = sd_get_drive_prefix(sd);
    const FRESULT fr = f_mount(&g_fs, g_drive, 1);
    if (fr != FR_OK) {
        printf("f_mount(%s) failed: %s (%d)\n", g_drive, FRESULT_str(fr), fr);
        dump_card_layout();
        return false;
    }
    printf("SD card mounted as %s\n", g_drive);
    return true;
}

int compare_names(const void* a, const void* b) {
    return strcmp(static_cast<const char*>(a), static_cast<const char*>(b));
}

// Collects /<class>/*.txt for every class, sorted by name.
int scan_recordings() {
    int total = 0;
    for (int c = 0; c < kClassCount; ++c) {
        g_file_count[c] = 0;
        char path[64];
        snprintf(path, sizeof path, "%s/%s", g_drive, kClassNames[c]);
        DIR dir;
        if (f_opendir(&dir, path) != FR_OK) {
            printf("  %-9s no folder %s\n", kClassNames[c], path);
            continue;
        }
        FILINFO info;
        while (f_readdir(&dir, &info) == FR_OK && info.fname[0] != '\0') {
            const size_t len = strlen(info.fname);
            if ((info.fattrib & AM_DIR) || len < 5 || len >= kMaxNameLen ||
                strcmp(info.fname + len - 4, ".txt") != 0) {
                continue;
            }
            if (g_file_count[c] == kMaxFilesPerClass) {
                printf("  %-9s more than %d files, ignoring the rest\n", kClassNames[c], kMaxFilesPerClass);
                break;
            }
            strcpy(g_files[c][g_file_count[c]++], info.fname);
        }
        f_closedir(&dir);
        qsort(g_files[c], g_file_count[c], kMaxNameLen, compare_names);
        printf("  %-9s %d recordings\n", kClassNames[c], g_file_count[c]);
        total += g_file_count[c];
    }
    return total;
}

// Parses "a,b,c" (optionally signed) into three ints. Returns false for the header / junk.
bool parse_line(const char* s, int32_t out[3]) {
    for (int i = 0; i < 3; ++i) {
        char* end = nullptr;
        const long v = strtol(s, &end, 10);
        if (end == s) {
            return false;
        }
        out[i] = static_cast<int32_t>(v);
        s = end;
        if (i < 2) {
            if (*s != ',') {
                return false;
            }
            ++s;
        }
    }
    return true;
}

// Loads up to kInputWindowLength samples into g_raw; returns the total sample
// count in the file (0 on error).
size_t load_recording(int cls, int idx) {
    char path[96];
    snprintf(path, sizeof path, "%s/%s/%s", g_drive, kClassNames[cls], g_files[cls][idx]);
    FIL f;
    if (f_open(&f, path, FA_READ) != FR_OK) {
        printf("cannot open %s\n", path);
        return 0;
    }
    char line[64];
    size_t n = 0;
    int32_t v[3];
    while (f_gets(line, sizeof line, &f) != nullptr) {
        if (!parse_line(line, v)) {
            continue;  // header "ax,ay,az"
        }
        if (n < static_cast<size_t>(kInputWindowLength)) {
            for (int a = 0; a < kNumAxes; ++a) {
                g_raw[n * kNumAxes + a] = static_cast<int16_t>(v[a]);
            }
        }
        ++n;
    }
    f_close(&f);
    return n;
}

bool is_held_out(int cls, const char* file) {
    char key[96];
    snprintf(key, sizeof key, "%s/%s", kClassNames[cls], file);
    for (int i = 0; i < kOwnHeldOutCount; ++i) {
        if (strcmp(kOwnHeldOut[i], key) == 0) return true;
    }
    return false;
}

struct Summary {
    int total = 0;
    int correct = 0;
    int test_total = 0;    // held-out recordings (never seen in training)
    int test_correct = 0;
    uint64_t total_us = 0;
    size_t arena = 0;
    int confusion[kClassCount][kClassCount] = {};
};

bool evaluate_model(int m, Summary* s) {
    const ModelEntry& model = kModels[m];
    printf("\n=== Model %d/%d: %s ===\n%s\n", m + 1, kModelCount, model.name, model.description);
    if (!g_runner.Load(model.data)) {
        return false;
    }
    s->arena = g_runner.arena_used_bytes();
    printf("model %u bytes, tensor arena used %u of %u bytes, input scale %.6f zero point %ld\n",
           model.size, static_cast<unsigned>(s->arena), static_cast<unsigned>(ModelRunner::arena_size()),
           g_runner.input_scale(), static_cast<long>(g_runner.input_zero_point()));

    for (int c = 0; c < kClassCount; ++c) {
        for (int i = 0; i < g_file_count[c]; ++i) {
            const size_t samples = load_recording(c, i);
            if (samples == 0 || !g_preprocessor.Prepare(g_raw, samples)) {
                printf("  %s/%s: unreadable, skipped\n", kClassNames[c], g_files[c][i]);
                continue;
            }
            g_preprocessor.Quantize(g_runner.input_data(), g_runner.input_scale(), g_runner.input_zero_point());
            uint32_t us = 0;
            const int pred = g_runner.Predict(&us);
            if (pred < 0) {
                return false;
            }
            const TfLiteTensor* out = g_runner.output();
            const float conf = out->params.scale * (out->data.uint8[pred] - out->params.zero_point);
            const bool test = is_held_out(c, g_files[c][i]);
            s->total++;
            s->correct += (pred == c);
            s->test_total += test;
            s->test_correct += test && (pred == c);
            s->total_us += us;
            s->confusion[c][pred]++;
            printf("  %-9s %-18s %s -> %-9s %s  p=%.2f  %6.3f ms  (%u samples)\n", kClassNames[c], g_files[c][i],
                   test ? "TEST " : "train", kClassNames[pred], pred == c ? "ok  " : "MISS", conf, us / 1000.0f,
                   static_cast<unsigned>(samples));
            printf("RESULT,%s,%s/%s,%s,%s,%s,%.3f\n", model.name, kClassNames[c], g_files[c][i],
                   test ? "test" : "train", kClassNames[c], kClassNames[pred], us / 1000.0f);
        }
    }

    const float acc = s->total ? 100.0f * s->correct / s->total : 0.0f;
    const float avg_ms = s->total ? s->total_us / 1000.0f / s->total : 0.0f;
    const float test_acc = s->test_total ? 100.0f * s->test_correct / s->test_total : 0.0f;
    printf("\n%s: held-out (TEST) accuracy %d/%d = %.1f %%, all own recordings %d/%d = %.1f %%\n", model.name,
           s->test_correct, s->test_total, test_acc, s->correct, s->total, acc);
    printf("total inference %.3f ms, average %.3f ms/gesture\n", s->total_us / 1000.0f, avg_ms);
    printf("confusion matrix (rows = true, columns = predicted):\n           ");
    for (int p = 0; p < kClassCount; ++p) {
        printf("%9s", kClassNames[p]);
    }
    printf("\n");
    for (int t = 0; t < kClassCount; ++t) {
        printf("  %-9s", kClassNames[t]);
        for (int p = 0; p < kClassCount; ++p) {
            printf("%9d", s->confusion[t][p]);
        }
        printf("\n");
    }
    printf("SUMMARY,%s,%d,%d,%.2f,%d,%d,%.2f,%.3f,%u,%u\n", model.name, s->test_total, s->test_correct, test_acc,
           s->total, s->correct, acc, avg_ms,
           static_cast<unsigned>(s->arena), model.size);
    return true;
}

void run_all() {
    status_led_rgb(0, 0, 64);  // blue: evaluating
    printf("\nRecordings on the SD card:\n");
    const int total = scan_recordings();
    if (total == 0) {
        printf("No recordings found. Record some with the data_collection firmware first.\n");
        status_led_rgb(64, 0, 0);
        return;
    }

    static Summary summaries[16];
    bool ok = true;
    for (int m = 0; m < kModelCount && m < 16; ++m) {
        summaries[m] = Summary();
        ok = evaluate_model(m, &summaries[m]) && ok;
    }

    printf("\n=== On-device summary (RP2350 @ %lu MHz, %d own gestures) ===\n",
           static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000), total);
    printf("| model | held-out accuracy | all own recordings | avg inference (ms) | tensor arena (B) | model size (B) |\n");
    printf("|---|---|---|---|---|---|\n");
    for (int m = 0; m < kModelCount && m < 16; ++m) {
        const Summary& s = summaries[m];
        printf("| %s | %.1f %% (%d/%d) | %.1f %% (%d/%d) | %.3f | %u | %u |\n", kModels[m].name,
               s.test_total ? 100.0f * s.test_correct / s.test_total : 0.0f, s.test_correct, s.test_total,
               s.total ? 100.0f * s.correct / s.total : 0.0f, s.correct, s.total,
               s.total ? s.total_us / 1000.0f / s.total : 0.0f, static_cast<unsigned>(s.arena), kModels[m].size);
    }
    status_led_rgb(ok ? 0 : 64, ok ? 64 : 0, 0);  // green: done, red: a model failed
}

void app_main() {
    status_led_init();
    status_led_rgb(64, 0, 0);  // red: starting
    wait_for_usb_console();

    printf("\n==== HA2 own-data test: %d model(s), window %d samples -> %d frames ====\n", kModelCount,
           kInputWindowLength, kTargetLength);
    if (!mount_sd()) {
        while (true) {
            tight_loop_contents();
        }
    }

    while (true) {
        run_all();
        printf("\nPress 'r' to run again.\n");
        int ch;
        do {
            ch = getchar_timeout_us(1000 * 1000);
        } while (ch != 'r' && ch != 'R');
    }
}

}  // namespace

int main() {
    stdio_init_all();
    run_on_core1(app_main);
}
