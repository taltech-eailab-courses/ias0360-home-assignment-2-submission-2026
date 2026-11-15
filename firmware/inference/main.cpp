// Digit demo: draw on the touchscreen, the Pico reads the digit by itself after a short countdown,
// shows the 28x28 image the network sees next to its answer, stores the drawing on the SD card and
// clears the screen for the next digit.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pico/stdlib.h"
#include "hardware/spi.h"

#include "ff.h"
#include "f_util.h"
#include "hw_config.h"
#include "sd_card.h"

#include "DEV_Config.h"
#include "LCD_Driver.h"
#include "LCD_GUI.h"

#include "model.h"
#include "model_settings.h"

constexpr int DRAW_LEFT = 10, DRAW_TOP = 40, DRAW_W = 280, DRAW_H = 240;
constexpr int PEN_RADIUS = 4;   // the line on the screen
constexpr int INK_RADIUS = 8;   // the stroke the network gets, as wide as an MNIST stroke
constexpr int VIEW_LEFT = 320, VIEW_TOP = 40, VIEW_SCALE = 5;
constexpr int PAUSE_MS = 500, TICK_MS = 500, TICKS = 3, SHOW_SECONDS = 3;
constexpr int BARS_LEFT = 372, BARS_TOP = 188, BAR_ROW = 13, BAR_W = 60;
constexpr uint16_t BAR_TRACK = 0xE71C;
constexpr int PRESS_THRESHOLD = 100;
constexpr int STABLE_RANGE = 50;

static uint8_t bitmap[DRAW_H][DRAW_W];
static uint8_t pixels[kMaxImageSize];
static Model ml_model;
static FATFS fs;
static bool sd_ready = false;

static uint16_t touch_adc(uint8_t command) {
    DEV_Digital_Write(TP_CS_PIN, 0);
    SPI4W_Write_Byte(command);
    Driver_Delay_us(200);
    uint16_t data = SPI4W_Read_Byte(0x00);
    data <<= 8;
    data |= SPI4W_Read_Byte(0x00);
    data |= SPI4W_Read_Byte(0x00);
    DEV_Digital_Write(TP_CS_PIN, 1);
    return data >> 3;
}

static uint16_t touch_adc_average(uint8_t command) {
    uint16_t samples[5];
    for (uint16_t& sample : samples) {
        sample = touch_adc(command);
        Driver_Delay_us(200);
    }
    std::sort(samples, samples + 5);
    return (samples[1] + samples[2] + samples[3]) / 3;
}

// The SD card's SDIO clock sits on the touch IRQ pin, so a press is read as pressure over SPI.
static bool touch_read(int* x, int* y) {
    spi_set_baudrate(SPI_PORT, 3000000);
    bool pressed = touch_adc(0xB0) > PRESS_THRESHOLD;
    int xad = 0, yad = 0;
    bool stable = false;
    if (pressed) {
        int x1 = touch_adc_average(0xD0), y1 = touch_adc_average(0x90);
        int x2 = touch_adc_average(0xD0), y2 = touch_adc_average(0x90);
        stable = std::abs(x1 - x2) < STABLE_RANGE && std::abs(y1 - y2) < STABLE_RANGE;
        xad = (x1 + x2) / 2;
        yad = (y1 + y2) / 2;
    }
    spi_set_baudrate(SPI_PORT, 18000000);
    if (!stable) {
        return false;
    }
    // Calibration of the Waveshare driver for the U2D_R2L scan direction.
    *x = static_cast<int>(-0.132906f * yad + 517);
    *y = static_cast<int>(0.087964f * xad - 20);
    return true;
}

static void pen_dot(int x, int y) {
    int left = std::max(x - PEN_RADIUS, 0), right = std::min(x + PEN_RADIUS, DRAW_W - 1);
    int top = std::max(y - PEN_RADIUS, 0), bottom = std::min(y + PEN_RADIUS, DRAW_H - 1);
    LCD_SetArealColor(DRAW_LEFT + left, DRAW_TOP + top, DRAW_LEFT + right + 1, DRAW_TOP + bottom + 1, BLACK);

    left = std::max(x - INK_RADIUS, 0);
    right = std::min(x + INK_RADIUS, DRAW_W - 1);
    top = std::max(y - INK_RADIUS, 0);
    bottom = std::min(y + INK_RADIUS, DRAW_H - 1);
    for (int row = top; row <= bottom; row++) {
        memset(&bitmap[row][left], 1, right - left + 1);
    }
}

static void pen_line(int x0, int y0, int x1, int y1) {
    int steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0)) / 3 + 1;
    for (int i = 1; i <= steps; i++) {
        pen_dot(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
    }
}

static void show_status(const char* text) {
    LCD_SetArealColor(DRAW_LEFT, 8, 300, 34, WHITE);
    GUI_DisString_EN(DRAW_LEFT, 10, text, &Font24, FONT_BACKGROUND, BLACK);
}

static void clear_screen() {
    LCD_Clear(WHITE);
    GUI_DrawRectangle(DRAW_LEFT - 2, DRAW_TOP - 2, DRAW_LEFT + DRAW_W + 1, DRAW_TOP + DRAW_H + 1,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_2X2);
    memset(bitmap, 0, sizeof bitmap);
    show_status("Draw a digit");
}

static void draw_until_idle() {
    int last_x = -1, last_y = -1, shown = -1;
    bool drawn = false;
    uint32_t last_touch_ms = 0;
    while (true) {
        int x, y;
        if (touch_read(&x, &y)) {
            x -= DRAW_LEFT;
            y -= DRAW_TOP;
            if (x >= 0 && x < DRAW_W && y >= 0 && y < DRAW_H) {
                if (last_x < 0) {
                    pen_dot(x, y);
                } else {
                    pen_line(last_x, last_y, x, y);
                }
                last_x = x;
                last_y = y;
                drawn = true;
                last_touch_ms = to_ms_since_boot(get_absolute_time());
                if (shown != -1) {
                    show_status("Draw a digit");
                    shown = -1;
                }
            }
        } else {
            last_x = -1;
        }
        if (!drawn) {
            continue;
        }
        int idle_ms = to_ms_since_boot(get_absolute_time()) - last_touch_ms - PAUSE_MS;
        if (idle_ms < 0) {
            continue;
        }
        int left = TICKS - idle_ms / TICK_MS;
        if (left <= 0) {
            return;
        }
        if (left != shown) {
            char text[20];
            snprintf(text, sizeof text, "Reading in %d", left);
            show_status(text);
            shown = left;
        }
    }
}

// Same steps as MNIST and run_ha2_experiments.py: crop, fit into 20x20, centre by mass in 28x28.
static void to_mnist(uint8_t* out) {
    int x0 = DRAW_W, x1 = -1, y0 = DRAW_H, y1 = -1;
    for (int y = 0; y < DRAW_H; y++) {
        for (int x = 0; x < DRAW_W; x++) {
            if (bitmap[y][x]) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    }
    int w = x1 - x0 + 1, h = y1 - y0 + 1, side = std::max(w, h);
    int nw = std::max(1, static_cast<int>(std::lround(w * 20.0f / side)));
    int nh = std::max(1, static_cast<int>(std::lround(h * 20.0f / side)));

    uint8_t small[20][20] = {};
    uint32_t mass = 0, mass_x = 0, mass_y = 0;
    for (int ny = 0; ny < nh; ny++) {
        int sy0 = y0 + ny * h / nh, sy1 = std::max(y0 + (ny + 1) * h / nh, sy0 + 1);
        for (int nx = 0; nx < nw; nx++) {
            int sx0 = x0 + nx * w / nw, sx1 = std::max(x0 + (nx + 1) * w / nw, sx0 + 1);
            int count = 0;
            for (int y = sy0; y < sy1; y++) {
                for (int x = sx0; x < sx1; x++) {
                    count += bitmap[y][x];
                }
            }
            small[ny][nx] = 255 * count / ((sy1 - sy0) * (sx1 - sx0));
            mass += small[ny][nx];
            mass_x += small[ny][nx] * nx;
            mass_y += small[ny][nx] * ny;
        }
    }
    int left = static_cast<int>(std::lround(13.5f - static_cast<float>(mass_x) / mass));
    int top = static_cast<int>(std::lround(13.5f - static_cast<float>(mass_y) / mass));

    memset(out, 0, kMaxImageSize);
    for (int ny = 0; ny < nh; ny++) {
        for (int nx = 0; nx < nw; nx++) {
            int tx = nx + left, ty = ny + top;
            if (tx >= 0 && tx < image_col_size && ty >= 0 && ty < image_row_size) {
                out[ty * image_col_size + tx] = small[ny][nx];
            }
        }
    }
}

static void show_network_view() {
    for (int row = 0; row < image_row_size; row++) {
        for (int col = 0; col < image_col_size; col++) {
            uint8_t v = pixels[row * image_col_size + col];
            uint16_t color = ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3);
            int x = VIEW_LEFT + col * VIEW_SCALE, y = VIEW_TOP + row * VIEW_SCALE;
            LCD_SetArealColor(x, y, x + VIEW_SCALE, y + VIEW_SCALE, color);
        }
    }
}

static void show_big_digit(int digit, int x, int y, int scale) {
    int row_bytes = Font24.Width / 8 + (Font24.Width % 8 ? 1 : 0);
    const uint8_t* glyph = &Font24.table[('0' + digit - ' ') * Font24.Height * row_bytes];
    for (int row = 0; row < Font24.Height; row++) {
        for (int col = 0; col < Font24.Width; col++) {
            if (glyph[row * row_bytes + col / 8] & (0x80 >> (col % 8))) {
                LCD_SetArealColor(x + col * scale, y + row * scale,
                                  x + (col + 1) * scale, y + (row + 1) * scale, BLUE);
            }
        }
    }
}

static void show_confidence_bars(const float* scores, int digit) {
    for (int d = 0; d < kCategoryCount; d++) {
        char text[8];
        int y = BARS_TOP + d * BAR_ROW;
        int bar_left = BARS_LEFT + 10;
        int width = static_cast<int>(scores[d] * BAR_W + 0.5f);
        COLOR color = d == digit ? BLUE : BLACK;
        snprintf(text, sizeof text, "%d", d);
        GUI_DisString_EN(BARS_LEFT, y, text, &Font12, FONT_BACKGROUND, color);
        LCD_SetArealColor(bar_left, y + 2, bar_left + BAR_W, y + 10, BAR_TRACK);
        if (width > 0) {
            LCD_SetArealColor(bar_left, y + 2, bar_left + width, y + 10, d == digit ? BLUE : GRAY);
        }
        snprintf(text, sizeof text, "%d%%", static_cast<int>(std::lround(scores[d] * 100)));
        GUI_DisString_EN(bar_left + BAR_W + 4, y, text, &Font12, FONT_BACKGROUND, color);
    }
}

static bool mount_sd() {
    if (!sd_init_driver()) {
        return false;
    }
    FRESULT fr = f_mount(&fs, "0:", 1);
    printf("f_mount -> %s (%d)\n", FRESULT_str(fr), fr);
    return fr == FR_OK;
}

static int next_free_index() {
    for (int i = 0; i < 1000; i++) {
        char path[24];
        FILINFO info;
        snprintf(path, sizeof path, "0:/digit_%03d.txt", i);
        if (f_stat(path, &info) == FR_NO_FILE) {
            return i;
        }
    }
    return 0;
}

static void save_to_sd(int index) {
    char path[24];
    FIL file;
    snprintf(path, sizeof path, "0:/digit_%03d.txt", index);
    if (f_open(&file, path, FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) {
        printf("SD write failed: %s\n", path);
        return;
    }
    for (int y = 0; y < DRAW_H; y++) {
        char row[DRAW_W];
        UINT written;
        for (int x = 0; x < DRAW_W; x++) {
            row[x] = bitmap[y][x] ? '1' : '0';
        }
        f_write(&file, row, DRAW_W, &written);
    }
    f_close(&file);
}

static void print_bitmap(int index) {
    printf("BITMAP,%d\n", index);
    for (int y = 0; y < DRAW_H; y++) {
        char row[DRAW_W + 1];
        for (int x = 0; x < DRAW_W; x++) {
            row[x] = bitmap[y][x] ? '1' : '0';
        }
        row[DRAW_W] = '\0';
        printf("%s\n", row);
    }
    printf("END\n");
}

static void read_digit(int index) {
    show_status("Reading...");
    to_mnist(pixels);
    show_network_view();

    ml_model.set_input_image(pixels);
    absolute_time_t start = get_absolute_time();
    int digit = ml_model.predict();
    int64_t inference_us = absolute_time_diff_us(start, get_absolute_time());
    float scores[kCategoryCount];
    ml_model.get_output_scores(scores, kCategoryCount);

    char text[24];
    GUI_DisString_EN(VIEW_LEFT, 14, "Network input", &Font16, FONT_BACKGROUND, BLACK);
    GUI_DisString_EN(300, 188, "Reads as", &Font12, FONT_BACKGROUND, BLACK);
    show_big_digit(digit, 300, 202, 3);
    snprintf(text, sizeof text, "%d%%", static_cast<int>(std::lround(scores[digit] * 100)));
    GUI_DisString_EN(300, 278, text, &Font20, FONT_BACKGROUND, BLUE);
    snprintf(text, sizeof text, "%.1f ms", inference_us / 1000.0);
    GUI_DisString_EN(300, 302, text, &Font12, FONT_BACKGROUND, BLACK);
    show_confidence_bars(scores, digit);

    printf("READ,%d,%d,%.1f,%lld\n", index, digit, scores[digit] * 100, (long long)inference_us);
    if (sd_ready) {
        save_to_sd(index);
    }
    print_bitmap(index);
}

int main() {
    System_Init();
    sleep_ms(2000);

    LCD_Init(U2D_R2L, 1000);
    LCD_SetBackLight(1000);
    LCD_Clear(WHITE);

    if (!ml_model.setup()) {
        GUI_DisString_EN(DRAW_LEFT, 10, "Model failed to load", &Font24, FONT_BACKGROUND, RED);
        printf("Failed to initialize ML model!\n");
        while (true) {
            tight_loop_contents();
        }
    }
    sd_ready = mount_sd();
    int index = sd_ready ? next_free_index() : 0;
    printf("Model initialized, sd %d, first file %d\n", sd_ready, index);

    while (true) {
        clear_screen();
        draw_until_idle();
        read_digit(index++);
        for (int left = SHOW_SECONDS; left > 0; left--) {
            char text[20];
            snprintf(text, sizeof text, "Next in %d", left);
            show_status(text);
            sleep_ms(1000);
        }
    }
}
