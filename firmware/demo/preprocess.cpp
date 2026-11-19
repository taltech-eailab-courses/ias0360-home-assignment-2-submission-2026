#include "preprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
constexpr int W = 280, H = 240, N = 28;
uint8_t cleaned[W * H];
uint8_t horizontal[20 * H];
uint8_t canvas[N * N], thick[N * N];
double weights[W];
int32_t coefficients[W];

// Lanczos-3 mit zwei 8-Bit-Durchläufen und 22-Bit-Koeffizienten wie Pillow.
// Algorithmusreferenz: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
double lanczos(double x) {
    if (x == 0) return 1;
    if (std::abs(x) >= 3) return 0;
    const double p = x * 3.14159265358979323846;
    return std::sin(p) / p * std::sin(p / 3) / (p / 3);
}

void kernel(int source, int destination, int index, int& first, int& count) {
    const double scale = double(source) / destination;
    const double filter_scale = std::max(1.0, scale);
    const double center = (index + 0.5) * scale;
    first = std::max(0, int(center - 3 * filter_scale + 0.5));
    const int end = std::min(source, int(center + 3 * filter_scale + 0.5));
    count = end - first;
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        weights[i] = lanczos((i + first - center + 0.5) / filter_scale);
        sum += weights[i];
    }
    for (int i = 0; i < count; ++i) {
        coefficients[i] = int32_t(std::round(weights[i] / sum * (1 << 22)));
    }
}

uint8_t clip(int32_t sum) {
    return uint8_t(std::clamp(sum >> 22, int32_t(0), int32_t(255)));
}

// Eine beschränkte Suche genügt: Jede Komponente ab acht Pixeln bleibt erhalten.
// Keine große rekursive Suche und kein zusätzlicher Puffer mit Pixelindizes.
void remove_specks(const uint8_t* ink) {
    std::memset(cleaned, 0, sizeof(cleaned));
    for (int p = 0; p < W * H; ++p) {
        if (!ink[p] || cleaned[p]) continue;
        int connected[8] = {p};
        int count = 1;
        for (int head = 0; head < count && count < 8; ++head) {
            const int x = connected[head] % W, y = connected[head] / W;
            for (int dy = -1; dy <= 1 && count < 8; ++dy) {
                for (int dx = -1; dx <= 1 && count < 8; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    const int q = ny * W + nx;
                    if (!ink[q]) continue;
                    bool known = false;
                    for (int i = 0; i < count; ++i) known |= connected[i] == q;
                    if (!known) connected[count++] = q;
                }
            }
        }
        for (int i = 0; i < count; ++i) cleaned[connected[i]] = count == 8 ? 255 : 1;
    }
    for (auto& value : cleaned) if (value != 255) value = 0;
}
}

bool preprocess_digit(const uint8_t* ink, float* output) {
    std::fill(output, output + N * N, 0.0f);
    remove_specks(ink);
    int left = W, right = -1, top = H, bottom = -1;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        if (!cleaned[y * W + x]) continue;
        left = std::min(left, x); right = std::max(right, x);
        top = std::min(top, y); bottom = std::max(bottom, y);
    }
    if (right < left) return false;
    const int width = right - left + 1, height = bottom - top + 1;
    const double scale = 20.0 / std::max(width, height);
    const int rw = std::max(1, int(std::nearbyint(width * scale)));
    const int rh = std::max(1, int(std::nearbyint(height * scale)));
    // Horizontal verkleinern, danach vertikal; Zwischenwerte bleiben 8-Bit.
    for (int x = 0; x < rw; ++x) {
        int first, count;
        kernel(width, rw, x, first, count);
        for (int y = 0; y < height; ++y) {
            int32_t sum = 1 << 21;
            for (int i = 0; i < count; ++i)
                sum += cleaned[(top + y) * W + left + first + i] * coefficients[i];
            horizontal[y * rw + x] = clip(sum);
        }
    }
    std::memset(canvas, 0, sizeof(canvas));
    const int ox = (N - rw) / 2, oy = (N - rh) / 2;
    for (int y = 0; y < rh; ++y) {
        int first, count;
        kernel(height, rh, y, first, count);
        for (int x = 0; x < rw; ++x) {
            int32_t sum = 1 << 21;
            for (int i = 0; i < count; ++i)
                sum += horizontal[(first + i) * rw + x] * coefficients[i];
            canvas[(oy + y) * N + ox + x] = clip(sum);
        }
    }
    // 3×3-Maximumfilter stärkt die dünnen Touchscreen-Striche.
    int mass = 0, mx = 0, my = 0, peak = 0;
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        uint8_t value = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (x + dx >= 0 && x + dx < N && y + dy >= 0 && y + dy < N)
                value = std::max(value, canvas[(y + dy) * N + x + dx]);
        }
        thick[y * N + x] = value;
        mass += value; mx += value * x; my += value * y;
        peak = std::max(peak, int(value));
    }
    if (!mass) return false;
    const int dx = int(std::nearbyint(13.5 - double(mx) / mass));
    const int dy = int(std::nearbyint(13.5 - double(my) / mass));
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int sx = x - dx, sy = y - dy;
        if (sx >= 0 && sx < N && sy >= 0 && sy < N)
            output[y * N + x] = float(thick[sy * N + sx]) / peak;
    }
    return true;
}
