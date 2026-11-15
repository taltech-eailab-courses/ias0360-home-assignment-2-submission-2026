#include "data_preprocessor.hpp"

#include <algorithm>
#include <cmath>

void DataPreprocessor::Normalize(const int16_t* raw, size_t length) {
    for (size_t t = 0; t < length; ++t) {
        for (int axis = 0; axis < kNumAxes; ++axis) {
            // Axis map; int32 so that -(-32768) does not overflow.
            int32_t counts = kAxisSign[axis] * static_cast<int32_t>(raw[t * kNumAxes + kAxisSource[axis]]);
            counts = std::min<int32_t>(32767, std::max<int32_t>(-32768, counts));
            const float g = static_cast<float>(counts) / kCountsPerG;
            window_[t * kNumAxes + axis] = (g - kAxisMean[axis]) / kAxisStd[axis];
        }
    }
}

size_t DataPreprocessor::PadWithTailMode(size_t length) {
    if (length == 0 || length >= static_cast<size_t>(kInputWindowLength)) {
        return length;
    }
    const size_t tail = std::min<size_t>(length, kPaddingModeWindow);
    const size_t tail_start = length - tail;

    for (int axis = 0; axis < kNumAxes; ++axis) {
        // Ties go to the most recent value.
        float mode = window_[(length - 1) * kNumAxes + axis];
        size_t best_count = 0;
        for (size_t a = length; a-- > tail_start;) {
            const float candidate = window_[a * kNumAxes + axis];
            size_t count = 0;
            for (size_t b = tail_start; b < length; ++b) {
                if (window_[b * kNumAxes + axis] == candidate) {
                    ++count;
                }
            }
            if (count > best_count) {
                best_count = count;
                mode = candidate;
            }
        }
        for (size_t t = length; t < static_cast<size_t>(kInputWindowLength); ++t) {
            window_[t * kNumAxes + axis] = mode;
        }
    }
    return kInputWindowLength;
}

void DataPreprocessor::Downsample() {
    for (int k = 0; k < kTargetLength; ++k) {
        const size_t begin = static_cast<size_t>(k) * kInputWindowLength / kTargetLength;
        const size_t end = std::max(begin + 1, static_cast<size_t>(k + 1) * kInputWindowLength / kTargetLength);
        for (int axis = 0; axis < kNumAxes; ++axis) {
            float sum = 0.0f;
            for (size_t t = begin; t < end; ++t) {
                sum += window_[t * kNumAxes + axis];
            }
            downsampled_[k * kNumAxes + axis] = sum / static_cast<float>(end - begin);
        }
    }
}

bool DataPreprocessor::Prepare(const int16_t* raw, size_t length) {
    if (raw == nullptr || length == 0) {
        return false;
    }
    const size_t n = std::min<size_t>(length, kInputWindowLength);  // CutTrailing
    Normalize(raw, n);
    PadWithTailMode(n);
    Downsample();
    return true;
}

void DataPreprocessor::Quantize(uint8_t* out, float scale, int32_t zero_point) const {
    for (int i = 0; i < kTargetLength * kNumAxes; ++i) {
        const int32_t q = static_cast<int32_t>(std::lround(downsampled_[i] / scale)) + zero_point;
        out[i] = static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, q)));
    }
}
