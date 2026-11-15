#ifndef DATA_PREPROCESSOR_HPP
#define DATA_PREPROCESSOR_HPP

#include <cstddef>
#include <cstdint>

#include "preprocessing_settings.hpp"

// Turns a raw IMU recording into the uint8 input tensor of a quantized model.
// Same algorithm as training/gesture_data.py (to_frames + normalize):
//
//   1. CutTrailing      keep at most kInputWindowLength samples
//   2. Normalize        axis map -> counts / kCountsPerG -> (g - mean) / std
//   3. PadWithTailMode  pad to kInputWindowLength with the per-axis mode of
//                       the last kPaddingModeWindow samples
//   4. Downsample       block-average kInputWindowLength -> kTargetLength
//   5. Quantize         float -> uint8 with the model input's scale/zero point
//
// Buffers are (ax, ay, az) interleaved: sample t, axis a is at t * kNumAxes + a.
// Prepare() runs stages 1-4 once; Quantize() fills the input of each model.
class DataPreprocessor {
public:
    // Stages 1-4. Returns false if `length` is 0.
    bool Prepare(const int16_t* raw, size_t length);

    // Stage 5 on the prepared frames: writes kTargetLength * kNumAxes bytes.
    void Quantize(uint8_t* out, float scale, int32_t zero_point) const;

    // Normalized, downsampled frames from the last Prepare() call.
    const float* frames() const { return downsampled_; }

private:
    void Normalize(const int16_t* raw, size_t length);
    size_t PadWithTailMode(size_t length);
    void Downsample();

    float window_[kInputWindowLength * kNumAxes];
    float downsampled_[kTargetLength * kNumAxes];
};

#endif  // DATA_PREPROCESSOR_HPP
