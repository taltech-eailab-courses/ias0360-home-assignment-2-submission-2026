#ifndef DATA_PREPROCESSOR_HPP
#define DATA_PREPROCESSOR_HPP

#include <cstdint>

#include "preprocessing_settings.hpp"

// Turns one raw IMU recording into the uint8 input of the model.
//
// The steps are the same as preprocess_recording() + normalize() in
// src/preprocessor.py, so the Pico sees the same input as the PC:
//   1. raw counts -> g, rotate into the training orientation
//   2. Butterworth low-pass, 20 Hz
//   3. low-pass again and linearly resample kRecordLength -> kTargetLength
//   4. normalise with the training mean/std
//   5. quantise to uint8 with the model's input scale/zero point
//
// `raw` holds kRecordLength samples, (ax, ay, az) interleaved.
// `output` gets kTargetLength * kNumAxes bytes in the same (time, axis) layout.
class DataPreprocessor {
public:
    void preprocess(const int16_t* raw, uint8_t* output,
                    float scale = kInputScale, int32_t zero_point = kInputZeroPoint);

private:
    float axis_[kRecordLength];   // one axis at a time, to keep RAM use low
};

#endif  // DATA_PREPROCESSOR_HPP
