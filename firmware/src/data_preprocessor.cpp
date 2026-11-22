#include "data_preprocessor.hpp"

#include <cmath>

// Cascade of second-order sections, in place. Same as scipy's
// sosfilt(sos, x, zi=sosfilt_zi(sos) * x[0]).
template <int Sections>
static void sos_filter(const float (&sos)[Sections][6], const float (&zi)[Sections][2],
                       float* x, int n)
{
    const float x0 = x[0];
    for (int s = 0; s < Sections; s++) {
        const float b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2];
        const float a1 = sos[s][4], a2 = sos[s][5];
        float z0 = zi[s][0] * x0;
        float z1 = zi[s][1] * x0;
        for (int i = 0; i < n; i++) {
            const float in = x[i];
            const float y = b0 * in + z0;
            z0 = b1 * in - a1 * y + z1;
            z1 = b2 * in - a2 * y;
            x[i] = y;
        }
    }
}

void DataPreprocessor::preprocess(const int16_t* raw, uint8_t* output,
                                  float scale, int32_t zero_point)
{
    for (int axis = 0; axis < kNumAxes; axis++) {
        // 1. counts -> g, rotated into the training orientation
        const float* r = kRotation[axis];
        for (int t = 0; t < kRecordLength; t++) {
            const int16_t* s = &raw[t * kNumAxes];
            axis_[t] = (r[0] * s[0] + r[1] * s[1] + r[2] * s[2]) / kCountsPerG;
        }

        // 2. and 3. low-pass filters
        sos_filter(kLowpassSos, kLowpassZi, axis_, kRecordLength);
        sos_filter(kResampleSos, kResampleZi, axis_, kRecordLength);

        for (int j = 0; j < kTargetLength; j++) {
            // 3. linear interpolation, like np.interp on 0..1 time axes
            const float pos = (float)j * (kRecordLength - 1) / (kTargetLength - 1);
            int i = (int)pos;
            if (i >= kRecordLength - 1) i = kRecordLength - 2;
            const float frac = pos - i;
            const float g = axis_[i] + frac * (axis_[i + 1] - axis_[i]);

            // 4. normalise, 5. quantise
            const float x = (g - kAxisMean[axis]) / kAxisStd[axis];
            long q = lroundf(x / scale) + zero_point;
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            output[j * kNumAxes + axis] = (uint8_t)q;
        }
    }
}
