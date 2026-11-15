#ifndef MODEL_RUNNER_HPP
#define MODEL_RUNNER_HPP

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"

namespace tflite {
class MicroInterpreter;
}

// One TFLite Micro interpreter in a shared static tensor arena. Load()
// replaces the previous model.
class ModelRunner {
public:
    // Returns false (and prints why) if the model cannot be used.
    bool Load(const unsigned char* model_data);

    uint8_t* input_data() const;
    float input_scale() const;
    int32_t input_zero_point() const;
    const TfLiteTensor* output() const;

    // Runs the model; returns the arg-max class index, or -1 on failure.
    // `elapsed_us` receives the Invoke() wall time.
    int Predict(uint32_t* elapsed_us);

    size_t arena_used_bytes() const;
    static size_t arena_size();

private:
    tflite::MicroInterpreter* interpreter_ = nullptr;
    TfLiteTensor* input_ = nullptr;
};

#endif  // MODEL_RUNNER_HPP
