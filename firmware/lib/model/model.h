#ifndef GESTURE_MODEL_H_
#define GESTURE_MODEL_H_

#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"

// The gesture CNN running in TFLite Micro.
class Model {
public:
    bool setup();                  // load the model and allocate the tensor arena
    int predict();                 // run inference, return the class index (-1 on error)

    uint8_t* input_data();         // uint8 input, kTargetLength * 3 bytes
    const TfLiteTensor* output_tensor();
    size_t arena_used_bytes();

    TfLiteTensor* input = nullptr;

private:
    tflite::MicroInterpreter* interpreter_ = nullptr;
};

#endif  // GESTURE_MODEL_H_
