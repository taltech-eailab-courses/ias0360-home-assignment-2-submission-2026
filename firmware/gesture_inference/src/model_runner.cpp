#include "model_runner.hpp"

#include <cstdio>
#include <new>

#include "pico/stdlib.h"
#include "preprocessing_settings.hpp"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr size_t kArenaSize = 48 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];

// Interpreter storage (placement-new, re-created per model).
alignas(tflite::MicroInterpreter) uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];

// Ops used by all architectures in training/models.py.
constexpr int kOpCount = 11;
tflite::MicroMutableOpResolver<kOpCount>& Resolver() {
    static tflite::MicroMutableOpResolver<kOpCount> resolver;
    static bool initialized = false;
    if (!initialized) {
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddMaxPool2D();
        resolver.AddMean();            // GlobalAveragePooling2D
        resolver.AddConcatenation();   // SqueezeNet Fire modules
        resolver.AddReshape();         // Flatten
        resolver.AddFullyConnected();  // Dense
        resolver.AddSoftmax();
        resolver.AddQuantize();        // uint8 input -> int8
        resolver.AddDequantize();
        resolver.AddTanh();
        initialized = true;
    }
    return resolver;
}

}  // namespace

bool ModelRunner::Load(const unsigned char* model_data) {
    if (interpreter_ != nullptr) {
        interpreter_->~MicroInterpreter();
        interpreter_ = nullptr;
        input_ = nullptr;
    }

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version %lu != supported %d\n",
               static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
        return false;
    }

    interpreter_ = new (g_interpreter_storage)
        tflite::MicroInterpreter(model, Resolver(), g_arena, kArenaSize);
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed (arena %u bytes too small or op missing)\n",
               static_cast<unsigned>(kArenaSize));
        return false;
    }

    input_ = interpreter_->input(0);
    const TfLiteTensor* out = interpreter_->output(0);
    const int expected_bytes = kTargetLength * kNumAxes;
    if (input_->type != kTfLiteUInt8 || static_cast<int>(input_->bytes) != expected_bytes) {
        printf("Unexpected input tensor: type=%d bytes=%u (want uint8, %d bytes)\n",
               input_->type, static_cast<unsigned>(input_->bytes), expected_bytes);
        return false;
    }
    if (out->type != kTfLiteUInt8 || out->dims->data[out->dims->size - 1] != kClassCount) {
        printf("Unexpected output tensor: type=%d classes=%d (want uint8, %d)\n",
               out->type, out->dims->data[out->dims->size - 1], kClassCount);
        return false;
    }
    return true;
}

uint8_t* ModelRunner::input_data() const { return input_ ? input_->data.uint8 : nullptr; }
float ModelRunner::input_scale() const { return input_ ? input_->params.scale : 0.0f; }
int32_t ModelRunner::input_zero_point() const { return input_ ? input_->params.zero_point : 0; }
const TfLiteTensor* ModelRunner::output() const { return interpreter_ ? interpreter_->output(0) : nullptr; }

int ModelRunner::Predict(uint32_t* elapsed_us) {
    if (interpreter_ == nullptr) {
        return -1;
    }
    const absolute_time_t start = get_absolute_time();
    const TfLiteStatus status = interpreter_->Invoke();
    if (elapsed_us != nullptr) {
        *elapsed_us = static_cast<uint32_t>(absolute_time_diff_us(start, get_absolute_time()));
    }
    if (status != kTfLiteOk) {
        printf("Invoke() failed\n");
        return -1;
    }
    // Arg-max of the uint8 scores.
    const TfLiteTensor* out = interpreter_->output(0);
    int best = 0;
    for (int c = 1; c < kClassCount; ++c) {
        if (out->data.uint8[c] > out->data.uint8[best]) {
            best = c;
        }
    }
    return best;
}

size_t ModelRunner::arena_used_bytes() const {
    return interpreter_ ? interpreter_->arena_used_bytes() : 0;
}

size_t ModelRunner::arena_size() { return kArenaSize; }
