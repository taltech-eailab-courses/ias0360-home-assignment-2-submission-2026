#include "model.h"

#include <cstdio>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gesture_model_data.h"
#include "model_settings.h"

bool Model::setup()
{
    const tflite::Model* model = tflite::GetModel(gesture_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version %lu, expected %d\n",
               (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Ops of the own GAP model (see the header of gesture_model_data.cpp).
    // GlobalAveragePooling becomes MEAN, the uint8 input/output adds QUANTIZE.
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();

    alignas(16) static uint8_t tensor_arena[kArenaSize];
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
    interpreter_ = &static_interpreter;

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return false;
    }
    input = interpreter_->input(0);
    return true;
}

int Model::predict()
{
    if (interpreter_->Invoke() != kTfLiteOk) {
        printf("Invoke() failed\n");
        return -1;
    }
    // argmax on the uint8 scores; quantisation keeps the order.
    const TfLiteTensor* output = interpreter_->output(0);
    int best = 0;
    for (int c = 1; c < CLASS_COUNT; c++) {
        if (output->data.uint8[c] > output->data.uint8[best]) best = c;
    }
    return best;
}

uint8_t* Model::input_data() { return input ? input->data.uint8 : nullptr; }

const TfLiteTensor* Model::output_tensor() { return interpreter_ ? interpreter_->output(0) : nullptr; }

size_t Model::arena_used_bytes() { return interpreter_ ? interpreter_->arena_used_bytes() : 0; }
