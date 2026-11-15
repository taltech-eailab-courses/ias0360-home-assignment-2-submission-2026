#include <algorithm>
#include <cmath>

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"
#include "model_settings.h"
#include "mnist_model_data.h"


Model::Model() :
    model(nullptr),
    interpreter(nullptr),
    input(nullptr),
    error_reporter(nullptr)
{
}

Model::~Model()
{
    // `interpreter`, `model`, and `input` all point at static storage set up
    // in setup() (see the `static` locals in setup()) -- they are not heap
    // allocated here, so there is nothing for this destructor to free.
}

int Model::setup()
{
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(mnist_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 0;
  }

  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  static uint8_t tensor_arena[arena_size];
  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, arena_size);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return 0;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  // Cache the input tensor's quantization parameters (baked into the
  // .tflite file by the converter) so set_input_image() below doesn't need
  // them hard-coded.
  input_scale_ = input->params.scale;
  input_zero_point_ = input->params.zero_point;

  return 1;
}

bool Model::set_input_image(const uint8_t* pixels) {
  if (input == nullptr) {
    return false;
  }

  for (int i = 0; i < kMaxImageSize; i++) {
    float normalized = pixels[i] / 255.0f;
    int quantized = static_cast<int>(std::lround(normalized / input_scale_)) + input_zero_point_;
    input->data.int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
  }

  return true;
}

bool Model::get_output_scores(float* out_scores, int count) {
  if (interpreter == nullptr) {
    return false;
  }
  TfLiteTensor* output = interpreter->output(0);
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  for (int i = 0; i < count; i++) {
    // Dequantize: real_value = (quantized - zero_point) * scale
    out_scores[i] = (static_cast<int>(output->data.int8[i]) - zero_point) * scale;
  }
  return true;
}

uint8_t* Model::input_data() {
  if (input == nullptr) {
    return nullptr;
  }
  return input->data.uint8;
}

int Model::byte_size() {
  if (input == nullptr) {
    return 0;
  }
  return input->bytes;
}

int Model::predict()
{
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke() failed");
    return -1;
  }

  TfLiteTensor* output = interpreter->output(0);

  int result = 0;
  for (int i = 1; i < kCategoryCount; i++) {
    if (output->data.int8[i] > output->data.int8[result]) {
      result = i;
    }
  }

  return result;
}
