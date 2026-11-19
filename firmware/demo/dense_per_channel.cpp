#include "dense_per_channel.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {
struct DenseData {
    tflite::FullyConnectedParams params;
    int32_t* multipliers;
    int* shifts;
};

void* init(TfLiteContext* context, const char*, size_t) {
    return context->AllocatePersistentBuffer(context, sizeof(DenseData));
}

TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node) {
    auto* micro = tflite::GetMicroContext(context);
    auto* input = micro->AllocateTempInputTensor(node, 0);
    auto* weights = micro->AllocateTempInputTensor(node, 1);
    auto* bias = micro->AllocateTempInputTensor(node, 2);
    auto* output = micro->AllocateTempOutputTensor(node, 0);
    TF_LITE_ENSURE(context, input && weights && bias && output && node->user_data);
    TF_LITE_ENSURE(context, input->type == kTfLiteInt8 && weights->type == kTfLiteInt8 &&
        bias->type == kTfLiteInt32 && output->type == kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, weights->dims->size, 2);
    TF_LITE_ENSURE_EQ(context, output->dims->size, 2);
    TF_LITE_ENSURE(context, weights->quantization.type == kTfLiteAffineQuantization);
    auto* quant = static_cast<TfLiteAffineQuantization*>(weights->quantization.params);
    const int channels = weights->dims->data[0];
    TF_LITE_ENSURE(context, quant && quant->scale && quant->zero_point);
    TF_LITE_ENSURE(context, quant->scale->size == channels || quant->scale->size == 1);
    TF_LITE_ENSURE(context, quant->scale->size == 1 || quant->quantized_dimension == 0);
    TF_LITE_ENSURE(context, input->params.scale > 0 && output->params.scale > 0);
    for (int i = 0; i < quant->zero_point->size; ++i)
        TF_LITE_ENSURE_EQ(context, quant->zero_point->data[i], 0);

    auto* data = static_cast<DenseData*>(node->user_data);
    data->multipliers = static_cast<int32_t*>(context->AllocatePersistentBuffer(context, channels * sizeof(int32_t)));
    data->shifts = static_cast<int*>(context->AllocatePersistentBuffer(context, channels * sizeof(int)));
    TF_LITE_ENSURE(context, data->multipliers && data->shifts);
    for (int i = 0; i < channels; ++i) {
        const double multiplier = double(input->params.scale) *
            quant->scale->data[quant->scale->size == 1 ? 0 : i] / output->params.scale;
        tflite::QuantizeMultiplier(multiplier, &data->multipliers[i], &data->shifts[i]);
    }
    data->params = {};
    data->params.input_offset = -input->params.zero_point;
    data->params.output_offset = output->params.zero_point;
    const auto* options = static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
    TF_LITE_ENSURE_STATUS(tflite::CalculateActivationRangeQuantized(context, options->activation,
        output, &data->params.quantized_activation_min, &data->params.quantized_activation_max));
    micro->DeallocateTempTfLiteTensor(input);
    micro->DeallocateTempTfLiteTensor(weights);
    micro->DeallocateTempTfLiteTensor(bias);
    micro->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus eval(TfLiteContext* context, TfLiteNode* node) {
    const auto* input = tflite::micro::GetEvalInput(context, node, 0);
    const auto* weights = tflite::micro::GetEvalInput(context, node, 1);
    const auto* bias = tflite::micro::GetEvalInput(context, node, 2);
    auto* output = tflite::micro::GetEvalOutput(context, node, 0);
    const auto& data = *static_cast<DenseData*>(node->user_data);
    // Die Bibliothek enthält bereits die richtige Referenzrechnung; ihr alter
    // CMSIS-Adapter verwendet für Dense jedoch nur einen gemeinsamen Faktor.
    tflite::reference_integer_ops::FullyConnectedPerChannel<int8_t, int8_t, int8_t, int32_t>(
        data.params, data.multipliers, data.shifts,
        tflite::micro::GetTensorShape(input), tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(weights), tflite::micro::GetTensorData<int8_t>(weights),
        tflite::micro::GetTensorShape(bias), tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output), tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
}
}

TFLMRegistration RegisterDensePerChannel() {
    return tflite::micro::RegisterOp(init, prepare, eval);
}
