#include <algorithm>
#include <cmath>
#include <cstdio>

#include "LCD_Touch.h"
#include "dense_per_channel.h"
#include "model_data.h"
#include "preprocess.h"
#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
alignas(16) uint8_t arena[64 * 1024];
float image[28 * 28];
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
int last_digit = -1;
unsigned last_ms = 0;
float last_score = 0;

void status(const char* text, COLOR color = BLACK) {
    GUI_DrawRectangle(0, 300, 479, 319, WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(4, 303, text, &Font12, WHITE, color);
}

void heading(bool drawing = false) {
    GUI_DrawRectangle(0, 0, 479, 44, WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    char text[80];
    if (drawing) std::snprintf(text, sizeof(text), "Erkannt: ...");
    else if (last_digit < 0) std::snprintf(text, sizeof(text), "Erkannt: -");
    else std::snprintf(text, sizeof(text), "Erkannt: %d", last_digit);
    GUI_DisString_EN(8, 0, text, &Font24, WHITE, BLUE);
    if (last_digit >= 0 && !drawing)
        std::snprintf(text, sizeof(text), "Modellwert %.0f%% | %u ms | H2 INT8", last_score * 100, last_ms);
    else std::snprintf(text, sizeof(text), "Eine Ziffer zeichnen | H2 INT8");
    GUI_DisString_EN(8, 28, text, &Font12, WHITE, BLACK);
}

bool setup_model() {
    const auto* model = tflite::GetModel(kDigitModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) return false;
    static tflite::MicroMutableOpResolver<8> ops;
    if (ops.AddConv2D() != kTfLiteOk || ops.AddMaxPool2D() != kTfLiteOk ||
        ops.AddShape() != kTfLiteOk || ops.AddStridedSlice() != kTfLiteOk ||
        ops.AddPack() != kTfLiteOk || ops.AddReshape() != kTfLiteOk ||
        ops.AddFullyConnected(RegisterDensePerChannel()) != kTfLiteOk || ops.AddSoftmax() != kTfLiteOk) return false;
    static tflite::MicroInterpreter runtime(model, ops, arena, sizeof(arena));
    interpreter = &runtime;
    if (interpreter->AllocateTensors() != kTfLiteOk) return false;
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (input->type != kTfLiteInt8 || input->bytes != 784 || input->params.scale <= 0 ||
        output->type != kTfLiteInt8 || output->bytes != 10) return false;
    return true;
}

bool predict(bool quantize = true) {
    const uint64_t started = time_us_64();
    for (int i = 0; quantize && i < 784; ++i) {
        const int q = int(std::round(image[i] / input->params.scale)) + input->params.zero_point;
        input->data.int8[i] = int8_t(std::clamp(q, -128, 127));
    }
    if (interpreter->Invoke() != kTfLiteOk) return false;
    last_digit = int(std::max_element(output->data.int8, output->data.int8 + 10) - output->data.int8);
    last_score = (int(output->data.int8[last_digit]) - output->params.zero_point) * output->params.scale;
    last_ms = unsigned((time_us_64() - started) / 1000);
    heading();
    std::printf("RESULT digit=%d score=%.6f inference_ms=%u scores=", last_digit, last_score, last_ms);
    for (int i = 0; i < 10; ++i) std::printf("%s%d", i ? "," : "", int(output->data.int8[i]));
    std::printf("\n");
    return true;
}

void recognize_drawing() {
    status("Erkenne ...", BLUE);
    if (!preprocess_digit(TP_SavedData(), image)) {
        last_digit = -1;
        heading();
        status("Bitte eine ganze Ziffer in den Rahmen zeichnen.");
        return;
    }
    if (predict()) status("CLEAR: neu zeichnen | READ: erkennen | Auto: 0.8 s");
    else { last_digit = -1; heading(); status("Fehler bei der Erkennung.", RED); }
}

void print_state() {
    std::printf("H2 DEMO READY v=1 model=head_8_16_8 format=int8 model_bytes=%u arena_used=%u board=%s digit=%d ink=%u\n",
        unsigned(sizeof(kDigitModel)), unsigned(interpreter->arena_used_bytes()), PICO_BOARD,
        last_digit, TP_InkPixels());
}

// Nur zur USB-Prüfung: I + 784 Graustufenbytes (bereits vorverarbeitet).
void serial_image(bool quantized = false) {
    for (int i = 0; i < 784; ++i) {
        const int value = getchar_timeout_us(1000000);
        if (value < 0) { std::printf("ERROR image timeout\n"); return; }
        if (quantized) input->data.int8[i] = int8_t(uint8_t(value));
        else image[i] = float(value) / 255;
    }
    if (!predict(!quantized)) std::printf("ERROR invoke\n");
}
}

int main() {
    System_Init();
    sleep_ms(1200);
    LCD_Init(SCAN_DIR_DFT, 1000);
    TP_Init(SCAN_DIR_DFT, nullptr, nullptr);
    TP_GetAdFac();
    TP_Dialog();
    TP_ConsumeClearRequest();
    // Die bekannte Aufnahmefläche bleibt gleich; die rechte Taste erkennt jetzt.
    GUI_DrawRectangle(SAVE_X0, BUTTON_Y0, SAVE_X1, BUTTON_Y1, BLUE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(408, 162, "READ", &Font16, BLUE, WHITE);
    heading();
    status("Lade H2-Modell ...");
    if (!setup_model()) {
        status("Modell konnte nicht geladen werden.", RED);
        while (true) {
            if (getchar_timeout_us(0) == '?') std::printf("H2 DEMO ERROR model setup\n");
            sleep_ms(10);
        }
    }
    status("CLEAR: neu zeichnen | READ: erkennen | Auto: 0.8 s");
    print_state();
    bool dirty = false, manual = false, usb_testing = false;
    uint64_t last_activity = TP_LastInputUs();
    while (true) {
        const int command = getchar_timeout_us(0);
        if (command == '?') print_state();
        else if (command == 'I') serial_image();
        else if (command == 'Q') { usb_testing = true; serial_image(true); }
        else if (command == 'C') {
            usb_testing = false;
            TP_ClearCanvas();
            dirty = false; manual = false; last_digit = -1;
            heading();
            status("CLEAR: neu zeichnen | READ: erkennen | Auto: 0.8 s");
        }
        if (usb_testing) { sleep_ms(2); continue; }
        TP_DrawBoard();
        if (TP_ConsumeClearRequest()) {
            dirty = false; manual = false; last_digit = -1;
            heading();
            status("CLEAR: neu zeichnen | READ: erkennen | Auto: 0.8 s");
        }
        if (TP_ConsumeSaveRequest()) manual = true;
        const auto activity = TP_LastInputUs();
        if (activity != last_activity) {
            last_activity = activity;
            if (TP_InkPixels() >= 8 && !dirty) { dirty = true; heading(true); }
        }
        if (!TP_IsTouchDown() && (manual || (dirty && time_us_64() - activity >= 800000))) {
            manual = false; dirty = false;
            recognize_drawing();
        }
        sleep_ms(2);
    }
}
