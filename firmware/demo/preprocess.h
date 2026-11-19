#pragma once
#include <cstdint>

// Gleiche Schritte wie H2/ml/prepare_data.py. Eingabe: 1 = Stift, 0 = leer.
// Ausgabe: 28×28, weißer Strich auf Schwarz, auf den Spitzenwert normiert.
bool preprocess_digit(const uint8_t* ink, float* output);
