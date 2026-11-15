| Optimization method | Inference average speed* | Memory size** (params / float32 / uint8 TFLite) | Validation accuracy (5-fold CV, provided) | Own data accuracy*** (uint8 TFLite) |
|---|---|---|---|---|
| Initial model trained on the provided dataset only (no own recordings in training) | 0.016 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 4.9 | 65.0 % (65/100) |
| Step 6 MobileNet-style CNN trained on the provided dataset only (no own recordings in training) | 0.014 ms | 10,308 / 40.3 KB / 28.0 KB | 99.0 % ± 2.0 | 60.0 % (60/100) |
| Initial model (lab 2.2 CNN: 8/16 filters, dense 16, ReLU, dropout 0.1; Adam lr=0.001, 30 epochs, batch 16) | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 96.0 % (96/100) |
| Number of epochs = 10; learning rate = 0.05 | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 92.0 % ± 5.1 | 85.0 % (85/100) |
| Number of epochs = 10 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 3.7 | 90.0 % (90/100) |
| Number of epochs = 60 | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 96.0 % (96/100) |
| Number of epochs = 100 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94.0 % (94/100) |
| Learning rate = 0.01 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 95.0 % ± 4.5 | 94.0 % (94/100) |
| Learning rate = 0.0003, 60 epochs | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95.0 % (95/100) |
| Optimizer SGD (momentum 0.9), lr = 0.01, 60 epochs | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95.0 % (95/100) |
| Optimizer RMSprop, lr = 0.001 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94.0 % (94/100) |
| Batch size = 8 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 94.0 % (94/100) |
| Batch size = 32, 60 epochs | 0.015 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 92.0 % (92/100) |
| Activation = tanh | 0.015 ms | 4,556 / 17.8 KB / 10.2 KB | 97.0 % ± 2.4 | 94.0 % (94/100) |
| No Dropout | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 93.0 % ± 7.5 | 95.0 % (95/100) |
| Dropout = 0.3 | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 94.0 % ± 5.8 | 95.0 % (95/100) |
| More neurons: 16/32 filters, dense 32 | 0.024 ms | 17,812 / 69.6 KB / 23.9 KB | 93.0 % ± 7.5 | 94.0 % (94/100) |
| One more Conv2D+MaxPool layer | 0.015 ms | 6,892 / 26.9 KB / 13.9 KB | 96.0 % ± 3.7 | 95.0 % (95/100) |
| Data augmentation x10 (time warp/shift, rotation +-15 deg, scale, noise) | 0.014 ms | 4,556 / 17.8 KB / 9.9 KB | 97.0 % ± 2.4 | 99.0 % (99/100) |
| Combined: 16/32 filters, dense 32, dropout 0.3, augmentation x10, 80 epochs | 0.024 ms | 17,812 / 69.6 KB / 23.9 KB | 98.0 % ± 2.4 | 98.0 % (98/100) |
| Step 6: MobileNet-style depthwise-separable CNN (augmentation x10, 80 epochs) | 0.014 ms | 10,308 / 40.3 KB / 28.0 KB | 98.0 % ± 2.4 | 100.0 % (100/100) |
| Step 6: SqueezeNet-style Fire-module CNN (augmentation x10, 80 epochs) | 0.017 ms | 8,180 / 32.0 KB / 27.7 KB | 99.0 % ± 2.0 | 100.0 % (100/100) |

\* Total uint8 TFLite prediction time divided by the number of gestures, measured on the training PC (held-out own data (100 gestures)). On-device (Pico 2 W / RP2350) times for the deployed models are printed by the `own_data_test` and `live_inference` firmware.
\*\* Total parameters as reported by `model.summary()` (incl. BatchNorm statistics, which are folded into the convolutions on conversion), float32 weight size, and the size of the deployed uint8 `.tflite` flatbuffer (weights + graph).
\*\*\* Every own recording is classified exactly once, by the fold model that did not train on it. Rows marked 'provided dataset only' never see own recordings in training; all other rows train on provided + own recordings of the other folds.
