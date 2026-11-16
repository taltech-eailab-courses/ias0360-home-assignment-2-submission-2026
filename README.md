# Home Assignment 2: MNIST digit recognition

This project trains and evaluates three TensorFlow/Keras models on MNIST
handwritten digits:

- a compact custom CNN;
- SqueezeNet with Fire modules;
- MobileNet-style depthwise-separable convolution blocks.

The project also contains Raspberry Pi Pico firmware for collecting digit
images from the touchscreen, downsampling them to 28 x 28 pixels, and saving
them to an SD card.

## Python model evaluation

Create or activate a Python environment containing TensorFlow, NumPy, and
Matplotlib. From this directory, run:

```sh
python3 main.py --model cnn
python3 main.py --model squeezenet
python3 main.py --model mobilenet
```

The default model is the custom CNN, so this is equivalent to:

```sh
python3 main.py
```

The script downloads MNIST through TensorFlow if it is not already cached,
trains the selected model for five epochs, evaluates it on the MNIST test
set and the files in `my_digits/`, and reports the average prediction time.
The custom digit files must be named `digit_0.txt` through `digit_9.txt` and
contain 28 rows of 28 binary pixel values.

The script also writes plots and exported models to `results/`. These are
generated outputs and are not required to compile the Pico firmware.

## Raspberry Pi Pico firmware

The firmware uses the LCD, touchscreen, SD-card driver, and Pico SDK from the
repository's `lab_1_1/lcd_sd_card_example` directory. The Pico SDK must
therefore be available at the path expected by
`lab_1_1/lcd_sd_card_example/pico_sdk_import.cmake`.

Build from this directory with:

```sh
cmake -S . -B build -DDIGIT_LABEL=1
cmake --build build -j2
```

`DIGIT_LABEL` selects the digit label written to the captured filenames and
must be an integer from 0 to 9. For another digit, reconfigure the same build
directory, for example:

```sh
cmake -S . -B build -DDIGIT_LABEL=2
cmake --build build -j2
```

The build produces `build/image_processing.uf2`. Flash this UF2 file to the
Pico using the normal Raspberry Pi Pico bootloader procedure.

## Firmware behavior and output

The firmware displays a 240 x 240 drawing area. After a digit is drawn and
the SAVE button is pressed, it stores:

```text
<digit>_<sample>_original.txt
<digit>_<sample>_downsampled.txt
```

The downsampled file contains a 28 x 28 binary image. Up to ten samples are
captured for the selected digit. The CLEAR button removes the current drawing
without saving it.
