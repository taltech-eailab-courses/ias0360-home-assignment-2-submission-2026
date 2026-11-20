# IAS0360 Home Assignment 2 (image domain)

Student: Reimo Rämman, 253440IACM

## Files
- models.py: baseline CNN, MobileNet-style and SqueezeNet-style models
- train_eval.py: training, evaluation and timing, writes results/results.csv
- run_all.py, run_more.py: the experiments from the report
- make_table.py: prints the results table
- parse_capture.py: converts the Pico touchscreen serial printout to PNG
- data_own_digits/: 10 digits drawn on the Pico touchscreen (28x28 grayscale)
- results/results.csv: results of all runs

## How to run
Python 3.12 with tf_keras, numpy and pillow (the course Docker container has them).

    python run_all.py
    python run_more.py
    python make_table.py

## Expected output
12 rows in results/results.csv. The best model is flat_4_8_16_lr003:
3,722 parameters (14.54 KB), 98.30% MNIST test accuracy, 8/10 own digits.

## Own data
The digits were drawn on the Pico touchscreen (lab 2.4 demo) and read from the USB
serial printout with parse_capture.py. They were not saved to the SD card.
