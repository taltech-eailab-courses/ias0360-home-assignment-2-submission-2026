# Actual touchscreen samples

Put one real, SD-card-collected image for each digit here:

```text
digit_0.png  digit_1.png  digit_2.png  digit_3.png  digit_4.png
digit_5.png  digit_6.png  digit_7.png  digit_8.png  digit_9.png
```

BMP, JPG, JPEG and PGM are also supported; use lowercase extensions. Keep only
one supported file per digit. Copy/rename files here rather than editing the
SD-card originals. Store original captures and collection notes under
`data/raw_sd_card/`. A blank image is rejected, not evaluated as a digit.

Use the required touchscreen and SD-card workflow. Synthetic unit-test images,
MNIST examples, and computer-generated digits do not meet the collection task.

The preprocessing code reads these sources, converts to grayscale, detects
background polarity, crops the strokes, preserves their aspect ratio, scales
inside a 20x20 region, and centers them in a 28x28 black canvas. Inputs become
float32 arrays with shape `(10,28,28,1)` and values in `[0,1]`.

The original files remain unchanged. Derived images and a 0-to-9 preview strip
are saved in the selected run's `own_digits/` directory. Inspect the preview
before interpreting accuracy. The preprocessing expects a single drawn digit
per image, not an entire touchscreen screenshot with buttons and labels.
