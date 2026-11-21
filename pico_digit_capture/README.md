# Pico touchscreen digit capture firmware

This firmware adapts the course LCD touchscreen example to collect the ten
real handwritten digit images required by Home Assignment 2. It saves each
touchscreen drawing as a portable grayscale PGM image on the SD card:
`digit_0.pgm` through `digit_9.pgm`.

## Build

The firmware uses the Raspberry Pi Pico SDK and the board/pin setup in
`config/hw_config.c`. From the assignment folder, build it with:

```bash
export PICO_SDK_PATH="$HOME/pico-sdk"
cmake -S pico_digit_capture -B /tmp/pico_digit_capture_build -G Ninja
cmake --build /tmp/pico_digit_capture_build
```

The resulting UF2 is `/tmp/pico_digit_capture_build/main.uf2`; a compiled copy
is included here as `main.uf2`. The project defaults to `PICO_BOARD=pico_w` as
in the course touchscreen sample. If your
board is a non-W Pico, configure with `-DPICO_BOARD=pico` instead.

## Export saved captures without a card reader

The separate `export_digits.uf2` reads the SD card only and sends the ten PGM
files over USB serial. Keep the card inserted in the Eval Board, put the Pico
in BOOTSEL mode, and copy `export_digits.uf2` to `RPI-RP2`. Then, from the
assignment folder, find the USB serial port and run the receiver:

```bash
ls /dev/cu.usbmodem*
python3 export_sd_digits.py /dev/cu.usbmodemXXXX
```

Replace `XXXX` with the port shown by `ls`. The script saves copies in both
`data/raw_sd_card/` and `data/own_digits/`, and refuses to overwrite files.
After recapturing, use `--refresh-own-digits` to replace only the evaluation
copies while preserving the first raw export.
The exporter does not format or write to the SD card. When it finishes, flash
the capture `main.uf2` again only if you want to use the touchscreen capture
program.

## Collect the ten digits

1. Back up any existing SD-card files. This firmware never formats the card;
   it replaces `digit_N.pgm` when that digit is successfully captured. Use a
   FAT-formatted card with enough free space.
2. Put the Pico into BOOTSEL mode and copy `main.uf2` to its `RPI-RP2` drive.
   It restarts after copying.
3. The screen shows `DRAW DIGIT: 0`. Draw one clear digit inside the box, then
   tap **SAVE** at the top-right. Wait for the screen to advance before drawing
   the next one. A blank canvas is rejected and keeps the current digit prompt.
   Tap **CLEAR** at the top-left to erase and retry before saving.
   Check the status at the bottom: it should say `SD READY` before collecting.
   `SD INIT FAILED` means the card was not mounted; `SD WRITE FAILED` means a
   file could not be saved (for example, that digit filename already exists).
4. Continue through digit 9. After the tenth successful save, the screen says
   `ALL DIGITS SAVED` and the SD card has `digit_0.pgm` through `digit_9.pgm`.
5. Power off before removing the SD card. Copy the ten PGM files unchanged to
   `data/raw_sd_card/` as the collection evidence.

Each file is a 280x240 PGM grayscale image. The assignment Python pipeline reads
PGM files and crops, scales and centers the ink into the model's 28x28 input.
Put a copy of each PGM file in `data/own_digits/`; the expected names are
`digit_0.pgm` through `digit_9.pgm`. Then run the Docker evaluation command in
the parent README.

## Hardware note

The firmware targets the Waveshare Pico-Eval-Board, SKU 20159. Its LCD uses SPI,
while its microSD slot uses SDIO: CLK GP5, CMD GP18, and data GP19–GP22. The
SDIO driver derives the clock and remaining data pins from GP19 as defined by
its PIO program. A successful compile does not prove the physical touchscreen,
SD slot, or card filesystem works; check the on-screen status on the actual
board.
