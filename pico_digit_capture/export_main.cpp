// Read-only USB export utility for the Pico-Eval-Board's SDIO card.
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "ff.h"
#include "sd_card.h"
#include "f_util.h"
#include "hw_config.h"

static FATFS fs;

static void send_text(const char *text)
{
    stdio_put_string(text, -1, false, false);
    stdio_flush();
}

int main()
{
    stdio_init_all();

    // Wait for the host export script to open the USB serial port.
    absolute_time_t deadline = make_timeout_time_ms(120000);
    while (!stdio_usb_connected() && !time_reached(deadline)) sleep_ms(20);
    if (!stdio_usb_connected()) while (true) tight_loop_contents();

    send_text("PICO_PGM_EXPORT_BEGIN\n");
    if (!sd_init_driver()) {
        send_text("EXPORT_ERROR sd_init_driver\n");
        while (true) tight_loop_contents();
    }

    sd_card_t *card = sd_get_by_num(0);
    if (!card) {
        send_text("EXPORT_ERROR no_card_config\n");
        while (true) tight_loop_contents();
    }
    const char *drive = sd_get_drive_prefix(card);
    if (!drive) {
        send_text("EXPORT_ERROR no_drive\n");
        while (true) tight_loop_contents();
    }

    FRESULT fr = f_mount(&fs, drive, 1);
    if (fr != FR_OK) {
        char message[64];
        snprintf(message, sizeof(message), "EXPORT_ERROR mount_%d\n", (int)fr);
        send_text(message);
        while (true) tight_loop_contents();
    }

    char buffer[1024];
    for (int digit = 0; digit < 10; ++digit) {
        char name[32];
        char path[64];
        snprintf(name, sizeof(name), "digit_%d.pgm", digit);
        snprintf(path, sizeof(path), "%s/%s", drive, name);

        FIL file;
        fr = f_open(&file, path, FA_READ);
        if (fr != FR_OK) {
            char message[64];
            snprintf(message, sizeof(message), "EXPORT_ERROR open_%s_%d\n", name, (int)fr);
            send_text(message);
            f_unmount(drive);
            while (true) tight_loop_contents();
        }

        char marker[48];
        snprintf(marker, sizeof(marker), "<<<BEGIN %s>>>\n", name);
        send_text(marker);
        do {
            UINT bytes_read = 0;
            fr = f_read(&file, buffer, sizeof(buffer), &bytes_read);
            if (fr != FR_OK) {
                char message[64];
                snprintf(message, sizeof(message), "EXPORT_ERROR read_%s_%d\n", name, (int)fr);
                send_text(message);
                f_close(&file);
                f_unmount(drive);
                while (true) tight_loop_contents();
            }
            if (bytes_read && stdio_put_string(buffer, (int)bytes_read, false, false) != (int)bytes_read) {
                send_text("EXPORT_ERROR usb_write\n");
                f_close(&file);
                f_unmount(drive);
                while (true) tight_loop_contents();
            }
        } while (!f_eof(&file));
        stdio_flush();

        fr = f_close(&file);
        if (fr != FR_OK) {
            send_text("EXPORT_ERROR close\n");
            f_unmount(drive);
            while (true) tight_loop_contents();
        }
        snprintf(marker, sizeof(marker), "\n<<<END %s>>>\n", name);
        send_text(marker);
    }

    f_unmount(drive);
    send_text("PICO_PGM_EXPORT_DONE\n");
    while (true) tight_loop_contents();
}
