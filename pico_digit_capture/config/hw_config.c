#include <assert.h>
#include "hw_config.h"

// Waveshare Pico-Eval-Board (SKU 20159) wires its microSD slot over SDIO:
// CLK=GP5, CMD=GP18, D0..D3=GP19..GP22. The SDIO PIO driver derives CLK/D1-D3
// from D0 using the offsets in rp2040_sdio.pio.
static sd_sdio_if_t sdio_ifs[] = {{
    .CMD_gpio = 18,
    .D0_gpio = 19,
    .SDIO_PIO = pio1,
    .DMA_IRQ_num = DMA_IRQ_1,
    .baud_rate = 10000000,
}};
static sd_card_t sd_cards[] = {{
    .type = SD_IF_SDIO,
    .sdio_if_p = &sdio_ifs[0],
    .use_card_detect = false,
}};
size_t sd_get_num(void){return count_of(sd_cards);}
sd_card_t *sd_get_by_num(size_t n){assert(n<sd_get_num());return n<sd_get_num()?&sd_cards[n]:NULL;}
