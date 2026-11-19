#include "LCD_Touch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern LCD_DIS sLCD_DIS;
extern uint8_t id;
static TP_DEV sTP_DEV;
static TP_DRAW sTP_Draw;

// Live "shadow" of what has been drawn (0 = empty, 1 = drawn)
static uint8_t sDrawShadow[BOX_H][BOX_W];
static bool sSaveRequested = false;
static bool sClearRequested = false;
static bool sTouchWasDown = false;
static unsigned sInkPixels = 0;
static uint64_t sLastInputUs = 0;


// Helper to record a pixel into the shadow buffer
static inline void Capture_SetPixel(uint16_t x, uint16_t y)
{
    if (x >= BOX_X0 && x < BOX_X1 && y >= BOX_Y0 && y < BOX_Y1) {
        uint8_t *pixel = &sDrawShadow[y - BOX_Y0][x - BOX_X0];
        if (!*pixel) {
            *pixel = 1;
            ++sInkPixels;
        }
    }
}

static uint16_t TP_Read_ADC(uint8_t CMD)
{
    uint16_t Data = 0;

    //A cycle of at least 400ns.
    DEV_Digital_Write(TP_CS_PIN, 0);

    SPI4W_Write_Byte(CMD);
    Driver_Delay_us(200);

    //	dont write 0xff, it will block xpt2046
    //Data = SPI4W_Read_Byte(0Xff);
    Data = SPI4W_Read_Byte(0X00);
    Data <<= 8; //7bit
    Data |= SPI4W_Read_Byte(0X00);
    //Data = SPI4W_Read_Byte(0Xff);
    Data >>= 3; //5bit
    DEV_Digital_Write(TP_CS_PIN, 1);
    return Data;
}

#define READ_TIMES 5 //Number of readings
#define LOST_NUM 1   //Discard value
static uint16_t
TP_Read_ADC_Average(uint8_t Channel_Cmd)
{
    uint8_t i, j;
    uint16_t Read_Buff[READ_TIMES];
    uint16_t Read_Sum = 0, Read_Temp = 0;
    //LCD SPI speed = 3 MHz
    spi_set_baudrate(SPI_PORT, 3000000);
    //Read and save multiple samples
    for (i = 0; i < READ_TIMES; i++)
    {
        Read_Buff[i] = TP_Read_ADC(Channel_Cmd);
        Driver_Delay_us(200);
    }
    //LCD SPI speed = 18 MHz
    spi_set_baudrate(SPI_PORT, 18000000);
    //Sort from small to large
    for (i = 0; i < READ_TIMES - 1; i++)
    {
        for (j = i + 1; j < READ_TIMES; j++)
        {
            if (Read_Buff[i] > Read_Buff[j])
            {
                Read_Temp = Read_Buff[i];
                Read_Buff[i] = Read_Buff[j];
                Read_Buff[j] = Read_Temp;
            }
        }
    }

    //Exclude the largest and the smallest
    for (i = LOST_NUM; i < READ_TIMES - LOST_NUM; i++)
        Read_Sum += Read_Buff[i];

    //Averaging
    Read_Temp = Read_Sum / (READ_TIMES - 2 * LOST_NUM);

    return Read_Temp;
}

static void TP_Read_ADC_XY(uint16_t *pXCh_Adc, uint16_t *pYCh_Adc)
{
    *pXCh_Adc = TP_Read_ADC_Average(0xD0);
    *pYCh_Adc = TP_Read_ADC_Average(0x90);
}

#define ERR_RANGE 50 //tolerance scope
static bool TP_Read_TwiceADC(uint16_t *pXCh_Adc, uint16_t *pYCh_Adc)
{
    uint16_t XCh_Adc1, YCh_Adc1, XCh_Adc2, YCh_Adc2;

    //Read the ADC values Read the ADC values twice
    TP_Read_ADC_XY(&XCh_Adc1, &YCh_Adc1);
    Driver_Delay_us(10);
    TP_Read_ADC_XY(&XCh_Adc2, &YCh_Adc2);
    Driver_Delay_us(10);

    //The ADC error used twice is greater than ERR_RANGE to take the average
    if (((XCh_Adc2 <= XCh_Adc1 && XCh_Adc1 < XCh_Adc2 + ERR_RANGE) ||
         (XCh_Adc1 <= XCh_Adc2 && XCh_Adc2 < XCh_Adc1 + ERR_RANGE)) &&
        ((YCh_Adc2 <= YCh_Adc1 && YCh_Adc1 < YCh_Adc2 + ERR_RANGE) ||
         (YCh_Adc1 <= YCh_Adc2 && YCh_Adc2 < YCh_Adc1 + ERR_RANGE)))
    {
        *pXCh_Adc = (XCh_Adc1 + XCh_Adc2) / 2;
        *pYCh_Adc = (YCh_Adc1 + YCh_Adc2) / 2;
        return true;
    }

    //The ADC error used twice is less than ERR_RANGE returns failed
    return false;
}

static uint8_t TP_Scan(uint8_t chCoordType)
{
    //In X, Y coordinate measurement, IRQ is disabled and output is low
    if (!DEV_Digital_Read(TP_IRQ_PIN))
    { //Press the button to press
        //Read the physical coordinates
        if (chCoordType)
        {
            TP_Read_TwiceADC(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint);
            //Read the screen coordinates
        }
        else if (TP_Read_TwiceADC(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint))
        {

            if (LCD_2_8 == id)
            {
                sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                  sTP_DEV.fXfac * sTP_DEV.Xpoint -
                                  sTP_DEV.iXoff;
                sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                  sTP_DEV.fYfac * sTP_DEV.Ypoint -
                                  sTP_DEV.iYoff;
            }
            else
            {
                //DEBUG("(Xad,Yad) = %d,%d\r\n",sTP_DEV.Xpoint,sTP_DEV.Ypoint);
                if (sTP_DEV.TP_Scan_Dir == R2L_D2U)
                { //Converts the result to screen coordinates
                    sTP_Draw.Xpoint = sTP_DEV.fXfac * sTP_DEV.Xpoint +
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sTP_DEV.fYfac * sTP_DEV.Ypoint +
                                      sTP_DEV.iYoff;
                }
                else if (sTP_DEV.TP_Scan_Dir == L2R_U2D)
                {
                    sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                      sTP_DEV.fXfac * sTP_DEV.Xpoint -
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                      sTP_DEV.fYfac * sTP_DEV.Ypoint -
                                      sTP_DEV.iYoff;
                }
                else if (sTP_DEV.TP_Scan_Dir == U2D_R2L)
                {
                    sTP_Draw.Xpoint = sTP_DEV.fXfac * sTP_DEV.Ypoint +
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sTP_DEV.fYfac * sTP_DEV.Xpoint +
                                      sTP_DEV.iYoff;
                }
                else
                {
                    sTP_Draw.Xpoint = sLCD_DIS.LCD_Dis_Column -
                                      sTP_DEV.fXfac * sTP_DEV.Ypoint -
                                      sTP_DEV.iXoff;
                    sTP_Draw.Ypoint = sLCD_DIS.LCD_Dis_Page -
                                      sTP_DEV.fYfac * sTP_DEV.Xpoint -
                                      sTP_DEV.iYoff;
                }
                // DEBUG("( x , y ) = %d,%d\r\n",sTP_Draw.Xpoint,sTP_Draw.Ypoint);
            }
        }
        if (0 == (sTP_DEV.chStatus & TP_PRESS_DOWN))
        { //Not being pressed
            sTP_DEV.chStatus = TP_PRESS_DOWN | TP_PRESSED;
            sTP_DEV.Xpoint0 = sTP_DEV.Xpoint;
            sTP_DEV.Ypoint0 = sTP_DEV.Ypoint;
        }
    }
    else
    {
        if (sTP_DEV.chStatus & TP_PRESS_DOWN)
        {                                  //0x80
            sTP_DEV.chStatus &= ~(1 << 7); //0x00
        }
        else
        {
            sTP_DEV.Xpoint0 = 0;
            sTP_DEV.Ypoint0 = 0;
            sTP_DEV.Xpoint = 0xffff;
            sTP_DEV.Ypoint = 0xffff;
        }
    }

    return (sTP_DEV.chStatus & TP_PRESS_DOWN);
}

void TP_GetAdFac(void)
{
    if (LCD_2_8 == id)
    {
        sTP_DEV.fXfac = 0.066626;
        sTP_DEV.fYfac = 0.089779;
        sTP_DEV.iXoff = -20;
        sTP_DEV.iYoff = -34;
    }
    else
    {
        if (sTP_DEV.TP_Scan_Dir == D2U_L2R)
        { //SCAN_DIR_DFT = D2U_L2R
            sTP_DEV.fXfac = -0.132443;
            sTP_DEV.fYfac = 0.089997;
            sTP_DEV.iXoff = 516;
            sTP_DEV.iYoff = -22;
        }
        else if (sTP_DEV.TP_Scan_Dir == L2R_U2D)
        {
            sTP_DEV.fXfac = 0.089697;
            sTP_DEV.fYfac = 0.134792;
            sTP_DEV.iXoff = -21;
            sTP_DEV.iYoff = -39;
        }
        else if (sTP_DEV.TP_Scan_Dir == R2L_D2U)
        {
            sTP_DEV.fXfac = 0.089915;
            sTP_DEV.fYfac = 0.133178;
            sTP_DEV.iXoff = -22;
            sTP_DEV.iYoff = -38;
        }
        else if (sTP_DEV.TP_Scan_Dir == U2D_R2L)
        {
            sTP_DEV.fXfac = -0.132906;
            sTP_DEV.fYfac = 0.087964;
            sTP_DEV.iXoff = 517;
            sTP_DEV.iYoff = -20;
        }
        else
        {
            LCD_Clear(LCD_BACKGROUND);
            GUI_DisString_EN(0, 60, "Does not support touch-screen \
                            calibration in this direction",
                             &Font16, FONT_BACKGROUND, RED);
        }
    }
}


void TP_Dialog(void)
{
    LCD_Clear(LCD_BACKGROUND);

    GUI_DrawRectangle(CLEAR_X0, BUTTON_Y0, CLEAR_X1, BUTTON_Y1,
                      BLUE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(22, 162, "CLEAR", &Font16, BLUE, WHITE);
    GUI_DrawRectangle(SAVE_X0, BUTTON_Y0, SAVE_X1, BUTTON_Y1,
                      BLUE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DisString_EN(408, 162, "SAVE", &Font16, BLUE, WHITE);
    TP_ClearCanvas();
    sClearRequested = true;
}

void TP_ClearCanvas(void)
{
    // Clear only the canvas, keeping the prompt and large side buttons visible.
    GUI_DrawRectangle(BOX_X0, BOX_Y0, BOX_X1, BOX_Y1,
                      WHITE, DRAW_FULL, DOT_PIXEL_1X1);
    GUI_DrawRectangle(BOX_X0 , BOX_Y0 ,
                      BOX_X1, BOX_Y1,
                      BLACK, DRAW_EMPTY, DOT_PIXEL_2X2);

    // NEW: also clear the shadow buffer that mirrors what's drawn
    memset(sDrawShadow, 0, sizeof(sDrawShadow));
    sInkPixels = 0;
    sLastInputUs = time_us_64();
    sSaveRequested = false;
}

void TP_Save(void)
{
    sSaveRequested = true;
}

bool TP_ConsumeSaveRequest(void)
{
    bool requested = sSaveRequested;
    sSaveRequested = false;
    return requested;
}

bool TP_ConsumeClearRequest(void)
{
    bool requested = sClearRequested;
    sClearRequested = false;
    return requested;
}

const uint8_t *TP_SavedData(void)
{
    return &sDrawShadow[0][0];
}

size_t TP_SavedDataLength(void)
{
    return sizeof(sDrawShadow);
}

unsigned TP_InkPixels(void) { return sInkPixels; }
uint64_t TP_LastInputUs(void) { return sLastInputUs; }
bool TP_IsTouchDown(void) { return sTouchWasDown; }

void TP_DrawBoard(void)
{
    //	sTP_DEV.chStatus &= ~(1 << 6);
    TP_Scan(0);
    bool down = (sTP_DEV.chStatus & TP_PRESS_DOWN) != 0;
    bool new_press = down && !sTouchWasDown;
    // Include pen-up, so a pause is measured from the end of the last touch.
    if (down || sTouchWasDown) sLastInputUs = time_us_64();
    sTouchWasDown = down;
    if (sTP_DEV.chStatus & TP_PRESS_DOWN)
    {
        spi_init(SPI_PORT, 10000000);

        if (new_press && sTP_Draw.Xpoint >= CLEAR_X0 && sTP_Draw.Xpoint <= CLEAR_X1 &&
            sTP_Draw.Ypoint >= BUTTON_Y0 && sTP_Draw.Ypoint <= BUTTON_Y1)
        {
            TP_ClearCanvas();
            sClearRequested = true;
        }
        else if (new_press && sTP_Draw.Xpoint >= SAVE_X0 && sTP_Draw.Xpoint <= SAVE_X1 &&
                 sTP_Draw.Ypoint >= BUTTON_Y0 && sTP_Draw.Ypoint <= BUTTON_Y1)
        {
            TP_Save();
        }

        sTP_Draw.Color = BLACK;


        if (sTP_Draw.Xpoint > BOX_X0 && sTP_Draw.Xpoint < BOX_X1 &&
            sTP_Draw.Ypoint > BOX_Y0 && sTP_Draw.Ypoint < BOX_Y1)
        {
            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint, sTP_Draw.Ypoint);

            GUI_DrawPoint(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint);

            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint + 1,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint, sTP_Draw.Ypoint + 1);

            GUI_DrawPoint(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint + 1,
                        sTP_Draw.Color, DOT_PIXEL_1X1, DOT_FILL_RIGHTUP);
            Capture_SetPixel(sTP_Draw.Xpoint + 1, sTP_Draw.Ypoint + 1);

            // The DOT_PIXEL_2X2 point covers the same area; no extra capture needed
            GUI_DrawPoint(sTP_Draw.Xpoint, sTP_Draw.Ypoint,
                        sTP_Draw.Color, DOT_PIXEL_2X2, DOT_FILL_RIGHTUP);
        }


        spi_init(SPI_PORT, 5000000);
    }
    SPI4W_Write_Byte(0xFF);
}

void TP_Init(LCD_SCAN_DIR Lcd_ScanDir, TP_DATA *tp_data_ptr, mutex_t *mutex)
{
    DEV_Digital_Write(TP_CS_PIN, 1);

    sTP_DEV.TP_Scan_Dir = Lcd_ScanDir;

    (void)tp_data_ptr;
    (void)mutex;

    TP_Read_ADC_XY(&sTP_DEV.Xpoint, &sTP_DEV.Ypoint);
}
