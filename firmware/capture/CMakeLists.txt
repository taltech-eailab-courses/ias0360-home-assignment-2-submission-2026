cmake_minimum_required(VERSION 3.13)
set(PICO_BOARD pico_w CACHE STRING "Board type")
get_filename_component(LAB_BASE "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include("${LAB_BASE}/pico_sdk_import.cmake")
project(h2_digit_capture C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

add_subdirectory("${LAB_BASE}/lib/config" config)
add_subdirectory("${LAB_BASE}/lib/font" font)
add_subdirectory("${LAB_BASE}/lib/sd_card_driver" sd_driver)

add_library(h2_lcd
    LCD_Touch.c
    "${LAB_BASE}/lib/lcd/LCD_Driver.c"
    "${LAB_BASE}/lib/lcd/LCD_GUI.c"
)
target_include_directories(h2_lcd PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${LAB_BASE}/lib/lcd"
    "${LAB_BASE}/lib/config"
    "${LAB_BASE}/lib/font"
)
target_link_libraries(h2_lcd PUBLIC config font pico_stdlib)

add_executable(h2_digit_capture main.cpp "${LAB_BASE}/config/hw_config.c")
target_compile_options(h2_digit_capture PRIVATE -Wall -Wextra -Wshadow)
target_compile_definitions(h2_digit_capture PRIVATE PICO_STACK_SIZE=0x1000 PICO_CORE1_STACK_SIZE=0x1000)
target_link_libraries(h2_digit_capture PRIVATE h2_lcd sd_custom_driver pico_stdlib pico_multicore pico_util)
pico_enable_stdio_usb(h2_digit_capture 1)
pico_enable_stdio_uart(h2_digit_capture 0)
pico_add_extra_outputs(h2_digit_capture)
