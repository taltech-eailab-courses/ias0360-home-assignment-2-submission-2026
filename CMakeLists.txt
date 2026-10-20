cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(home_assignment_2 C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_subdirectory(lib/config)
add_subdirectory(lib/font)
add_subdirectory(lib/lcd)
add_subdirectory(lib/sd_card_driver build_sd)

include_directories(
    config
    lib/config
    lib/font
    lib/lcd
    lib/sd_card_driver/include
    lib/sd_card_driver/ff15/source
)

add_executable(home_assignment_2
    config/hw_config.c
    src/main.c
)

target_compile_definitions(home_assignment_2 PRIVATE
    PICO_STACK_SIZE=0x1000
    PICO_CORE1_STACK_SIZE=0x1000
)

target_compile_options(home_assignment_2 PRIVATE
    -Wall
    -Wextra
)

target_link_libraries(home_assignment_2
    pico_stdlib
    pico_multicore
    pico_util
    hardware_i2c
    hardware_spi
    hardware_dma
    hardware_pio
    hardware_clocks
    hardware_adc
    sd_custom_driver
    m
    config
    font
    lcd
)

target_link_options(home_assignment_2 PRIVATE
    "-Wl,--print-memory-usage"
)

pico_enable_stdio_usb(home_assignment_2 1)
pico_enable_stdio_uart(home_assignment_2 0)

pico_add_extra_outputs(home_assignment_2)
