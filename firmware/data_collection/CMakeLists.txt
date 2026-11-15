cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)
project(ha2_data_collection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

set(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../common)
add_subdirectory(${COMMON_DIR}/icm20948 icm20948)
add_subdirectory(${COMMON_DIR}/sd_card_driver sd_card_driver)

add_executable(data_collection main.cpp ${COMMON_DIR}/hw_config.c)
target_include_directories(data_collection PRIVATE ${COMMON_DIR} ${COMMON_DIR}/icm20948)
target_compile_definitions(data_collection PRIVATE
    PICO_STACK_SIZE=0x1000
    PICO_CORE1_STACK_SIZE=0x1000
)
target_link_libraries(data_collection
    pico_stdlib
    pico_multicore
    pico_util
    hardware_pio
    hardware_i2c
    icm20948
    sd_custom_driver
)
pico_generate_pio_header(data_collection ${COMMON_DIR}/ws2812.pio)
pico_enable_stdio_usb(data_collection 1)
pico_enable_stdio_uart(data_collection 0)
pico_add_extra_outputs(data_collection)
