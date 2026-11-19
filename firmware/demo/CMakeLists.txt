cmake_minimum_required(VERSION 3.13)
set(PICO_BOARD pico_w CACHE STRING "Board type")
get_filename_component(LAB_BASE "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(CAPTURE "${LAB_BASE}/capture")
set(TFLMICRO_PATH "${LAB_BASE}/deps/course/lab_2_4/pico_sw/lcd/lib/pico-tflmicro" CACHE PATH "Course pico-tflmicro source")
include("${LAB_BASE}/pico_sdk_import.cmake")
project(h2_digit_demo C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

add_subdirectory("${LAB_BASE}/lib/config" config)
add_subdirectory("${LAB_BASE}/lib/font" font)
add_subdirectory("${TFLMICRO_PATH}" tflmicro EXCLUDE_FROM_ALL)
add_library(demo_lcd
    "${CAPTURE}/LCD_Touch.c"
    "${LAB_BASE}/lib/lcd/LCD_Driver.c"
    "${LAB_BASE}/lib/lcd/LCD_GUI.c")
target_include_directories(demo_lcd PUBLIC "${CAPTURE}" "${LAB_BASE}/lib/lcd"
    "${LAB_BASE}/lib/config" "${LAB_BASE}/lib/font")
target_link_libraries(demo_lcd PUBLIC config font pico_stdlib)

add_executable(h2_digit_demo main.cpp preprocess.cpp dense_per_channel.cpp)
target_compile_options(h2_digit_demo PRIVATE -Wall -Wextra -Wshadow -fno-rtti -fno-exceptions)
target_compile_definitions(h2_digit_demo PRIVATE PICO_STACK_SIZE=0x1000 PICO_CORE1_STACK_SIZE=0x1000)
target_link_libraries(h2_digit_demo PRIVATE demo_lcd pico-tflmicro pico_stdlib)
pico_enable_stdio_usb(h2_digit_demo 1)
pico_enable_stdio_uart(h2_digit_demo 0)
pico_set_program_name(h2_digit_demo "H2 digit recognition")
pico_set_program_version(h2_digit_demo "1.0")
pico_add_extra_outputs(h2_digit_demo)
