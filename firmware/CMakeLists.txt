cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico_w)
include(pico_sdk_import.cmake)
project(hw2_gesture C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

# preprocessing_settings.hpp is used by the model library and the targets
include_directories(src lib/model lib/imu lib/pico-tflmicro)

add_subdirectory(lib/pico-tflmicro)
add_subdirectory(lib/imu)
add_subdirectory(lib/model)

add_library(data_preprocessor src/data_preprocessor.cpp)
target_link_libraries(data_preprocessor pico_stdlib)

# main: live gesture recognition from the IMU
# own_data_test: runs the 20 own recordings stored in flash, for comparison with the PC
add_executable(main main.cpp)
add_executable(own_data_test own_data_test.cpp)

target_link_libraries(main model data_preprocessor icm20948 pico-tflmicro
                      pico_stdlib hardware_pio hardware_i2c)
target_link_libraries(own_data_test model data_preprocessor pico-tflmicro pico_stdlib)
pico_generate_pio_header(main ${CMAKE_CURRENT_LIST_DIR}/src/ws2812.pio)

foreach(target main own_data_test)
    set_target_properties(${target} PROPERTIES
        COMPILE_FLAGS "-fno-rtti -fno-exceptions -fno-threadsafe-statics")
    set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-Wl,--print-memory-usage")
    pico_enable_stdio_usb(${target} 1)
    pico_enable_stdio_uart(${target} 0)
    pico_add_extra_outputs(${target})
endforeach()
