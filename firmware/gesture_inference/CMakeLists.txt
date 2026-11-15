cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)
project(ha2_gesture_inference C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
pico_sdk_init()

set(COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../common)

# TensorFlow Lite Micro from the lab repository (this repo is in <lab repo>/submission/).
# Override with -DPICO_TFLMICRO_PATH=/path/to/pico-tflmicro
set(PICO_TFLMICRO_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../../../lab_2_4/pico_sw/imu/lib/pico-tflmicro"
    CACHE PATH "Path to the pico-tflmicro library (lab_2_4/pico_sw/imu/lib/pico-tflmicro)")
if(NOT EXISTS "${PICO_TFLMICRO_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "pico-tflmicro not found at '${PICO_TFLMICRO_PATH}'.\n"
        "Pass -DPICO_TFLMICRO_PATH=<ias0360-lab-excercises-2026>/lab_2_4/pico_sw/imu/lib/pico-tflmicro")
endif()
add_subdirectory(${PICO_TFLMICRO_PATH} pico-tflmicro EXCLUDE_FROM_ALL)

add_subdirectory(${COMMON_DIR}/icm20948 icm20948)
add_subdirectory(${COMMON_DIR}/sd_card_driver sd_card_driver)

# Preprocessing + TFLM runner + all exported models, shared by every target.
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
add_library(gesture_core STATIC
    src/data_preprocessor.cpp
    src/model_runner.cpp
    ${MODEL_SOURCES}
)
target_include_directories(gesture_core PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/models
)
target_link_libraries(gesture_core PUBLIC pico_stdlib pico-tflmicro)
target_compile_options(gesture_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)

function(ha2_target name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${COMMON_DIR} ${COMMON_DIR}/icm20948)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)
    # The application runs on core 1 (src/run_on_core1.hpp).
    target_link_libraries(${name} pico_multicore)
    pico_generate_pio_header(${name} ${COMMON_DIR}/ws2812.pio)
    pico_enable_stdio_usb(${name} 1)
    pico_enable_stdio_uart(${name} 0)
    set_property(TARGET ${name} APPEND_STRING PROPERTY LINK_FLAGS "-Wl,--print-memory-usage")
    pico_add_extra_outputs(${name})
endfunction()

# Step 5 on the device: evaluate all models on the own recordings stored on the SD card.
ha2_target(own_data_test src/own_data_test.cpp)
target_sources(own_data_test PRIVATE ${COMMON_DIR}/hw_config.c)
target_link_libraries(own_data_test gesture_core hardware_pio sd_custom_driver)

# Live demo: record a gesture with the IMU and classify it with all models.
ha2_target(live_inference src/live_inference.cpp)
target_link_libraries(live_inference gesture_core hardware_pio hardware_i2c icm20948)
