cmake_minimum_required(VERSION 3.7)
project(assignment2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/c_runtime_api.cc
        src/c_runtime_api.h
        src/cpu_device_api.cc
        src/cpu_device_api.h
        src/cuda_device_api.cc
        src/cuda_device_api.h
        src/device_api.h
        src/dlarray.h
        src/runtime_base.h)

add_executable(assignment2 ${SOURCE_FILES})