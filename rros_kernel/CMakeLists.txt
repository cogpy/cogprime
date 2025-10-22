cmake_minimum_required(VERSION 3.10)
project(RROS_CognitiveKernel VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/episodes)
include_directories(${CMAKE_SOURCE_DIR}/modules)
include_directories(${CMAKE_SOURCE_DIR}/memory)
include_directories(${CMAKE_SOURCE_DIR}/attention)
include_directories(${CMAKE_SOURCE_DIR}/interfaces)

# Core library
file(GLOB_RECURSE CORE_SOURCES "core/*.cpp")
file(GLOB_RECURSE EPISODE_SOURCES "episodes/*.cpp")
file(GLOB_RECURSE MODULE_SOURCES "modules/*.cpp")
file(GLOB_RECURSE MEMORY_SOURCES "memory/*.cpp")
file(GLOB_RECURSE ATTENTION_SOURCES "attention/*.cpp")

add_library(rros_kernel STATIC
    ${CORE_SOURCES}
    ${EPISODE_SOURCES}
    ${MODULE_SOURCES}
    ${MEMORY_SOURCES}
    ${ATTENTION_SOURCES}
)

# Main executable
add_executable(rros_main interfaces/main.cpp)
target_link_libraries(rros_main rros_kernel)

# Tests
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} rros_kernel)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Installation
install(TARGETS rros_kernel rros_main
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)