cmake_minimum_required(VERSION 3.15)
project(cogprime VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_TESTS "Build CogPrime C++ tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cog INTERFACE)
target_include_directories(cog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cog INTERFACE cxx_std_11)

add_library(cogprime INTERFACE)
target_link_libraries(cogprime INTERFACE cog)
add_library(cogprime::cogprime ALIAS cogprime)

if(BUILD_TESTS)
    enable_testing()

    add_executable(cogprime_core_tests tests/unit/test_core.cpp)
    target_link_libraries(cogprime_core_tests PRIVATE cogprime)
    add_test(NAME cogprime_core_tests COMMAND cogprime_core_tests)

    add_executable(cogprime_prime_tests tests/unit/test_prime.cpp)
    target_link_libraries(cogprime_prime_tests PRIVATE cogprime)
    add_test(NAME cogprime_prime_tests COMMAND cogprime_prime_tests)
endif()

install(DIRECTORY include/cog DESTINATION include)
