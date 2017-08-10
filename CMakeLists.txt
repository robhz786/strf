cmake_minimum_required (VERSION 3.0)
enable_testing()

project (BoostStringify)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/out_${CMAKE_BUILD_TYPE})

add_subdirectory(test test)
add_subdirectory(example example)
add_subdirectory(performance performance)
add_subdirectory(build build)

