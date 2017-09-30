cmake_minimum_required (VERSION 3.0)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMAKE_BUILD_TYPE set to Relase")
endif (NOT CMAKE_BUILD_TYPE)

enable_testing()

project (BoostStringify)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/out/${CMAKE_BUILD_TYPE})

add_subdirectory(test test)
add_subdirectory(example example)
add_subdirectory(performance performance)
add_subdirectory(performance/fmt-4.0.0 fmt)
add_subdirectory(build build)

