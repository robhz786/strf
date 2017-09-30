cmake_minimum_required (VERSION 3.0)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMAKE_BUILD_TYPE set to Relase")
endif (NOT CMAKE_BUILD_TYPE)

enable_testing()

project (BoostStringify)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  add_compile_options("-O3")
endif(${CMAKE_BUILD_TYPE} STREQUAL "Release")


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/out/${CMAKE_BUILD_TYPE})

add_subdirectory(build build)
add_subdirectory(test test)
add_subdirectory(example example)

if (BOOST_STRINGIFY_BUILD_PERFORMANCE_TESTS)
add_subdirectory(performance performance)
add_subdirectory(performance/fmt-4.0.0 fmt)
endif (BOOST_STRINGIFY_BUILD_PERFORMANCE_TESTS)


