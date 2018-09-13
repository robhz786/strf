#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.5)
project (Boost.Stringify CXX)
enable_testing()

option(STRINGIFY_BUILD_TESTS "Build Boost.Stringify unit test" OFF)
option(STRINGIFY_BUILD_EXAMPLES "Test Boost.Stringify examples files" OFF)
option(STRINGIFY_BUILD_PERFORMANCE_TESTS "Build Boost.Stringify performace test" OFF)

message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")

add_subdirectory(build build)

if (STRINGIFY_BUILD_TESTS)
  add_subdirectory(test test)
endif(STRINGIFY_BUILD_TESTS)

if (STRINGIFY_BUILD_EXAMPLES)
  add_subdirectory(example example)
endif (STRINGIFY_BUILD_EXAMPLES)

if (STRINGIFY_BUILD_PERFORMANCE_TESTS)
  add_subdirectory(performance performance)
  add_subdirectory(performance/fmt-5.2.0 fmt)
endif (STRINGIFY_BUILD_PERFORMANCE_TESTS)

