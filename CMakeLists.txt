#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.8)
project (strf CXX CUDA)
enable_testing()

option(
  STRF_BUILD_TESTS
  "Build  unit test"
  ${STRF_BUILD_TESTS} )
option(
 STRF_BUILD_EXAMPLES
 "Test examples files"
  ${STRF_BUILD_EXAMPLES})
option(
  STRF_BUILD_BENCHMARKS
  "Build performace test"
  ${STRF_BUILD_BENCHMARKS})

#set(
#  STRF_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../.."
#  CACHE FILEPATH "installation path" )
set(
  STRF_INSTALL_SUBDIR "${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}"
  CACHE STRING
  "subdirectory of STRF_INSTALL_PREFIX where the binaries and .cmake file will be installed")

#set(CMAKE_INSTALL_PREFIX ${STRF_INSTALL_PREFIX})

add_library(strf-header-only INTERFACE)
add_library(strf STATIC src/strf.cpp)

target_compile_definitions(
  strf
  PUBLIC STRF_SEPARATE_COMPILATION )

target_include_directories(
  strf-header-only
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )
target_include_directories(
  strf
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )

set(features cxx_return_type_deduction cxx_decltype_auto)
target_compile_features(strf PUBLIC ${features})
target_compile_features(strf-header-only INTERFACE ${features})

set_property(
  TARGET strf-header-only
  PROPERTY EXPORT_NAME strf-header-only)
set_property(
  TARGET strf
  PROPERTY EXPORT_NAME strf)

install(
  TARGETS strf-header-only strf
  EXPORT strf
  DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>")
install(
  EXPORT strf
  FILE strf.cmake
  NAMESPACE strf::
  DESTINATION "${STRF_INSTALL_SUBDIR}/cmake"  )


install(
  FILES include/strf.hpp
  DESTINATION ./strf/ )
install(
  DIRECTORY include/strf
  DESTINATION ./strf/
  USE_SOURCE_PERMISSIONS )

if (STRF_BUILD_TESTS)
  add_subdirectory(test test)
endif(STRF_BUILD_TESTS)

if (STRF_BUILD_EXAMPLES)
  add_subdirectory(example example)
endif (STRF_BUILD_EXAMPLES)

if (STRF_BUILD_BENCHMARKS)
  add_subdirectory(performance performance)
  add_subdirectory(externals/fmt fmt)
endif (STRF_BUILD_BENCHMARKS)
