#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.5)
project (stringify CXX)
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

set(
  STRF_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CACHE FILEPATH "installation path" )
set(
  STRF_INSTALL_SUBDIR "${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}"
  CACHE STRING
  "subdirectory of STRF_INSTALL_PREFIX where the binaries and .cmake file will be installed")

set(CMAKE_INSTALL_PREFIX ${STRF_INSTALL_PREFIX})

add_library(stringify-header-only INTERFACE)
add_library(stringify STATIC build/stringify.cpp)

target_compile_definitions(
  stringify
  PUBLIC STRF_SEPARATE_COMPILATION )

target_include_directories(
  stringify-header-only
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )
target_include_directories(
  stringify
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )

set(features cxx_return_type_deduction cxx_decltype_auto)
target_compile_features(stringify PUBLIC ${features})
target_compile_features(stringify-header-only INTERFACE ${features})

set_property(
  TARGET stringify-header-only
  PROPERTY EXPORT_NAME stringify-header-only)
set_property(
  TARGET stringify
  PROPERTY EXPORT_NAME stringify)

#install(
#  TARGETS stringify-header-only stringify
#  EXPORT stringify
#  DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>")
#install(
#  EXPORT stringify
#  FILE stringify.cmake
##  NAMESPACE boost::
#  DESTINATION "${STRF_INSTALL_SUBDIR}/cmake"  )
#

#install(
#  FILES include/stringify.hpp
#  DESTINATION ./boost/ )
#install(
#  DIRECTORY include/stringify
#  DESTINATION ./boost/
#  USE_SOURCE_PERMISSIONS )
#
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
