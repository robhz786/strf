#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.5)
project (boost_stringify CXX)
enable_testing()

option(
  BOOST_STRINGIFY_BUILD_TESTS
  "Build Boost.Stringify unit test"
  ${BOOST_BUILD_TESTS} )
option(
 BOOST_STRINGIFY_BUILD_EXAMPLES
 "Test Boost.Stringify examples files"
  ${BOOST_BUILD_EXAMPLES})
option(
  BOOST_STRINGIFY_BUILD_BENCHMARKS
  "Build Boost.Stringify performace test"
  ${BOOST_BUILD_BENCHMARKS})

set(
  BOOST_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CACHE FILEPATH "installation path" )
set(
  BOOST_INSTALL_SUBDIR "${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}"
  CACHE STRING
  "subdirectory of BOOST_INSTALL_PREFIX where the binaries and .cmake file will be installed")

set(CMAKE_INSTALL_PREFIX ${BOOST_INSTALL_PREFIX})

add_library(boost_stringify-header-only INTERFACE)
add_library(boost_stringify STATIC build/stringify.cpp)

target_compile_definitions(
  boost_stringify
  PUBLIC BOOST_STRINGIFY_SEPARATE_COMPILATION )

target_include_directories(
  boost_stringify-header-only
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )
target_include_directories(
  boost_stringify
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:.> )

set(features cxx_return_type_deduction cxx_decltype_auto)
target_compile_features(boost_stringify PUBLIC ${features})
target_compile_features(boost_stringify-header-only INTERFACE ${features})

set_property(
  TARGET boost_stringify-header-only
  PROPERTY EXPORT_NAME stringify-header-only)
set_property(
  TARGET boost_stringify
  PROPERTY EXPORT_NAME stringify)

install(
  TARGETS boost_stringify-header-only boost_stringify
  EXPORT boost_stringify
  DESTINATION "${BOOST_INSTALL_SUBDIR}/$<CONFIG>")
install(
  EXPORT boost_stringify
  FILE boost_stringify.cmake
  NAMESPACE boost::
  DESTINATION "${BOOST_INSTALL_SUBDIR}/cmake"  )

install(
  FILES include/boost/stringify.hpp
  DESTINATION ./boost/ )
install(
  DIRECTORY include/boost/stringify
  DESTINATION ./boost/
  USE_SOURCE_PERMISSIONS )

if (BOOST_STRINGIFY_BUILD_TESTS)
  add_subdirectory(test test)
endif(BOOST_STRINGIFY_BUILD_TESTS)

if (BOOST_STRINGIFY_BUILD_EXAMPLES)
  add_subdirectory(example example)
endif (BOOST_STRINGIFY_BUILD_EXAMPLES)

if (BOOST_STRINGIFY_BUILD_BENCHMARKS)
  add_subdirectory(performance performance)
  add_subdirectory(performance/fmt-5.2.0 fmt)
endif (BOOST_STRINGIFY_BUILD_BENCHMARKS)
