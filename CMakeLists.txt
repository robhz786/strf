#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.5)
project (boost_stringify CXX)
enable_testing()

option(BOOST_STRINGIFY_TESTS "Build Boost.Stringify unit test" ${BOOST_TESTS})
option(BOOST_STRINGIFY_EXAMPLES "Test Boost.Stringify examples files" ${BOOST_EXAMPLES})
option(BOOST_STRINGIFY_PERFORMANCE "Build Boost.Stringify performace test" ${BOOST_PERFORMANCE})
set(
  BOOST_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CACHE FILEPATH "installation path" )
set(
  BOOST_INSTALL_SUBDIR "${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}"
  CACHE STRING
  "subdirectory of BOOST_INSTALL_PREFIX where the binaries and .cmake file will be installed")

set(CMAKE_INSTALL_PREFIX ${BOOST_INSTALL_PREFIX})
#set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version")

add_library(boost_stringify-header-only INTERFACE)
add_library(boost_stringify STATIC build/stringify.cpp)

target_compile_definitions(
  boost_stringify
  PUBLIC BOOST_STRINGIFY_NOT_HEADER_ONLY )

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
  PROPERTY EXPORT_NAME stringify-static)

install(
  TARGETS boost_stringify-header-only boost_stringify
  EXPORT boost_stringify
  DESTINATION "./${BOOST_INSTALL_SUBDIR}/$<CONFIG>")
install(
  EXPORT boost_stringify
  FILE boost_stringify.cmake
  NAMESPACE boost::
  DESTINATION "./${BOOST_INSTALL_SUBDIR}/cmake"  )

install(
  FILES include/boost/stringify.hpp
  DESTINATION ./boost/ )
install(
  DIRECTORY include/boost/stringify
  DESTINATION ./boost/
  USE_SOURCE_PERMISSIONS )

if (BOOST_STRINGIFY_TESTS)
  add_subdirectory(test test)
endif(BOOST_STRINGIFY_TESTS)

if (BOOST_STRINGIFY_EXAMPLES)
  add_subdirectory(example example)
endif (BOOST_STRINGIFY_EXAMPLES)

if (BOOST_STRINGIFY_PERFORMANCE)
  add_subdirectory(performance performance)
  add_subdirectory(performance/fmt-5.2.0 fmt)
endif (BOOST_STRINGIFY_PERFORMANCE)
