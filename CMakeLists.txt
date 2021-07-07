#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.11)
option(
  STRF_CUDA_SUPPORT
  "Build the static library for CUDA device-side use"
  ${STRF_CUDA_SUPPORT} )
option(
  STRF_FREESTANDING
  "Avoid as must as possible using hosted standard headers."
  ${STRF_FREESTANDING} )
option(
  STRF_WITH_CSTRING
  "Use header <cstring> even when STRF_FREESTANDING is ON."
  ${STRF_WITH_CSTRING} )
option(
  STRF_BUILD_TESTS
  "Build unit tests"
  ${STRF_BUILD_TESTS} )
option(
 STRF_BUILD_EXAMPLES
 "Test example programs"
  ${STRF_BUILD_EXAMPLES})

project (
  strf 
  VERSION 0.14.0
  LANGUAGES CXX)

if (STRF_CUDA_SUPPORT)
  enable_language(CUDA)
endif ()

if (STRF_BUILD_TESTS OR STRF_BUILD_EXAMPLES)
  enable_testing()
endif (STRF_BUILD_TESTS OR STRF_BUILD_EXAMPLES)

add_library(strf-header-only INTERFACE)
add_library(strf-static-lib  STATIC src/strf.cpp)

target_compile_features(strf-header-only INTERFACE cxx_std_11)
target_compile_features(strf-static-lib  PUBLIC    cxx_std_11)

add_library(strf::strf-static-lib  ALIAS strf-static-lib)
add_library(strf::strf-header-only ALIAS strf-header-only)

target_compile_definitions(
  strf-static-lib
  PUBLIC STRF_SEPARATE_COMPILATION )

if (STRF_FREESTANDING)
  target_compile_definitions(strf-static-lib PUBLIC STRF_FREESTANDING)
  target_compile_definitions(strf-header-only INTERFACE STRF_FREESTANDING)
endif (STRF_FREESTANDING)

if (STRF_WITH_CSTRING)
  target_compile_definitions(strf-static-lib PUBLIC STRF_WITH_CSTRING)
  target_compile_definitions(strf-header-only INTERFACE STRF_WITH_CSTRING)
endif (STRF_WITH_CSTRING)
  
target_include_directories(
  strf-header-only
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_include_directories(
  strf-static-lib
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )


set_property(
  TARGET strf-header-only
  PROPERTY EXPORT_NAME strf-header-only)
set_property(
  TARGET strf-static-lib
  PROPERTY EXPORT_NAME strf-static-lib)
set_property(
  TARGET strf-static-lib
  PROPERTY OUTPUT_NAME strf )

set(TARGETS_TO_INSTALL strf-header-only strf-static-lib)

if (${STRF_CUDA_SUPPORT})
  add_library(strf-cuda STATIC src/strf.cu)
  target_include_directories(
    strf-cuda
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_compile_features(strf-cuda PUBLIC cxx_std_11)
  set_property(
    TARGET strf-cuda
    PROPERTY EXPORT_NAME strf-cuda)
  list(APPEND TARGETS_TO_INSTALL strf-cuda)
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "strf-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

include(GNUInstallDirs)

install(
  TARGETS ${TARGETS_TO_INSTALL}
  EXPORT strf
  RUNTIME DESTINATION  "${CMAKE_INSTALL_BINDIR}"
  ARCHIVE DESTINATION  "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION  "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
  EXPORT strf
  FILE "strf-config.cmake"
  NAMESPACE strf::
  DESTINATION "lib/cmake/strf" )

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/strf-config-version.cmake"
  DESTINATION "lib/cmake/strf" )

# TODO: These should be pulled into the installation directory by some target...
install(
  FILES include/strf.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )
install(
  DIRECTORY include/strf
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

if (STRF_BUILD_TESTS)
  add_subdirectory(tests tests)
endif(STRF_BUILD_TESTS)

if (STRF_BUILD_EXAMPLES)
  add_subdirectory(examples examples)
endif (STRF_BUILD_EXAMPLES)

