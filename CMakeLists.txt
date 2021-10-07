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
  STRF_BUILD_CH32_WIDTH_AND_GCB_PROP
  "Enable specifically test of file include/strf/detail/ch32_width_and_gcb_prop"
  ${STRF_BUILD_CH32_WIDTH_AND_GCB_PROP})
option(
 STRF_BUILD_EXAMPLES
 "Test example programs"
  ${STRF_BUILD_EXAMPLES})

set(STRF_VERSION 0.15.3)
project (
  strf
  VERSION ${STRF_VERSION}
  LANGUAGES CXX)

if (STRF_CUDA_SUPPORT)
  enable_language(CUDA)
endif ()

if (STRF_BUILD_TESTS OR STRF_BUILD_EXAMPLES)
  enable_testing()
endif (STRF_BUILD_TESTS OR STRF_BUILD_EXAMPLES)

# https://cmake.org/cmake/help/git-stage/guide/importing-exporting/index.html
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

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
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_include_directories(
  strf-static-lib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

set_property(
  TARGET strf-header-only
  PROPERTY EXPORT_NAME strf-header-only)

set_target_properties(
  strf-static-lib
  PROPERTIES
  OUTPUT_NAME strf
  EXPORT_NAME strf-static-lib
  VERSION ${STRF_VERSION})

set(TARGETS_TO_INSTALL strf-header-only strf-static-lib)

if (${STRF_CUDA_SUPPORT})  
  add_library(strf-cuda STATIC src/strf.cu)
  target_include_directories(
    strf-cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> )
  #target_compile_features(strf-cuda PUBLIC cxx_std_11)
  set_target_properties(
    strf-cuda
    PROPERTIES
    VERSION ${STRF_VERSION}
    CUDA_SEPARABLE_COMPILATION ON
    EXPORT_NAME strf-cuda )
  target_compile_definitions(
    strf-cuda
    PUBLIC STRF_CUDA_SEPARATE_COMPILATION )
  list(APPEND TARGETS_TO_INSTALL strf-cuda)
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/strf-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strf
)

write_basic_package_version_file(
  "strf-config-version.cmake"
  VERSION ${STRF_VERSION}
  COMPATIBILITY SameMinorVersion)

install(
  TARGETS ${TARGETS_TO_INSTALL}
  EXPORT strf-targets
  ARCHIVE  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
  EXPORT strf-targets
  NAMESPACE strf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strf )

install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/strf-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/strf-config-version.cmake"
  DESTINATION "lib/cmake/strf" )

install(
  FILES include/strf.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )
install(
  DIRECTORY include/strf
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

if (STRF_BUILD_TEST_CH32_WIDTH_AND_GCB_PROP)

  set(icu_dir ${CMAKE_CURRENT_SOURCE_DIR}/externals/icu)
  set(icu_lib_dir ${icu_dir}/lib)
  set(icu_include_dir ${icu_dir}/include)

  add_library(icudata SHARED IMPORTED)
  add_library(icuuc   SHARED IMPORTED)

  set_property(TARGET icudata PROPERTY IMPORTED_LOCATION ${icu_lib_dir}/libicudata.so)
  set_property(TARGET icuuc   PROPERTY IMPORTED_LOCATION ${icu_lib_dir}/libicuuc.so)

  target_include_directories(icudata INTERFACE ${icu_include_dir})
  target_include_directories(icuuc   INTERFACE ${icu_include_dir})

endif (STRF_BUILD_TEST_CH32_WIDTH_AND_GCB_PROP)

if (STRF_BUILD_TESTS OR STRF_BUILD_CH32_WIDTH_AND_GCB_PROP)
  add_subdirectory(tests tests)
endif(STRF_BUILD_TESTS OR STRF_BUILD_CH32_WIDTH_AND_GCB_PROP)

if (STRF_BUILD_EXAMPLES)
  add_subdirectory(examples examples)
endif (STRF_BUILD_EXAMPLES)


