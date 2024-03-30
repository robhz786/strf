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
  STRF_BUILD_FUZZY_TESTS
  "Build fuzzy tests"
  ${STRF_BUILD_FUZZY_TESTS} )
option(
  STRF_BUILD_CH32_WIDTH_AND_GCB_PROP
  "Enable specifically test of file include/strf/detail/ch32_width_and_gcb_prop"
  ${STRF_BUILD_CH32_WIDTH_AND_GCB_PROP})
option(
 STRF_BUILD_EXAMPLES
 "Test example programs"
  ${STRF_BUILD_EXAMPLES})

set(STRF_VERSION 0.16.0)
project (
  Strf
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

add_library(StrfHeaderOnly INTERFACE)
add_library(StrfStatic  STATIC src/strf.cpp)

target_compile_features(StrfHeaderOnly INTERFACE cxx_std_14)
target_compile_features(StrfStatic  PUBLIC    cxx_std_14)

add_library(Strf::StrfStatic  ALIAS StrfStatic)
add_library(Strf::StrfHeaderOnly ALIAS StrfHeaderOnly)

target_compile_definitions(
  StrfStatic
  PUBLIC STRF_SEPARATE_COMPILATION )

if (STRF_FREESTANDING)
  target_compile_definitions(StrfStatic PUBLIC STRF_FREESTANDING)
  target_compile_definitions(StrfHeaderOnly INTERFACE STRF_FREESTANDING)
endif (STRF_FREESTANDING)

if (STRF_WITH_CSTRING)
  target_compile_definitions(StrfStatic PUBLIC STRF_WITH_CSTRING)
  target_compile_definitions(StrfHeaderOnly INTERFACE STRF_WITH_CSTRING)
endif (STRF_WITH_CSTRING)

target_include_directories(
  StrfHeaderOnly
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_include_directories(
  StrfStatic
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

set_property(
  TARGET StrfHeaderOnly
  PROPERTY EXPORT_NAME StrfHeaderOnly)

set_target_properties(
  StrfStatic
  PROPERTIES
  OUTPUT_NAME strf
  EXPORT_NAME StrfStatic
  VERSION ${STRF_VERSION})

set(TARGETS_TO_INSTALL StrfHeaderOnly StrfStatic)

if (${STRF_CUDA_SUPPORT})  
  add_library(StrfCuda STATIC src/strf.cu)
  add_library(Strf::StrfCuda ALIAS StrfCuda)
  target_include_directories(
    StrfCuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> )
  #target_compile_features(StrfCuda PUBLIC cxx_std_14)
  set_target_properties(
    StrfCuda
    PROPERTIES
    VERSION ${STRF_VERSION}
    CUDA_SEPARABLE_COMPILATION ON
    EXPORT_NAME StrfCuda )
  target_compile_definitions(
    StrfCuda
    PUBLIC STRF_CUDA_SEPARATE_COMPILATION )
  list(APPEND TARGETS_TO_INSTALL StrfCuda)
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/StrfConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strf
)

write_basic_package_version_file(
  "StrfConfigVersion.cmake"
  VERSION ${STRF_VERSION}
  COMPATIBILITY SameMinorVersion)

install(
  TARGETS ${TARGETS_TO_INSTALL}
  EXPORT StrfTargets
  ARCHIVE  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
  EXPORT StrfTargets
  NAMESPACE Strf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/strf )

install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/StrfConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/StrfConfigVersion.cmake"
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


