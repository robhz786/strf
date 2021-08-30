#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)

if (STRF_BUILD_TEST_CH32_WIDTH_AND_GCB_PROP)

  add_executable(test-ch32_width_and_gcb_prop ch32_width_and_gcb_prop.cpp)
  target_link_libraries(test-ch32_width_and_gcb_prop strf-header-only icudata icuuc)
  set_target_properties(test-ch32_width_and_gcb_prop PROPERTIES OUTPUT_NAME ch32_width_and_gcb_prop)
  add_test(
    NAME ch32_width_and_gcb_prop COMMAND
    ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${icu_lib_dir}
    ${CMAKE_CURRENT_BINARY_DIR}/ch32_width_and_gcb_prop )

endif ()

if (STRF_BUILD_TESTS)

  set(sources_freestanding
    cstr_writer.cpp
    dynamic_charset.cpp
    encode_char.cpp
    encode_fill.cpp
    facets_pack.cpp
    facets_pack_merge.cpp
    input_bool.cpp
    input_char.cpp
    input_char32.cpp
    input_facets_pack.cpp
    input_float.cpp
    input_int.cpp
    input_range.cpp
    input_string.cpp
    override_printable.cpp
    destination.cpp
    join.cpp
    miscellaneous.cpp
    numpunct.cpp
    range_writer.cpp
    reserve.cpp
    single_byte_charsets.cpp
    tr_string.cpp
    utf8_to_utf16.cpp
    utf8_to_utf32.cpp
    utf8.cpp
    utf16_to_utf32.cpp
    utf16_to_utf8.cpp
    utf16.cpp
    utf32_to_utf8.cpp
    utf32_to_utf16.cpp
    utf32.cpp
    width_calculator.cpp
    std_width_calculator.cpp
    width_t.cpp
  )

  set(sources_hosted
    locale.cpp
    cfile_writer.cpp
    streambuf_writer.cpp
    string_writer.cpp )

  if (STRF_FREESTANDING)
    set(sources ${sources_freestanding})
  else ()
    set(sources ${sources_freestanding} ${sources_hosted})
  endif ()

  add_executable(test-header-only main.cpp ${sources})
  add_executable(test-static-lib  main.cpp ${sources})

  target_link_libraries(test-header-only   strf-header-only)
  target_link_libraries(test-static-lib    strf-static-lib)

  set_target_properties(test-header-only  PROPERTIES OUTPUT_NAME header-only)

  set_target_properties(test-static-lib   PROPERTIES OUTPUT_NAME static-lib)

  target_link_libraries(test-header-only  strf-header-only)
  target_link_libraries(test-static-lib   strf-static-lib)

  add_test(NAME run-tests-header-only COMMAND  header-only)
  add_test(NAME run-tests-static-lib  COMMAND  static-lib)

  if (${STRF_CUDA_SUPPORT})

    add_library(cuda-main-header-only STATIC main_cuda.cu)
    add_library(cuda-main-static-lib STATIC main_cuda.cu)

    target_include_directories(cuda-main-header-only PRIVATE ./)
    target_include_directories(cuda-main-static-lib  PRIVATE ./)

    target_link_libraries(cuda-main-header-only strf-header-only)
    target_link_libraries(cuda-main-static-lib strf-cuda)

    set_target_properties(cuda-main-header-only PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

    set_target_properties(cuda-main-static-lib  PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

  endif (${STRF_CUDA_SUPPORT})

  function(define_strf_cuda_test basename)

    configure_file(${basename}.cpp ${basename}.cu COPYONLY)

    add_executable(cuda-${basename}-header-only ${basename}.cu)
    add_executable(cuda-${basename}-static-lib  ${basename}.cu)

    target_include_directories(cuda-${basename}-header-only PRIVATE ./)
    target_include_directories(cuda-${basename}-static-lib  PRIVATE ./)

    target_compile_definitions(cuda-${basename}-header-only PUBLIC
      "STRF_TEST_FUNC=__device__" )
    target_compile_definitions(cuda-${basename}-static-lib  PUBLIC
      "STRF_TEST_FUNC=__device__" )

    set_target_properties(
      cuda-${basename}-header-only
      PROPERTIES OUTPUT_NAME cuda-${basename}-header-only)
    set_target_properties(
      cuda-${basename}-static-lib
      PROPERTIES OUTPUT_NAME cuda-${basename}-static-lib )

    target_link_libraries(
      cuda-${basename}-header-only
      cuda-main-header-only
      strf-header-only )
    target_link_libraries(
      cuda-${basename}-static-lib
      cuda-main-static-lib
      strf-cuda )

    set_target_properties(cuda-${basename}-header-only PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(cuda-${basename}-static-lib  PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

    add_test(
      NAME run-cuda-${basename}-header-only
      COMMAND cuda-${basename}-header-only)
    add_test(
      NAME run-cuda-${basename}-static-lib
      COMMAND cuda-${basename}-static-lib )

  endfunction()

  if (${STRF_CUDA_SUPPORT})

    # define_strf_cuda_test(dynamic_charset) # to-do
    define_strf_cuda_test(cstr_writer)
    define_strf_cuda_test(encode_char)
    define_strf_cuda_test(encode_fill)
    define_strf_cuda_test(facets_pack)
    define_strf_cuda_test(facets_pack_merge)
    define_strf_cuda_test(input_bool)
    define_strf_cuda_test(input_char)
    define_strf_cuda_test(input_char32)
    define_strf_cuda_test(input_facets_pack)
    define_strf_cuda_test(input_float)
    define_strf_cuda_test(input_int)
    define_strf_cuda_test(input_range)
    define_strf_cuda_test(input_string)
    define_strf_cuda_test(override_printable)
    define_strf_cuda_test(destination)
    define_strf_cuda_test(join)
    define_strf_cuda_test(miscellaneous)
    define_strf_cuda_test(numpunct)
    define_strf_cuda_test(range_writer)
    define_strf_cuda_test(reserve)
    define_strf_cuda_test(single_byte_charsets)
    define_strf_cuda_test(tr_string)
    define_strf_cuda_test(utf8_to_utf16)
    define_strf_cuda_test(utf8_to_utf32)
    define_strf_cuda_test(utf8)
    define_strf_cuda_test(utf16_to_utf32)
    define_strf_cuda_test(utf16_to_utf8)
    define_strf_cuda_test(utf16)
    define_strf_cuda_test(utf32_to_utf8)
    define_strf_cuda_test(utf32_to_utf16)
    define_strf_cuda_test(utf32)
    define_strf_cuda_test(width_calculator)
    define_strf_cuda_test(std_width_calculator)
    define_strf_cuda_test(width_t)

  endif (${STRF_CUDA_SUPPORT})
endif (STRF_BUILD_TESTS)
