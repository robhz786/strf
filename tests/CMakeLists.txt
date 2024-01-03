#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)

if (STRF_BUILD_FUZZY_TESTS)
    enable_language(C)
    add_library(Ryu STATIC
      "externals/ryu/ryu/f2s.c"
      "externals/ryu/ryu/d2s.c"
      "externals/ryu/ryu/d2fixed.c")
    target_include_directories(Ryu PUBLIC externals/ryu)
    add_executable(test_fuzzy_input_float fuzzy_input_float.cpp)
    target_link_libraries(test_fuzzy_input_float Ryu Strf::StrfHeaderOnly)
endif (STRF_BUILD_FUZZY_TESTS)

if (STRF_BUILD_TEST_CH32_WIDTH_AND_GCB_PROP)
  add_executable(test_ch32_width_and_gcb_prop ch32_width_and_gcb_prop.cpp)
  target_link_libraries(test_ch32_width_and_gcb_prop Strf::StrfHeaderOnly icudata icuuc)
  add_test(
    NAME ch32_width_and_gcb_prop COMMAND
    ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${icu_lib_dir}
    ${CMAKE_CURRENT_BINARY_DIR}/test_ch32_width_and_gcb_prop )
endif (STRF_BUILD_TEST_CH32_WIDTH_AND_GCB_PROP)

if (STRF_BUILD_TESTS)
  set(sources_freestanding
    add_and_override_printable_types.cpp
    cstr_destination.cpp
    dynamic_charset.cpp
    decode_encode_to_ptr.cpp
    decode_encode_to_dst.cpp
    unsafe_decode_encode.cpp
    transcode.cpp
    unsafe_transcode.cpp
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
    input_tr_string.cpp
    override_printable.cpp
    output_buffer.cpp
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
    string_writer.cpp
    output_buffer_iterator.cpp )

  if (STRF_FREESTANDING)
    set(sources ${sources_freestanding})
  else ()
    set(sources ${sources_freestanding} ${sources_hosted})
  endif ()

  add_executable(TestHeaderOnly main.cpp ${sources})
  add_executable(TestStatic     main.cpp ${sources})

  target_link_libraries(TestHeaderOnly   Strf::StrfHeaderOnly)
  target_link_libraries(TestStatic       Strf::StrfStatic)

  set_target_properties(TestHeaderOnly  PROPERTIES OUTPUT_NAME test_header_only)
  set_target_properties(TestStatic      PROPERTIES OUTPUT_NAME test_static_lib)

  target_link_libraries(TestHeaderOnly  Strf::StrfHeaderOnly)
  target_link_libraries(TestStatic      Strf::StrfStatic)

  add_test(NAME header_only COMMAND  TestHeaderOnly)
  add_test(NAME static_lib  COMMAND  TestStatic)

  if (${STRF_CUDA_SUPPORT})

    add_library(CudaMainHeaderOnly STATIC main_cuda.cu)
    add_library(CudaMainStatic     STATIC main_cuda.cu)

    target_include_directories(CudaMainHeaderOnly PRIVATE ./)
    target_include_directories(CudaMainStatic     PRIVATE ./)

    target_link_libraries(CudaMainHeaderOnly Strf::StrfHeaderOnly)
    target_link_libraries(CudaMainStatic     Strf::StrfCuda)

    set_target_properties(CudaMainHeaderOnly PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

    set_target_properties(CudaMainStatic     PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

  endif (${STRF_CUDA_SUPPORT})

  function(define_strf_cuda_test basename)

    configure_file(${basename}.cpp ${basename}.cu COPYONLY)

    add_executable(test_cuda_${basename}_header_only ${basename}.cu)
    add_executable(test_cuda_${basename}_static_lib  ${basename}.cu)

    target_include_directories(test_cuda_${basename}_header_only PRIVATE ./)
    target_include_directories(test_cuda_${basename}_static_lib  PRIVATE ./)

    target_compile_definitions(test_cuda_${basename}_header_only PUBLIC
      "STRF_TEST_FUNC=__device__" )
    target_compile_definitions(test_cuda_${basename}_static_lib  PUBLIC
      "STRF_TEST_FUNC=__device__" )

    set_target_properties(
      test_cuda_${basename}_header_only
      PROPERTIES OUTPUT_NAME cuda_${basename}_header_only)
    set_target_properties(
      test_cuda_${basename}_static_lib
      PROPERTIES OUTPUT_NAME cuda_${basename}_static_lib )

    target_link_libraries(
      test_cuda_${basename}_header_only
      CudaMainHeaderOnly
      Strf::StrfHeaderOnly )
    target_link_libraries(
      test_cuda_${basename}_static_lib
      CudaMainStatic
      Strf::StrfCuda )

    set_target_properties(test_cuda_${basename}_header_only PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(test_cuda_${basename}_static_lib  PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON)

    add_test(
      NAME    cuda_${basename}_header_only
      COMMAND test_cuda_${basename}_header_only)
    add_test(
      NAME    cuda_${basename}_static_lib
      COMMAND test_cuda_${basename}_static_lib )

  endfunction()

  if (${STRF_CUDA_SUPPORT})

    # define_strf_cuda_test(dynamic_charset) # to-do
    define_strf_cuda_test(transcode)
    define_strf_cuda_test(unsafe_transcode)
    define_strf_cuda_test(add_and_override_printable_types)
    define_strf_cuda_test(cstr_destination)
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
    define_strf_cuda_test(output_buffer)
    define_strf_cuda_test(join)
    define_strf_cuda_test(miscellaneous)
    define_strf_cuda_test(numpunct)
    define_strf_cuda_test(range_writer)
    define_strf_cuda_test(reserve)
    define_strf_cuda_test(single_byte_charsets)
    define_strf_cuda_test(tr_string)
    define_strf_cuda_test(input_tr_string)
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
