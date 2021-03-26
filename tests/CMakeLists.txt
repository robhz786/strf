#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (strf_test)
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
  outbuff.cpp
  join.cpp
  miscellaneous.cpp
  numpunct.cpp
  range_writer.cpp
  reserve.cpp
  single_byte_encodings.cpp
  tr_string.cpp
  utf_to_utf.cpp
  width_calculator.cpp
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

add_executable(test-header-only main.cpp test_utils.cpp ${sources})
add_executable(test-static-lib  main.cpp test_utils.cpp ${sources})

target_link_libraries(test-header-only   strf-header-only)
target_link_libraries(test-static-lib    strf)

set_target_properties(test-header-only  PROPERTIES OUTPUT_NAME header-only)

set_target_properties(test-static-lib   PROPERTIES OUTPUT_NAME static-lib)

target_link_libraries(test-header-only  strf-header-only)
target_link_libraries(test-static-lib   strf)

add_test(NAME run-tests-header-only COMMAND  header-only)
add_test(NAME run-tests-static-lib  COMMAND  static-lib)

set(tests_on_cuda_device
   cstr_writer
#  dynamic_charset # to-do
   encode_char
   encode_fill
   facets_pack
   facets_pack_merge
   input_bool
   input_char
   input_char32
   input_facets_pack
   input_float
   input_int
   input_range
   input_string
   override_printable
   outbuff
   join
   miscellaneous
   numpunct
   range_writer
   reserve
   single_byte_encodings
   tr_string
   utf_to_utf
   width_calculator
   width_t
  )

if (${STRF_CUDA_SUPPORT})

  foreach(src ${tests_on_cuda_device})
    configure_file(${src}.cpp ${src}.cu COPYONLY)
    list(APPEND cuda_sources ${src}.cu)
  endforeach()

  add_executable(test-cuda-header-only main_cuda.cu ${cuda_sources})
  add_executable(test-cuda-static-lib  main_cuda.cu ${cuda_sources})

  target_include_directories(test-cuda-header-only PRIVATE ./)
  target_include_directories(test-cuda-static-lib  PRIVATE ./)

  target_compile_definitions(test-cuda-header-only PUBLIC "STRF_TEST_FUNC=__device__")
  target_compile_definitions(test-cuda-static-lib  PUBLIC "STRF_TEST_FUNC=__device__")

  target_compile_options(test-cuda-header-only PRIVATE --expt-relaxed-constexpr)
  target_compile_options(test-cuda-static-lib  PRIVATE --expt-relaxed-constexpr)

  set_target_properties(test-cuda-header-only PROPERTIES OUTPUT_NAME kernel-header-only)
  set_target_properties(test-cuda-static-lib  PROPERTIES OUTPUT_NAME kernel-static-lib )

  target_link_libraries(test-cuda-header-only strf-header-only)
  target_link_libraries(test-cuda-static-lib  strf-cuda )

  set_target_properties(test-cuda-header-only PROPERTIES
    CXX_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test-cuda-static-lib  PROPERTIES
    CXX_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON)

  add_test(NAME run-test-cuda-header-only COMMAND kernel-header-only)
  add_test(NAME run-test-cuda-static-lib  COMMAND kernel-static-lib )

endif (${STRF_CUDA_SUPPORT})

