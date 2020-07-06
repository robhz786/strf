#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (strf_test)
set(sources_freestanding
  tr_string.cpp
  dynamic_charset.cpp
  encode_char.cpp
  encode_fill.cpp
  input_bool.cpp
  input_char.cpp
  input_float.cpp
  input_int.cpp
  input_string.cpp
  input_facets_pack.cpp
  input_range.cpp
  numpunct.cpp
  join.cpp
  facets_pack_merge.cpp
  facets_pack.cpp
  reserve.cpp
  width_t.cpp
  width_calculator.cpp
  utf_to_utf.cpp
  single_byte_encodings.cpp
  cstr_writer.cpp  )

set(sources_hosted
  locale.cpp
  cfile_writer.cpp
  streambuf_writer.cpp
  string_writer.cpp )

set(sources
  ${sources_freestanding}
  ${sources_hosted} )

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


if (${STRF_CUDA_SUPPORT})
  foreach(
    t
    cuda_inclusion
    write_within_kernel )

    add_executable(test-${t}-header-only   ${t}.cu)
    add_executable(test-${t}-static-lib    ${t}.cu)

    set_target_properties(test-${t}-header-only PROPERTIES CXX_STANDARD 14)
    set_target_properties(test-${t}-static-lib  PROPERTIES CXX_STANDARD 14)

    set_target_properties(test-${t}-header-only PROPERTIES OUTPUT_NAME ${t}-header-only)
    set_target_properties(test-${t}-static-lib  PROPERTIES OUTPUT_NAME ${t}-static-lib)

    # Note: Users of strf in CUDA device-side code may want to add this
    # assembly option to their build options as well, since by default, ptxas
    # complains on kernels with dynamic memory allocation.
    target_compile_options(test-${t}-header-only PRIVATE --ptxas-options --suppress-stack-size-warning)
    target_compile_options(test-${t}-static-lib PRIVATE --ptxas-options --suppress-stack-size-warning)

    target_link_libraries(test-${t}-header-only   strf-header-only)
    target_link_libraries(test-${t}-static-lib    strf-cuda)

    add_test(NAME run-test-${t}-header-only COMMAND ${t}-header-only)
    add_test(NAME run-test-${t}-static-lib  COMMAND ${t}-static-lib)
  endforeach(t)
endif()
