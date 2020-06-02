#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (strf_test)

set(tests
  tr_string
  dynamic_charset
  encode_char
  encode_fill
  input_bool
  input_char
  input_float
  input_int
  input_string
  input_facets_pack
  input_range
  numpunct
  join
  facets_pack_merge
  facets_pack
  reserve
  width_t
  width_calculator
  utf_to_utf
  single_byte_encodings
  cstr_writer )

if (NOT ${STRF_FREESTANDING})
  list(APPEND tests
    cfile_writer
    streambuf_writer
    string_writer )
endif()

foreach(t IN LISTS tests)

  add_executable(test-${t}-header-only   ${t}.cpp)
  add_executable(test-${t}-static-lib    ${t}.cpp)

  target_link_libraries(test-${t}-header-only   strf-header-only)
  target_link_libraries(test-${t}-static-lib    strf)

  set_target_properties(test-${t}-header-only  PROPERTIES OUTPUT_NAME ${t}-header-only)
  set_target_properties(test-${t}-static-lib   PROPERTIES OUTPUT_NAME ${t}-static-lib)

  add_test(NAME run-test-${t}-header-only COMMAND  ${t}-header-only)
  add_test(NAME run-test-${t}-static-lib  COMMAND  ${t}-static-lib)

endforeach(t)

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
