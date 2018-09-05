#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (unit_tests)

include_directories("../../../")

foreach(
  t
  assembly_string_test
  input_string_test
  input_int_test
  input_char_test
  input_char32_test
  input_facets_pack_test
  input_range_test
  numpunct_test
  join_test
  output_string_test
  output_char_ptr_test
  output_streambuf_test
  output_FILE_test
  facets_pack_merge_test
  facets_pack_test
  reserve_test
  width_calculator_test
  transcoding_test
  )

  add_executable(${t}-header-only   v0/${t}.cpp)
  add_executable(${t}-static-lib    v0/${t}.cpp)

  target_link_libraries(${t}-header-only   boost_stringify-header-only)
  target_link_libraries(${t}-static-lib    boost_stringify)

  add_test(run-${t}-header-only     ${t}-header-only)
  add_test(run-${t}-static-lib      ${t}-static-lib)

endforeach(t)

