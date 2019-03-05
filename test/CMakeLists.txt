#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (boost_stringify_test)

include_directories("../../../")

foreach(
  t
  assembly_string
  constrained_fpe
  encode_char
  encode_fill
  input_string
  input_char
  input_int
#  input_char32
  input_facets_pack
  input_range
  numpunct
  join
  output_string
  output_char_ptr
  output_streambuf
  output_FILE
  facets_pack_merge
  facets_pack
  reserve
  width_calculator
  utf_to_utf
  single_byte_encodings
  )

  add_executable(test-${t}-header-only   ${t}.cpp)
  add_executable(test-${t}-static-lib    ${t}.cpp)

  target_link_libraries(test-${t}-header-only   boost_stringify-header-only)
  target_link_libraries(test-${t}-static-lib    boost_stringify)

  add_test(run-test-${t}-header-only   test-${t}-header-only)
  add_test(run-test-${t}-static-lib    test-${t}-static-lib)

endforeach(t)

