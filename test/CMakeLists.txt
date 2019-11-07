#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (stringify_test)

# include_directories("../include")

foreach(
  t
  tr_string
#  dispatcher_with_move_only_arg
  encode_char
  encode_fill
  input_string
  input_char
  input_int
  input_float
#  input_char32
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
  cstr_writer
  cfile_writer
  streambuf_writer
  string_writer
  )

  add_executable(test-${t}-header-only   ${t}.cpp)
  add_executable(test-${t}-static-lib    ${t}.cpp)

  target_link_libraries(test-${t}-header-only   stringify-header-only)
  target_link_libraries(test-${t}-static-lib    stringify)

  add_test(run-test-${t}-header-only   test-${t}-header-only)
  add_test(run-test-${t}-static-lib    test-${t}-static-lib)

endforeach(t)

