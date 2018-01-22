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
  input_ftuple_test
  join_test
  output_string_test
  output_char_ptr_test
  output_streambuf_test
  output_FILE_test
  ftuple_merge_test
  ftuple_test
  reserve_test
  throw_on_error_code_test
  utf8_encoding_test
  utf8_decoding_test
  utf16_encoding_test
  utf16_decoding_test
  )

  add_executable(${t}_header_only   v0/${t}.cpp)
  add_executable(${t}_static_lib    v0/${t}.cpp)

  target_link_libraries(${t}_header_only   boost_stringify_header_only)
  target_link_libraries(${t}_static_lib    boost_stringify)

  add_test(run-${t}_header_only     ${t}_header_only)
  add_test(run-${t}_static_lib      ${t}_static_lib)

endforeach(t)

