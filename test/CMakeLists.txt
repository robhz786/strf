cmake_minimum_required (VERSION 3.0)
project (unit_tests)

include_directories("../../../")

add_executable(input_string_test v0/input_string_test.cpp)
add_executable(input_int_test v0/input_int_test.cpp)
add_executable(input_char_test v0/input_char_test.cpp)
add_executable(input_ftuple_test v0/input_ftuple_test.cpp)
add_executable(join_test v0/join_test.cpp)
add_executable(output_types_test v0/output_types_test.cpp)
add_executable(ftuple_merge_test v0/ftuple_merge_test.cpp)

add_test(run_input_string_test input_string_test)
add_test(run_input_int_test input_int_test)
add_test(run_input_char_test input_char_test)
add_test(run_input_ftuple_test input_ftuple_test)
add_test(run_join_test join_test) 
add_test(run_output_types_test output_types_test)
add_test(run_ftuple_merge_test ftuple_merge_test)


