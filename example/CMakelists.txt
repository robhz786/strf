cmake_minimum_required (VERSION 3.0)
enable_testing()

project (example)


include_directories("../../../")

add_executable(facets v0/facets.cpp)
add_executable(ftuple_as_input_arg v0/ftuple_as_input_arg.cpp)
add_executable(joins v0/joins.cpp)
add_executable(trivial_formatting v0/trivial_formatting.cpp)
add_executable(trivial_sample v0/trivial_sample.cpp)

add_test(NAME run_facets COMMAND facets)
add_test(NAME run_ftuple_as_input_arg COMMAND ftuple_as_input_arg)
add_test(NAME run_joins COMMAND joins)
add_test(NAME run_trivial_formatting COMMAND trivial_formatting)
add_test(NAME run_trivial_sample COMMAND trivial_sample)

