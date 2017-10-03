cmake_minimum_required (VERSION 3.0)
enable_testing()

project (example)

add_executable(facets               v0/facets.cpp)
add_executable(ftuple_as_input_arg  v0/ftuple_as_input_arg.cpp)
add_executable(joins                v0/joins.cpp)
add_executable(trivial_formatting   v0/trivial_formatting.cpp)
add_executable(trivial_sample       v0/trivial_sample.cpp)

target_link_libraries(facets               boost_stringify)
target_link_libraries(ftuple_as_input_arg  boost_stringify)
target_link_libraries(joins                boost_stringify)
target_link_libraries(trivial_formatting   boost_stringify)
target_link_libraries(trivial_sample       boost_stringify)

add_test(NAME run_facets               COMMAND facets)
add_test(NAME run_ftuple_as_input_arg  COMMAND ftuple_as_input_arg)
add_test(NAME run_joins                COMMAND joins)
add_test(NAME run_trivial_formatting   COMMAND trivial_formatting)
add_test(NAME run_trivial_sample       COMMAND trivial_sample)

