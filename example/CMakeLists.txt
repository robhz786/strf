cmake_minimum_required (VERSION 3.0)
enable_testing()

project (example)

remove_definitions(-DNDEBUG)

foreach(x
    assembly_string
    custom_decoder
    custom_decoding_error_handling
    custom_encoding_error_handling
    decoding_facets
    facets
    ftuple_as_input_arg
    joins
    syntax_samples
    trivial_sample)

  Add_executable(${x} v0/${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  add_test(NAME run_${x} COMMAND ${x})  

endforeach(x)

