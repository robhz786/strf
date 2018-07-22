cmake_minimum_required (VERSION 3.0)
enable_testing()

project (example)

remove_definitions(-DNDEBUG)

foreach(x
    assembly_string
    encoding_conversion
    custom_width_calculation
    facets
    format_functions
    facets_pack_as_input
    numpunct_facets
    width_calculation_in_encoding_conversion
    joins
    range_input
    syntax_samples
    trivial_sample)

  Add_executable(${x} v0/${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  add_test(NAME run_${x} COMMAND ${x})  

endforeach(x)

