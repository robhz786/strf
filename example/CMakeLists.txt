cmake_minimum_required (VERSION 3.0)
enable_testing()

project (example)

remove_definitions(-DNDEBUG)

foreach(x
    syntax_samples
    trivial_sample
    assembly_string
    encoding_conversion
    format_functions
    custom_width_calculation
    facets
    facets_pack_as_input
    numpunct_facets
    width_calculation_in_encoding_conversion
    joins
    range_input
    extend_input_ipv4
    extend_input_ipv6
    extend_input_base64
    extend_input_template ) 

  Add_executable(${x} v0/${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  add_test(NAME run_${x} COMMAND ${x})  

endforeach(x)


find_package(Qt5Core QUIET)

if (Qt5Core_FOUND)

  add_executable(extend_ouput_toQString v0/extend_output_toQString.cpp)
  target_link_libraries(extend_ouput_toQString Qt5::Core boost_stringify)
  add_test(NAME run_extend_ouput_toQString COMMAND extend_ouput_toQString)
#  set_target_properties(extend_ouput_toQString PROPERTIES CXX_STANDARD 14)
  
endif(Qt5Core_FOUND)
