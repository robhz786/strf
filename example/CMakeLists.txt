#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#


cmake_minimum_required (VERSION 3.0)
enable_testing()

project (boost_stringify_example)

remove_definitions(-DNDEBUG)

foreach(x
    overview
    syntax_samples
    trivial_sample
    assembly_string
    # encoding_conversion
    format_functions
    custom_width_calculation
    facets
    # facets_pack_as_input
    numpunct_facets
    width_calculation_in_encoding_conversion
    joins
    # range_input
    # extend_input_ipv4
    # extend_input_ipv6
    # extend_input_base64
    # extend_input_template
    ) 

  Add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  add_test(NAME run_${x} COMMAND ${x})  

endforeach(x)


find_package(Qt5Core QUIET)

if (Qt5Core_FOUND)

  add_executable(extend_ouput_toQString extend_output_toQString.cpp)
  target_link_libraries(extend_ouput_toQString Qt5::Core boost_stringify)
  add_test(NAME run_extend_ouput_toQString COMMAND extend_ouput_toQString)
  
endif(Qt5Core_FOUND)
