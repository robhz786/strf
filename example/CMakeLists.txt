#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
enable_testing()

project (strf_example)

remove_definitions(-DNDEBUG)

foreach(x
    overview
    readme_md
    tr_string
    encoding_conversion
    format_functions
    facets_pack_as_input
    joins
    range_input
    extend_input_ipv4
    extend_input_base64
    extend_input_template
    ) 

  Add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} strf)
  add_test(NAME run_${x} COMMAND ${x})  

endforeach(x)


find_package(Qt5Core QUIET)

if (Qt5Core_FOUND)

  add_executable(toQString toQString.cpp)
  target_link_libraries(toQString Qt5::Core strf)
  add_test(NAME run_toQString COMMAND toQString)

  add_executable(appendQString appendQString.cpp)
  target_link_libraries(appendQString Qt5::Core strf)
  add_test(NAME run_appendQString COMMAND appendQString)


  
endif(Qt5Core_FOUND)
