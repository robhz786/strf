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
    facets_pack_as_input
    joins
    range_input
    extend_input_ipv4
    extend_input_point2D
    extend_input_ipv6
    extend_input_base64
    extend_input_std_complex
    override_bool
    std_complex_printer
    extend_input_template
    )

  Add_executable(example-${x} ${x}.cpp)
  target_link_libraries(example-${x} strf)
  add_test(NAME run_example-${x} COMMAND ${x})
  set_target_properties(example-${x} PROPERTIES OUTPUT_NAME ${x})
endforeach(x)

if (${STRF_CUDA_SUPPORT})
  add_executable(example-cuda cuda.cu)
  target_link_libraries(example-cuda strf-header-only)
  add_test(NAME run_example-cuda COMMAND cuda)
  set_target_properties(example-cuda PROPERTIES OUTPUT_NAME cuda)
endif (${STRF_CUDA_SUPPORT})

find_package(Qt5Core QUIET)

if (Qt5Core_FOUND)

  add_executable(example-toQString toQString.cpp)
  target_link_libraries(example-toQString Qt5::Core strf)
  add_test(NAME run_example-toQString COMMAND toQString)

  add_executable(example-appendQString appendQString.cpp)
  target_link_libraries(example-appendQString Qt5::Core strf)
  add_test(NAME run_example-appendQString COMMAND appendQString)

  set_target_properties(example-toQString     PROPERTIES OUTPUT_NAME toQString)
  set_target_properties(example-appendQString PROPERTIES OUTPUT_NAME appendQString)

endif(Qt5Core_FOUND)
