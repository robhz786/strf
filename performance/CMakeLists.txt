
cmake_minimum_required (VERSION 2.6)
project (performance_tests)

add_definitions(-DFMT_HEADER_ONLY=1)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

foreach(
    x
    to_char_ptr
    to_FILE
    width_calculation
    from_utf32
    to_utf32
    utf8_to_utf16
    utf16_to_utf8
    )

# add_executable(${X}_ho    ${X}.cpp)
# target_link_libraries(${X}_ho boost_stringify_header_only fmt::fmt-header-only)
# install(TARGETS ${X}_ho   DESTINATION "./")

  add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} fmt::fmt boost_stringify)
  install(TARGETS ${x} DESTINATION "./")
  
endforeach(x)
  
