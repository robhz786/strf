
cmake_minimum_required (VERSION 2.6)
project (performance_tests)

add_definitions(-DFMT_HEADER_ONLY=1)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(to_char_ptr    to_char_ptr.cpp)
target_link_libraries(to_char_ptr boost_stringify fmt::fmt)
install(TARGETS to_char_ptr   DESTINATION "./")

add_executable(to_char_ptr_ho    to_char_ptr.cpp)
target_link_libraries(to_char_ptr_ho boost_stringify_header_only fmt::fmt-header-only)
install(TARGETS to_char_ptr_ho   DESTINATION "./")


if (NOT MSVC)

  add_executable(utf32_to_utf8  utf32_to_utf8.cpp)
  target_link_libraries(utf32_to_utf8 boost_stringify)
  install(TARGETS utf32_to_utf8 DESTINATION "./")

  add_executable(utf8_to_utf32  utf8_to_utf32.cpp)
  target_link_libraries(utf8_to_utf32 boost_stringify)
  install(TARGETS utf8_to_utf32 DESTINATION "./")

  add_executable(utf8_to_utf16  utf8_to_utf16.cpp)
  target_link_libraries(utf8_to_utf16 boost_stringify)
  install(TARGETS utf8_to_utf16 DESTINATION "./")

  add_executable(utf16_to_utf8  utf16_to_utf8.cpp)
  target_link_libraries(utf16_to_utf8 boost_stringify)
  install(TARGETS utf16_to_utf8 DESTINATION "./")

  
endif (NOT MSVC)
