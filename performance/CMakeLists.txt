
cmake_minimum_required (VERSION 2.6)
project (performance_tests)

foreach(x to_char_ptr to_FILE)

  add_executable(${x}_ho   ${x}.cpp)
  target_link_libraries(${x}_ho boost_stringify_header_only fmt::fmt-header-only)
  install(TARGETS ${x}_ho   DESTINATION "./")

  add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} fmt::fmt boost_stringify)
  install(TARGETS ${x} DESTINATION "./")
  
endforeach(x)


foreach(x width_calculation from_utf32 to_utf32 utf8_to_utf16 utf16_to_utf8)

  add_executable(${x}_ho    ${x}.cpp)
  target_link_libraries(${x}_ho boost_stringify_header_only)
  install(TARGETS ${x}_ho   DESTINATION "./")

  add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  install(TARGETS ${x} DESTINATION "./")
  
endforeach(x)
  
