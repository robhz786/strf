
cmake_minimum_required (VERSION 2.6)
project (performance_tests)

foreach(
    x
    to_char_ptr
    to_FILE
    to_string
)

  add_executable(${x}_ho   ${x}.cpp)
  target_link_libraries(${x}_ho boost_stringify-header-only fmt::fmt-header-only)
  install(TARGETS ${x}_ho   DESTINATION "$<CONFIG>/performance_tests")

  add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} fmt::fmt boost_stringify)
  install(TARGETS ${x} DESTINATION "$<CONFIG>/performance_tests")
  
endforeach(x)



foreach(
    x
    width_calculation
    from_utf32
    to_utf32
    utf8_to_utf16
    utf16_to_utf8
)

  add_executable(${x}_ho    ${x}.cpp)
  target_link_libraries(${x}_ho boost_stringify-header-only)
  install(TARGETS ${x}_ho   DESTINATION "$<CONFIG>/performance_tests")

  add_executable(${x} ${x}.cpp)
  target_link_libraries(${x} boost_stringify)
  install(TARGETS ${x} DESTINATION "$<CONFIG>/performance_tests")
  
endforeach(x)
  
