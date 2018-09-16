
cmake_minimum_required (VERSION 2.6)
project (boost_stringify_performance)

foreach(
    x
    to_char_ptr
    to_FILE
    to_string
)

  add_executable(perfomance_${x}_ho   ${x}.cpp)
  target_link_libraries(perfomance_${x}_ho boost_stringify-header-only fmt::fmt-header-only)
  set_target_properties(perfomance_${x}_ho PROPERTIES OUTPUT_NAME ${x}_ho)
  install(
    TARGETS perfomance_${x}_ho
    DESTINATION "./${BOOST_INSTALL_SUBDIR}/$<CONFIG>/performance_tests" )

  add_executable(perfomance_${x} ${x}.cpp)
  target_link_libraries(perfomance_${x} fmt::fmt boost_stringify)
  set_target_properties(perfomance_${x} PROPERTIES OUTPUT_NAME ${x})
  install(
    TARGETS perfomance_${x}
    DESTINATION "./${BOOST_INSTALL_SUBDIR}/$<CONFIG>/performance_tests")
  
endforeach(x)



foreach(
    x
    width_calculation
    from_utf32
    to_utf32
    utf8_to_utf16
    utf16_to_utf8
)

  add_executable(perfomance_${x}_ho    ${x}.cpp)
  target_link_libraries(perfomance_${x}_ho boost_stringify-header-only)
  set_target_properties(perfomance_${x}_ho PROPERTIES OUTPUT_NAME ${x}_ho)
  install(
    TARGETS perfomance_${x}_ho
    DESTINATION "./${BOOST_INSTALL_SUBDIR}/$<CONFIG>/performance_tests")

  add_executable(perfomance_${x} ${x}.cpp)
  target_link_libraries(perfomance_${x} boost_stringify)
  set_target_properties(perfomance_${x} PROPERTIES OUTPUT_NAME ${x})
  install(
    TARGETS perfomance_${x}
    DESTINATION "./${BOOST_INSTALL_SUBDIR}/$<CONFIG>/performance_tests")
  
endforeach(x)
  
