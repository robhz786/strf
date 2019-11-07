
cmake_minimum_required (VERSION 2.6)
project (stringify_performance)

foreach(
    x
    to_char_ptr
    to_string
    input_float
    input_int
    tr_string
)

  add_executable(performance-${x}-header-only   ${x}.cpp)
  target_link_libraries(
    performance-${x}-header-only
    stringify-header-only
    fmt::fmt-header-only )
  set_target_properties(
    performance-${x}-header-only
    PROPERTIES OUTPUT_NAME
    ${x}-header-only )
#  install(
#    TARGETS performance-${x}-header-only
#    DESTINATION "${BOOST_INSTALL_SUBDIR}/$<CONFIG>/benchmarks" )

  add_executable(performance-${x} ${x}.cpp)
  target_link_libraries(performance-${x} fmt::fmt stringify)
  set_target_properties(performance-${x} PROPERTIES OUTPUT_NAME ${x})
#  install(
#    TARGETS performance-${x}
#    DESTINATION "${BOOST_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)



foreach(
    x
    range
    join
    width_calculation
#    from_utf32
#    to_utf32
    utf8_to_utf16
    utf16_to_utf8
)

  add_executable(performance-${x}-header-only    ${x}.cpp)
  target_link_libraries(
    performance-${x}-header-only
    stringify-header-only )
  set_target_properties(
    performance-${x}-header-only
    PROPERTIES OUTPUT_NAME
    ${x}-header-only )
  install(
    TARGETS performance-${x}-header-only
    DESTINATION "${BOOST_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")

  add_executable(performance-${x} ${x}.cpp)
  target_link_libraries(performance-${x} stringify)
  set_target_properties(performance-${x} PROPERTIES OUTPUT_NAME ${x})
  install(
    TARGETS performance-${x}
    DESTINATION "${BOOST_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)
  
