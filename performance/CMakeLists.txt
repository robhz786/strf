
cmake_minimum_required (VERSION 2.6)
project (strf_performance)

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
    strf-header-only
    fmt::fmt-header-only
    benchmark )
  set_target_properties(
    performance-${x}-header-only
    PROPERTIES OUTPUT_NAME
    ${x}-header-only )
  install(
    TARGETS performance-${x}-header-only
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks" )

  add_executable(performance-${x} ${x}.cpp)
  target_link_libraries(performance-${x} strf fmt::fmt benchmark)
  set_target_properties(performance-${x} PROPERTIES OUTPUT_NAME ${x})
  install(
    TARGETS performance-${x}
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)



foreach(
    x
    range
    join
    width_calculation
    utf8_to_utf16
    utf16_to_utf8
)

  add_executable(performance-${x}-header-only    ${x}.cpp)
  target_link_libraries(
    performance-${x}-header-only
    strf-header-only
    benchmark )
  set_target_properties(
    performance-${x}-header-only
    PROPERTIES OUTPUT_NAME
    ${x}-header-only )
  install(
    TARGETS performance-${x}-header-only
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")

  add_executable(performance-${x} ${x}.cpp)
  target_link_libraries(performance-${x} strf benchmark)
  set_target_properties(performance-${x} PROPERTIES OUTPUT_NAME ${x})
  install(
    TARGETS performance-${x}
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)
  
