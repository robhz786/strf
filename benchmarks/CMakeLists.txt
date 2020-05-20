cmake_minimum_required (VERSION 2.6)

foreach(
    x
    to_char_ptr
    to_string
    input_float
    input_int
    tr_string
)

  add_executable(bench-${x}-header-only   ${x}.cpp)
  target_link_libraries(
    bench-${x}-header-only
    strf-header-only
    fmt::fmt-header-only
    benchmark )
  set_target_properties(
    bench-${x}-header-only
    PROPERTIES OUTPUT_NAME ${x}-header-only )
  install(
    TARGETS bench-${x}-header-only
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks" )

  add_executable(bench-${x}-static-lib ${x}.cpp)
  target_link_libraries(bench-${x}-static-lib strf fmt::fmt benchmark)
  set_target_properties(bench-${x}-static-lib PROPERTIES OUTPUT_NAME ${x}-static-lib)
#  install(
#    TARGETS bench-${x}
#    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)

foreach(
    x
    range
    join
    width_calculation
    utf8_to_utf16
    utf16_to_utf8
    numpunct
)

  add_executable(bench-${x}-header-only    ${x}.cpp)
  target_link_libraries(
    bench-${x}-header-only
    strf-header-only
    benchmark )
  set_target_properties(
    bench-${x}-header-only
    PROPERTIES OUTPUT_NAME
    ${x}-header-only )
  install(
    TARGETS bench-${x}-header-only
    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")

  add_executable(bench-${x}-static-lib ${x}.cpp)
  target_link_libraries(bench-${x}-static-lib strf benchmark)
  set_target_properties(bench-${x}-static-lib PROPERTIES OUTPUT_NAME ${x}-static-lib)
#  install(
#    TARGETS bench-${x}-static-lib
#    DESTINATION "${STRF_INSTALL_SUBDIR}/$<CONFIG>/benchmarks")
  
endforeach(x)
