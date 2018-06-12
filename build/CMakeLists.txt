cmake_minimum_required (VERSION 3.0)
project (build)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release" AND NOT MSVC)
  add_compile_options("-O3")
endif(${CMAKE_BUILD_TYPE} STREQUAL "Release" AND NOT MSVC)

set(stringify_hpp                 ../../../boost/stringify.hpp)
file(GLOB headers_v0              ../../../boost/stringify/v0/*.hpp)
file(GLOB headers_v0_facets       ../../../boost/stringify/v0/facets/*.hpp)
file(GLOB headers_v0_input_types  ../../../boost/stringify/v0/input_types/*.hpp)
file(GLOB headers_v0_output_types ../../../boost/stringify/v0/output_types/*.hpp)
file(GLOB headers_v0_detail       ../../../boost/stringify/v0/detail/*.hpp)

add_library(
    boost_stringify
    STATIC
    stringify.cpp

    ${stringify_hpp}
    ${headers_v0}
    ${headers_v0_facets}
    ${headers_v0_input_types}
    ${headers_v0_output_types}
    ${headers_v0_detail}
)

source_group("Header Files"                              FILES  ${stringify_hpp})
source_group("Header Files\\stringify\\v0"               FILES  ${headers_v0})
source_group("Header Files\\stringify\\v0\\facets"       FILES  ${headers_v0_facets})
source_group("Header Files\\stringify\\v0\\input_types"  FILES  ${headers_v0_input_types})
source_group("Header Files\\stringify\\v0\\output_types" FILES  ${headers_v0_output_types})
source_group("Header Files\\stringify\\v0\\detail"       FILES  ${headers_v0_detail})

target_compile_definitions(boost_stringify INTERFACE BOOST_STRINGIFY_NOT_HEADER_ONLY)
target_include_directories(boost_stringify PUBLIC ../../../)

add_library(boost_stringify_header_only INTERFACE)
target_include_directories(boost_stringify_header_only INTERFACE ../../../)

target_compile_features(boost_stringify PUBLIC ${features})
target_compile_features(boost_stringify_header_only INTERFACE ${features})