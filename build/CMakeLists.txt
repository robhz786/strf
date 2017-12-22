cmake_minimum_required (VERSION 3.0)
project (build)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release" AND NOT MSVC)
  add_compile_options("-O3")
endif(${CMAKE_BUILD_TYPE} STREQUAL "Release" AND NOT MSVC)

add_library(boost_stringify STATIC stringify.cpp)
target_compile_definitions(boost_stringify PUBLIC BOOST_STRINGIFY_NOT_HEADER_ONLY)
target_include_directories(boost_stringify PUBLIC ../../../)

add_library(boost_stringify_header_only INTERFACE)
target_include_directories(boost_stringify_header_only INTERFACE ../../../)

target_compile_features(boost_stringify PUBLIC ${features})
target_compile_features(boost_stringify_header_only INTERFACE ${features})