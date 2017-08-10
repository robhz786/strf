cmake_minimum_required (VERSION 3.0)
project (build)

add_library(boost_stringify STATIC stringify.cpp)

target_compile_definitions(boost_stringify PUBLIC BOOST_STRINGIFY_NOT_HEADER_ONLY)

target_include_directories(boost_stringify PUBLIC ../../../)
