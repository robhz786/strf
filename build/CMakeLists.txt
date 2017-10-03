cmake_minimum_required (VERSION 3.0)
project (build)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3") 

set(features
  cxx_aggregate_default_initializers
  cxx_alias_templates
  cxx_decltype_auto
  cxx_extern_templates
  cxx_variable_templates
)

add_library(boost_stringify STATIC stringify.cpp)
target_compile_definitions(boost_stringify PUBLIC BOOST_STRINGIFY_NOT_HEADER_ONLY)
target_include_directories(boost_stringify PUBLIC ../../../)

add_library(boost_stringify_header_only INTERFACE)
target_include_directories(boost_stringify_header_only INTERFACE ../../../)

target_compile_features(boost_stringify PUBLIC ${features})
target_compile_features(boost_stringify_header_only INTERFACE ${features})