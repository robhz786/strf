#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#

cmake_minimum_required (VERSION 3.0)
project (build)

set(stringify_hpp                 ../../../boost/stringify.hpp)
file(GLOB headers_v0              ../../../boost/stringify/v0/*.hpp)
file(GLOB headers_v0_facets       ../../../boost/stringify/v0/facets/*.hpp)
file(GLOB headers_v0_input_types  ../../../boost/stringify/v0/input_types/*.hpp)
file(GLOB headers_v0_output_types ../../../boost/stringify/v0/output_types/*.hpp)
file(GLOB headers_v0_detail       ../../../boost/stringify/v0/detail/*.hpp)

source_group("Header Files"                              FILES  ${stringify_hpp})
source_group("Header Files\\stringify\\v0"               FILES  ${headers_v0})
source_group("Header Files\\stringify\\v0\\facets"       FILES  ${headers_v0_facets})
source_group("Header Files\\stringify\\v0\\input_types"  FILES  ${headers_v0_input_types})
source_group("Header Files\\stringify\\v0\\output_types" FILES  ${headers_v0_output_types})
source_group("Header Files\\stringify\\v0\\detail"       FILES  ${headers_v0_detail})

set(
  headers
  ${stringify_hpp}
  ${headers_v0}
  ${headers_v0_facets}
  ${headers_v0_input_types}
  ${headers_v0_output_types}
  ${headers_v0_detail}
)

add_library(boost_stringify STATIC stringify.cpp ${headers})
add_library(boost_stringify-header-only INTERFACE)
add_library(boost::stringify ALIAS boost_stringify)
add_library(boost::stringify-header-only ALIAS boost_stringify-header-only)

target_compile_definitions(boost_stringify INTERFACE BOOST_STRINGIFY_NOT_HEADER_ONLY)
target_include_directories(
  boost_stringify
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(
  boost_stringify-header-only
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
  $<INSTALL_INTERFACE:include>
)

set(features cxx_return_type_deduction cxx_decltype_auto)
target_compile_features(boost_stringify PUBLIC ${features})
target_compile_features(boost_stringify-header-only INTERFACE ${features})

set_property(
  TARGET boost_stringify
  PROPERTY EXPORT_NAME stringify
)
set_property(
  TARGET boost_stringify-header-only
  PROPERTY EXPORT_NAME stringify-header-only
)

install(TARGETS boost_stringify EXPORT boost_stringify DESTINATION "$<CONFIG>")
install(TARGETS boost_stringify-header-only EXPORT boost_stringify DESTINATION "$<CONFIG>")
install(
  EXPORT boost_stringify
  FILE BoostStringify.cmake
  NAMESPACE boost::
  DESTINATION cmake)
install(DIRECTORY ../include/ DESTINATION include USE_SOURCE_PERMISSIONS)

