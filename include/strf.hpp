#ifndef STRF_HPP_INCLUDED
#define STRF_HPP_INCLUDED

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if defined(_MSC_VER)
#pragma warning ( push )

#if ! defined(__cpp_if_constexpr)
#pragma warning ( disable : 4127 )
#endif

// #pragma warning ( disable : 4061 )
// #pragma warning ( disable : 4365 )
// #pragma warning ( disable : 4514 )
// #pragma warning ( disable : 4571 )
// #pragma warning ( disable : 4623 )
// #pragma warning ( disable : 4625 )
// #pragma warning ( disable : 4626 )
// #pragma warning ( disable : 4627 )
// #pragma warning ( disable : 4710 )
// #pragma warning ( disable : 4820 )
// #pragma warning ( disable : 5026 )
// #pragma warning ( disable : 5027 )
// #pragma warning ( disable : 5045 )
#endif // defined(_MSC_VER)

#include <strf/detail/single_byte_charsets.hpp>

//
// Input types
//
#include <strf/detail/printable_types/bool.hpp>
#include <strf/detail/printable_types/int.hpp>
#include <strf/detail/printable_types/char.hpp>
#include <strf/detail/printable_types/float.hpp>
#include <strf/detail/printable_types/string.hpp>
#include <strf/detail/printable_types/join.hpp>
#include <strf/detail/printable_types/facets_pack.hpp>
#include <strf/detail/printable_types/range.hpp>

#include <strf/printer.hpp>
#include <strf/precalculate.hpp>

#if defined(_MSC_VER)
#pragma warning ( pop )
#endif // defined(_MSC_VER)

#endif
