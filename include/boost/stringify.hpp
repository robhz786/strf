#ifndef BOOST_STRINGIFY_HPP_INCLUDED
#define BOOST_STRINGIFY_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if defined(_MSC_VER)
#pragma warning ( push )
#pragma warning ( disable : 4127 )
#endif // defined(_MSC_VER)

#include <boost/stringify/v0/dynamic_join_printer.hpp>
#include <boost/stringify/v0/detail/input_types/fmt_int.hpp>
#include <boost/stringify/v0/detail/input_types/char.hpp>
#include <boost/stringify/v0/detail/input_types/float.hpp>
#include <boost/stringify/v0/detail/input_types/string.hpp>
#include <boost/stringify/v0/detail/input_types/cv_string.hpp>
#include <boost/stringify/v0/detail/input_types/join.hpp>
#include <boost/stringify/v0/detail/input_types/facets_pack.hpp>
#include <boost/stringify/v0/detail/input_types/range.hpp>

#include <boost/stringify/v0/detail/output_types/char_ptr.hpp>
#include <boost/stringify/v0/detail/output_types/std_string.hpp>
#include <boost/stringify/v0/detail/output_types/FILE.hpp>
#include <boost/stringify/v0/detail/output_types/std_streambuf.hpp>

#if defined(_MSC_VER)
#pragma warning ( pop )
#endif // defined(_MSC_VER)

#endif
