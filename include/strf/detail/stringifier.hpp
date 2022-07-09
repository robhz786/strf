#ifndef STRF_DETAIL_STRINGIFIER_HPP
#define STRF_DETAIL_STRINGIFIER_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/destination.hpp>

namespace strf {

template <typename CharT>
class stringifier
{
public:
    using char_type = CharT;

    STRF_HD virtual ~stringifier()
    {
    }

    STRF_HD virtual void print_to(strf::destination<CharT>& dest) const = 0;
};

template <typename CharT>
using arg_printer
STRF_DEPRECATED_MSG("arg_printer was renamed to stringifier")
= stringifier<CharT>;


namespace detail {

#if defined(__cpp_fold_expressions)

template <typename CharT, typename... Stringifiers>
inline STRF_HD void write_args( strf::destination<CharT>& dest
                              , const Stringifiers&... stringifiers )
{
    (... , stringifiers.print_to(dest));
}

#else // defined(__cpp_fold_expressions)

template <typename CharT>
inline STRF_HD void write_args(strf::destination<CharT>&)
{
}

template <typename CharT, typename Stringifier, typename... Stringifiers>
inline STRF_HD void write_args
    ( strf::destination<CharT>& dest
    , const Stringifier& stringifier
    , const Stringifiers&... stringifiers )
{
    stringifier.print_to(dest);
    if (dest.good()) {
        write_args<CharT>(dest, stringifiers...);
    }
}

#endif // defined(__cpp_fold_expressions)

} // namespace detail

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

// template <typename CharIn>
// struct tr_string_input_tag: strf::string_input_tag<CharIn>
// {
// };

// template <typename CharIn>
// struct is_tr_string_of
// {
//     template <typename T>
//     using fn = std::is_same<strf::tr_string_input_tag<CharIn>, T>;
// };

// template <typename T>
// struct is_tr_string: std::false_type
// {
// };

// template <typename CharIn>
// struct is_tr_string<strf::is_tr_string_of<CharIn>> : std::true_type
// {
// };

} // namespace strf

#endif  // STRF_DETAIL_STRINGIFIER_HPP

