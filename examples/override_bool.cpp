//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "strf/to_string.hpp"

struct my_bool_override
{
    using category = strf::print_override_c;

    template <typename CharT, typename PrePrinting, typename FPack, typename... T>
    constexpr auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::value_with_formatters<T...> x ) const noexcept
    {
        bool value = x.value();
        return strf::make_arg_printer_input<CharT>
            ( pre
            , fp
            , strf::conv(false_true_strings[value], strf::utf_t<char>{})
                .set_alignment_format(x.get_alignment_format()) );
    }

    const char* false_true_strings[2] = {"false", "true"};
};

static_assert(strf::is_overridable<bool>, "bool not overridable");

template <typename T>
struct is_bool: std::is_same<T, strf::override_tag<bool>> {};

constexpr auto italian_bool = strf::constrain<is_bool>(my_bool_override{{"falso", "vero"}});

int main()
{
    auto str = strf::to_string.with(italian_bool)
        (true, '/', false, '/', 1, '/', 0, '/', 1.0, '/', 0.0, '/', (void*)0);
    assert(str == "vero/falso/1/0/1/0/0x0");

    // with formatting
    str = strf::to_string.with(italian_bool)
        ( strf::center(true, 10, '.'), '/'
        , strf::center(false, 10, '.') );
    assert(str == "...vero.../..falso...");

    // what happens when you don't constrain an overrider facet:
    str = strf::to_string.with(my_bool_override{{"falso", "vero"}})
        (true, '/', false, '/', 1, '/', 0, '/', 1.0, '/', 0.0, '/', (void*)0);
    assert(str == "vero/falso/vero/falso/vero/falso/falso");

    return 0;
}

