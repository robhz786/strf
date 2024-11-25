//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifdef __clang__
#pragma clang diagnostic ignored "-Wfloat-conversion"
#endif

#include "strf/to_string.hpp"

struct my_bool_overrider
{
    using category = strf::printable_overrider_c_of<bool>;

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    constexpr auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , bool value ) const noexcept
    {
        return strf::make_printer<CharT>
            ( pre
            , fp
            , strf::unsafe_transcode(false_true_strings[value]) );
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    constexpr auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::value_and_format<T...> x ) const noexcept
    {
        const bool value = static_cast<bool>(x.value());
        return strf::make_printer<CharT>
            ( pre
            , fp
            , strf::unsafe_transcode(false_true_strings[value])
                .set_alignment_format(x.get_alignment_format()) );
    }

    const char* false_true_strings[2] = {"false", "true"};
};

static_assert(strf::is_printable_and_overridable_v<bool>, "bool not overridable");

constexpr auto italian_bool = my_bool_overrider{{"falso", "vero"}};

int main()
{
    auto str = strf::to_string.with(italian_bool)
        (true, '/', false, '/', 1, '/', 0, '/', 1.0, '/', 0.0, '/', static_cast<void*>(nullptr));
    assert(str == "vero/falso/1/0/1/0/0x0");

    // with formatting
    str = strf::to_string.with(italian_bool)
        ( strf::center(true, 10, '.'), '/'
        , strf::center(false, 10, '.') );
    assert(str == "...vero.../..falso...");

    return 0;
}

