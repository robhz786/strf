#ifndef STRF_DETAIL_FACETS_LETTERCASE_HPP
#define STRF_DETAIL_FACETS_LETTERCASE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {

enum class lettercase : std::uint16_t { lower = 0x2020, mixed = 0x20, upper = 0 };

constexpr lettercase lowercase = lettercase::lower;
constexpr lettercase mixedcase = lettercase::mixed;
constexpr lettercase uppercase = lettercase::upper;

struct lettercase_c
{
    static constexpr bool constrainable = true;
    constexpr static STRF_HD strf::lettercase get_default() noexcept
    {
        return strf::lettercase::lower;
    }
};

template <typename Facet>
struct facet_traits;

template <>
struct facet_traits<strf::lettercase>
{
    using category = strf::lettercase_c;
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_LETTERCASE_HPP

