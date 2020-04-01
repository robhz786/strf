#ifndef STRF_DETAIL_FACETS_LETTERCASE_HPP
#define STRF_DETAIL_FACETS_LETTERCASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>

namespace strf {

enum class lettercase { lower = 0, mixed = 1, upper = 3 };

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
class facet_traits;

template <>
class facet_traits<strf::lettercase>
{
public:
    using category = strf::lettercase_c;
    static constexpr bool store_by_value = true;
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_LETTERCASE_HPP

