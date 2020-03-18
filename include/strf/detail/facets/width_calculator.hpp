#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>

namespace strf {

struct width_calculator_c;

class fast_width final
{
public:
    using category = width_calculator_c;

    template <typename Charset>
    STRF_HD strf::width_t width
        ( const Charset&
        , strf::underlying_char_type<Charset::char_size> ) const noexcept
    {
        return 1;
    }

    template <typename Charset>
    constexpr STRF_HD strf::width_t width
        ( const Charset&
        , strf::width_t
        , const strf::underlying_char_type<Charset::char_size>*
        , std::size_t str_len
        , strf::invalid_seq_policy
        , strf::surrogate_policy ) const noexcept
    {
        if (str_len < INT16_MAX) {
            return static_cast<std::int16_t>(str_len);
        }
        return strf::width_max;
    }
};

class width_as_fast_u32len final
{
public:
    using category = width_calculator_c;

    template <typename Charset>
    constexpr STRF_HD strf::width_t width
        ( const Charset&
        , strf::underlying_char_type<Charset::char_size> ) const noexcept
    {
        return 1;
    }

    template <typename Charset>
    STRF_HD strf::width_t width
        ( const Charset& cs
        , strf::width_t limit
        , const strf::underlying_char_type<Charset::char_size>* str
        , std::size_t str_len
        , strf::invalid_seq_policy
        , strf::surrogate_policy ) const
    {
        (void) str;

        if (limit > 0) {
            auto res = cs.codepoints_fast_count(str, str + str_len, limit.ceil());
            if (res.count < INT16_MAX) {
                return static_cast<std::int16_t>(res.count);
            }
            return strf::width_max;
        }
        return 0;
    }
};

struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static constexpr STRF_HD strf::fast_width get_default() noexcept
    {
        return {};
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace strf

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

