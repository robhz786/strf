#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/encoding.hpp>

namespace strf {

template <std::size_t CharSize> struct width_calculator_c;
template <std::size_t CharSize> class width_calculator;

template <std::size_t CharSize>
class width_calculator
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;
    using category = strf::width_calculator_c<CharSize>;

    virtual STRF_HD strf::width_t width_of
        ( char_type ch
        , const strf::underlying_encoding<CharSize>& enc ) const = 0;

    virtual STRF_HD strf::width_t width
        ( strf::width_t limit
        , const char_type* str
        , std::size_t str_len
        , const strf::underlying_encoding<CharSize>& enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const = 0;
};

template <std::size_t CharSize>
class fast_width final: public strf::width_calculator<CharSize>
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    STRF_HD strf::width_t width_of
        ( char_type ch, const strf::underlying_encoding<CharSize>& enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }

    STRF_HD strf::width_t width
        ( strf::width_t limit, const char_type* str, std::size_t str_len
        , const strf::underlying_encoding<CharSize>& enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const override
    {
        (void) limit;
        (void) str;
        (void) enc;
        (void) enc_err;
        (void) allow_surr;

        if (str_len < INT16_MAX) {
            return static_cast<std::int16_t>(str_len);
        }
        return strf::width_max;
    }
};

template <std::size_t CharSize>
class width_as_u32len final: public strf::width_calculator<CharSize>
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    virtual STRF_HD strf::width_t width_of
        ( char_type ch, const strf::underlying_encoding<CharSize>& enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }


    STRF_HD strf::width_t width
        ( strf::width_t limit, const char_type* str, std::size_t str_len
        , const strf::underlying_encoding<CharSize>& enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const override
    {
        (void) limit;
        (void) str;
        (void) enc;
        (void) enc_err;
        (void) allow_surr;

        if (limit > 0) {
            auto count = enc.codepoints_count(str, str + str_len, limit.ceil());
            if (count < INT16_MAX) {
                return static_cast<std::int16_t>(count);
            }
            return strf::width_max;
        }
        return 0;
    }
};

template <std::size_t CharSize>
struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static STRF_HD const strf::fast_width<CharSize>& get_default()
    {
        static const strf::fast_width<CharSize> x{};
        return x;
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace strf

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

