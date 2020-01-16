#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/encoding.hpp>

STRF_NAMESPACE_BEGIN

template <typename CharT> struct width_calculator_c;
template <typename CharT> class width_calculator;

template <typename CharT>
class width_calculator
{
public:

    using category = strf::width_calculator_c<CharT>;

    virtual STRF_HD strf::width_t width_of
        ( CharT ch
        , strf::encoding<CharT> enc ) const = 0;

    virtual STRF_HD strf::width_t width
        ( strf::width_t limit
        , const CharT* str
        , std::size_t str_len
        , strf::encoding<CharT> enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const = 0;
};

template <typename CharT>
class fast_width final: public strf::width_calculator<CharT>
{
public:

    STRF_HD strf::width_t width_of
        ( CharT ch, strf::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }

    STRF_HD strf::width_t width
        ( strf::width_t limit
        , const CharT* str
        , std::size_t str_len
        , strf::encoding<CharT> enc
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

template <typename CharT>
class width_as_u32len final: public strf::width_calculator<CharT>
{
public:

    virtual STRF_HD strf::width_t width_of
        ( CharT ch, strf::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }


    STRF_HD strf::width_t width
        ( strf::width_t limit
        , const CharT* str
        , std::size_t str_len
        , strf::encoding<CharT> enc
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

template <typename CharT>
struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static STRF_HD const strf::fast_width<CharT>& get_default()
    {
        static const strf::fast_width<CharT> x{};
        return x;
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

