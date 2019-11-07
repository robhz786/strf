#ifndef STRF_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>

STRF_NAMESPACE_BEGIN

template <typename CharT> struct width_calculator_c;
template <typename CharT> class width_calculator;

template <typename CharT>
class width_calculator
{
public:

    using category = strf::width_calculator_c<CharT>;

    virtual strf::width_t width_of
        ( CharT ch
        , strf::encoding<CharT> enc ) const = 0;

    virtual strf::width_t width
        ( strf::width_t limit
        , const CharT* str
        , std::size_t str_len
        , strf::encoding<CharT> enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) const = 0;
};

template <typename CharT>
class width_as_len final: public strf::width_calculator<CharT>
{
public:

    strf::width_t width_of
        ( CharT ch, strf::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }

    strf::width_t width
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

        if (str_len < INT16_MAX)
        {
            return static_cast<std::int16_t>(str_len);
        }
        return strf::width_t_max;
    }
};

template <typename CharT>
class width_as_u32len final: public strf::width_calculator<CharT>
{
public:

    virtual strf::width_t width_of
        ( CharT ch, strf::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }


    strf::width_t width
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

        if (limit > 0)
        {
            auto count = enc.codepoints_count(str, str + str_len, limit.ceil());
            if (count < INT16_MAX)
            {
                return static_cast<std::int16_t>(count);
            }
            return strf::width_t_max;
        }
        return 0;
    }
};

template <typename CharT>
struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static const strf::width_as_len<CharT>& get_default()
    {
        static const strf::width_as_len<CharT> x{};
        return x;
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

STRF_NAMESPACE_END

#endif  // STRF_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

