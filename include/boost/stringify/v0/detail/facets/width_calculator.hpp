#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> struct width_calculator_c;
template <typename CharT> class width_calculator;

template <typename CharT>
class width_calculator
{
public:

    using category = stringify::v0::width_calculator_c<CharT>;

    virtual stringify::v0::width_t width_of
        ( CharT ch
        , stringify::v0::encoding<CharT> enc ) const = 0;

    virtual stringify::v0::width_t width
        ( stringify::v0::width_t limit
        , const CharT* str
        , std::size_t str_len
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr ) const = 0;
};

template <typename CharT>
class width_as_len final: public stringify::v0::width_calculator<CharT>
{
public:

    stringify::v0::width_t width_of
        ( CharT ch, stringify::v0::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }

    stringify::v0::width_t width
        ( stringify::v0::width_t limit
        , const CharT* str
        , std::size_t str_len
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr ) const override
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
        return stringify::v0::width_t_max;
    }
};

template <typename CharT>
class width_as_u32len final: public stringify::v0::width_calculator<CharT>
{
public:

    virtual stringify::v0::width_t width_of
        ( CharT ch, stringify::v0::encoding<CharT> enc ) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }


    stringify::v0::width_t width
        ( stringify::v0::width_t limit
        , const CharT* str
        , std::size_t str_len
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr ) const override
    {
        (void) limit;
        (void) str;
        (void) enc;
        (void) enc_err;
        (void) allow_surr;

        auto count = enc.codepoints_count(str, str + str_len, limit.ceil());
        if (count < INT16_MAX)
        {
            return static_cast<std::int16_t>(count);
        }
        return stringify::v0::width_t_max;
    }
};

template <typename CharT>
struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static const stringify::v0::width_as_len<CharT>& get_default()
    {
        static const stringify::v0::width_as_len<CharT> x{};
        return x;
    }
};

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

