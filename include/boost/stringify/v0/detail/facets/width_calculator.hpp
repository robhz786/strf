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

// namespace detail {
//
// class width_accumulator: public stringify::v0::basic_outbuf<char32_t>
// {
// public:
//
//     width_accumulator(width_calc_func func, int limit)
//         : stringify::v0::basic_outbuf<char32_t>(_buff, _buff + _buff_size)
//         , _func(func)
//         , _limit(limit)
//     {
//     }
//
//     void recycle() override;
//
//     int get_result()
//     {
//         if (_limit > 0)
//         {
//             _sum += _func(_limit, _buff, this->pos());
//             this->set_pos(_buff);
//         }
//         return _sum;
//     }
//
// private:
//
//     constexpr static std::size_t _buff_size = 16;
//     char32_t _buff[_buff_size];
//     width_calc_func _func;
//     const int _limit;
//     int _sum = 0;
// };
//
// #if ! defined(BOOST_STRINGIFY_OMIT_IMPL)
//
// BOOST_STRINGIFY_INLINE void width_accumulator::recycle()
// {
//     auto p = this->pos();
//     this->set_pos(_buff);
//     if (this->good())
//     {
//         try
//         {
//             _sum += _func(_limit - _sum, _buff, p);
//             this->set_good(_sum < _limit);
//         }
//         catch(...)
//         {
//             this->set_good(false);
//         }
//     }
// }
//
// #endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)
//
// } // namespace detail

template <typename CharT>
class width_calculator
{
public:

    using category = stringify::v0::width_calculator_c<CharT>;

    virtual int width_of(CharT ch, stringify::v0::encoding<CharT> enc) const = 0;

    virtual int width( int limit
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

    int width_of(CharT ch, stringify::v0::encoding<CharT> enc) const override
    {
        (void)ch;
        (void)enc;
        return 1;
    }

    int width( int limit
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
        return str_len;
    }
};

template <typename CharT>
class width_as_u32len final: public stringify::v0::width_calculator<CharT>
{
public:

    virtual int width_of(CharT ch, stringify::v0::encoding<CharT> enc) const
    {
        (void)ch;
        (void)enc;
        return 1;
    }


    int width( int limit
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
        return static_cast<int>(enc.codepoints_count( str
                                                    , str + str_len
                                                    , limit ));
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

// inline stringify::v0::width_calculator width_as_len()
// {
//     return stringify::v0::width_calculator
//         { stringify::v0::width_calculation_type::as_len };
// }

// inline stringify::v0::width_calculator width_as_u32len()
// {
//     return stringify::v0::width_calculator
//         { stringify::v0::width_calculation_type::as_u32len };
// }

// inline stringify::v0::width_calculator width_as
//     (stringify::v0::width_calc_func func)
// {
//     return stringify::v0::width_calculator {func};
// }


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

