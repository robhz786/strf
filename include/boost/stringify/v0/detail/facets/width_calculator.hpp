#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct width_calculator_category;
class width_calculator;

typedef int (*char_width_calculator)(char32_t);

enum class width_calculation_type : std::size_t
{
    as_len,
    as_u32len
};

namespace detail {

class width_decrementer: public stringify::v0::output_buffer<char32_t>
{
public:

    width_decrementer(char_width_calculator func, int width)
        : stringify::v0::output_buffer<char32_t>(_buff, _buff + _buff_size)
        , _func(func)
        , _remaining_width(width)
    {
    }

    bool recycle() override;

    std::size_t get_result()
    {
        recycle();
        return _remaining_width > 0 ? _remaining_width : 0;
    }

private:

    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    char32_t _buff[_buff_size];
    char_width_calculator _func;
    int _remaining_width = 0;
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE bool width_decrementer::recycle()
{
    for(auto it = _buff; it < this->pos() && _remaining_width > 0; ++it)
    {
       _remaining_width -= _func(*it);
    }
    this->set_pos(_buff);
    return _remaining_width > 0;
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

class width_calculator
{
public:

    using category = stringify::v0::width_calculator_category;

    explicit width_calculator
    ( const stringify::v0::width_calculation_type calc_type
    )
        : _type(calc_type)
    {
    }

    explicit width_calculator
    ( const stringify::v0::char_width_calculator calc_function
    )
        : _ch_wcalc(calc_function)
    {
    }

    width_calculator(const width_calculator& cp) = default;

    int width_of(char32_t ch) const;

    template <typename CharT>
    int width_of(CharT ch, stringify::v0::encoding<CharT> enc) const
    {
        if ( _type == stringify::width_calculation_type::as_len
          || _type == stringify::width_calculation_type::as_u32len )
        {
            return 1;
        }
        BOOST_STRINGIFY_IF_CONSTEXPR(std::is_same<CharT, char32_t>::value)
        {
            if (enc.id() == stringify::v0::encoding_id::eid_utf32)
            {
                return _ch_wcalc(ch);
            }
        }
        return _ch_wcalc(enc.decode_single_char(ch));
    }

    template <typename CharIn>
    int remaining_width
        ( int width
        , const CharIn* str
        , std::size_t str_len
        , stringify::v0::encoding<CharIn> enc
        , const stringify::v0::encoding_policy epoli ) const
    {
        if (_type == stringify::width_calculation_type::as_len)
        {
            return str_len > static_cast<std::size_t>(width)
                ? 0
                : width - static_cast<int>(str_len);
        }
        else if(_type == stringify::width_calculation_type::as_u32len)
        {
            auto count = enc.codepoints_count(str, str + str_len, width);
            BOOST_ASSERT((std::ptrdiff_t)width >= (std::ptrdiff_t)count);
            return width - (int) count;
        }
        else
        {
            stringify::v0::detail::width_decrementer acc(_ch_wcalc, width);
            enc.to_u32().transcode( acc, str, str + str_len
                                  , epoli.err_hdl(), epoli.allow_surr() );
            return acc.get_result();
        }
    }

private:

    union
    {
        stringify::v0::width_calculation_type _type;
        stringify::v0::char_width_calculator _ch_wcalc;
    };

    static_assert( sizeof(stringify::v0::width_calculation_type) >=
                   sizeof(stringify::v0::char_width_calculator)
                 , "");
};


struct width_calculator_category
{
    static constexpr bool constrainable = true;

    static stringify::v0::width_calculator get_default()
    {
        return stringify::v0::width_calculator{nullptr};
    }
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char>
    ( int width
    , const char* str
    , std::size_t str_len
    , stringify::v0::encoding<char> conv
    , stringify::v0::encoding_policy epoli ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char16_t>
    ( int width
    , const char16_t* str
    , std::size_t str_len
    , stringify::v0::encoding<char16_t> conv
    , stringify::v0::encoding_policy epoli ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char32_t>
    ( int width
    , const char32_t* str
    , std::size_t str_len
    , stringify::v0::encoding<char32_t> conv
    , stringify::v0::encoding_policy epoli ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<wchar_t>
    ( int width
    , const wchar_t* str
    , std::size_t str_len
    , stringify::v0::encoding<wchar_t> conv
    , stringify::v0::encoding_policy epoli ) const;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)



BOOST_STRINGIFY_INLINE int width_calculator::width_of(char32_t ch) const
{
    if ( _type == stringify::width_calculation_type::as_len
      || _type == stringify::width_calculation_type::as_u32len )
    {
        return 1;
    }
    else
    {
        return _ch_wcalc(ch);
    }
}


// BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
//     ( int width
//     , const char32_t* begin
//     , const char32_t* end
//     ) const
// {
//     if ( _type == stringify::width_calculation_type::as_len
//       || _type == stringify::width_calculation_type::as_u32len )
//     {
//         std::size_t str_len = end - begin;
//         if(str_len >= (std::size_t)(width))
//         {
//             return 0;
//         }
//         return width - static_cast<int>(str_len);
//     }
//     else
//     {
//         for(auto it = begin; it < end && width > 0; ++it)
//         {
//             width -= _ch_wcalc(*it);
//         }
//         return width > 0 ? width : 0;
//     }
// }

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

inline stringify::v0::width_calculator width_as_len()
{
    return stringify::v0::width_calculator
        { stringify::v0::width_calculation_type::as_len };
}

inline stringify::v0::width_calculator width_as_u32len()
{
    return stringify::v0::width_calculator
        { stringify::v0::width_calculation_type::as_u32len };
}

inline stringify::v0::width_calculator width_as
    (stringify::v0::char_width_calculator func)
{
    return stringify::v0::width_calculator {func};
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

