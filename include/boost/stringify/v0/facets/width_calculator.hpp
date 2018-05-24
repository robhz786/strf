#ifndef BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/facets/encodings.hpp>
#include <boost/assert.hpp>
#include <string>
#include <limits>
#include <algorithm>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct width_calculator_category;
class width_calculator;


typedef int (*char_width_calculator)(char32_t);

namespace detail{

class width_decrementer: public stringify::v0::u32output
{
public:

    width_decrementer
        ( int initial_width
        , const stringify::v0::char_width_calculator wc
        , const stringify::v0::error_signal& err_sig
        )
        : m_wcalc(wc)
        , m_err_sig(err_sig)
        , m_width(initial_width)
    {
    }

    ~width_decrementer()
    {
    }

    stringify::v0::cv_result put32(char32_t ch) override;

    bool signalize_error() override;

    int get_remaining_width() const
    {
        return m_width > 0 ? m_width : 0;
    }

private:

    const stringify::v0::char_width_calculator m_wcalc;
    const stringify::v0::error_signal& m_err_sig;
    int m_width = 0;

};

} // namespaced detail

enum class width_calculation_type : std::size_t
{
    as_length,
    as_codepoints_count
};


class width_calculator
{

public:

    using category = stringify::v0::width_calculator_category;

    explicit width_calculator
    ( const stringify::v0::width_calculation_type calc_type
    )
        : m_type(calc_type)
    {
    }

    explicit width_calculator
    ( const stringify::v0::char_width_calculator calc_function
    )
        : m_ch_wcalc(calc_function)
    {
    }

    width_calculator(const width_calculator& cp) = default;

    int width_of(char32_t ch) const;

    int remaining_width
        ( int width
        , const char32_t* begin
        , const char32_t* end
        ) const;

    template <typename CharIn>
    int remaining_width
        ( int width
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::decoder<CharIn>& conv
        , const stringify::v0::error_signal& err_sig
        , bool keep_surrogates
        ) const
    {
        if (m_type == stringify::width_calculation_type::as_length)
        {
            std::size_t str_len = end - begin;
            return str_len > static_cast<std::size_t>(width)
                ? 0
                : width - static_cast<int>(str_len);
        }
        else if(m_type == stringify::width_calculation_type::as_codepoints_count)
        {
            return static_cast<int>(conv.remaining_codepoints_count(width, begin, end));
        }
        else
        {
            detail::width_decrementer decrementer{width, *m_ch_wcalc, err_sig};
            (void) conv.decode(decrementer, begin, end, keep_surrogates);
            return decrementer.get_remaining_width();
        }
    }

private:

    static int unique_char_width(char32_t)
    {
        return 1;
    }

    union
    {
        stringify::v0::width_calculation_type m_type;
        stringify::v0::char_width_calculator m_ch_wcalc;
    };

    static_assert(sizeof(stringify::v0::width_calculation_type) >= sizeof(stringify::v0::char_width_calculator), "");
};


struct width_calculator_category
{
    static constexpr bool constrainable = true;

    static const stringify::v0::width_calculator& get_default()
    {
        static stringify::v0::width_calculator x {nullptr};
        return x;
    }
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char>
    ( int width
    , const char* begin
    , const char* end
    , const stringify::v0::decoder<char>& conv
    , const stringify::v0::error_signal& err_sig
    , bool keep_surrogates
    ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char16_t>
    ( int width
    , const char16_t* begin
    , const char16_t* end
    , const stringify::v0::decoder<char16_t>& conv
    , const stringify::v0::error_signal& err_sig
    , bool keep_surrogates
    ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<char32_t>
    ( int width
    , const char32_t* begin
    , const char32_t* end
    , const stringify::v0::decoder<char32_t>& conv
    , const stringify::v0::error_signal& err_sig
    , bool keep_surrogates
    ) const;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
int width_calculator::remaining_width<wchar_t>
    ( int width
    , const wchar_t* str
    , const wchar_t* end
    , const stringify::v0::decoder<wchar_t>& conv
    , const stringify::v0::error_signal& err_sig
    , bool keep_surrogates
    ) const;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

namespace detail {

BOOST_STRINGIFY_INLINE stringify::v0::cv_result width_decrementer::put32(char32_t ch)
{
    m_width -= m_wcalc(ch);
    return m_width > 0
        ? stringify::v0::cv_result::success
        : stringify::v0::cv_result::insufficient_space;
}

BOOST_STRINGIFY_INLINE bool width_decrementer::signalize_error()
{
    if (m_err_sig.has_char())
    {
        return put32(m_err_sig.get_char()) == stringify::v0::cv_result::success;
    }
    return true; 
}

} // namespace detail

BOOST_STRINGIFY_INLINE int width_calculator::width_of(char32_t ch) const
{
    if ( m_type == stringify::width_calculation_type::as_length
      || m_type == stringify::width_calculation_type::as_codepoints_count )
    {
        return 1;
    }
    else
    {
        return m_ch_wcalc(ch);
    }
}


BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const char32_t* begin
    , const char32_t* end
    ) const
{
    if ( m_type == stringify::width_calculation_type::as_length
      || m_type == stringify::width_calculation_type::as_codepoints_count )
    {
        std::size_t str_len = end - begin;
        if(str_len >= (std::size_t)(width))
        {
            return 0;
        }
        return width - static_cast<int>(str_len);
    }
    else
    {
        for(auto it = begin; it < end && width > 0; ++it)
        {
            width -= m_ch_wcalc(*it);
        }
        return width > 0 ? width : 0;
    }
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

namespace detail{

inline int char_width_aways_one(char32_t)
{
    return 1;
}

}

inline stringify::v0::width_calculator width_as_length()
{
    return stringify::v0::width_calculator
        { stringify::v0::width_calculation_type::as_length };
}

inline stringify::v0::width_calculator width_as_codepoints_count()
{
    return stringify::v0::width_calculator
        { stringify::v0::width_calculation_type::as_codepoints_count };
}

inline stringify::v0::width_calculator width_as
    (stringify::v0::char_width_calculator func)
{
    return stringify::v0::width_calculator {func};
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

