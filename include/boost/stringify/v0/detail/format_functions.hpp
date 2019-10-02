#ifndef BOOST_STRINGIFY_V0_DETAIL_FORMAT_FUNCTIONS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FORMAT_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template
    < class QFromFmt
    , class QToFmt
    , template <class, class ...> class ValueWithFmt
    , class ValueType
    , class ... QFmts >
struct mp_replace_fmt
{
    template <class QF>
    using f = std::conditional_t< std::is_same<QFromFmt, QF>::value, QToFmt, QF >;

    using type = ValueWithFmt< ValueType, f<QFmts> ... >;
};

template <class QFmt, class T>
struct fmt_helper_impl;

template
    < class QFmt
    , template <class, class ...> class ValueWithFmt
    , class ValueType
    , class ... QFmts>
struct fmt_helper_impl<QFmt, ValueWithFmt<ValueType, QFmts ...> >
{
    using derived_type = ValueWithFmt<ValueType, QFmts ...>;

    template <class QToFmt>
    using adapted_derived_type =
        typename stringify::v0::detail::mp_replace_fmt
            < QFmt, QToFmt, ValueWithFmt, ValueType, QFmts ...>::type;
};

template <class QFmt>
struct fmt_helper_impl<QFmt, void>
{
    using derived_type = typename QFmt::template fn<void>;

    template <class QToFmt>
    using adapted_derived_type = typename QToFmt::template fn<void>;
};

} // namespace detail

template <typename QFmt, typename Der>
using fmt_helper = stringify::v0::detail::fmt_helper_impl<QFmt, Der>;

template <typename QFmt, typename Der>
using fmt_derived
= typename stringify::v0::fmt_helper<QFmt, Der>::derived_type;

template <typename Der, typename QFmtFrom, typename QFmtTo>
using fmt_replace = typename stringify::v0::fmt_helper<QFmtFrom, Der>
    ::template adapted_derived_type<QFmtTo>;

template <typename ValueType, class ... Fmts>
class value_with_format;

template <typename ValueType, class ... Fmts>
class value_with_format
    : public Fmts::template fn<value_with_format<ValueType, Fmts ...>> ...
{
public:

    template <typename U>
    using replace_value_type = stringify::v0::value_with_format<U, Fmts ...>;

    template <typename ... OhterFmts>
    using replace_fmts = stringify::v0::value_with_format<ValueType, OhterFmts ...>;

    constexpr value_with_format(const value_with_format&) = default;
    constexpr value_with_format(value_with_format&&) = default;

    explicit constexpr value_with_format(const ValueType& v)
        : _value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , const stringify::v0::value_with_format<OtherValueType, Fmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>>& >(f) )
        ...
        , _value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , stringify::v0::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , _value(static_cast<ValueType&&>(v))
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( const stringify::v0::value_with_format<ValueType, OtherFmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>& >(f) )
        ...
        , _value(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( stringify::v0::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , _value(static_cast<ValueType&&>(f.value()))
    {
    }

    constexpr const ValueType& value() const
    {
        return _value;
    }

    constexpr ValueType& value()
    {
        return _value;
    }

private:

    ValueType _value;
};

enum class alignment {left, right, internal, center};

namespace detail
{
  template <class T> class alignment_format_impl;
  template <class T> class empty_alignment_format_impl;
}

struct alignment_format
{
    template <class T>
    using fn = stringify::v0::detail::alignment_format_impl<T>;
};

struct empty_alignment_format
{
    template <class T>
    using fn = stringify::v0::detail::empty_alignment_format_impl<T>;
};

namespace detail {

template <class T = void>
class alignment_format_impl
{
    using derived_type = stringify::v0::fmt_derived<alignment_format, T>;

    derived_type& as_derived_ref()
    {
        derived_type* d =  static_cast<derived_type*>(this);
        return *d;
    }

    derived_type&& as_derived_rval_ref()
    {
        derived_type* d =  static_cast<derived_type*>(this);
        return static_cast<derived_type&&>(*d);
    }

public:

    constexpr alignment_format_impl()
    {
        static_assert
            ( std::is_base_of<alignment_format_impl, derived_type>::value
            , "T must be void or derive from alignment_format_impl<T>" );
    }

    constexpr alignment_format_impl(const alignment_format_impl&) = default;

    template <typename U>
    constexpr alignment_format_impl(const alignment_format_impl<U>& u)
        : _fill(u.fill())
        , _width(u.width())
        , _alignment(u.alignment())
    {
    }

    template <typename U>
    constexpr alignment_format_impl(const empty_alignment_format_impl<U>&)
    {
    }

    constexpr derived_type&& operator<(int width) &&
    {
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& operator<(int width) &
    {
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& operator>(int width) &&
    {
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& operator>(int width) &
    {
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& operator^(int width) &&
    {
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& operator^(int width) &
    {
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& operator%(int width) &&
    {
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& operator%(int width) &
    {
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& left(int width) &&
    {
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& left(int width) &
    {
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& right(int width) &&
    {
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& right(int width) &
    {
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& center(int width) &&
    {
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& center(int width) &
    {
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& internal(int width) &&
    {
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& internal(int width) &
    {
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& left(int width, char32_t fill_char) &&
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& left(int width, char32_t fill_char) &
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::left;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& right(int width, char32_t fill_char) &&
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& right(int width, char32_t fill_char) &
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::right;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& center(int width, char32_t fill_char) &&
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& center(int width, char32_t fill_char) &
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::center;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& internal(int width, char32_t fill_char) &&
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type& internal(int width, char32_t fill_char) &
    {
        _fill = fill_char;
        _alignment = stringify::v0::alignment::internal;
        _width = width;
        return as_derived_ref();
    }
    constexpr derived_type&& fill(char32_t ch) &&
    {
        _fill = ch;
        return as_derived_rval_ref();
    }
    constexpr derived_type& fill(char32_t ch) &
    {
        _fill = ch;
        return as_derived_ref();
    }
    constexpr derived_type&& width(int w) &&
    {
        _width = w;
        return as_derived_rval_ref();
    }
    constexpr derived_type& width(int w) &
    {
        _width = w;
        return as_derived_ref();
    }
    constexpr int width() const
    {
        return _width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return _alignment;
    }
    constexpr char32_t fill() const
    {
        return _fill;
    }

private:

    template <typename>
    friend class alignment_format_impl;

    char32_t _fill = U' ';
    int _width = 0;
    stringify::v0::alignment _alignment = stringify::v0::alignment::right;
};

template <class T>
class empty_alignment_format_impl
{
    using helper = stringify::v0::fmt_helper<empty_alignment_format, T>;
    using derived_type = typename helper::derived_type;
    using adapted_derived_type
    = typename helper::template adapted_derived_type<stringify::v0::alignment_format>;

    constexpr adapted_derived_type make_adapted() const
    {
        return adapted_derived_type{static_cast<const derived_type&>(*this)};
    }

public:

    constexpr empty_alignment_format_impl()
    {
    }

    constexpr empty_alignment_format_impl(const empty_alignment_format_impl&)
        = default;

    template <typename U>
    constexpr empty_alignment_format_impl(const empty_alignment_format_impl<U>&)
    {
    }

    ~empty_alignment_format_impl()
    {
    }

    constexpr adapted_derived_type operator<(int width) const
    {
        return make_adapted() < width;
    }
    constexpr adapted_derived_type operator>(int width) const
    {
        return make_adapted() > width;
    }
    constexpr adapted_derived_type operator^(int width) const
    {
        return make_adapted() ^ width;
    }
    constexpr adapted_derived_type operator%(int width) const
    {
        return make_adapted() % width;
    }
    constexpr auto fill(char32_t ch) const
    {
        return make_adapted().fill(ch);
    }
    constexpr adapted_derived_type width(int w) const
    {
        return make_adapted().width(w);
    }

    constexpr int width() const
    {
        return 0;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return stringify::v0::alignment::right;
    }
    constexpr char32_t fill() const
    {
        return U' ';
    }
};

} // namespace detail



template <typename T>
constexpr auto fmt(const T& value)
{
    return make_fmt(stringify::v0::tag{}, value);
}

template <typename T>
constexpr auto hex(const T& value)
{
    return fmt(value).hex();
}

template <typename T>
constexpr auto dec(const T& value)
{
    return fmt(value).dec();
}

template <typename T>
constexpr auto oct(const T& value)
{
    return fmt(value).oct();
}

template <typename T>
constexpr auto left(const T& value, int width)
{
    return fmt(value) < width;
}

template <typename T>
constexpr auto right(const T& value, int width)
{
    return fmt(value) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width)
{
    return fmt(value) % width;
}

template <typename T>
constexpr auto center(const T& value, int width)
{
    return fmt(value) ^ width;
}

template <typename T>
constexpr auto left(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) < width;
}

template <typename T>
constexpr auto right(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) > width;
}

template <typename T>
constexpr auto internal(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) % width;
}

template <typename T>
constexpr auto center(const T& value, int width, char32_t fill)
{
    return fmt(value).fill(fill) ^ width;
}

template <typename T, typename I>
constexpr auto multi(const T& value, I count)
{
    return fmt(value).multi(count);
}

template <typename T>
constexpr auto fixed(const T& value)
{
    return fmt(value).fixed();
}

template <typename T>
constexpr auto sci(const T& value)
{
    return fmt(value).sci();
}

template <typename T, typename P>
constexpr auto fixed(const T& value, P precision)
{
    return fmt(value).fixed().p(precision);
}

template <typename T, typename P>
constexpr auto sci(const T& value, P precision)
{
    return fmt(value).sci().p(precision);
}



BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FORMAT_FUNCTIONS_HPP

