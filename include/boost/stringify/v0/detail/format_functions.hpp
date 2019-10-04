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
struct fmt_replace_impl2
{
    template <class QF>
    using f = std::conditional_t< std::is_same<QFromFmt, QF>::value, QToFmt, QF >;

    using type = ValueWithFmt< ValueType, f<QFmts> ... >;
};

template <class QFmt, class T>
struct fmt_replace_impl;

template
    < class QFmt
    , template <class, class ...> class ValueWithFmt
    , class ValueType
    , class ... QFmts>
struct fmt_replace_impl<QFmt, ValueWithFmt<ValueType, QFmts ...> >
{
    template <class QToFmt>
    using type_tmpl =
        typename stringify::v0::detail::fmt_replace_impl2
            < QFmt, QToFmt, ValueWithFmt, ValueType, QFmts ...>::type;
};

} // namespace detail

template <typename Der, typename QFmtFrom, typename QFmtTo>
using fmt_replace
    = typename stringify::v0::detail::fmt_replace_impl<QFmtFrom, Der>
    ::template type_tmpl<QFmtTo>;

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

struct alignment_format_data
{
    char32_t fill = U' ';
    int width = 0;
    stringify::v0::alignment alignment = stringify::v0::alignment::right;
};

constexpr bool operator==( stringify::v0::alignment_format_data d1
                         , stringify::v0::alignment_format_data d2 )
{
    return d1.fill == d2.fill
        && d1.width == d2.width
        && d1.alignment == d2.alignment;
}

template <bool Active, class T>
class alignment_format_fn;

template <bool Active>
struct alignment_format_q
{
    template <class T>
    using fn = stringify::v0::alignment_format_fn<Active, T>;
};

using alignment_format = stringify::v0::alignment_format_q<true>;
using empty_alignment_format = stringify::v0::alignment_format_q<false>;

template <class T>
class alignment_format_fn<true, T>
{
    using derived_type = T;

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

    constexpr alignment_format_fn() = default;

    template <bool B, typename U>
    constexpr explicit alignment_format_fn
        ( const stringify::v0::alignment_format_fn<B, U>& u ) noexcept
        : _data(u.get_alignment_format_data())
    {
    }

    constexpr explicit alignment_format_fn(stringify::v0::alignment_format_data data)
        : _data(data)
    {
    }

    constexpr derived_type&& operator<(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment::left;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type&& operator>(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment::right;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type&& operator^(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment::center;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type&& operator%(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment::internal;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr derived_type&& fill(char32_t ch) && noexcept
    {
        _data.fill = ch;
        return as_derived_rval_ref();
    }
    constexpr int width() const
    {
        return _data.width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return _data.alignment;
    }
    constexpr char32_t fill() const
    {
        return _data.fill;
    }

    constexpr alignment_format_data get_alignment_format_data() const noexcept
    {
        return _data;
    }

private:

    stringify::v0::alignment_format_data _data;
};

template <class T>
class alignment_format_fn<false, T>
{
    using derived_type = T;
    using adapted_derived_type = stringify::v0::fmt_replace
            < T
            , stringify::v0::alignment_format_q<false>
            , stringify::v0::alignment_format_q<true> >;

    constexpr adapted_derived_type make_adapted() const
    {
        return adapted_derived_type{static_cast<const derived_type&>(*this)};
    }

public:

    constexpr alignment_format_fn() noexcept = default;

    template <typename U>
    constexpr explicit alignment_format_fn(const alignment_format_fn<false, U>&) noexcept
    {
    }

    ~alignment_format_fn()
    {
    }

    constexpr adapted_derived_type operator<(int width) const noexcept
    {
        return make_adapted() < width;
    }
    constexpr adapted_derived_type operator>(int width) const noexcept
    {
        return make_adapted() > width;
    }
    constexpr adapted_derived_type operator^(int width) const noexcept
    {
        return make_adapted() ^ width;
    }
    constexpr adapted_derived_type operator%(int width) const noexcept
    {
        return make_adapted() % width;
    }
    constexpr auto fill(char32_t ch) const noexcept
    {
        return make_adapted().fill(ch);
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
    constexpr alignment_format_data get_alignment_format_data() const noexcept
    {
        return {};
    }
};



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

