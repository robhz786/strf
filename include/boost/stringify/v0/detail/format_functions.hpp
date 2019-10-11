#ifndef BOOST_STRINGIFY_V0_DETAIL_FORMAT_FUNCTIONS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FORMAT_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <cstring>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template
    < class From
    , class To
    , template <class ...> class List
    , class ... T >
struct fmt_replace_impl2
{
    template <class U>
    using f = std::conditional_t<std::is_same<From, U>::value, To, U>;

    using type = List<f<T> ...>;
};

template <class From, class List>
struct fmt_replace_impl;

template
    < class From
    , template <class ...> class List
    , class ... T>
struct fmt_replace_impl<From, List<T ...> >
{
    template <class To>
    using type_tmpl =
        typename stringify::v0::detail::fmt_replace_impl2
            < From, To, List, T...>::type;
};

} // namespace detail

template <typename List, typename From, typename To>
using fmt_replace
    = typename stringify::v0::detail::fmt_replace_impl<From, List>
    ::template type_tmpl<To>;

template <typename ValueType, class ... Fmts>
class value_with_format;

template <typename ValueType, class ... Fmts>
class value_with_format
    : public Fmts::template fn<value_with_format<ValueType, Fmts ...>> ...
{
public:

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

template <bool Active>
struct alignment_format_q;

enum class alignment_e {left, right, internal, center};

struct alignment_format_data
{
    char32_t fill = U' ';
    int width = 0;
    stringify::v0::alignment_e alignment = stringify::v0::alignment_e::right;
};

constexpr bool operator==( stringify::v0::alignment_format_data lhs
                         , stringify::v0::alignment_format_data rhs ) noexcept
{
    return lhs.fill == rhs.fill
        && lhs.width == rhs.width
        && lhs.alignment == rhs.alignment ;
}

constexpr bool operator!=( stringify::v0::alignment_format_data lhs
                         , stringify::v0::alignment_format_data rhs ) noexcept
{
    return ! (lhs == rhs);
}

template <bool Active, class T>
class alignment_format_fn
{
    T& as_derived_ref()
    {
        T* d =  static_cast<T*>(this);
        return *d;
    }

    T&& as_derived_rval_ref()
    {
        T* d =  static_cast<T*>(this);
        return static_cast<T&&>(*d);
    }

public:

    constexpr alignment_format_fn() noexcept = default;

    template <bool B, typename U>
    constexpr explicit alignment_format_fn
        ( const stringify::v0::alignment_format_fn<B, U>& u ) noexcept
        : _data(u.get_alignment_format_data())
    {
    }

    constexpr T&& operator<(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment_e::left;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator>(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment_e::right;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator^(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment_e::center;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator%(int width) && noexcept
    {
        _data.alignment = stringify::v0::alignment_e::internal;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& fill(char32_t ch) && noexcept
    {
        _data.fill = ch;
        return as_derived_rval_ref();
    }
    constexpr int width() const noexcept
    {
        return _data.width;
    }
    constexpr stringify::v0::alignment_e alignment() const noexcept
    {
        return _data.alignment;
    }
    constexpr char32_t fill() const noexcept
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
        return adapted_derived_type{static_cast<const T&>(*this)};
    }

public:

    constexpr alignment_format_fn() noexcept = default;

    template <typename U>
    constexpr explicit alignment_format_fn(const alignment_format_fn<false, U>&) noexcept
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

    constexpr int width() const noexcept
    {
        return 0;
    }
    constexpr stringify::v0::alignment_e alignment() const noexcept
    {
        return stringify::v0::alignment_e::right;
    }
    constexpr char32_t fill() const noexcept
    {
        return U' ';
    }
    constexpr alignment_format_data get_alignment_format_data() const noexcept
    {
        return {};
    }
};

template <bool Active>
struct alignment_format_q
{
    template <class T>
    using fn = stringify::v0::alignment_format_fn<Active, T>;
};

using alignment_format = stringify::v0::alignment_format_q<true>;
using empty_alignment_format = stringify::v0::alignment_format_q<false>;


template <class T>
class quantity_format_fn
{
public:

    constexpr quantity_format_fn() noexcept = default;

    template <typename U>
    constexpr explicit quantity_format_fn(const quantity_format_fn<U>& u) noexcept
        : _count(u.count())
    {
    }

    constexpr T&& multi(int count) && noexcept
    {
        _count = count;
        return static_cast<T&&>(*this);
    }
    constexpr int count() const noexcept
    {
        return _count;
    }

private:

    int _count = 1;
};

struct quantity_format
{
    template <class T>
    using fn = stringify::v0::quantity_format_fn<T>;
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

