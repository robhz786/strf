#ifndef STRF_DETAIL_FORMAT_FUNCTIONS_HPP
#define STRF_DETAIL_FORMAT_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>
#include <cstring>

STRF_NAMESPACE_BEGIN

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
        typename strf::detail::fmt_replace_impl2
            < From, To, List, T...>::type;
};

template <typename FmtA, typename FmtB, typename ValueWithFormat>
struct fmt_forward_switcher
{
    template <typename FmtAInit>
    static const typename FmtB::template fn<ValueWithFormat>&
    f(const FmtAInit&, const ValueWithFormat& v)
    {
        return v;
    }

    template <typename FmtAInit>
    static typename FmtB::template fn<ValueWithFormat>&&
    f(const FmtAInit&, ValueWithFormat&& v)
    {
        return v;
    }
};

template <typename FmtA, typename ValueWithFormat>
struct fmt_forward_switcher<FmtA, FmtA, ValueWithFormat>
{
    template <typename FmtAInit>
    static constexpr FmtAInit&&
    f(std::remove_reference_t<FmtAInit>& fa,  const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }

    template <typename FmtAInit>
    static constexpr FmtAInit&&
    f(std::remove_reference_t<FmtAInit>&& fa, const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }
};


} // namespace detail

template <typename List, typename From, typename To>
using fmt_replace
    = typename strf::detail::fmt_replace_impl<From, List>
    ::template type_tmpl<To>;

template <typename ValueType, class ... Fmts>
class value_with_format;

template <typename ValueType, class ... Fmts>
class value_with_format
    : public Fmts::template fn<value_with_format<ValueType, Fmts ...>> ...
{
public:

    template <typename ... OhterFmts>
    using replace_fmts = strf::value_with_format<ValueType, OhterFmts ...>;

    constexpr value_with_format(const value_with_format&) = default;
    constexpr value_with_format(value_with_format&&) = default;

    explicit constexpr value_with_format(const ValueType& v)
        : _value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , const strf::value_with_format<OtherValueType, Fmts...>& f )
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
        , strf::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , _value(static_cast<ValueType&&>(v))
    {
    }

    template <typename ... F, typename ... FInit>
    constexpr value_with_format
        ( const ValueType& v
        , strf::tag<F...>
        , FInit&& ... finit )
        : F::template fn<value_with_format<ValueType, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , _value(v)
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( const strf::value_with_format<ValueType, OtherFmts...>& f )
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
        ( strf::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , _value(static_cast<ValueType&&>(f.value()))
    {
    }

    template <typename Fmt, typename FmtInit, typename ... OtherFmts>
    constexpr value_with_format
        ( const strf::value_with_format<ValueType, OtherFmts...>& f
        , strf::tag<Fmt>
        , FmtInit&& fmt_init )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( strf::detail::fmt_forward_switcher
                  < Fmt
                  , Fmts
                  , strf::value_with_format<ValueType, OtherFmts...> >
              :: template f<FmtInit>(fmt_init, f) )
            ...
        , _value(f.value())
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

enum class text_alignment {left, right, split, center};

struct alignment_format_data
{
    char32_t fill = U' ';
    std::int16_t width = 0;
    strf::text_alignment alignment = strf::text_alignment::right;
};

constexpr bool operator==( strf::alignment_format_data lhs
                         , strf::alignment_format_data rhs ) noexcept
{
    return lhs.fill == rhs.fill
        && lhs.width == rhs.width
        && lhs.alignment == rhs.alignment ;
}

constexpr bool operator!=( strf::alignment_format_data lhs
                         , strf::alignment_format_data rhs ) noexcept
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

    constexpr explicit alignment_format_fn
        ( strf::alignment_format_data data) noexcept
        : _data(data)
    {
    }

    template <bool B, typename U>
    constexpr explicit alignment_format_fn
        ( const strf::alignment_format_fn<B, U>& u ) noexcept
        : _data(u.get_alignment_format_data())
    {
    }

    constexpr T&& operator<(std::int16_t width) && noexcept
    {
        _data.alignment = strf::text_alignment::left;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator>(std::int16_t width) && noexcept
    {
        _data.alignment = strf::text_alignment::right;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator^(std::int16_t width) && noexcept
    {
        _data.alignment = strf::text_alignment::center;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& operator%(std::int16_t width) && noexcept
    {
        _data.alignment = strf::text_alignment::split;
        _data.width = width;
        return as_derived_rval_ref();
    }
    constexpr T&& fill(char32_t ch) && noexcept
    {
        _data.fill = ch;
        return as_derived_rval_ref();
    }
    constexpr std::int16_t width() const noexcept
    {
        return _data.width;
    }
    constexpr strf::text_alignment alignment() const noexcept
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

    strf::alignment_format_data _data;
};

template <class T>
class alignment_format_fn<false, T>
{
    using derived_type = T;
    using adapted_derived_type = strf::fmt_replace
            < T
            , strf::alignment_format_q<false>
            , strf::alignment_format_q<true> >;

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

    constexpr adapted_derived_type operator<(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::left } };
    }
    constexpr adapted_derived_type operator>(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::right } };
    }
    constexpr adapted_derived_type operator^(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::center } };
    }
    constexpr adapted_derived_type operator%(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::split } };
    }
    constexpr auto fill(char32_t ch) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ ch } };
    }

    constexpr std::int16_t width() const noexcept
    {
        return 0;
    }
    constexpr strf::text_alignment alignment() const noexcept
    {
        return strf::text_alignment::right;
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
    using fn = strf::alignment_format_fn<Active, T>;
};

using alignment_format = strf::alignment_format_q<true>;
using empty_alignment_format = strf::alignment_format_q<false>;


template <class T>
class quantity_format_fn
{
public:

    constexpr quantity_format_fn(std::size_t count) noexcept
        : _count(count)
    {
    }
    constexpr quantity_format_fn() noexcept = default;

    template <typename U>
    constexpr explicit quantity_format_fn(const quantity_format_fn<U>& u) noexcept
        : _count(u.count())
    {
    }

    constexpr T&& multi(std::size_t count) && noexcept
    {
        _count = count;
        return static_cast<T&&>(*this);
    }
    constexpr std::size_t count() const noexcept
    {
        return _count;
    }

private:

    std::size_t _count = 1;
};

struct quantity_format
{
    template <class T>
    using fn = strf::quantity_format_fn<T>;
};

template <typename T>
constexpr auto fmt(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(make_fmt(strf::rank<5>{}, value))>>
{
    return make_fmt(strf::rank<5>{}, value);
}

template <typename T>
constexpr auto hex(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).hex())>>
{
    return fmt(value).hex();
}

template <typename T>
constexpr auto dec(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).dec())>>
{
    return fmt(value).dec();
}

template <typename T>
constexpr auto oct(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).oct())>>
{
    return fmt(value).oct();
}

template <typename T>
constexpr auto left(const T& value, std::int16_t width)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value) < width)>>
{
    return fmt(value) < width;
}

template <typename T>
constexpr auto right(const T& value, std::int16_t width)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value) > width)>>
{
    return fmt(value) > width;
}

template <typename T>
constexpr auto split(const T& value, std::int16_t width)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value) % width)>>
{
    return fmt(value) % width;
}

template <typename T>
constexpr auto center(const T& value, std::int16_t width)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value) ^ width)>>
{
    return fmt(value) ^ width;
}

template <typename T>
constexpr auto left(const T& value, std::int16_t width, char32_t fill)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fill(fill) < width)>>
{
    return fmt(value).fill(fill) < width;
}

template <typename T>
constexpr auto right(const T& value, std::int16_t width, char32_t fill)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fill(fill) > width)>>
{
    return fmt(value).fill(fill) > width;
}

template <typename T>
constexpr auto split(const T& value, std::int16_t width, char32_t fill)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fill(fill) % width)>>
{
    return fmt(value).fill(fill) % width;
}

template <typename T>
constexpr auto center(const T& value, std::int16_t width, char32_t fill)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fill(fill) ^ width)>>
{
    return fmt(value).fill(fill) ^ width;
}

template <typename T, typename I>
constexpr auto multi(const T& value, I count)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).multi(count))>>
{
    return fmt(value).multi(count);
}

template <typename T>
constexpr auto fixed(const T& value) -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fixed())>>
{
    return fmt(value).fixed();
}

template <typename T>
constexpr auto sci(const T& value) -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).sci())>>
{
    return fmt(value).sci();
}

template <typename T, typename P>
constexpr auto fixed(const T& value, P precision)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).fixed().p(precision))>>
{
    return fmt(value).fixed().p(precision);
}

template <typename T, typename P>
constexpr auto sci(const T& value, P precision)
    -> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).sci().p(precision))>>
{
    return fmt(value).sci().p(precision);
}

template <typename T>
constexpr auto cv(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).cv())>>
{
    return fmt(value).convert_charset(); // defined in no_cv_format_fn
}

template <typename T, typename E>
constexpr auto cv(const T& value, const E& e)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).cv(e))>>
{
    return fmt(value).convert_charset(e);  // defined in no_cv_format_fn
}

template <typename T>
constexpr auto sani(const T& value)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).sani())>>
{
    return fmt(value).sanitize_charset();  // defined in no_cv_format_fn
}

template <typename T, typename E>
constexpr auto sani(const T& value, const E& e)
-> std::remove_cv_t<std::remove_reference_t<decltype(fmt(value).sani(e))>>
{
    return fmt(value).sanitize_charset(e);  // defined in no_cv_format_fn
}


STRF_NAMESPACE_END

#endif  // STRF_DETAIL_FORMAT_FUNCTIONS_HPP

