#ifndef STRF_DETAIL_FORMAT_FUNCTIONS_HPP
#define STRF_DETAIL_FORMAT_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>
#include <cstring>


namespace strf {

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
    static STRF_HD const typename FmtB::template fn<ValueWithFormat>&
    f(const FmtAInit&, const ValueWithFormat& v)
    {
        return v;
    }

    template <typename FmtAInit>
    static STRF_HD typename FmtB::template fn<ValueWithFormat>&&
    f(const FmtAInit&, ValueWithFormat&& v)
    {
        return v;
    }
};

template <typename FmtA, typename ValueWithFormat>
struct fmt_forward_switcher<FmtA, FmtA, ValueWithFormat>
{
    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
    f(std::remove_reference_t<FmtAInit>& fa,  const ValueWithFormat&)
    {
        return static_cast<FmtAInit&&>(fa);
    }

    template <typename FmtAInit>
    static constexpr STRF_HD FmtAInit&&
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

    using value_type = ValueType;

    template <typename ... OhterFmts>
    using replace_fmts = strf::value_with_format<ValueType, OhterFmts ...>;

    explicit constexpr STRF_HD value_with_format(const ValueType& v)
        : value_(v)
    {
    }

    template <typename OtherValueType>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , const strf::value_with_format<OtherValueType, Fmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherValueType>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , strf::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , value_(static_cast<ValueType&&>(v))
    {
    }

    template <typename ... F, typename ... FInit>
    constexpr STRF_HD value_with_format
        ( const ValueType& v
        , strf::tag<F...>
        , FInit&& ... finit )
        : F::template fn<value_with_format<ValueType, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD value_with_format
        ( const strf::value_with_format<ValueType, OtherFmts...>& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD value_with_format
        ( strf::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , value_(static_cast<ValueType&&>(f.value()))
    {
    }

    template <typename Fmt, typename FmtInit, typename ... OtherFmts>
    constexpr STRF_HD value_with_format
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
        , value_(f.value())
    {
    }

    constexpr STRF_HD const ValueType& value() const
    {
        return value_;
    }

    constexpr STRF_HD ValueType& value()
    {
        return value_;
    }

private:

    ValueType value_;
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

constexpr STRF_HD bool operator==( strf::alignment_format_data lhs
                                 , strf::alignment_format_data rhs ) noexcept
{
    return lhs.fill == rhs.fill
        && lhs.width == rhs.width
        && lhs.alignment == rhs.alignment ;
}

constexpr STRF_HD bool operator!=( strf::alignment_format_data lhs
                                 , strf::alignment_format_data rhs ) noexcept
{
    return ! (lhs == rhs);
}

template <bool Active, class T>
class alignment_format_fn
{
    STRF_HD T& as_derived_ref()
    {
        T* d =  static_cast<T*>(this);
        return *d;
    }

    STRF_HD T&& as_derived_rval_ref()
    {
        T* d =  static_cast<T*>(this);
        return static_cast<T&&>(*d);
    }

public:

    constexpr STRF_HD alignment_format_fn() noexcept
    {
    }

    constexpr STRF_HD explicit alignment_format_fn
        ( strf::alignment_format_data data) noexcept
        : data_(data)
    {
    }

    template <bool B, typename U>
    constexpr STRF_HD explicit alignment_format_fn
        ( const strf::alignment_format_fn<B, U>& u ) noexcept
        : data_(u.get_alignment_format_data())
    {
    }

    constexpr STRF_HD T&& operator<(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::left;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator>(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::right;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator^(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::center;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& operator%(std::int16_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::split;
        data_.width = width;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& fill(char32_t ch) && noexcept
    {
        data_.fill = ch;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD T&& set(alignment_format_data data) && noexcept
    {
        data_ = data;
        return as_derived_rval_ref();
    }
    constexpr STRF_HD std::int16_t width() const noexcept
    {
        return data_.width;
    }
    constexpr STRF_HD strf::text_alignment alignment() const noexcept
    {
        return data_.alignment;
    }
    constexpr STRF_HD char32_t fill() const noexcept
    {
        return data_.fill;
    }

    constexpr STRF_HD alignment_format_data get_alignment_format_data() const noexcept
    {
        return data_;
    }

private:

    strf::alignment_format_data data_;
};

template <class T>
class alignment_format_fn<false, T>
{
    using derived_type = T;
    using adapted_derived_type = strf::fmt_replace
            < T
            , strf::alignment_format_q<false>
            , strf::alignment_format_q<true> >;

    constexpr STRF_HD adapted_derived_type make_adapted() const
    {
        return adapted_derived_type{static_cast<const T&>(*this)};
    }

public:

    constexpr STRF_HD alignment_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit alignment_format_fn(const alignment_format_fn<false, U>&) noexcept
    {
    }

    constexpr STRF_HD adapted_derived_type operator<(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::left } };
    }
    constexpr STRF_HD adapted_derived_type operator>(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::right } };
    }
    constexpr STRF_HD adapted_derived_type operator^(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::center } };
    }
    constexpr STRF_HD adapted_derived_type operator%(std::int16_t width) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ U' '
                                         , width
                                         , strf::text_alignment::split } };
    }
    constexpr STRF_HD auto fill(char32_t ch) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<alignment_format_q<true>>{}
            , strf::alignment_format_data{ ch } };
    }
    constexpr STRF_HD auto set(strf::alignment_format_data data) const noexcept
    {
        return adapted_derived_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::alignment_format_q<true>>{}
            , data };
    }
    constexpr STRF_HD std::int16_t width() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD strf::text_alignment alignment() const noexcept
    {
        return strf::text_alignment::right;
    }
    constexpr STRF_HD char32_t fill() const noexcept
    {
        return U' ';
    }
    constexpr STRF_HD alignment_format_data get_alignment_format_data() const noexcept
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

    constexpr STRF_HD quantity_format_fn(std::size_t count) noexcept
        : count_(count)
    {
    }

    constexpr STRF_HD quantity_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit quantity_format_fn(const quantity_format_fn<U>& u) noexcept
        : count_(u.count())
    {
    }

    constexpr STRF_HD T&& multi(std::size_t count) && noexcept
    {
        count_ = count;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD std::size_t count() const noexcept
    {
        return count_;
    }

private:

    std::size_t count_ = 1;
};

struct quantity_format
{
    template <class T>
    using fn = strf::quantity_format_fn<T>;
};

template <typename>
inline void get_fmt_traits() {}

template <typename T>
struct fmt_traits
    : decltype(get_fmt_traits(strf::tag<>{}, std::declval<T>()))
{
};

template <typename FmtType>
struct make_fmt_traits
{
    using fmt_type = FmtType;
};

template <typename T>
using fmt_type = typename fmt_traits<T>::fmt_type;

inline namespace format_functions {

template <typename T>
constexpr STRF_HD strf::fmt_type<T> fmt(const T& value)
{
    using fmt_value_type = typename strf::fmt_type<T>::value_type;
    return strf::fmt_type<T>{fmt_value_type{value}};
}

template <typename T>
constexpr STRF_HD auto hex(const T& value)
{
    return fmt(value).hex();
}

template <typename T>
constexpr STRF_HD auto dec(const T& value)
{
    return fmt(value).dec();
}

template <typename T>
constexpr STRF_HD auto oct(const T& value)
{
    return fmt(value).oct();
}

template <typename T>
constexpr STRF_HD auto bin(const T& value)
{
    return fmt(value).bin();
}

template <typename T>
constexpr STRF_HD auto left(const T& value, std::int16_t width)
{
    return fmt(value) < width;
}

template <typename T>
constexpr STRF_HD auto right(const T& value, std::int16_t width)
{
    return fmt(value) > width;
}

template <typename T>
constexpr STRF_HD auto split(const T& value, std::int16_t width)
{
    return fmt(value) % width;
}

template <typename T>
constexpr STRF_HD auto center(const T& value, std::int16_t width)
{
    return fmt(value) ^ width;
}

template <typename T>
constexpr STRF_HD auto left(const T& value, std::int16_t width, char32_t fill)
{
    return fmt(value).fill(fill) < width;
}

template <typename T>
constexpr STRF_HD auto right(const T& value, std::int16_t width, char32_t fill)
{
    return fmt(value).fill(fill) > width;
}

template <typename T>
constexpr STRF_HD auto split(const T& value, std::int16_t width, char32_t fill)
{
    return fmt(value).fill(fill) % width;
}

template <typename T>
constexpr STRF_HD auto center(const T& value, std::int16_t width, char32_t fill)
{
    return fmt(value).fill(fill) ^ width;
}

template <typename T, typename I>
constexpr STRF_HD auto multi(const T& value, I count)
{
    return fmt(value).multi(count);
}

template <typename T>
constexpr STRF_HD auto fixed(const T& value) 
{
    return fmt(value).fixed();
}

template <typename T>
constexpr STRF_HD auto sci(const T& value)
{
    return fmt(value).sci();
}

template <typename T, typename P>
constexpr STRF_HD auto fixed(const T& value, P precision)
{
    return fmt(value).fixed().p(precision);
}

template <typename T, typename P>
constexpr STRF_HD auto sci(const T& value, P precision)
{
    return fmt(value).sci().p(precision);
}

template <typename T>
constexpr STRF_HD auto conv(const T& value)
{
    return fmt(value).convert_encoding(); // defined in no_conv_format_fn
}

template <typename T, typename E>
constexpr STRF_HD auto conv(const T& value, const E& e)
{
    return fmt(value).convert_encoding(e);  // defined in no_conv_format_fn
}

template <typename T>
constexpr STRF_HD auto sani(const T& value)
{
    return fmt(value).sanitize_encoding();  // defined in no_conv_format_fn

}
template <typename T, typename E>
constexpr STRF_HD auto sani(const T& value, const E& e)
{
    return fmt(value).sanitize_encoding(e);  // defined in no_conv_format_fn
}

} // inline namespace format_functions
} // namespace strf


#endif  // STRF_DETAIL_FORMAT_FUNCTIONS_HPP

