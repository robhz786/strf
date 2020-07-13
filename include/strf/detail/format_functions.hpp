#ifndef STRF_DETAIL_FORMAT_FUNCTIONS_HPP
#define STRF_DETAIL_FORMAT_FUNCTIONS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/standard_lib_functions.hpp> // strf::detail::tag_invoke

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

template <bool HasAlignment>
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

template <class T, bool HasAlignment>
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

    template <typename U, bool B>
    constexpr STRF_HD explicit alignment_format_fn
        ( const strf::alignment_format_fn<U, B>& u ) noexcept
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
class alignment_format_fn<T, false>
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
    constexpr STRF_HD explicit alignment_format_fn(const alignment_format_fn<U, false>&) noexcept
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

template <bool HasAlignment>
struct alignment_format_q
{
    template <class T>
    using fn = strf::alignment_format_fn<T, HasAlignment>;
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

struct fmt_tag {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::detail::tag_invoke(*(const fmt_tag*)0, value)))
        -> decltype(strf::detail::tag_invoke(*(const fmt_tag*)0, value))
    {
        return strf::detail::tag_invoke(*this, value);
    }
};

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

inline namespace format_functions {

template <typename T>
constexpr STRF_HD auto fmt(T&& value)
    noexcept(noexcept(strf::detail::tag_invoke(strf::fmt_tag{}, value)))
    -> decltype(strf::detail::tag_invoke(strf::fmt_tag{}, value))
{
    return strf::detail::tag_invoke(strf::fmt_tag{}, value);
}

template <typename T>
constexpr STRF_HD auto hex(T&& value)
    noexcept(noexcept(strf::fmt(value).hex()))
    -> std::remove_reference_t<decltype(strf::fmt(value).hex())>
{
    return strf::fmt(value).hex();
}

template <typename T>
constexpr STRF_HD auto dec(T&& value)
    noexcept(noexcept(strf::fmt(value).dec()))
    -> std::remove_reference_t<decltype(strf::fmt(value).dec())>
{
    return strf::fmt(value).dec();
}

template <typename T>
constexpr STRF_HD auto oct(T&& value)
    noexcept(noexcept(strf::fmt(value).oct()))
    -> std::remove_reference_t<decltype(strf::fmt(value).oct())>
{
    return strf::fmt(value).oct();
}

template <typename T>
constexpr STRF_HD auto bin(T&& value)
    noexcept(noexcept(strf::fmt(value).bin()))
    -> std::remove_reference_t<decltype(strf::fmt(value).bin())>
{
    return strf::fmt(value).bin();
}

template <typename T>
constexpr STRF_HD auto fixed(T&& value)
    noexcept(noexcept(strf::fmt(value).fixed()))
    -> std::remove_reference_t<decltype(strf::fmt(value).fixed())>
{
    return strf::fmt(value).fixed();
}

template <typename T>
    constexpr STRF_HD auto fixed(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).fixed().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).fixed().p(precision))>
{
    return strf::fmt(value).fixed().p(precision);
}

template <typename T>
constexpr STRF_HD auto sci(T&& value)
    noexcept(noexcept(strf::fmt(value).sci()))
    -> std::remove_reference_t<decltype(strf::fmt(value).sci())>
{
    return strf::fmt(value).sci();
}

template <typename T>
constexpr STRF_HD auto sci(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).sci().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).sci().p(precision))>
{
    return strf::fmt(value).sci().p(precision);
}

template <typename T>
constexpr STRF_HD auto gen(T&& value)
    noexcept(noexcept(strf::fmt(value).gen()))
    -> std::remove_reference_t<decltype(strf::fmt(value).gen())>
{
    return strf::fmt(value).gen();
}

template <typename T>
constexpr STRF_HD auto gen(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt(value).gen().p(precision)))
    -> std::remove_reference_t<decltype(strf::fmt(value).gen().p(precision))>
{
    return strf::fmt(value).gen().p(precision);
}

template <typename T, typename C>
constexpr STRF_HD auto multi(T&& value, C&& count)
    noexcept(noexcept(strf::fmt(value).multi(count)))
    -> std::remove_reference_t<decltype(strf::fmt(value).multi(count))>
{
    return strf::fmt(value).multi(count);
}

template <typename T>
constexpr STRF_HD auto conv(T&& value)
    noexcept(noexcept(strf::fmt(value).convert_encoding()))
    -> std::remove_reference_t<decltype(strf::fmt(value).convert_encoding())>
{
    return strf::fmt(value).convert_encoding();
}
template <typename T, typename E>
    constexpr STRF_HD auto conv(T&& value, E&& enc)
    noexcept(noexcept(strf::fmt(value).convert_from_encoding(enc)))
    -> std::remove_reference_t<decltype(strf::fmt(value).convert_from_encoding(enc))>
{
    return strf::fmt(value).convert_from_encoding(enc);
}

template <typename T>
constexpr STRF_HD auto sani(T&& value)
    noexcept(noexcept(strf::fmt(value).sanitize_encoding()))
    -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_encoding())>
{
    return strf::fmt(value).sanitize_encoding();
}
template <typename T, typename E>
    constexpr STRF_HD auto sani(T&& value, E&& enc)
    noexcept(noexcept(strf::fmt(value).sanitize_from_encoding(enc)))
    -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_from_encoding(enc))>
{
    return strf::fmt(value).sanitize_from_encoding(enc);
}

template <typename T>
constexpr STRF_HD auto right(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) > width))
    -> std::remove_reference_t<decltype(strf::fmt(value) > width)>
{
    return strf::fmt(value) > width;
}

template <typename T>
constexpr STRF_HD auto right(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) > width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) > width)>
{
    return strf::fmt(value).fill(fill) > width;
}

template <typename T>
constexpr STRF_HD auto left(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) < width))
    -> std::remove_reference_t<decltype(strf::fmt(value) < width)>
{
    return strf::fmt(value) < width;
}
    
template <typename T>
constexpr STRF_HD auto left(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) < width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) < width)>
{
    return strf::fmt(value).fill(fill) < width;
}

template <typename T>
constexpr STRF_HD auto center(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) ^ width))
    -> std::remove_reference_t<decltype(strf::fmt(value) ^ width)>
{
    return strf::fmt(value) ^ width;
}
template <typename T>
constexpr STRF_HD auto center(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) ^ width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) ^ width)>
{
    return strf::fmt(value).fill(fill) ^ width;
}

template <typename T>
constexpr STRF_HD auto split(T&& value, std::int16_t width)
    noexcept(noexcept(strf::fmt(value) % width))
    -> std::remove_reference_t<decltype(strf::fmt(value) % width)>
{
    return strf::fmt(value) % width;
}

template <typename T>
constexpr STRF_HD auto split(T&& value, std::int16_t width, char32_t fill)
    noexcept(noexcept(strf::fmt(value).fill(fill) % width))
    -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) % width)>
{
    return strf::fmt(value).fill(fill) % width;
}
       
} // inline namespace format_functions

#else  // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

inline namespace format_functions {
constexpr fmt_tag fmt {};
} // inline namespace format_functions

namespace detail_format_functions {

struct hex_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).hex()))
        -> std::remove_reference_t<decltype(strf::fmt(value).hex())>
    {
        return strf::fmt(value).hex();
    }
};

struct dec_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).dec()))
        -> std::remove_reference_t<decltype(strf::fmt(value).dec())>
    {
        return strf::fmt(value).dec();
    }
};

struct oct_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).oct()))
        -> std::remove_reference_t<decltype(strf::fmt(value).oct())>
    {
        return strf::fmt(value).oct();
    }
};

struct bin_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).bin()))
        -> std::remove_reference_t<decltype(strf::fmt(value).bin())>
    {
        return strf::fmt(value).bin();
    }
};

struct fixed_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).fixed()))
        -> std::remove_reference_t<decltype(strf::fmt(value).fixed())>
    {
        return strf::fmt(value).fixed();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).fixed().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).fixed().p(precision))>
    {
        return strf::fmt(value).fixed().p(precision);
    }
};

struct sci_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).sci()))
        -> std::remove_reference_t<decltype(strf::fmt(value).sci())>
    {
        return strf::fmt(value).sci();
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).sci().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).sci().p(precision))>
    {
        return strf::fmt(value).sci().p(precision);
    }
};

struct gen_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).gen()))
        -> std::remove_reference_t<decltype(strf::fmt(value).gen())>
    {
        return strf::fmt(value).gen();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt(value).gen().p(precision)))
        -> std::remove_reference_t<decltype(strf::fmt(value).gen().p(precision))>
    {
        return strf::fmt(value).gen().p(precision);
    }
};

struct multi_fn {
    template <typename T, typename C>
    constexpr STRF_HD auto operator()(T&& value, C&& count) const
        noexcept(noexcept(strf::fmt(value).multi(count)))
        -> std::remove_reference_t<decltype(strf::fmt(value).multi(count))>
    {
        return strf::fmt(value).multi(count);
    }
};

struct conv_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).convert_encoding()))
        -> std::remove_reference_t<decltype(strf::fmt(value).convert_encoding())>
    {
        return strf::fmt(value).convert_encoding();
    }
    template <typename T, typename E>
        constexpr STRF_HD auto operator()(T&& value, E&& enc) const
        noexcept(noexcept(strf::fmt(value).convert_from_encoding(enc)))
        -> std::remove_reference_t<decltype(strf::fmt(value).convert_from_encoding(enc))>
    {
        return strf::fmt(value).convert_from_encoding(enc);
    }
};

struct sani_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt(value).sanitize_encoding()))
        -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_encoding())>
    {
        return strf::fmt(value).sanitize_encoding();
    }
    template <typename T, typename E>
    constexpr STRF_HD auto operator()(T&& value, E&& enc) const
        noexcept(noexcept(strf::fmt(value).sanitize_from_encoding(enc)))
        -> std::remove_reference_t<decltype(strf::fmt(value).sanitize_from_encoding(enc))>
    {
        return strf::fmt(value).sanitize_from_encoding(enc);
    }
};

struct right_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) > width))
        -> std::remove_reference_t<decltype(strf::fmt(value) > width)>
    {
        return strf::fmt(value) > width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) > width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) > width)>
    {
        return strf::fmt(value).fill(fill) > width;
    }
};

struct left_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) < width))
        -> std::remove_reference_t<decltype(strf::fmt(value) < width)>
    {
        return strf::fmt(value) < width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) < width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) < width)>
    {
        return strf::fmt(value).fill(fill) < width;
    }
};

struct center_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) ^ width))
        -> std::remove_reference_t<decltype(strf::fmt(value) ^ width)>
    {
        return strf::fmt(value) ^ width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) ^ width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) ^ width)>
    {
        return strf::fmt(value).fill(fill) ^ width;
    }
};

struct split_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width) const
        noexcept(noexcept(strf::fmt(value) % width))
        -> std::remove_reference_t<decltype(strf::fmt(value) % width)>
    {
        return strf::fmt(value) % width;
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, std::int16_t width, char32_t fill) const
        noexcept(noexcept(strf::fmt(value).fill(fill) % width))
        -> std::remove_reference_t<decltype(strf::fmt(value).fill(fill) % width)>
    {
        return strf::fmt(value).fill(fill) % width;
    }
};

} // namespace detail_format_functions

inline namespace format_functions {

constexpr strf::detail_format_functions::hex_fn    hex {};
constexpr strf::detail_format_functions::dec_fn    dec {};
constexpr strf::detail_format_functions::oct_fn    oct {};
constexpr strf::detail_format_functions::bin_fn    bin {};
constexpr strf::detail_format_functions::fixed_fn  fixed {};
constexpr strf::detail_format_functions::sci_fn    sci {};
constexpr strf::detail_format_functions::gen_fn    gen {};
constexpr strf::detail_format_functions::multi_fn  multi {};
constexpr strf::detail_format_functions::conv_fn   conv {};
constexpr strf::detail_format_functions::sani_fn   sani {};
constexpr strf::detail_format_functions::right_fn  right {};
constexpr strf::detail_format_functions::left_fn   left {};
constexpr strf::detail_format_functions::center_fn center {};
constexpr strf::detail_format_functions::split_fn  split {};

} // inline namespace format_functions

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename T>
using fmt_type = std::remove_cv_t<std::remove_reference_t<decltype(strf::fmt(std::declval<T>()))>>;

} // namespace strf

#endif  // STRF_DETAIL_FORMAT_FUNCTIONS_HPP

