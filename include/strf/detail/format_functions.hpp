#ifndef STRF_DETAIL_FORMAT_FUNCTIONS_HPP
#define STRF_DETAIL_FORMAT_FUNCTIONS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>

namespace strf {

enum class showsign {negative_only = 0, positive_also = '+', fill_instead_of_positive = ' '};

template <bool HasAlignment>
struct alignment_formatter_q;

enum class text_alignment {left, right, center};

struct alignment_format
{

#if __cplusplus < 201402L

    constexpr STRF_HD alignment_format
        ( char32_t fill_ = U' '
        , strf::width_t width_ = 0
        , strf::text_alignment alignment_ = strf::text_alignment::right ) noexcept
        : fill(fill_)
        , width(width_)
        , alignment(alignment_)
    {
    }

#endif // __cplusplus < 201402L

    char32_t fill = U' ';
    strf::width_t width = 0;
    strf::text_alignment alignment = strf::text_alignment::right;
};

struct default_alignment_format
{
    static constexpr char32_t fill = U' ';
    static constexpr strf::width_t width = 0;
    static constexpr strf::text_alignment alignment = strf::text_alignment::right;

    constexpr STRF_HD operator strf::alignment_format () const noexcept
    {
        return {};
    }
};

template <class T, bool HasAlignment>
class alignment_formatter_fn;

template <class T>
class alignment_formatter_fn<T, true>
{
    STRF_HD T&& move_self_downcast_()
    {
        T* d =  static_cast<T*>(this);
        return static_cast<T&&>(*d);
    }

public:

    constexpr STRF_HD alignment_formatter_fn() noexcept
    {
    }

    constexpr STRF_HD explicit alignment_formatter_fn
        ( strf::alignment_format data) noexcept
        : data_(data)
    {
    }

    template <typename U, bool B>
    constexpr STRF_HD explicit alignment_formatter_fn
        ( const strf::alignment_formatter_fn<U, B>& u ) noexcept
        : data_(u.get_alignment_format())
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator<(strf::width_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::left;
        data_.width = width;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator>(strf::width_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::right;
        data_.width = width;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator^(strf::width_t width) && noexcept
    {
        data_.alignment = strf::text_alignment::center;
        data_.width = width;
        return move_self_downcast_();
    }
    template < typename CharT >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill(CharT ch) && noexcept
    {
        static_assert( strf::is_char<CharT>::value // issue #19
                     , "Refusing non-char argument to set the fill character, "
                       "since one may pass 0 instead of '0' by accident." );
        data_.fill = ch;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_alignment_format(strf::alignment_format data) && noexcept
    {
        data_ = data;
        return move_self_downcast_();
    }
    constexpr STRF_HD strf::width_t width() const noexcept
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

    constexpr STRF_HD alignment_format get_alignment_format() const noexcept
    {
        return data_;
    }

private:

    strf::alignment_format data_;
};

template <class T>
class alignment_formatter_fn<T, false>
{
    using derived_type = T;
    using adapted_derived_type = strf::fmt_replace
            < T
            , strf::alignment_formatter_q<false>
            , strf::alignment_formatter_q<true> >;

public:

    constexpr STRF_HD alignment_formatter_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit alignment_formatter_fn(const alignment_formatter_fn<U, false>&) noexcept
    {
    }

    constexpr STRF_HD adapted_derived_type operator<(strf::width_t width) const noexcept
    {
        return adapted_derived_type
            { self_downcast_()
            , strf::tag<alignment_formatter_q<true>>{}
            , strf::alignment_format{U' ', width, strf::text_alignment::left} };
    }
    constexpr STRF_HD adapted_derived_type operator>(strf::width_t width) const noexcept
    {
        return adapted_derived_type
            { self_downcast_()
            , strf::tag<alignment_formatter_q<true>>{}
            , strf::alignment_format{U' ', width, strf::text_alignment::right} };
    }
    constexpr STRF_HD adapted_derived_type operator^(strf::width_t width) const noexcept
    {
        return adapted_derived_type
            { self_downcast_()
            , strf::tag<alignment_formatter_q<true>>{}
            , strf::alignment_format{U' ', width, strf::text_alignment::center} };
    }
    template <typename CharT>
    constexpr STRF_HD adapted_derived_type fill(CharT ch) const noexcept
    {
        static_assert( strf::is_char<CharT>::value // issue #19
                     , "Refusing non-char argument to set the fill character, "
                       "since one may pass 0 instead of '0' by accident." );
        return adapted_derived_type
            { self_downcast_()
            , strf::tag<alignment_formatter_q<true>>{}
            , strf::alignment_format{static_cast<char32_t>(ch)} };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    T&& set_alignment_format(strf::default_alignment_format) && noexcept
    {
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&  set_alignment_format(strf::default_alignment_format) & noexcept
    {
        return self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD const T& set_alignment_format(strf::default_alignment_format) const & noexcept
    {
        return self_downcast_();
    }
    constexpr STRF_HD adapted_derived_type set_alignment_format(strf::alignment_format data) const & noexcept
    {
        return adapted_derived_type
            { self_downcast_()
            , strf::tag<strf::alignment_formatter_q<true>>{}
            , data };
    }
    constexpr static STRF_HD strf::default_alignment_format get_alignment_format() noexcept
    {
        return {};
    }
    constexpr STRF_HD strf::width_t width() const noexcept
    {
        return 0;//get_alignment_format().width;
    }
    constexpr STRF_HD strf::text_alignment alignment() const noexcept
    {
        return get_alignment_format().alignment;
    }
    constexpr STRF_HD char32_t fill() const noexcept
    {
        return get_alignment_format().fill;
    }

private:

    STRF_HD constexpr const T& self_downcast_() const
    {
        //const T* base_ptr = static_cast<const T*>(this);
        return *static_cast<const T*>(this);
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
};

template <bool HasAlignment>
struct alignment_formatter_q
{
    template <class T>
    using fn = strf::alignment_formatter_fn<T, HasAlignment>;
};

using dynamic_alignment_formatter = strf::alignment_formatter_q<true>;
using alignment_formatter = strf::alignment_formatter_q<false>;


template <class T>
class quantity_formatter_fn
{
public:

    constexpr STRF_HD explicit quantity_formatter_fn(std::size_t count) noexcept
        : count_(count)
    {
    }

    constexpr STRF_HD quantity_formatter_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit quantity_formatter_fn(const quantity_formatter_fn<U>& u) noexcept
        : count_(u.count())
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& multi(std::size_t count) && noexcept
    {
        count_ = count;
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
    constexpr STRF_HD std::size_t count() const noexcept
    {
        return count_;
    }

private:

    std::size_t count_ = 1;
};

struct quantity_formatter
{
    template <class T>
    using fn = strf::quantity_formatter_fn<T>;
};


inline namespace format_functions {

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename T>
constexpr STRF_HD auto hex(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).hex()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).hex())>
{
    return strf::fmt((T&&)value).hex();
}

template <typename T>
constexpr STRF_HD auto dec(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).dec()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).dec())>
{
    return strf::fmt((T&&)value).dec();
}

template <typename T>
constexpr STRF_HD auto oct(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).oct()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).oct())>
{
    return strf::fmt((T&&)value).oct();
}

template <typename T>
constexpr STRF_HD auto bin(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).bin()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).bin())>
{
    return strf::fmt((T&&)value).bin();
}

template <typename T>
constexpr STRF_HD auto fixed(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).fixed()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fixed())>
{
    return strf::fmt((T&&)value).fixed();
}

template <typename T>
    constexpr STRF_HD auto fixed(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt((T&&)value).fixed().p(precision)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fixed().p(precision))>
{
    return strf::fmt((T&&)value).fixed().p(precision);
}

template <typename T>
constexpr STRF_HD auto sci(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).sci()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sci())>
{
    return strf::fmt((T&&)value).sci();
}

template <typename T>
constexpr STRF_HD auto sci(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt((T&&)value).sci().p(precision)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sci().p(precision))>
{
    return strf::fmt((T&&)value).sci().p(precision);
}

template <typename T>
constexpr STRF_HD auto gen(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).gen()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).gen())>
{
    return strf::fmt((T&&)value).gen();
}

template <typename T>
constexpr STRF_HD auto gen(T&& value, unsigned precision)
    noexcept(noexcept(strf::fmt((T&&)value).gen().p(precision)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).gen().p(precision))>
{
    return strf::fmt((T&&)value).gen().p(precision);
}

template <typename T, typename C>
constexpr STRF_HD auto multi(T&& value, C&& count)
    noexcept(noexcept(strf::fmt((T&&)value).multi(count)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).multi(count))>
{
    return strf::fmt((T&&)value).multi(count);
}

template <typename T>
constexpr STRF_HD auto transcode(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).transcode()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode())>
{
    return strf::fmt((T&&)value).transcode();
}

template <typename T, typename Charset>
    constexpr STRF_HD auto transcode(T&& value, Charset&& charset)
    noexcept(noexcept(strf::fmt((T&&)value).transcode_from(charset)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode_from(charset))>
{
    return strf::fmt((T&&)value).transcode_from(charset);
}

template <typename T>
constexpr STRF_HD auto conv(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).transcode()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode())>
{
    return strf::fmt((T&&)value).transcode();
}

template <typename T, typename Charset>
    constexpr STRF_HD auto conv(T&& value, Charset&& charset)
    noexcept(noexcept(strf::fmt((T&&)value).transcode_from(charset)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode_from(charset))>
{
    return strf::fmt((T&&)value).transcode_from(charset);
}

template <typename T>
constexpr STRF_HD auto sani(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).sanitize_charset()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sanitize_charset())>
{
    return strf::fmt((T&&)value).sanitize_charset();
}

template <typename T, typename Charset>
    constexpr STRF_HD auto sani(T&& value, Charset&& charset)
    noexcept(noexcept(strf::fmt((T&&)value).sanitize_from_charset(charset)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sanitize_from_charset(charset))>
{
    return strf::fmt((T&&)value).sanitize_from_charset(charset);
}

template <typename T>
constexpr STRF_HD auto right(T&& value, strf::width_t width)
    noexcept(noexcept(strf::fmt((T&&)value) > width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) > width)>
{
    return strf::fmt((T&&)value) > width;
}

template <typename T, typename CharT>
constexpr STRF_HD auto right(T&& value, strf::width_t width, CharT fill)
    noexcept(noexcept(strf::fmt((T&&)value).fill(fill) > width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) > width)>
{
    return strf::fmt((T&&)value).fill(fill) > width;
}

template <typename T>
constexpr STRF_HD auto left(T&& value, strf::width_t width)
    noexcept(noexcept(strf::fmt((T&&)value) < width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) < width)>
{
    return strf::fmt((T&&)value) < width;
}

template <typename T, typename CharT>
constexpr STRF_HD auto left(T&& value, strf::width_t width, CharT fill)
    noexcept(noexcept(strf::fmt((T&&)value).fill(fill) < width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) < width)>
{
    return strf::fmt((T&&)value).fill(fill) < width;
}

template <typename T>
constexpr STRF_HD auto center(T&& value, strf::width_t width)
    noexcept(noexcept(strf::fmt((T&&)value) ^ width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) ^ width)>
{
    return strf::fmt((T&&)value) ^ width;
}

template <typename T, typename CharT>
constexpr STRF_HD auto center(T&& value, strf::width_t width, CharT fill)
    noexcept(noexcept(strf::fmt((T&&)value).fill(fill) ^ width))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) ^ width)>
{
    return strf::fmt((T&&)value).fill(fill) ^ width;
}

template <typename T>
constexpr STRF_HD auto pad0(T&& value, decltype(strf::fmt((T&&)value).pad0width()) width)
    noexcept(noexcept(strf::fmt((T&&)value).pad0(width)))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).pad0(width))>
{
    return strf::fmt((T&&)value).pad0(width);
}

template <typename T>
constexpr STRF_HD auto punct(T&& value)
    noexcept(noexcept(strf::fmt((T&&)value).punct()))
    -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).punct())>
{
    return strf::fmt((T&&)value).punct();
}

#else  // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

namespace detail_format_functions {

struct hex_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).hex()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).hex())>
    {
        return strf::fmt((T&&)value).hex();
    }
};

struct dec_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).dec()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).dec())>
    {
        return strf::fmt((T&&)value).dec();
    }
};

struct oct_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).oct()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).oct())>
    {
        return strf::fmt((T&&)value).oct();
    }
};

struct bin_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).bin()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).bin())>
    {
        return strf::fmt((T&&)value).bin();
    }
};

struct fixed_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).fixed()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fixed())>
    {
        return strf::fmt((T&&)value).fixed();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt((T&&)value).fixed().p(precision)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fixed().p(precision))>
    {
        return strf::fmt((T&&)value).fixed().p(precision);
    }
};

struct sci_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).sci()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sci())>
    {
        return strf::fmt((T&&)value).sci();
    }
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt((T&&)value).sci().p(precision)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sci().p(precision))>
    {
        return strf::fmt((T&&)value).sci().p(precision);
    }
};

struct gen_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).gen()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).gen())>
    {
        return strf::fmt((T&&)value).gen();
    }
    template <typename T>
        constexpr STRF_HD auto operator()(T&& value, unsigned precision) const
        noexcept(noexcept(strf::fmt((T&&)value).gen().p(precision)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).gen().p(precision))>
    {
        return strf::fmt((T&&)value).gen().p(precision);
    }
};

struct multi_fn {
    template <typename T, typename C>
    constexpr STRF_HD auto operator()(T&& value, C&& count) const
        noexcept(noexcept(strf::fmt((T&&)value).multi(count)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).multi(count))>
    {
        return strf::fmt((T&&)value).multi(count);
    }
};

struct transcode_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).transcode()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode())>
    {
        return strf::fmt((T&&)value).transcode();
    }
    template <typename T, typename Charset>
        constexpr STRF_HD auto operator()(T&& value, Charset&& charset) const
        noexcept(noexcept(strf::fmt((T&&)value).transcode_from(charset)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).transcode_from(charset))>
    {
        return strf::fmt((T&&)value).transcode_from(charset);
    }
};

struct sani_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value) const
        noexcept(noexcept(strf::fmt((T&&)value).sanitize_charset()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sanitize_charset())>
    {
        return strf::fmt((T&&)value).sanitize_charset();
    }
    template <typename T, typename Charset>
    constexpr STRF_HD auto operator()(T&& value, Charset&& charset) const
        noexcept(noexcept(strf::fmt((T&&)value).sanitize_from_charset(charset)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).sanitize_from_charset(charset))>
    {
        return strf::fmt((T&&)value).sanitize_from_charset(charset);
    }
};

struct right_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width) const
        noexcept(noexcept(strf::fmt((T&&)value) > width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) > width)>
    {
        return strf::fmt((T&&)value) > width;
    }
    template <typename T, typename CharT>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width, CharT fill) const
        noexcept(noexcept(strf::fmt((T&&)value).fill(fill) > width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) > width)>
    {
        return strf::fmt((T&&)value).fill(fill) > width;
    }
};

struct left_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width) const
        noexcept(noexcept(strf::fmt((T&&)value) < width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) < width)>
    {
        return strf::fmt((T&&)value) < width;
    }
    template <typename T, typename CharT>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width, CharT fill) const
        noexcept(noexcept(strf::fmt((T&&)value).fill(fill) < width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) < width)>
    {
        return strf::fmt((T&&)value).fill(fill) < width;
    }
};

struct center_fn {
    template <typename T>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width) const
        noexcept(noexcept(strf::fmt((T&&)value) ^ width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value) ^ width)>
    {
        return strf::fmt((T&&)value) ^ width;
    }
    template <typename T, typename CharT>
    constexpr STRF_HD auto operator()(T&& value, strf::width_t width, CharT fill) const
        noexcept(noexcept(strf::fmt((T&&)value).fill(fill) ^ width))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).fill(fill) ^ width)>
    {
        return strf::fmt((T&&)value).fill(fill) ^ width;
    }
};

struct pad0_fn {
    template <typename T, typename W>
    constexpr STRF_HD auto operator() (T&& value, W width) const
        noexcept(noexcept(strf::fmt((T&&)value).pad0(width)))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).pad0(width))>
    {
        return strf::fmt((T&&)value).pad0(width);
    }
};

struct punct_fn {
    template <typename T>
    constexpr STRF_HD auto operator()
        ( T&& value ) const
        noexcept(noexcept(strf::fmt((T&&)value).punct()))
        -> strf::detail::remove_cvref_t<decltype(strf::fmt((T&&)value).punct())>
    {
        return strf::fmt((T&&)value).punct();
    }
};

} // namespace detail_format_functions

constexpr strf::detail_format_functions::hex_fn    hex {};
constexpr strf::detail_format_functions::dec_fn    dec {};
constexpr strf::detail_format_functions::oct_fn    oct {};
constexpr strf::detail_format_functions::bin_fn    bin {};
constexpr strf::detail_format_functions::fixed_fn  fixed {};
constexpr strf::detail_format_functions::sci_fn    sci {};
constexpr strf::detail_format_functions::gen_fn    gen {};
constexpr strf::detail_format_functions::multi_fn  multi {};
constexpr strf::detail_format_functions::right_fn  right {};
constexpr strf::detail_format_functions::left_fn   left {};
constexpr strf::detail_format_functions::center_fn center {};
constexpr strf::detail_format_functions::pad0_fn   pad0 {};
constexpr strf::detail_format_functions::punct_fn  punct {};
constexpr strf::detail_format_functions::transcode_fn   transcode {};
STRF_DEPRECATED_MSG("conv was renamed to transcode")
constexpr strf::detail_format_functions::transcode_fn   conv {};
constexpr strf::detail_format_functions::sani_fn        sani {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

} // inline namespace format_functions


namespace detail {



} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_FORMAT_FUNCTIONS_HPP

