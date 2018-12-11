#ifndef BOOST_STRINGIFY_V0_BASIC_TYPES_HPP
#define BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/expected.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

constexpr std::size_t min_buff_size = 60;

struct tag {};

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
        : m_value(v)
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
        , m_value(v)
    {
    }

    template <typename OtherValueType>
    constexpr value_with_format
        ( const ValueType& v
        , const stringify::v0::value_with_format<OtherValueType, Fmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_with_format<OtherValueType, Fmts...>> &&>(f) )
        ...
        , m_value(static_cast<ValueType&&>(v))
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
        , m_value(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr value_with_format
        ( const stringify::v0::value_with_format<ValueType, OtherFmts...>&& f )
        : Fmts::template fn<value_with_format<ValueType, Fmts...>>
            ( static_cast
              < typename OtherFmts
             :: template fn<value_with_format<ValueType, OtherFmts ...>>&& >(f) )
        ...
        , m_value(static_cast<ValueType&&>(f.value()))
    {
    }

    constexpr const ValueType& value() const
    {
        return m_value;
    }

private:

    ValueType m_value;
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
        : m_fill(u.fill())
        , m_width(u.width())
        , m_alignment(u.alignment())
    {
    }

    template <typename U>
    constexpr alignment_format_impl(const empty_alignment_format_impl<U>&)
    {
    }

    ~alignment_format_impl()
    {
    }

    constexpr derived_type&& operator<(int width) &&
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator<(int width) &
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator>(int width) &&
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator>(int width) &
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator^(int width) &&
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator^(int width) &
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& operator%(int width) &&
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& operator%(int width) &
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& fill(char32_t ch) &&
    {
        m_fill = ch;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& fill(char32_t ch) &
    {
        m_fill = ch;
        return static_cast<derived_type&>(*this);
    }
    constexpr derived_type&& width(int w) &&
    {
        m_width = w;
        return static_cast<derived_type&&>(*this);
    }
    constexpr derived_type& width(int w) &
    {
        m_width = w;
        return static_cast<derived_type&>(*this);
    }
    constexpr int width() const
    {
        return m_width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return m_alignment;
    }
    constexpr char32_t fill() const
    {
        return m_fill;
    }

private:

    template <typename>
    friend class alignment_format_impl;

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
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
    constexpr adapted_derived_type fill(char32_t ch) const
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

template <typename CharOut>
struct buff_it
{
    buff_it& operator=(const buff_it& other)
    {
        it = other.it;
        end = other.end;
        return *this;
    }

    CharOut* it;
    CharOut* end;
};

template <typename It>
using expected_buff_it = stringify::v0::expected
    < stringify::v0::buff_it<It>
    , std::error_code >;

template <typename CharOut>
class buffer_recycler
{
public:

    using char_type = CharOut;

    virtual stringify::v0::expected_buff_it<CharOut> recycle(CharOut* it) = 0;
};

template <typename CharOut>
class printer
{
public:

    virtual ~printer()
    {
    }

    virtual stringify::v0::expected_buff_it<CharOut> write
        ( stringify::v0::buff_it<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler ) const = 0;

    virtual std::size_t necessary_size() const = 0;

    virtual int remaining_width(int w) const = 0;
};


template <typename CharOut>
class printers_receiver
{
public:

    virtual ~printers_receiver()
    {
    }

    virtual bool put(const stringify::v0::printer<CharOut>& ) = 0;
};

inline std::error_code encoding_error()
{
    return std::make_error_code(std::errc::illegal_byte_sequence);
}

namespace detail {

template <typename CharOut>
class width_subtracter: public printers_receiver<CharOut>
{
public:

    width_subtracter(int w)
        : m_width(w)
    {
    }

    bool put(const stringify::v0::printer<CharOut>& p) override;

    int remaining_width() const
    {
        return m_width;
    }

private:

    int m_width;
};

template <typename CharOut>
bool width_subtracter<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    m_width = p.remaining_width(m_width);
    return m_width > 0;
}

template <typename CharOut>
class necessary_size_sum: public printers_receiver<CharOut>
{
public:

    necessary_size_sum() = default;

    bool put(const stringify::v0::printer<CharOut>& p) override;

    std::size_t accumulated_length() const
    {
        return m_len;
    }

private:

    std::size_t m_len = 0;
};

template <typename CharOut>
bool necessary_size_sum<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    m_len += p.necessary_size();
    return true;
}

template <typename CharOut>
class serial_writer: public printers_receiver<CharOut>
{
public:

    serial_writer() = default;

    bool put(const stringify::v0::printer<CharOut>& p) override;
};

template <typename CharOut>
bool serial_writer<CharOut>::put(const stringify::v0::printer<CharOut>& p)
{
    p.write();
    return true;
}

} // namespace detail

// template <typename CharOut>
// class dynamic_join_printer: public stringify::v0::printer<CharOut>
// {
// public:

//     dynamic_join_printer(stringify::v0::output_writer<CharOut>& out)
//         : m_out(out)
//     {
//     }

//     std::size_t necessary_size() const override;

//     void write() const override;

//     int remaining_width(int w) const override;

// protected:

//     virtual stringify::v0::alignment_format::fn<void> formatting() const;

//     virtual void compose(stringify::v0::printers_receiver<CharOut>& out) const = 0;

// private:

//     void write_with_fill(int fillcount) const;

//     void write_without_fill() const;

//     stringify::v0::output_writer<CharOut>& m_out;
// };

// template <typename CharOut>
// stringify::v0::alignment_format::fn<void>
// dynamic_join_printer<CharOut>::formatting() const
// {
//     return {};
// }

// template <typename CharOut>
// std::size_t dynamic_join_printer<CharOut>::necessary_size() const
// {
//     std::size_t fill_len = 0;
//     const auto fmt = formatting();
//     if(fmt.width() > 0)
//     {
//         stringify::v0::detail::width_subtracter<CharOut> wds{fmt.width()};
//         compose(wds);
//         std::size_t fillcount = wds.remaining_width();
//         fill_len = m_out.necessary_size(fmt.fill()) * fillcount;
//     }

//     stringify::v0::detail::necessary_size_sum<CharOut> s;
//     compose(s);
//     return s.accumulated_length() + fill_len;
// }

// template <typename CharOut>
// int dynamic_join_printer<CharOut>::remaining_width(int w) const
// {
//     const auto fmt_width = formatting().width();
//     if (fmt_width > w)
//     {
//         return 0;
//     }

//     stringify::v0::detail::width_subtracter<CharOut> s{w};
//     compose(s);
//     int rw = s.remaining_width();
//     return (w - rw < fmt_width) ? (w - fmt_width) : rw;
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write() const
// {
//     auto fmt = formatting();
//     auto fillcount = fmt.width();
//     if(fillcount > 0)
//     {
//         stringify::v0::detail::width_subtracter<CharOut> wds{fillcount};
//         compose(wds);
//         fillcount = wds.remaining_width();
//     }
//     if(fillcount > 0)
//     {
//         write_with_fill(fillcount);
//     }
//     else
//     {
//         write_without_fill();
//     }
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write_without_fill() const
// {
//     stringify::v0::detail::serial_writer<CharOut> s;
//     compose(s);
// }

// template <typename CharOut>
// void dynamic_join_printer<CharOut>::write_with_fill(int fillcount) const
// {
//     auto fmt = formatting();
//     switch (fmt.alignment())
//     {
//         case stringify::v0::alignment::left:
//         {
//             write_without_fill();
//             m_out.put32(fillcount, fmt.fill());
//             break;
//         }
//         case stringify::v0::alignment::center:
//         {
//             auto halfcount = fillcount / 2;
//             m_out.put32(halfcount, fmt.fill());
//             write_without_fill();
//             m_out.put32(fillcount - halfcount, fmt.fill());
//             break;
//         }
//         //case stringify::v0::alignment::internal:
//         //case stringify::v0::alignment::right:
//         default:
//         {
//             m_out.put32(fillcount, fmt.fill());
//             write_without_fill();
//         }
//     }
// }


template <typename T>
struct identity
{
    using type = T;
};

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharIn>
struct asm_string_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: stringify::v0::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_asm_string_of
{
    template <typename T>
    using fn = std::is_same<stringify::v0::asm_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_asm_string: std::false_type
{
};

template <typename CharIn>
struct is_asm_string<stringify::v0::is_asm_string_of<CharIn>> : std::true_type
{
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

#endif

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

