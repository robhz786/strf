#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP
#define BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <boost/stringify/v0/facets/encodings.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename Iterator>
struct range_p
{
    using iterator = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    Iterator begin;
    Iterator end;
    const char32_t* separator_begin = nullptr;
    const char32_t* separator_end = nullptr;
};

template <typename Iterator> class range_with_formatting;

template
    < typename Iterator
    , typename ChildClass
    , typename value_type = typename Iterator::value_type
    , typename value_with_formatting
      = decltype(stringify_fmt(std::declval<const value_type>()))
    >
using range_format
    = typename value_with_formatting::template other<ChildClass>;


template <typename Iterator>
class range_with_formatting
    : public stringify::v0::detail::range_format
        < Iterator
        , range_with_formatting<Iterator>
        >
{
public:

    using CharT = char32_t;

    template <typename T>
    using other = stringify::v0::detail::range_format<Iterator, T>;

    range_with_formatting
        ( Iterator begin
        , Iterator end
        , const CharT* separator_begin
        , const CharT* separator_end
        )
        : m_begin(begin)
        , m_end(end)
        , m_sep_begin(separator_begin)
        , m_sep_end(separator_end)
    {
    }

    Iterator begin() const
    {
        return m_begin;
    }
    Iterator end() const
    {
        return m_end;
    }

private:

    Iterator m_begin;
    Iterator m_end;
    const CharT* m_sep_begin;
    const CharT* m_sep_end;
};

} // namespace detail


template <typename CharT, typename FPack, typename Iterator>
class range_printer: public printer<CharT>
{
public:
    using writer_type = stringify::v0::output_writer<CharT>;
    using iterator = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    //using fmt_type = decltype(stringify_fmt(std::declval<const value_type&>()));

    // range_printer
    //     ( const FPack& ft
    //     , iterator begin
    //     , iterator end
    //     , const fmt_type& fmt
    //     )
    //     : m_ft(ft)
    //     , m_begin(begin)
    //     , m_end(end)
    //     , m_fmt(fmt)
    // {
    // }

    range_printer
        ( writer_type& ow
        , const FPack& ft
        , iterator begin
        , iterator end
        )
        : m_out(ow)
        , m_ft(ft)
        , m_begin(begin)
        , m_end(end)
    {
    }

    std::size_t length() const override
    {
        std::size_t len = 0;
        for(auto it = m_begin; it < m_end; ++it)
        {
            len += stringify_make_printer<CharT, FPack>(m_out, m_ft, *it).length();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_begin; it < m_end && w > 0; ++it)
        {
            w = stringify_make_printer<CharT, FPack>(m_out, m_ft, *it).remaining_width(w);
        }
        return w;
    }

    void write() const override
    {
        for(auto it = m_begin; it < m_end; ++it)
        {
            stringify_make_printer<CharT, FPack>(m_out, m_ft, *it).write();
        }
    }

private:

    stringify::v0::output_writer<CharT>& m_out;
    const FPack& m_ft;
    iterator m_begin;
    iterator m_end;
};


template <typename CharT, typename FPack, typename Iterator>
class fmt_range_printer: public printer<CharT>
{
public:
    using writer_type = stringify::v0::output_writer<CharT>;
    using iterator = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using fmt_type = stringify::v0::detail::range_with_formatting<Iterator>;

    fmt_range_printer
        ( writer_type& ow
        , const FPack& ft
        , const fmt_type& fmt
        )
        : m_out(ow)
        , m_ft(ft)
        , m_fmt(fmt)
    {
    }

    std::size_t length() const override
    {
        std::size_t len = 0;
        for(const auto& value : m_fmt)
        {
            // auto rebinded_fmt = stringify_fmt(value).format_as(m_fmt);
            // auto printer = stringify_make_printer<CharT, FPack>(m_ft, rebinded_fmt);
            // len += printer.length();
            len += make_printer(value).length();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_fmt.begin(); it < m_fmt.end() && w > 0; ++it)
        {
            w = make_printer(*it).remaining_width(w);
        }
        return w;
    }

    void write() const override
    {
        for(const auto& value : m_fmt)
        {
            make_printer(value).write();
        }
    }

private:

    auto make_printer(const value_type& value) const
    {
        return stringify_make_printer<CharT, FPack>
            ( m_out, m_ft, apply_fmt(stringify_fmt(value)) );
    }

    template <typename ElemFmtWithValue>
    ElemFmtWithValue&& apply_fmt(ElemFmtWithValue&& fmt_with_value) const
    {
        // This functions aims just to check if the return type of
        // ElemFmtWithValue::format_as is correct, producing a compile
        // error message easier to understand.

        return std::move(fmt_with_value.format_as(m_fmt));
    }

    stringify::v0::output_writer<CharT>& m_out;
    const FPack& m_ft;
    fmt_type m_fmt;
};


template <typename CharT, typename FPack, typename Iterator>
inline stringify::v0::range_printer<CharT, FPack, Iterator>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , stringify::v0::detail::range_p<Iterator> r )
{
    return {out, ft, r.begin, r.end};
}


template <typename CharT, typename FPack, typename Iterator>
inline stringify::v0::fmt_range_printer<CharT, FPack, Iterator>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , const stringify::v0::detail::range_with_formatting<Iterator>& fmt)
{
    return {out, ft, fmt};
}


template <typename Iterator>
inline stringify::v0::detail::range_with_formatting<Iterator>
stringify_fmt(stringify::v0::detail::range_p<Iterator> r)
{
    return {r.begin, r.end, r.separator_begin, r.separator_end};
}

template <typename T>
stringify::v0::detail::range_p<typename T::const_iterator> iterate(const T& container)
{
    return {container.begin(), container.end()};
}

template <typename T>
inline stringify::v0::detail::range_with_formatting<typename T::const_iterator>
fmt_iterate(const T& container)
{
    return {container.begin(), container.end(), nullptr, nullptr};
}




BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP

