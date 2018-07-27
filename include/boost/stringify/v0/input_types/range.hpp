#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP
#define BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <boost/stringify/v0/facets/encodings.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename ForwardIt>
struct range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
};

template <typename ForwardIt, typename CharIn>
struct sep_range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
    const CharIn* separator_begin;
    const CharIn* separator_end;
};

} // namespace detail


template <typename CharT, typename FPack, typename ForwardIt>
class range_printer: public printer<CharT>
{
public:
    using writer_type = stringify::v0::output_writer<CharT>;
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    range_printer
        ( writer_type& ow
        , const FPack& fp
        , iterator begin
        , iterator end
        )
        : m_out(ow)
        , m_fp(fp)
        , m_begin(begin)
        , m_end(end)
    {
    }

    std::size_t length() const override
    {
        std::size_t len = 0;
        for(auto it = m_begin; it != m_end; ++it)
        {
            len += stringify_make_printer<CharT, FPack>(m_out, m_fp, *it).length();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_begin; it != m_end && w > 0; ++it)
        {
            w = stringify_make_printer<CharT, FPack>(m_out, m_fp, *it).remaining_width(w);
        }
        return w;
    }

    void write() const override
    {
        for(auto it = m_begin; it != m_end; ++it)
        {
            stringify_make_printer<CharT, FPack>(m_out, m_fp, *it).write();
        }
    }

private:

    stringify::v0::output_writer<CharT>& m_out;
    const FPack& m_fp;
    iterator m_begin;
    iterator m_end;
};


template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
class sep_range_printer: public printer<CharOut>
{
    using sep_tag = stringify::v0::range_separator_input_tag<CharIn>;

public:

    using writer_type = stringify::v0::output_writer<CharOut>;
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    sep_range_printer
        ( writer_type& ow
        , const FPack& fp
        , iterator begin
        , iterator end
        , const CharIn* sep
        , const CharIn* sep_end
        )
        : m_out(ow)
        , m_fp(fp)
        , m_begin(begin)
        , m_end(end)
        , m_sep_begin(sep)
        , m_sep_end(sep_end)
        , m_sw(ow, get_facet<stringify::v0::encoding_category<CharIn>>(fp), false)
    {
    }

    std::size_t length() const override;

    int remaining_width(int w) const override;

    void write() const override;

private:

    writer_type& m_out;
    const FPack& m_fp;
    iterator m_begin;
    iterator m_end;
    const CharIn* m_sep_begin;
    const CharIn* m_sep_end;
    const stringify::v0::string_writer<CharIn, CharOut> m_sw;

    template <typename Category>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
int sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::remaining_width(int w) const
{
    const auto& wcalc = get_facet<stringify::v0::width_calculator_category>(m_fp);
    const auto encoding = get_facet<stringify::v0::encoding_category<CharIn>>(m_fp);
    const auto& decoder = encoding.decoder();

    int sep_width = -1;
    bool not_first = false;
    for(auto it = m_begin; it != m_end && w > 0; ++it)
    {
        w = stringify_make_printer<CharOut, FPack>(m_out, m_fp, *it).remaining_width(w);
        if (not_first && w > 0)
        {
            if (sep_width == -1)
            {
                int w_prev = w;
                w = wcalc.remaining_width
                    ( w
                    , m_sep_begin
                    , m_sep_end
                    , decoder
                    , m_sw.on_error()
                    , m_sw.allow_surrogates() );
                sep_width = w_prev - w;
            }
            else if(sep_width < w)
            {
                w -= sep_width;
            }
            else
            {
                w = 0;
            }
        }
        not_first = true;
    }
    return w;
}

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
std::size_t sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::length() const
{
    std::size_t len = 0;
    std::size_t sep_len = -1;
    for(auto it = m_begin; it != m_end; ++it)
    {
        len += stringify_make_printer<CharOut, FPack>(m_out, m_fp, *it).length();
        if (sep_len == static_cast<std::size_t>(-1))
        {
            sep_len = m_sw.length(m_sep_begin, m_sep_end);
        }
        else
        {
            len += sep_len;
        }
    }
    return len;
}

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
void sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::write() const
{
    bool not_first = false;
    for(auto it = m_begin; it != m_end; ++it)
    {
        if (not_first)
        {
            m_sw.write(m_sep_begin, m_sep_end);
        }
        not_first = true;
        stringify_make_printer<CharOut, FPack>(m_out, m_fp, *it).write();
    }
}

namespace detail{

template <typename ForwardIt>
struct fmt_range_helper
{
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using fmt_value_type = decltype(stringify_fmt(std::declval<const value_type>()));

    template <typename T>
    using formatting = typename fmt_value_type::template fmt_other<T>;

};

template <typename ForwardIt>
class fmt_range
    : public stringify::v0::detail::fmt_range_helper<ForwardIt>
        ::template formatting<fmt_range<ForwardIt>>
{
public:

    using original_fmt
        = typename stringify::v0::detail::fmt_range_helper<ForwardIt>
        ::fmt_value_type;

    fmt_range(ForwardIt begin, ForwardIt end)
        : m_begin(begin)
        , m_end(end)
    {
    }

    ForwardIt begin() const
    {
        return m_begin;
    }
    ForwardIt end() const
    {
        return m_end;
    }

private:

    ForwardIt m_begin;
    ForwardIt m_end;
};

template <typename ForwardIt, typename CharIn>
class fmt_sep_range
    : public stringify::v0::detail::fmt_range_helper<ForwardIt>
        ::template formatting<fmt_sep_range<ForwardIt, CharIn> >
{
public:

    using original_fmt
        = typename stringify::v0::detail::fmt_range_helper<ForwardIt>
        ::fmt_value_type;

    fmt_sep_range
        ( ForwardIt begin
        , ForwardIt end
        , const CharIn* sep_begin
        , const CharIn* sep_end )
        : m_begin(begin)
        , m_end(end)
        , m_sep_begin(sep_begin)
        , m_sep_end(sep_end)
    {
    }

    ForwardIt begin() const
    {
        return m_begin;
    }
    ForwardIt end() const
    {
        return m_end;
    }

    const CharIn* sep_begin() const
    {
        return m_sep_begin;
    }
    const CharIn* sep_end() const
    {
        return m_sep_end;
    }

private:

    ForwardIt m_begin;
    ForwardIt m_end;
    const CharIn* m_sep_begin;
    const CharIn* m_sep_end;
};

} // namespace detail


template <typename CharOut, typename FPack, typename ForwardIt>
class fmt_range_printer: public printer<CharOut>
{
public:
    using writer_type = stringify::v0::output_writer<CharOut>;
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using fmt_type = stringify::v0::detail::fmt_range<ForwardIt>;
    using original_fmt_type = typename fmt_type::original_fmt;

    fmt_range_printer
        ( writer_type& ow
        , const FPack& fp
        , const fmt_type& fmt )
        : m_out(ow)
        , m_fp(fp)
        , m_fmt(fmt)
    {
    }

    std::size_t length() const override
    {
        std::size_t len = 0;
        for(const auto& value : m_fmt)
        {
            len += make_printer(value).length();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_fmt.begin(); it != m_fmt.end() && w > 0; ++it)
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
        return stringify_make_printer<CharOut, FPack>
            ( m_out, m_fp, original_fmt_type{value, m_fmt} );
    }

    stringify::v0::output_writer<CharOut>& m_out;
    const FPack& m_fp;
    fmt_type m_fmt;
};

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
class fmt_sep_range_printer: public printer<CharOut>
{
    using sep_tag = stringify::v0::range_separator_input_tag<CharIn>;

public:

    using writer_type = stringify::v0::output_writer<CharOut>;
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using fmt_type = stringify::v0::detail::fmt_sep_range<ForwardIt, CharIn>;
    using original_fmt_type = typename fmt_type::original_fmt;

    fmt_sep_range_printer
        ( writer_type& ow
        , const FPack& fp
        , const fmt_type& fmt
        )
        : m_out(ow)
        , m_fp(fp)
        , m_fmt(fmt)
        , m_sw(ow, get_facet<stringify::v0::encoding_category<CharIn>>(fp), false)
    {
    }

    std::size_t length() const override;

    int remaining_width(int w) const override;

    void write() const override;

private:

    writer_type& m_out;
    const FPack& m_fp;
    fmt_type m_fmt;
    const stringify::v0::string_writer<CharIn, CharOut> m_sw;

    auto make_printer(const value_type& value) const
    {
        return stringify_make_printer<CharOut, FPack>
            ( m_out, m_fp, original_fmt_type{value, m_fmt} );
    }

    template <typename Category>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
std::size_t fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::length() const
{
    std::size_t len = 0;
    std::size_t sep_len = -1;
    for(const auto& value : m_fmt)
    {
        len += make_printer(value).length();
        if (sep_len == static_cast<std::size_t>(-1))
        {
            sep_len = m_sw.length(m_fmt.sep_begin(), m_fmt.sep_end());
        }
        else
        {
            len += sep_len;
        }
    }
    return len;
}

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
int fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::remaining_width(int w) const
{
    const auto& wcalc = get_facet<stringify::v0::width_calculator_category>(m_fp);
    const auto encoding = get_facet<stringify::v0::encoding_category<CharIn>>(m_fp);
    const auto& decoder = encoding.decoder();

    int sep_width = -1;
    bool not_first = false;
    for(auto it = m_fmt.begin(); it != m_fmt.end() && w > 0; ++it)
    {
        w = make_printer(*it).remaining_width(w);
        if (not_first && w > 0)
        {
            if (sep_width == -1)
            {
                int w_prev = w;
                w = wcalc.remaining_width
                    ( w
                    , m_fmt.sep_begin()
                    , m_fmt.sep_end()
                    , decoder
                    , m_sw.on_error()
                    , m_sw.allow_surrogates() );
                sep_width = w_prev - w;
            }
            else if(sep_width < w)
            {
                w -= sep_width;
            }
            else
            {
                w = 0;
            }
        }
        not_first = true;
    }
    return w;
}

template <typename CharOut, typename CharIn, typename FPack, typename ForwardIt>
void fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::write() const
{
    bool not_first = false;
    for(const auto& value : m_fmt)
    {
        if (not_first)
        {
            m_sw.write(m_fmt.sep_begin(), m_fmt.sep_end());
        }
        not_first = true;
        make_printer(value).write();
    }
}


template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::range_printer<CharOut, FPack, ForwardIt>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , stringify::v0::detail::range_p<ForwardIt> r )
{
    return {out, fp, r.begin, r.end};
}

template <typename CharOut, typename FPack, typename ForwardIt, typename CharIn>
inline stringify::v0::sep_range_printer<CharOut, CharIn, FPack, ForwardIt>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , stringify::v0::detail::sep_range_p<ForwardIt, CharIn> r )
{
    return {out, fp, r.begin, r.end, r.separator_begin, r.separator_end};
}

template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::fmt_range_printer<CharOut, FPack, ForwardIt>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::detail::fmt_range<ForwardIt>& fmt)
{
    return {out, fp, fmt};
}

template <typename CharOut, typename FPack, typename ForwardIt, typename CharIn>
inline stringify::v0::fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::detail::fmt_sep_range<ForwardIt, CharIn>& fmt)
{
    return {out, fp, fmt};
}

template <typename ForwardIt>
inline stringify::v0::detail::fmt_range<ForwardIt>
stringify_fmt(stringify::v0::detail::range_p<ForwardIt> r)
{
    return {r.begin, r.end};
}

template <typename ForwardIt, typename CharIn>
inline stringify::v0::detail::fmt_sep_range<ForwardIt, CharIn>
stringify_fmt(stringify::v0::detail::sep_range_p<ForwardIt, CharIn> r)
{
    return {r.begin, r.end, r.separator_begin, r.separator_end};
}


template <typename ForwardIt>
stringify::v0::detail::range_p<ForwardIt>
iterate(ForwardIt begin, ForwardIt end)
{
    return {begin, end};
}

template <typename ForwardIt, typename CharIn>
stringify::v0::detail::sep_range_p<ForwardIt, CharIn>
iterate(ForwardIt begin, ForwardIt end, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return {begin, end, sep, sep + sep_len};
}

template <typename ForwardIt>
stringify::v0::detail::fmt_range<ForwardIt>
fmt_iterate(ForwardIt begin, ForwardIt end)
{
    return {begin, end};
}

template <typename ForwardIt, typename CharIn>
stringify::v0::detail::fmt_sep_range<ForwardIt, CharIn>
fmt_iterate(ForwardIt begin, ForwardIt end, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return {begin, end, sep + sep_len};
}


template <typename Range>
stringify::v0::detail::range_p<typename Range::const_iterator>
range(const Range& range)
{
    using namespace std;
    return {begin(range), end(range)};
}

template <typename T, std::size_t N>
stringify::v0::detail::range_p<const T*>
range(T (&array)[N])
{
    return {&array[0], &array[0] + N};
}

template <typename Range, typename CharIn>
stringify::v0::detail::sep_range_p<typename Range::const_iterator, CharIn>
range(const Range& range, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    return {begin(range), end(range), sep, sep + sep_len};
}

template <typename T, std::size_t N, typename CharIn>
stringify::v0::detail::sep_range_p<const T*, CharIn>
range(T (&array)[N], const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return {&array[0], &array[0] + N, sep, sep + sep_len};
}


template <typename Range>
inline stringify::v0::detail::fmt_range
    < typename Range::const_iterator >
fmt_range(const Range& range)
{
    using namespace std;
    return {begin(range), end(range)};
}

template <typename T, std::size_t N>
inline stringify::v0::detail::fmt_range<const T*>
fmt_range(T (&array)[N])
{
    using namespace std;
    return {&array[0], &array[0] + N};
}

template <typename Range, typename CharIn>
inline stringify::v0::detail::fmt_sep_range
    < typename Range::const_iterator, CharIn >
fmt_range(const Range& range, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    return {begin(range), end(range), sep, sep + sep_len};
}

template <typename T, std::size_t N, typename CharIn>
inline stringify::v0::detail::fmt_sep_range<const T*, CharIn>
fmt_range(T (&array)[N], const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    return {&array[0], &array[0] + N, sep, sep + sep_len};
}



BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_INPUT_TYPES_RANGE_HPP

