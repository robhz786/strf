#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <boost/stringify/v0/detail/facets/encoding.hpp>

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
    const CharIn* sep_begin;
    const CharIn* sep_end;
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

    std::size_t necessary_size() const override
    {
        std::size_t len = 0;
        for(auto it = m_begin; it != m_end; ++it)
        {
            len += make_printer<CharT, FPack>(m_out, m_fp, *it)
                .necessary_size();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        for(auto it = m_begin; it != m_end && w > 0; ++it)
        {
            w = make_printer<CharT, FPack>(m_out, m_fp, *it).remaining_width(w);
        }
        return w;
    }

    void write() const override
    {
        for(auto it = m_begin; it != m_end; ++it)
        {
            make_printer<CharT, FPack>(m_out, m_fp, *it).write();
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
        , m_sw( ow
              , get_facet<stringify::v0::encoding_category<CharIn>>(fp)
              , false )
    {
    }

    std::size_t necessary_size() const override;

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
int sep_range_printer<CharOut, CharIn, FPack, ForwardIt>
    ::remaining_width(int w) const
{
    const auto& wcalc
        = get_facet<stringify::v0::width_calculator_category>(m_fp);
    const auto encoding
        = get_facet<stringify::v0::encoding_category<CharIn>>(m_fp);
    const auto& decoder
        = encoding.decoder();

    int sep_width = -1;
    bool not_first = false;
    for(auto it = m_begin; it != m_end && w > 0; ++it)
    {
        w = make_printer<CharOut, FPack>(m_out, m_fp, *it).remaining_width(w);
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
                    , m_sw.on_encoding_error()
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
std::size_t
sep_range_printer<CharOut, CharIn, FPack, ForwardIt>::necessary_size() const
{
    std::size_t len = 0;
    std::size_t sep_len = -1;
    for(auto it = m_begin; it != m_end; ++it)
    {
        len += make_printer<CharOut, FPack>(m_out, m_fp, *it).necessary_size();
        if (sep_len == static_cast<std::size_t>(-1))
        {
            sep_len = m_sw.necessary_size(m_sep_begin, m_sep_end);
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
        make_printer<CharOut, FPack>(m_out, m_fp, *it).write();
    }
}

namespace detail{

template <typename ForwardIt>
struct fmt_range_helper
{
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using fmt_value_type
        = decltype(make_fmt( stringify::v0::tag{}
                           , std::declval<const value_type>() ));
    template <typename T>
    using formatting = typename fmt_value_type::template replace_value_type<T>;

};

template <typename ForwardIt>
using range_with_format
= typename stringify::v0::detail::fmt_range_helper<ForwardIt>
    ::template formatting<stringify::v0::detail::range_p<ForwardIt>>;

template <typename ForwardIt, typename CharIn>
using sep_range_with_format
= typename stringify::v0::detail::fmt_range_helper<ForwardIt>
    ::template formatting<stringify::v0::detail::sep_range_p<ForwardIt, CharIn>>;

} // namespace detail


template
    < typename CharOut
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
class fmt_range_printer: public printer<CharOut>
{
    using helper = stringify::v0::detail::fmt_range_helper<ForwardIt>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using fmt_type = stringify::v0::detail::range_with_format<ForwardIt>;
    using fmt_type_adapted = typename fmt_type::template replace_fmts<Fmts...>;
    using value_fmt_type = typename helper::fmt_value_type;
    using value_fmt_type_adapted
    = typename value_fmt_type::template replace_fmts<Fmts...>;

public:

    fmt_range_printer
        ( writer_type& ow
        , const FPack& fp
        , const fmt_type_adapted& fmt )
        : m_out(ow)
        , m_fp(fp)
        , m_fmt(fmt)
    {
    }

    std::size_t necessary_size() const override
    {
        std::size_t len = 0;
        auto r = m_fmt.value();
        for(auto it = r.begin; it != r.end; ++it)
        {
            len += make_printer<CharOut, FPack>
                ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
                .necessary_size();
        }
        return len;
    }

    int remaining_width(int w) const override
    {
        auto r = m_fmt.value();
        for(auto it = r.begin; it != r.end && w > 0; ++it)
        {
            w = make_printer<CharOut, FPack>
                ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
                .remaining_width(w);
        }
        return w;
    }

    void write() const override
    {
        auto r = m_fmt.value();
        for(auto it = r.begin; it != r.end; ++it)
        {
            make_printer<CharOut, FPack>
                ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
                .write();
        }
    }

private:

    stringify::v0::output_writer<CharOut>& m_out;
    const FPack& m_fp;
    fmt_type_adapted m_fmt;
};

template
    < typename CharOut
    , typename CharIn
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
class fmt_sep_range_printer: public printer<CharOut>
{
    using sep_tag = stringify::v0::range_separator_input_tag<CharIn>;
    using helper = stringify::v0::detail::fmt_range_helper<ForwardIt>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using fmt_type
    = stringify::v0::detail::sep_range_with_format<ForwardIt, CharIn>;
    using fmt_type_adapted
    = typename fmt_type::template replace_fmts<Fmts...>;
    using value_fmt_type = typename helper::fmt_value_type;
    using value_fmt_type_adapted
    = typename value_fmt_type::template replace_fmts<Fmts...>;

public:

    fmt_sep_range_printer
        ( writer_type& ow
        , const FPack& fp
        , const fmt_type_adapted& fmt
        )
        : m_out(ow)
        , m_fp(fp)
        , m_fmt(fmt)
        , m_sw(ow, get_facet<stringify::v0::encoding_category<CharIn>>(fp), false)
    {
    }

    std::size_t necessary_size() const override;

    int remaining_width(int w) const override;

    void write() const override;

private:

    writer_type& m_out;
    const FPack& m_fp;
    fmt_type_adapted m_fmt;
    const stringify::v0::string_writer<CharIn, CharOut> m_sw;

    template <typename Category>
    const auto& get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, sep_tag>();
    }
};

template
    < typename CharOut
    , typename CharIn
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts  >
std::size_t
fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt, Fmts ...>
::necessary_size() const
{
    std::size_t len = 0;
    std::size_t sep_len = -1;
    auto r = m_fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        len += make_printer<CharOut, FPack>
            ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
            .necessary_size();
        if (sep_len == static_cast<std::size_t>(-1))
        {
            sep_len = m_sw.necessary_size(r.sep_begin, r.sep_end);
        }
        else
        {
            len += sep_len;
        }
    }
    return len;
}

template
    < typename CharOut
    , typename CharIn
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
int
fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt, Fmts ...>
::remaining_width(int w) const
{
    const auto& wcalc = get_facet<stringify::v0::width_calculator_category>(m_fp);
    const auto encoding
        = get_facet<stringify::v0::encoding_category<CharIn>>(m_fp);
    const auto& decoder = encoding.decoder();

    int sep_width = -1;
    bool not_first = false;
    auto r = m_fmt.value();
    for(auto it = r.begin; it != r.end && w > 0; ++it)
    {
        w = make_printer<CharOut, FPack>
            ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
            .remaining_width(w);
        if (not_first && w > 0)
        {
            if (sep_width == -1)
            {
                int w_prev = w;
                w = wcalc.remaining_width
                    ( w
                    , r.sep_begin
                    , r.sep_end
                    , decoder
                    , m_sw.on_encoding_error()
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

template
    < typename CharOut
    , typename CharIn
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
void
fmt_sep_range_printer<CharOut, CharIn, FPack, ForwardIt, Fmts ...>
::write() const
{
    bool not_first = false;
    auto r = m_fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        if (not_first)
        {
            m_sw.write(r.sep_begin, r.sep_end);
        }
        not_first = true;
        make_printer<CharOut, FPack>
            ( m_out, m_fp, value_fmt_type_adapted{{*it}, m_fmt} )
            .write();
    }
}


template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::range_printer<CharOut, FPack, ForwardIt>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , stringify::v0::detail::range_p<ForwardIt> r )
{
    return {out, fp, r.begin, r.end};
}

template <typename CharOut, typename FPack, typename ForwardIt, typename CharIn>
inline stringify::v0::sep_range_printer<CharOut, CharIn, FPack, ForwardIt>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , stringify::v0::detail::sep_range_p<ForwardIt, CharIn> r )
{
    return {out, fp, r.begin, r.end, r.sep_begin, r.sep_end};
}

template
    < typename CharOut
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
inline stringify::v0::fmt_range_printer<CharOut, FPack, ForwardIt, Fmts...>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::value_with_format
        < stringify::v0::detail::range_p<ForwardIt>
        , Fmts ... >& fmt )
{
    return {out, fp, fmt};
}

template
    < typename CharOut
    , typename FPack
    , typename ForwardIt
    , typename CharIn
    , typename ... Fmts >
inline stringify::v0::fmt_sep_range_printer
    < CharOut
    , CharIn
    , FPack
    , ForwardIt
    , Fmts... >
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::value_with_format
        < stringify::v0::detail::sep_range_p<ForwardIt, CharIn>
        , Fmts ... >& fmt )
{
    return {out, fp, fmt};
}

template <typename ForwardIt>
inline stringify::v0::detail::range_with_format<ForwardIt>
make_fmt(stringify::v0::tag, stringify::v0::detail::range_p<ForwardIt> r)
{
    return stringify::v0::detail::range_with_format<ForwardIt>{{r.begin, r.end}};
}

template <typename ForwardIt, typename CharIn>
inline stringify::v0::detail::sep_range_with_format<ForwardIt, CharIn>
make_fmt
    ( stringify::v0::tag
    , stringify::v0::detail::sep_range_p<ForwardIt, CharIn> r )
{
    return stringify::v0::detail::sep_range_with_format<ForwardIt, CharIn>
        {{r.begin, r.end, r.sep_begin, r.sep_end}};
}

template <typename ForwardIt>
inline auto range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::detail::range_p<ForwardIt>{begin, end};
}

template <typename ForwardIt, typename CharIn>
inline auto range(ForwardIt begin, ForwardIt end, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return stringify::v0::detail::sep_range_p<ForwardIt, CharIn>
        {{begin, end, sep, sep + sep_len}};
}

template <typename ForwardIt>
inline auto fmt_range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::detail::range_with_format<ForwardIt>{{begin, end}};
}

template <typename ForwardIt, typename CharIn>
inline auto fmt_range(ForwardIt begin, ForwardIt end, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return stringify::v0::detail::sep_range_with_format<ForwardIt, CharIn>
        {{begin, end, sep + sep_len}};
}


template <typename Range, typename It = typename Range::const_iterator>
inline auto range(const Range& range)
{
    using namespace std;
    return stringify::v0::detail::range_p<It>{begin(range), end(range)};
}

template <typename T, std::size_t N>
inline auto range(T (&array)[N])
{
    return stringify::v0::detail::range_p<const T*>{&array[0], &array[0] + N};
}

template <typename Range, typename CharIn>
inline auto range(const Range& range, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    return stringify::v0::detail::sep_range_p
        <typename Range::const_iterator, CharIn>
        {begin(range), end(range), sep, sep + sep_len};
}

template <typename T, std::size_t N, typename CharIn>
inline auto range(T (&array)[N], const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    return stringify::v0::detail::sep_range_p<const T*, CharIn>
        {&array[0], &array[0] + N, sep, sep + sep_len};
}


template <typename Range, typename It = typename Range::const_iterator>
inline stringify::v0::detail::range_with_format<It>
 fmt_range(const Range& range)
{
    using namespace std;
    stringify::v0::detail::range_p<It> r{begin(range), end(range)};
    return stringify::v0::detail::range_with_format<It>{r};
}

template <typename T, std::size_t N>
inline auto fmt_range(T (&array)[N])
{
    using namespace std;
    using fmt_type = stringify::v0::detail::range_with_format<const T*>;
    return fmt_type{{&array[0], &array[0] + N}};
}

template
    < typename Range
    , typename CharIn
    , typename It = typename Range::const_iterator >
inline auto fmt_range(const Range& range, const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    stringify::v0::detail::sep_range_p<It, CharIn> r
    { begin(range), end(range), sep, sep + sep_len };
    return stringify::v0::detail::sep_range_with_format<It, CharIn>{r};
}

template <typename T, std::size_t N, typename CharIn>
inline auto fmt_range(T (&array)[N], const CharIn* sep)
{
    std::size_t sep_len = std::char_traits<CharIn>::length(sep);
    using namespace std;
    return stringify::v0::detail::sep_range_with_format<const T*, CharIn>
        { {&array[0], &array[0] + N, sep, sep + sep_len} };
}



BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP

