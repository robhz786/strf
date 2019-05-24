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
    std::size_t sep_len;
};


template <typename CharT, typename FPack, typename ForwardIt>
class range_printer: public printer<CharT>
{
public:
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    range_printer
        ( const FPack& fp
        , iterator begin
        , iterator end )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
    {
    }

    std::size_t necessary_size() const override;

    int width(int limit) const override;

    bool write(stringify::v0::output_buffer<CharT>& ob) const override;

private:

    const FPack& _fp;
    iterator _begin;
    iterator _end;
};

template <typename CharT, typename FPack, typename ForwardIt>
std::size_t range_printer<CharT, FPack, ForwardIt>::necessary_size() const
{
    std::size_t len = 0;
    for(auto it = _begin; it != _end; ++it)
    {
        len += make_printer<CharT, FPack>(_fp, *it).necessary_size();
    }
    return len;
}

template <typename CharT, typename FPack, typename ForwardIt>
int range_printer<CharT, FPack, ForwardIt>::width(int limit) const
{
    int sum = 0;
    for(auto it = _begin; it != _end && sum < limit; ++it)
    {
        sum += make_printer<CharT, FPack>(_fp, *it).width(limit - sum);
    }
    return sum;
}

template <typename CharT, typename FPack, typename ForwardIt>
bool range_printer<CharT, FPack, ForwardIt>::write
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    for(auto it = _begin; it != _end; ++it)
    {
        if ( ! make_printer<CharT, FPack>(_fp, *it).write(ob))
        {
            return false;
        }
    }
    return true;
}

template <typename CharT, typename FPack, typename ForwardIt>
class sep_range_printer: public printer<CharT>
{
    using sep_tag = stringify::v0::range_separator_input_tag<CharT>;

public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    sep_range_printer
        ( const FPack& fp
        , iterator begin
        , iterator end
        , const CharT* sep
        , std::size_t sep_len )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
        , _sep_begin(sep)
        , _sep_len(sep_len)
    {
    }

    std::size_t necessary_size() const override;

    int width(int limit) const override;

    bool write(stringify::v0::output_buffer<CharT>& ob) const override;

private:

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    const CharT* _sep_begin;
    std::size_t _sep_len;

    template <typename Category>
    decltype(auto) get_facet(const FPack& fp) const
    {
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename ForwardIt>
int sep_range_printer<CharT, FPack, ForwardIt>::width(int limit) const
{
    std::size_t count = 0;
    int sum = 0;
    for(auto it = _begin; it != _end && sum < limit; ++it)
    {
        sum += make_printer<CharT, FPack>(_fp, *it).width(limit - sum);
        ++ count;
    }
    if (count > 1 && sum < limit)
    {
        decltype(auto) wcalc
            = get_facet<stringify::v0::width_calculator_c>(_fp);
        decltype(auto) encoding
            = get_facet<stringify::v0::encoding_c<CharT>>(_fp);
        auto enc_err = get_facet<stringify::v0::encoding_error_c>(_fp);
        auto allow_surr = get_facet<stringify::v0::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( (limit - sum), _sep_begin, _sep_len
                             , encoding, enc_err, allow_surr );
        sum += dw * (count - 1);
    }
    return sum;
}

template <typename CharT, typename FPack, typename ForwardIt>
std::size_t sep_range_printer<CharT, FPack, ForwardIt>::necessary_size() const
{
    std::size_t len = 0;
    std::size_t count = 0;
    for(auto it = _begin; it != _end; ++it)
    {
        len += make_printer<CharT, FPack>(_fp, *it).necessary_size();
        ++count;
    }
    if (count > 1)
    {
        return len + _sep_len * (count - 1);
    }
    return len;
}

template <typename CharT, typename FPack, typename ForwardIt>
bool sep_range_printer<CharT, FPack, ForwardIt>::write
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
    auto it = _begin;
    if (it == _end)
    {
        return true;
    }
    if ( ! make_printer<CharT, FPack>(_fp, *it).write(ob))
    {
        return false;
    }
    while (++it != _end)
    {
        if ( ! detail::write_str( ob, _sep_begin, _sep_len )
          || ! make_printer<CharT, FPack>(_fp, *it).write(ob) )
        {
            return false;
        }
    }
    return true;
}

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

template <typename ForwardIt, typename CharT>
using sep_range_with_format
= typename stringify::v0::detail::fmt_range_helper<ForwardIt>
    ::template formatting<stringify::v0::detail::sep_range_p<ForwardIt, CharT>>;

template
    < typename CharOut
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
class fmt_range_printer: public printer<CharOut>
{
    using helper = stringify::v0::detail::fmt_range_helper<ForwardIt>;
    using fmt_type = stringify::v0::detail::range_with_format<ForwardIt>;
    using fmt_type_adapted = typename fmt_type::template replace_fmts<Fmts...>;
    using value_fmt_type = typename helper::fmt_value_type;
    using value_fmt_type_adapted
    = typename value_fmt_type::template replace_fmts<Fmts...>;

public:

    fmt_range_printer
        ( const FPack& fp
        , const fmt_type_adapted& fmt )
        : _fp(fp)
        , _fmt(fmt)
    {
    }

    std::size_t necessary_size() const override;

    int width(int lim) const override;

    bool write(stringify::v0::output_buffer<CharOut>& ob) const override;

private:

    const FPack& _fp;
    fmt_type_adapted _fmt;
};


template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
std::size_t
fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::necessary_size() const
{
    std::size_t len = 0;
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        len += make_printer<CharOut, FPack>
            ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
            .necessary_size();
    }
    return len;
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
int fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::width(int lim) const
{
    auto r = _fmt.value();
    int sum = 0;
    for(auto it = r.begin; it != r.end && sum < lim; ++it)
    {
        sum += make_printer<CharOut, FPack>
            ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
            .width(lim - sum);
    }
    return sum;
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
bool fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::write
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        if ( ! make_printer<CharOut, FPack>
             ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
             .write(ob) )
        {
            return false;
        }
    }
    return true;
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
class fmt_sep_range_printer: public printer<CharT>
{
    using sep_tag = stringify::v0::range_separator_input_tag<CharT>;
    using helper = stringify::v0::detail::fmt_range_helper<ForwardIt>;
    using fmt_type
    = stringify::v0::detail::sep_range_with_format<ForwardIt, CharT>;
    using fmt_type_adapted
    = typename fmt_type::template replace_fmts<Fmts...>;
    using value_fmt_type = typename helper::fmt_value_type;
    using value_fmt_type_adapted
    = typename value_fmt_type::template replace_fmts<Fmts...>;

public:

    fmt_sep_range_printer
        ( const FPack& fp
        , const fmt_type_adapted& fmt )
        : _fp(fp)
        , _fmt(fmt)
    {
    }

    std::size_t necessary_size() const override;

    int width(int limit) const override;

    bool write(stringify::v0::output_buffer<CharT>& ob) const override;

private:

    const FPack& _fp;
    fmt_type_adapted _fmt;

    template <typename Category>
    static decltype(auto) get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, sep_tag>();
    }
};

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts  >
std::size_t fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>
::necessary_size() const
{
    std::size_t len = 0;
    auto r = _fmt.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it)
    {
        len += make_printer<CharT, FPack>
            ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
            .necessary_size();
        ++ count;
    }
    if (count > 1)
    {
        len += (count - 1) * r.sep_len;
    }
    return len;
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
int fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>::width
    (int limit) const
{
    auto r = _fmt.value();
    std::size_t count = 0;
    int sum = 0;
    for(auto it = r.begin; it != r.end && sum < limit; ++it)
    {
        sum += make_printer<CharT, FPack>
            ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
            .width(limit - sum);
        ++ count;
    }
    if (count > 1 && sum < limit)
    {
        decltype(auto) wcalc
            = get_facet<stringify::v0::width_calculator_c>(_fp);
        decltype(auto) encoding
            = get_facet<stringify::v0::encoding_c<CharT>>(_fp);
        auto enc_err = get_facet<stringify::v0::encoding_error_c>(_fp);
        auto allow_surr = get_facet<stringify::v0::surrogate_policy_c>(_fp);

        int dw = wcalc.width( (limit - sum)
                            , r.sep_begin, r.sep_len
                            , encoding, enc_err, allow_surr );
        sum += dw * (count - 1);
    }
    return sum;
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
bool fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>
::write( stringify::v0::output_buffer<CharT>& ob ) const
{
    auto r = _fmt.value();
    auto it = r.begin;
    if (it != r.end)
    {
        if ( make_printer<CharT, FPack>
               (_fp, value_fmt_type_adapted{{*it}, _fmt})
               .write(ob) )
        {
            while(++it != r.end)
            {
                if ( ! detail::write_str(ob, r.sep_begin, r.sep_len)
                  || ! make_printer<CharT, FPack>
                         ( _fp, value_fmt_type_adapted{{*it}, _fmt} )
                         .write(ob) )
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
    return true;
}


template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::detail::range_printer<CharOut, FPack, ForwardIt>
make_printer
    ( const FPack& fp
    , stringify::v0::detail::range_p<ForwardIt> r )
{
    return {fp, r.begin, r.end};
}

template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::detail::sep_range_printer<CharOut, FPack, ForwardIt>
make_printer
    ( const FPack& fp
    , stringify::v0::detail::sep_range_p<ForwardIt, CharOut> r )
{
    return {fp, r.begin, r.end, r.sep_begin, r.sep_len};
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
inline
stringify::v0::detail::fmt_range_printer<CharOut, FPack, ForwardIt, Fmts...>
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
        < stringify::v0::detail::range_p<ForwardIt>
        , Fmts ... >& fmt )
{
    return {fp, fmt};
}

template
    < typename CharOut
    , typename FPack
    , typename ForwardIt
    , typename ... Fmts >
inline stringify::v0::detail::fmt_sep_range_printer< CharOut, FPack
                                                   , ForwardIt , Fmts... >
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
        < stringify::v0::detail::sep_range_p<ForwardIt, CharOut>
        , Fmts ... >& fmt )
{
    return {fp, fmt};
}

template <typename ForwardIt>
inline stringify::v0::detail::range_with_format<ForwardIt>
make_fmt(stringify::v0::tag, stringify::v0::detail::range_p<ForwardIt> r)
{
    return stringify::v0::detail::range_with_format<ForwardIt>{{r.begin, r.end}};
}

template <typename ForwardIt, typename CharT>
inline stringify::v0::detail::sep_range_with_format<ForwardIt, CharT>
make_fmt
    ( stringify::v0::tag
    , stringify::v0::detail::sep_range_p<ForwardIt, CharT> r )
{
    return stringify::v0::detail::sep_range_with_format<ForwardIt, CharT>
        {{r.begin, r.end, r.sep_begin, r.sep_len}};
}

} // namespace detail

template <typename ForwardIt>
inline auto range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::detail::range_p<ForwardIt>{begin, end};
}

template <typename ForwardIt, typename CharT>
inline auto range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::detail::sep_range_p<ForwardIt, CharT>
        {{begin, end, sep, sep_len}};
}

template <typename ForwardIt>
inline auto fmt_range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::detail::range_with_format<ForwardIt>{{begin, end}};
}

template <typename ForwardIt, typename CharT>
inline auto fmt_range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::detail::sep_range_with_format<ForwardIt, CharT>
        {{begin, end, sep, sep_len}};
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

template <typename Range, typename CharT>
inline auto range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    return stringify::v0::detail::sep_range_p
        <typename Range::const_iterator, CharT>
        {begin(range), end(range), sep, sep_len};
}

template <typename T, std::size_t N, typename CharT>
inline auto range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::detail::sep_range_p<const T*, CharT>
        {&array[0], &array[0] + N, sep, sep_len};
}

template <typename Range, typename It = typename Range::const_iterator>
inline
stringify::v0::detail::range_with_format<It> fmt_range(const Range& range)
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
    , typename CharT
    , typename It = typename Range::const_iterator >
inline auto fmt_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    stringify::v0::detail::sep_range_p<It, CharT> r
    { begin(range), end(range), sep, sep_len };
    return stringify::v0::detail::sep_range_with_format<It, CharT>{r};
}

template <typename T, std::size_t N, typename CharT>
inline auto fmt_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    return stringify::v0::detail::sep_range_with_format<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP

