#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <boost/stringify/v0/detail/facets/encoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

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

namespace detail {

template <typename List, typename T>
struct mp_replace_front_impl;

template < typename NewFirst
         , template <typename...> class List
         , typename First
         , typename ... Rest >
struct mp_replace_front_impl<List<First, Rest...>, NewFirst>
{
    using type = List<NewFirst, Rest...>;
};

template <typename T, typename List>
using mp_replace_front
    = typename stringify::v0::detail::mp_replace_front_impl<T, List>::type;

} // namespace detail

template < typename Iterator
         , typename V  = typename std::iterator_traits<Iterator>::value_type
         , typename VF = decltype( make_fmt( stringify::v0::tag{}
                                           , std::declval<const V&>()) ) >
using range_with_format
    = stringify::v0::detail::mp_replace_front
        < VF, stringify::v0::range_p<Iterator> >;

template < typename Iterator
         , typename CharT
         , typename V  = typename std::iterator_traits<Iterator>::value_type
         , typename VF = decltype( make_fmt( stringify::v0::tag{}
                                           , std::declval<const V&>()) ) >
using sep_range_with_format
    = stringify::v0::detail::mp_replace_front
        < VF, stringify::v0::sep_range_p<Iterator, CharT> >;

namespace detail {

template <typename CharT, typename FPack, typename ForwardIt>
class range_printer: public printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    range_printer(const FPack& fp, iterator begin, iterator end)
        : _fp(fp)
        , _begin(begin)
        , _end(end)
    {
    }

    std::size_t necessary_size() const override;

    int width(int limit) const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

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
void range_printer<CharT, FPack, ForwardIt>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    for(auto it = _begin; it != _end; ++it)
    {
        make_printer<CharT, FPack>(_fp, *it).write(ob);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
class sep_range_printer: public printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    sep_range_printer( const FPack& fp
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

    void write(boost::basic_outbuf<CharT>& ob) const override;

private:

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    const CharT* _sep_begin;
    std::size_t _sep_len;

    template <typename Category>
    decltype(auto) get_facet(const FPack& fp) const
    {
        using sep_tag = stringify::v0::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename ForwardIt>
int sep_range_printer<CharT, FPack, ForwardIt>::width(int limit) const
{
    int count = 0;
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
void sep_range_printer<CharT, FPack, ForwardIt>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    auto it = _begin;
    if (it != _end)
    {
        make_printer<CharT, FPack>(_fp, *it).write(ob);
        while (++it != _end)
        {
            boost::write(ob, _sep_begin, _sep_len);
            make_printer<CharT, FPack>(_fp, *it).write(ob);
        }
    }
}

template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
class fmt_range_printer: public printer<CharOut>
{
    using _value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using _value_fmt_type
        = decltype( make_fmt( stringify::v0::tag{}
                            , std::declval<const _value_type&>()) );
    using _value_fmt_type_adapted
        = typename _value_fmt_type::template replace_fmts<Fmts...>;

    using _fmt_type = detail::mp_replace_front
        < _value_fmt_type
        , stringify::v0::range_p<ForwardIt> >;

    using _fmt_type_adapted = detail::mp_replace_front
        < _value_fmt_type_adapted
        , stringify::v0::range_p<ForwardIt> >;

public:

    fmt_range_printer(const FPack& fp, const _fmt_type_adapted& fmt)
        : _fp(fp)
        , _fmt(fmt)
    {
    }

    std::size_t necessary_size() const override;

    int width(int lim) const override;

    void write(boost::basic_outbuf<CharOut>& ob) const override;

private:

    const FPack& _fp;
    _fmt_type_adapted _fmt;
};


template < typename CharOut
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
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
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
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
            .width(lim - sum);
    }
    return sum;
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::write
    ( boost::basic_outbuf<CharOut>& ob ) const
{
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        make_printer<CharOut, FPack>
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
            .write(ob);
    }
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
class fmt_sep_range_printer: public printer<CharT>
{
    using _value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using _value_fmt_type
        = decltype( make_fmt( stringify::v0::tag{}
                            , std::declval<const _value_type&>()) );
    using _value_fmt_type_adapted
        = typename _value_fmt_type::template replace_fmts<Fmts...>;

    using _fmt_type = detail::mp_replace_front
        < _value_fmt_type
        , stringify::v0::sep_range_p<ForwardIt, CharT> >;

    using _fmt_type_adapted = detail::mp_replace_front
        < _value_fmt_type_adapted
        , stringify::v0::sep_range_p<ForwardIt, CharT> >;

public:

    fmt_sep_range_printer(const FPack& fp, const _fmt_type_adapted& fmt)
        : _fp(fp)
        , _fmt(fmt)
    {
    }

    std::size_t necessary_size() const override;

    int width(int limit) const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

private:

    const FPack& _fp;
    _fmt_type_adapted _fmt;

    template <typename Category>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using sep_tag = stringify::v0::range_separator_input_tag<CharT>;
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
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
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
    int count = 0;
    int sum = 0;
    for(auto it = r.begin; it != r.end && sum < limit; ++it)
    {
        sum += make_printer<CharT, FPack>
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
            .width(limit - sum);
        ++ count;
    }
    if (count > 1 && sum < limit)
    {
        decltype(auto) wcalc
            = _get_facet<stringify::v0::width_calculator_c>(_fp);
        decltype(auto) encoding
            = _get_facet<stringify::v0::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<stringify::v0::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<stringify::v0::surrogate_policy_c>(_fp);

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
void fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>
::write( boost::basic_outbuf<CharT>& ob ) const
{
    auto r = _fmt.value();
    auto it = r.begin;
    if (it != r.end)
    {
        make_printer<CharT, FPack>
            ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
            .write(ob);
        while(++it != r.end)
        {
            boost::write(ob, r.sep_begin, r.sep_len);
            make_printer<CharT, FPack>
                ( _fp, _value_fmt_type_adapted{{*it}, _fmt} )
                .write(ob);
        }
    }
}

} // namespace detail

template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::detail::range_printer<CharOut, FPack, ForwardIt>
make_printer( const FPack& fp
            , stringify::v0::range_p<ForwardIt> r )
{
    return {fp, r.begin, r.end};
}

template <typename CharOut, typename FPack, typename ForwardIt>
inline stringify::v0::detail::sep_range_printer<CharOut, FPack, ForwardIt>
make_printer( const FPack& fp
            , stringify::v0::sep_range_p<ForwardIt, CharOut> r )
{
    return {fp, r.begin, r.end, r.sep_begin, r.sep_len};
}

template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
inline
stringify::v0::detail::fmt_range_printer<CharOut, FPack, ForwardIt, Fmts...>
make_printer( const FPack& fp
            , const stringify::v0::value_with_format
                < stringify::v0::range_p<ForwardIt>
                , Fmts ... >& fmt )
{
    return {fp, fmt};
}

template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
inline stringify::v0::detail::fmt_sep_range_printer< CharOut, FPack
                                                   , ForwardIt , Fmts... >
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
        < stringify::v0::sep_range_p<ForwardIt, CharOut>
        , Fmts ... >& fmt )
{
    return {fp, fmt};
}

template <typename ForwardIt>
inline stringify::v0::range_with_format<ForwardIt>
make_fmt(stringify::v0::tag, stringify::v0::range_p<ForwardIt> r)
{
    return stringify::v0::range_with_format<ForwardIt>{{r.begin, r.end}};
}

template <typename ForwardIt, typename CharT>
inline stringify::v0::sep_range_with_format<ForwardIt, CharT>
make_fmt( stringify::v0::tag
        , stringify::v0::sep_range_p<ForwardIt, CharT> r )
{
    return stringify::v0::sep_range_with_format<ForwardIt, CharT>
        {{r.begin, r.end, r.sep_begin, r.sep_len}};
}

template <typename ForwardIt>
inline auto range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::range_p<ForwardIt>{begin, end};
}

template <typename ForwardIt, typename CharT>
inline auto range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::sep_range_p<ForwardIt, CharT>
        {{begin, end, sep, sep_len}};
}

template <typename Range, typename It = typename Range::const_iterator>
inline auto range(const Range& range)
{
    using namespace std;
    return stringify::v0::range_p<It>{begin(range), end(range)};
}

template <typename T, std::size_t N>
inline auto range(T (&array)[N])
{
    return stringify::v0::range_p<const T*>{&array[0], &array[0] + N};
}

template <typename Range, typename CharT>
inline auto range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    return stringify::v0::sep_range_p
        <typename Range::const_iterator, CharT>
        {begin(range), end(range), sep, sep_len};
}

template <typename T, std::size_t N, typename CharT>
inline auto range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::sep_range_p<const T*, CharT>
        {&array[0], &array[0] + N, sep, sep_len};
}

template <typename ForwardIt>
inline auto fmt_range(ForwardIt begin, ForwardIt end)
{
    return stringify::v0::range_with_format<ForwardIt>{{begin, end}};
}

template <typename ForwardIt, typename CharT>
inline auto fmt_range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return stringify::v0::sep_range_with_format<ForwardIt, CharT>
        {{begin, end, sep, sep_len}};
}

template <typename Range, typename It = typename Range::const_iterator>
inline
stringify::v0::range_with_format<It> fmt_range(const Range& range)
{
    using namespace std;
    stringify::v0::range_p<It> r{begin(range), end(range)};
    return stringify::v0::range_with_format<It>{r};
}

template <typename T, std::size_t N>
inline auto fmt_range(T (&array)[N])
{
    using namespace std;
    using fmt_type = stringify::v0::range_with_format<const T*>;
    return fmt_type{{&array[0], &array[0] + N}};
}

template < typename Range
         , typename CharT
         , typename It = typename Range::const_iterator >
inline auto fmt_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    stringify::v0::sep_range_p<It, CharT> r
    { begin(range), end(range), sep, sep_len };
    return stringify::v0::sep_range_with_format<It, CharT>{r};
}

template <typename T, std::size_t N, typename CharT>
inline auto fmt_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    return stringify::v0::sep_range_with_format<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_RANGE_HPP

