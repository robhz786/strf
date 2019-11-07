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

    template <typename Preview>
    range_printer( const FPack& fp
                 , Preview& preview
                 , iterator begin
                 , iterator end )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
    {
        _preview(preview);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    void _preview(stringify::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
};

template <typename CharT, typename FPack, typename ForwardIt>
template <typename Preview, typename >
void range_printer<CharT, FPack, ForwardIt>::_preview(Preview& preview) const
{
    for(auto it = _begin; it != _end; ++it)
    {
        make_printer<CharT, FPack>(_fp, preview, *it);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
void range_printer<CharT, FPack, ForwardIt>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    stringify::print_preview<false, false> no_preview;
    for(auto it = _begin; it != _end; ++it)
    {
        make_printer<CharT, FPack>(_fp, no_preview, *it).print_to(ob);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
class sep_range_printer: public printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    sep_range_printer( const FPack& fp
                     , Preview& preview
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
        _preview(preview);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    void _preview(stringify::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    const CharT* _sep_begin;
    std::size_t _sep_len;

    template <typename Category>
    decltype(auto) _get_facet(const FPack& fp) const
    {
        using sep_tag = stringify::v0::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It>
template <typename Preview, typename>
void sep_range_printer<CharT, FPack, It>::_preview(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = _begin; it != _end; ++it)
    {
        make_printer<CharT, FPack>(_fp, preview, *it);
        ++ count;
        if ( ! Preview::size_required && preview.remaining_width() <= 0)
        {
            return;
        }
    }
    if (count < 2)
    {
        return;
    }
    if (Preview::width_required)
    {
        decltype(auto) wcalc
            = _get_facet<stringify::v0::width_calculator_c<CharT>>(_fp);
        decltype(auto) encoding
            = _get_facet<stringify::v0::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<stringify::v0::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<stringify::v0::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( preview.remaining_width(), _sep_begin, _sep_len
                             , encoding, enc_err, allow_surr );
        if (dw != 0)
        {
            if (count > UINT32_MAX)
            {
                preview.clear_remaining_width();
            }
            else
            {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required)
    {
        preview.add_size((count - 1) * _sep_len);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
void sep_range_printer<CharT, FPack, ForwardIt>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    stringify::print_preview<false, false> no_preview;
    auto it = _begin;
    if (it != _end)
    {
        make_printer<CharT, FPack>(_fp, no_preview, *it).print_to(ob);
        while (++it != _end)
        {
            stringify::v0::write(ob, _sep_begin, _sep_len);
            make_printer<CharT, FPack>(_fp, no_preview, *it).print_to(ob);
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

    template <typename Preview>
    fmt_range_printer( const FPack& fp
                     , Preview& preview
                     , const _fmt_type_adapted& fmt)
        : _fp(fp)
        , _fmt(fmt)
    {
        _preview(preview);
    }

    void print_to(stringify::v0::basic_outbuf<CharOut>& ob) const override;

private:

    void _preview(stringify::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    void _preview(Preview& preview) const;

    const FPack& _fp;
    _fmt_type_adapted _fmt;
};


template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
template <typename Preview, typename >
void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::_preview
    ( Preview& preview ) const
{
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        make_printer<CharOut, FPack>( _fp
                                    , preview
                                    , _value_fmt_type_adapted{{*it}, _fmt} );
    }
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::print_to
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    stringify::print_preview<false, false> no_preview;
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it)
    {
        make_printer<CharOut, FPack>( _fp
                                    , no_preview
                                    , _value_fmt_type_adapted{{*it}, _fmt} )
            .print_to(ob);
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

    template <typename Preview>
    fmt_sep_range_printer( const FPack& fp
                         , Preview& preview
                         , const _fmt_type_adapted& fmt )
        : _fp(fp)
        , _fmt(fmt)
    {
        _preview(preview);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    void _preview(stringify::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    void _preview(Preview& preview) const;

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
        , typename ... Fmts >
template <typename Preview, typename>
void fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>::_preview
    ( Preview& preview ) const
{
    auto r = _fmt.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it)
    {
        make_printer<CharT, FPack>( _fp
                                  , preview
                                  , _value_fmt_type_adapted{{*it}, _fmt} );
        ++ count;
        if ( ! Preview::size_required && preview.remaining_width() <= 0)
        {
            return;
        }
    }
    if (count < 2)
    {
        return;
    }
    if (Preview::width_required)
    {
        decltype(auto) wcalc
            = _get_facet<stringify::v0::width_calculator_c<CharT>>(_fp);
        decltype(auto) encoding
            = _get_facet<stringify::v0::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<stringify::v0::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<stringify::v0::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( preview.remaining_width(), r.sep_begin, r.sep_len
                             , encoding, enc_err, allow_surr );
        if (dw != 0)
        {
            if (count > UINT32_MAX)
            {
                preview.clear_remaining_width();
            }
            else
            {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required)
    {
        preview.add_size((count - 1) * r.sep_len);
    }
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
void fmt_sep_range_printer<CharT, FPack, ForwardIt, Fmts ...>
::print_to( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    stringify::print_preview<false, false> no_preview;
    auto r = _fmt.value();
    auto it = r.begin;
    if (it != r.end)
    {
        make_printer<CharT, FPack>
            ( _fp, no_preview, _value_fmt_type_adapted{{*it}, _fmt} )
            .print_to(ob);
        while(++it != r.end)
        {
            stringify::v0::write(ob, r.sep_begin, r.sep_len);
            make_printer<CharT, FPack>
                ( _fp, no_preview, _value_fmt_type_adapted{{*it}, _fmt} )
                .print_to(ob);
        }
    }
}

} // namespace detail

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt >
inline stringify::v0::detail::range_printer<CharOut, FPack, ForwardIt>
make_printer( const FPack& fp
            , Preview& preview
            , stringify::v0::range_p<ForwardIt> r )
{
    return {fp, preview, r.begin, r.end};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt >
inline stringify::v0::detail::sep_range_printer<CharOut, FPack, ForwardIt>
make_printer( const FPack& fp
            , Preview& preview
            , stringify::v0::sep_range_p<ForwardIt, CharOut> r )
{
    return {fp, preview, r.begin, r.end, r.sep_begin, r.sep_len};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename ... Fmts >
inline
stringify::v0::detail::fmt_range_printer<CharOut, FPack, ForwardIt, Fmts...>
make_printer( const FPack& fp
            , Preview& preview
            , const stringify::v0::value_with_format
                < stringify::v0::range_p<ForwardIt>
                , Fmts ... >& fmt )
{
    return {fp, preview, fmt};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename ... Fmts >
inline stringify::v0::detail::fmt_sep_range_printer< CharOut, FPack
                                                   , ForwardIt , Fmts... >
make_printer( const FPack& fp
            , Preview& preview
            , const stringify::v0::value_with_format
                < stringify::v0::sep_range_p<ForwardIt, CharOut>
                , Fmts ... >& fmt )
{
    return {fp, preview, fmt};
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

