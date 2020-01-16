#ifndef STRF_DETAIL_INPUT_TYPES_RANGE_HPP
#define STRF_DETAIL_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <strf/detail/facets/encoding.hpp>

namespace strf {

template <typename ForwardIt>
struct range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
};

template <typename ForwardIt, typename CharIn>
struct separated_range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
    const CharIn* sep_begin;
    std::size_t sep_len;
};

template <typename ForwardIt, typename UnaryOp>
struct transformed_range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
    UnaryOp op;
};

template <typename ForwardIt, typename CharIn, typename UnaryOp>
struct separated_transformed_range_p
{
    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    ForwardIt begin;
    ForwardIt end;
    const CharIn* sep_begin;
    std::size_t sep_len;
    UnaryOp op;
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
    = typename strf::detail::mp_replace_front_impl<T, List>::type;

} // namespace detail

template < typename Iterator
         , typename V  = typename std::iterator_traits<Iterator>::value_type
         , typename VF = decltype( make_fmt( strf::rank<5>{}
                                           , std::declval<const V&>()) ) >
using range_with_format
    = strf::detail::mp_replace_front
        < VF, strf::range_p<Iterator> >;

template < typename Iterator
         , typename CharT
         , typename V  = typename std::iterator_traits<Iterator>::value_type
         , typename VF = decltype( make_fmt( strf::rank<5>{}
                                           , std::declval<const V&>()) ) >
using sep_range_with_format
    = strf::detail::mp_replace_front
        < VF, strf::separated_range_p<Iterator, CharT> >;

namespace detail {

template <typename CharT, typename FPack, typename ForwardIt>
class range_printer: public printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    STRF_HD range_printer( const FPack& fp
                 , Preview& preview
                 , iterator begin
                 , iterator end )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
    {
        _preview(preview);
    }

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
};

template <typename CharT, typename FPack, typename ForwardIt>
template <typename Preview, typename >
STRF_HD void range_printer<CharT, FPack, ForwardIt>::_preview(Preview& preview) const
{
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, preview, *it);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
STRF_HD void range_printer<CharT, FPack, ForwardIt>::print_to
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{},_fp, no_preview, *it).print_to(ob);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
class separated_range_printer: public printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    STRF_HD separated_range_printer
        ( const FPack& fp
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

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    const CharT* _sep_begin;
    std::size_t _sep_len;

    template <typename Category>
    decltype(auto) _get_facet(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It>
template <typename Preview, typename>
STRF_HD void separated_range_printer<CharT, FPack, It>::_preview(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, preview, *it);
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc
            = _get_facet<strf::width_calculator_c<CharT>>(_fp);
        decltype(auto) encoding
            = _get_facet<strf::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<strf::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<strf::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( preview.remaining_width(), _sep_begin, _sep_len
                             , encoding, enc_err, allow_surr );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * _sep_len);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
STRF_HD void separated_range_printer<CharT, FPack, ForwardIt>::print_to
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    auto it = _begin;
    if (it != _end) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, no_preview, *it)
            .print_to(ob);
        while (++it != _end) {
            strf::write(ob, _sep_begin, _sep_len);
            make_printer<CharT, FPack>(strf::rank<5>{}, _fp, no_preview, *it)
                .print_to(ob);
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
        = decltype( make_fmt( strf::rank<5>{}
                            , std::declval<const _value_type&>()) );
    using _value_fmt_type_adapted
        = typename _value_fmt_type::template replace_fmts<Fmts...>;

    using _fmt_type = detail::mp_replace_front
        < _value_fmt_type
        , strf::range_p<ForwardIt> >;

    using _fmt_type_adapted = detail::mp_replace_front
        < _value_fmt_type_adapted
        , strf::range_p<ForwardIt> >;

public:

    template <typename Preview>
    STRF_HD fmt_range_printer( const FPack& fp
                     , Preview& preview
                     , const _fmt_type_adapted& fmt)
        : _fp(fp)
        , _fmt(fmt)
    {
        _preview(preview);
    }

    STRF_HD void print_to(strf::basic_outbuf<CharOut>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    _fmt_type_adapted _fmt;
};


template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
template <typename Preview, typename >
STRF_HD void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::_preview
    ( Preview& preview ) const
{
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharOut, FPack>( strf::rank<5>{}
                                    , _fp
                                    , preview
                                    , _value_fmt_type_adapted{{*it}, _fmt} );
    }
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::print_to
    ( strf::basic_outbuf<CharOut>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    auto r = _fmt.value();
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharOut, FPack>( strf::rank<5>{}
                                    , _fp
                                    , no_preview
                                    , _value_fmt_type_adapted{{*it}, _fmt} )
            .print_to(ob);
    }
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
class fmt_separated_range_printer: public printer<CharT>
{
    using _value_type = typename std::iterator_traits<ForwardIt>::value_type;
    using _value_fmt_type
        = decltype( make_fmt( strf::rank<5>{}
                            , std::declval<const _value_type&>()) );
    using _value_fmt_type_adapted
        = typename _value_fmt_type::template replace_fmts<Fmts...>;

    using _fmt_type = detail::mp_replace_front
        < _value_fmt_type
        , strf::separated_range_p<ForwardIt, CharT> >;

    using _fmt_type_adapted = detail::mp_replace_front
        < _value_fmt_type_adapted
        , strf::separated_range_p<ForwardIt, CharT> >;

public:

    template <typename Preview>
    STRF_HD fmt_separated_range_printer( const FPack& fp
                         , Preview& preview
                         , const _fmt_type_adapted& fmt )
        : _fp(fp)
        , _fmt(fmt)
    {
        _preview(preview);
    }

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    _fmt_type_adapted _fmt;

    template <typename Category>
    static inline STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
template <typename Preview, typename>
STRF_HD void fmt_separated_range_printer<CharT, FPack, ForwardIt, Fmts ...>::_preview
    ( Preview& preview ) const
{
    auto r = _fmt.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharT, FPack>( strf::rank<5>{}
                                  , _fp
                                  , preview
                                  , _value_fmt_type_adapted{{*it}, _fmt} );
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc
            = _get_facet<strf::width_calculator_c<CharT>>(_fp);
        decltype(auto) encoding
            = _get_facet<strf::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<strf::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<strf::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( preview.remaining_width(), r.sep_begin, r.sep_len
                             , encoding, enc_err, allow_surr );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * r.sep_len);
    }
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
STRF_HD void fmt_separated_range_printer<CharT, FPack, ForwardIt, Fmts ...>
::print_to( strf::basic_outbuf<CharT>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    auto r = _fmt.value();
    auto it = r.begin;
    if (it != r.end) {
        make_printer<CharT, FPack>
            ( strf::rank<5>{}, _fp, no_preview, _value_fmt_type_adapted{{*it}, _fmt} )
            .print_to(ob);
        while(++it != r.end) {
            strf::write(ob, r.sep_begin, r.sep_len);
            make_printer<CharT, FPack>
                ( strf::rank<5>{}, _fp, no_preview, _value_fmt_type_adapted{{*it}, _fmt} )
                .print_to(ob);
        }
    }
}

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
class transformed_range_printer: public strf::printer<CharT>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    transformed_range_printer( const FPack& fp
                             , Preview& preview
                             , iterator begin
                             , iterator end
                             , UnaryOp op )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
        , _op(op)
    {
        _preview(preview);
    }

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    UnaryOp _op;
};

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
template <typename Preview, typename >
STRF_HD void transformed_range_printer<CharT, FPack, ForwardIt, UnaryOp>
    ::_preview(Preview& preview) const
{
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, preview, _op(*it));
    }
}

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, ForwardIt, UnaryOp>::print_to
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{},_fp, no_preview, _op(*it))
            .print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer: public printer<CharT>
{
public:

    using iterator = It;
    using value_type = typename std::iterator_traits<It>::value_type;

    template <typename Preview>
    STRF_HD sep_transformed_range_printer
        ( const FPack& fp
        , Preview& preview
        , iterator begin
        , iterator end
        , const CharT* sep
        , std::size_t sep_len
        , UnaryOp op )
        : _fp(fp)
        , _begin(begin)
        , _end(end)
        , _sep_begin(sep)
        , _sep_len(sep_len)
        , _op(op)
    {
        _preview(preview);
    }

    STRF_HD void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    STRF_HD void _preview(strf::print_preview<false, false>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void _preview(Preview& preview) const;

    const FPack& _fp;
    iterator _begin;
    iterator _end;
    const CharT* _sep_begin;
    std::size_t _sep_len;
    UnaryOp _op;

    template <typename Category>
    STRF_HD decltype(auto) _get_facet(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template <typename Preview, typename>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::_preview(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = _begin; it != _end; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, preview, _op(*it));
        ++ count;
        STRF_IF_CONSTEXPR (!Preview::size_required) {
            if (preview.remaining_width() <= 0) {
                return;
            }
        }
    }
    if (count < 2) {
        return;
    }
    if (Preview::width_required) {
        decltype(auto) wcalc
            = _get_facet<strf::width_calculator_c<CharT>>(_fp);
        decltype(auto) encoding
            = _get_facet<strf::encoding_c<CharT>>(_fp);
        auto enc_err = _get_facet<strf::encoding_error_c>(_fp);
        auto allow_surr = _get_facet<strf::surrogate_policy_c>(_fp);

        auto dw = wcalc.width( preview.remaining_width(), _sep_begin, _sep_len
                             , encoding, enc_err, allow_surr );
        if (dw != 0) {
            if (count > UINT32_MAX) {
                preview.clear_remaining_width();
            } else {
                preview.checked_subtract_width
                    ( checked_mul(dw, static_cast<std::uint32_t>(count - 1)) );
            }
        }
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * _sep_len);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::print_preview<false, false> no_preview;
    auto it = _begin;
    if (it != _end) {
        make_printer<CharT, FPack>(strf::rank<5>{}, _fp, no_preview, _op(*it))
            .print_to(ob);
        while (++it != _end) {
            strf::write(ob, _sep_begin, _sep_len);
            make_printer<CharT, FPack>(strf::rank<5>{}, _fp, no_preview, _op(*it))
                .print_to(ob);
        }
    }
}

} // namespace detail

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt >
inline STRF_HD strf::detail::range_printer<CharOut, FPack, ForwardIt>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::range_p<ForwardIt> r )
{
    return {fp, preview, r.begin, r.end};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt >
inline STRF_HD strf::detail::separated_range_printer<CharOut, FPack, ForwardIt>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::separated_range_p<ForwardIt, CharOut> r )
{
    return {fp, preview, r.begin, r.end, r.sep_begin, r.sep_len};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename ... Fmts >
inline STRF_HD
strf::detail::fmt_range_printer<CharOut, FPack, ForwardIt, Fmts...>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::value_with_format
                < strf::range_p<ForwardIt>
                , Fmts ... >& fmt )
{
    return {fp, preview, fmt};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename ... Fmts >
inline STRF_HD strf::detail::fmt_separated_range_printer< CharOut, FPack
                                                        , ForwardIt , Fmts... >
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::value_with_format
                < strf::separated_range_p<ForwardIt, CharOut>
                , Fmts ... >& fmt )
{
    return {fp, preview, fmt};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename UnaryOp >
inline STRF_HD strf::detail::transformed_range_printer< CharOut, FPack
                                                      , ForwardIt, UnaryOp >
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::transformed_range_p<ForwardIt, UnaryOp>& r )
{
    return {fp, preview, r.begin, r.end, r.op};
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename ForwardIt
         , typename UnaryOp >
inline STRF_HD strf::detail::sep_transformed_range_printer< CharOut, FPack
                                                          , ForwardIt, UnaryOp >
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::separated_transformed_range_p<ForwardIt, CharOut, UnaryOp>& r )
{
    return {fp, preview, r.begin, r.end, r.sep_begin, r.sep_len, r.op};
}

template <typename ForwardIt>
inline STRF_HD strf::range_with_format<ForwardIt>
make_fmt(strf::rank<1>, strf::range_p<ForwardIt> r)
{
    return strf::range_with_format<ForwardIt>{{r.begin, r.end}};
}

template <typename ForwardIt, typename CharT>
inline STRF_HD strf::sep_range_with_format<ForwardIt, CharT>
make_fmt( strf::rank<1>
        , strf::separated_range_p<ForwardIt, CharT> r )
{
    return strf::sep_range_with_format<ForwardIt, CharT>
        {{r.begin, r.end, r.sep_begin, r.sep_len}};
}

template <typename ForwardIt, typename UnaryOp>
void make_fmt
    ( strf::rank<1>
    , strf::transformed_range_p<ForwardIt, UnaryOp> ) = delete;

template <typename ForwardIt, typename CharT, typename UnaryOp>
void make_fmt
    ( strf::rank<1>
    , strf::separated_transformed_range_p<ForwardIt, CharT, UnaryOp> ) = delete;

template <typename ForwardIt>
inline STRF_HD auto range(ForwardIt begin, ForwardIt end)
{
    return strf::range_p<ForwardIt>{begin, end};
}

template <typename ForwardIt, typename CharT>
inline STRF_HD auto separated_range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::separated_range_p<ForwardIt, CharT>
        {begin, end, sep, sep_len};
}

template <typename Range, typename It = typename Range::const_iterator>
inline STRF_HD auto range(const Range& range)
{
    using namespace std;
    return strf::range_p<It>{begin(range), end(range)};
}

template <typename T, std::size_t N>
inline STRF_HD auto range(T (&array)[N])
{
    return strf::range_p<const T*>{&array[0], &array[0] + N};
}

template <typename Range, typename CharT>
inline STRF_HD auto separated_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    return strf::separated_range_p
        <typename Range::const_iterator, CharT>
        {begin(range), end(range), sep, sep_len};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::separated_range_p<const T*, CharT>
        {&array[0], &array[0] + N, sep, sep_len};
}

template <typename ForwardIt>
inline STRF_HD auto fmt_range(ForwardIt begin, ForwardIt end)
{
    return strf::range_with_format<ForwardIt>{{begin, end}};
}

template <typename ForwardIt, typename CharT>
inline STRF_HD auto fmt_separated_range(ForwardIt begin, ForwardIt end, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::sep_range_with_format<ForwardIt, CharT>
        {{begin, end, sep, sep_len}};
}

template <typename Range, typename It = typename Range::const_iterator>
inline STRF_HD
strf::range_with_format<It> fmt_range(const Range& range)
{
    using namespace std;
    strf::range_p<It> r{begin(range), end(range)};
    return strf::range_with_format<It>{r};
}

template <typename T, std::size_t N>
inline STRF_HD auto fmt_range(T (&array)[N])
{
    using namespace std;
    using fmt_type = strf::range_with_format<const T*>;
    return fmt_type{{&array[0], &array[0] + N}};
}

template < typename Range
         , typename CharT
         , typename It = typename Range::const_iterator >
inline STRF_HD auto fmt_separated_range(const Range& range, const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    strf::separated_range_p<It, CharT> r
    { begin(range), end(range), sep, sep_len };
    return strf::sep_range_with_format<It, CharT>{r};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto fmt_separated_range(T (&array)[N], const CharT* sep)
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    using namespace std;
    return strf::sep_range_with_format<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

template < typename It
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto range(It begin, It end, UnaryOp op)
{
    return strf::transformed_range_p<It, UnaryOp>{begin, end, op};
}

template < typename Range
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto range(const Range& range, UnaryOp op)
{
    using namespace std;
    return strf::transformed_range_p<It, UnaryOp>{begin(range), end(range), op};
}

template < typename T
         , std::size_t N
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto range(T (&array)[N], UnaryOp op)
{
    return strf::transformed_range_p<const T*, UnaryOp>
        {&array[0], &array[0] + N, op};
}

template < typename It
         , typename CharT
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto separated_range(It begin, It end, const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return strf::separated_transformed_range_p<It, CharT, UnaryOp>
        { begin, end, sep, sep_len, op };
}

template < typename Range
         , typename CharT
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto separated_range(const Range& range, const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    using namespace std;
    return strf::separated_transformed_range_p<It, CharT, UnaryOp>
        { begin(range), end(range), sep, sep_len, op };
}

template < typename T
         , std::size_t N
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep, UnaryOp op)
{
    std::size_t sep_len = std::char_traits<CharT>::length(sep);
    return strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
        { &array[0], &array[0] + N, sep, sep_len, op };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_RANGE_HPP

