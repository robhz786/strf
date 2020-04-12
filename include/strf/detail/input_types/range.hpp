#ifndef STRF_DETAIL_INPUT_TYPES_RANGE_HPP
#define STRF_DETAIL_INPUT_TYPES_RANGE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <strf/detail/facets/charset.hpp>

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
class range_printer: public strf::printer<sizeof(CharT)>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    STRF_HD range_printer( const FPack& fp, Preview& preview, strf::range_p<ForwardIt> r
                         , strf::tag<CharT> = strf::tag<CharT>{} )
        : fp_(fp)
        , begin_(r.begin)
        , end_(r.end)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharT)>& ob) const override;

private:

    STRF_HD void preview_
        (strf::print_preview<strf::preview_size::no, strf::preview_width::no>&) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
};

template <typename CharT, typename FPack, typename ForwardIt>
template <typename Preview, typename >
STRF_HD void range_printer<CharT, FPack, ForwardIt>::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, preview, *it);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
STRF_HD void range_printer<CharT, FPack, ForwardIt>::print_to
    ( strf::underlying_outbuf<sizeof(CharT)>& ob ) const
{
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{},fp_, no_preview, *it).print_to(ob);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
class separated_range_printer: public strf::printer<sizeof(CharT)>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    STRF_HD separated_range_printer
        ( const FPack& fp
        , Preview& preview
        , strf::separated_range_p<ForwardIt, CharT> r
        , strf::tag<CharT> = strf::tag<CharT>{} )
        : fp_(fp)
        , begin_(r.begin)
        , end_(r.end)
        , sep_begin_(r.sep_begin)
        , sep_len_(r.sep_len)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharT)>& ob) const override;

private:

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;

    template <typename Category>
    decltype(auto) get_facet_(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It>
template <typename Preview, typename>
STRF_HD void separated_range_printer<CharT, FPack, It>::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, preview, *it);
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
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        using uchar = strf::underlying_char_type<sizeof(CharT)>;
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , reinterpret_cast<const uchar*>(sep_begin_)
                                 , sep_len_
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
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
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename ForwardIt>
STRF_HD void separated_range_printer<CharT, FPack, ForwardIt>::print_to
    ( strf::underlying_outbuf<sizeof(CharT)>& ob ) const
{
    using uchar = strf::underlying_char_type<sizeof(CharT)>;
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    auto it = begin_;
    if (it != end_) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, no_preview, *it)
            .print_to(ob);
        while (++it != end_) {
            strf::write(ob, reinterpret_cast<const uchar*>(sep_begin_), sep_len_);
            make_printer<CharT, FPack>(strf::rank<5>{}, fp_, no_preview, *it)
                .print_to(ob);
        }
    }
}

template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
class fmt_range_printer: public strf::printer<sizeof(CharOut)>
{
    using value_type_ = typename std::iterator_traits<ForwardIt>::value_type;
    using value_fmt_type_
        = decltype( make_fmt( strf::rank<5>{}
                            , std::declval<const value_type_&>()) );
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_ = detail::mp_replace_front
        < value_fmt_type_
        , strf::range_p<ForwardIt> >;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::range_p<ForwardIt> >;

public:

    template <typename Preview>
    STRF_HD fmt_range_printer
        ( const FPack& fp
        , Preview& preview
        , const fmt_type_adapted_& fmt
        , strf::tag<CharOut> = strf::tag<CharOut>{} )
        : fp_(fp)
        , fmt_(fmt)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharOut)>& ob) const override;

private:

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;
};


template < typename CharOut
         , typename FPack
         , typename ForwardIt
         , typename ... Fmts >
template <typename Preview, typename >
STRF_HD void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharOut, FPack>( strf::rank<5>{}
                                    , fp_
                                    , preview
                                    , value_fmt_type_adapted_{{*it}, fmt_} );
    }
}

template< typename CharOut
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharOut, FPack, ForwardIt, Fmts ...>::print_to
    ( strf::underlying_outbuf<sizeof(CharOut)>& ob ) const
{
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharOut, FPack>( strf::rank<5>{}
                                    , fp_
                                    , no_preview
                                    , value_fmt_type_adapted_{{*it}, fmt_} )
            .print_to(ob);
    }
}

template< typename CharT
        , typename FPack
        , typename ForwardIt
        , typename ... Fmts >
class fmt_separated_range_printer: public strf::printer<sizeof(CharT)>
{
    using value_type_ = typename std::iterator_traits<ForwardIt>::value_type;
    using value_fmt_type_
        = decltype( make_fmt( strf::rank<5>{}
                            , std::declval<const value_type_&>()) );
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_ = detail::mp_replace_front
        < value_fmt_type_
        , strf::separated_range_p<ForwardIt, CharT> >;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::separated_range_p<ForwardIt, CharT> >;

public:

    template <typename Preview>
    STRF_HD fmt_separated_range_printer
        ( const FPack& fp
        , Preview& preview
        , const fmt_type_adapted_& fmt
        , strf::tag<CharT> = strf::tag<CharT>{} )
        : fp_(fp)
        , fmt_(fmt)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharT)>& ob) const override;

private:

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;

    template <typename Category>
    static inline STRF_HD decltype(auto) get_facet_(const FPack& fp)
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
STRF_HD void fmt_separated_range_printer<CharT, FPack, ForwardIt, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it) {
        make_printer<CharT, FPack>( strf::rank<5>{}
                                  , fp_
                                  , preview
                                  , value_fmt_type_adapted_{{*it}, fmt_} );
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
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        using uchar = strf::underlying_char_type<sizeof(CharT)>;
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , reinterpret_cast<const uchar*>(r.sep_begin)
                                 , r.sep_len
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
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
::print_to( strf::underlying_outbuf<sizeof(CharT)>& ob ) const
{
    using uchar = strf::underlying_char_type<sizeof(CharT)>;
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    auto r = fmt_.value();
    auto it = r.begin;
    if (it != r.end) {
        make_printer<CharT, FPack>
            ( strf::rank<5>{}, fp_, no_preview, value_fmt_type_adapted_{{*it}, fmt_} )
            .print_to(ob);
        while(++it != r.end) {
            strf::write(ob, reinterpret_cast<const uchar*>(r.sep_begin), r.sep_len);
            make_printer<CharT, FPack>
                ( strf::rank<5>{}, fp_, no_preview, value_fmt_type_adapted_{{*it}, fmt_} )
                .print_to(ob);
        }
    }
}

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
class transformed_range_printer: public strf::printer<sizeof(CharT)>
{
public:

    using iterator = ForwardIt;
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;

    template <typename Preview>
    transformed_range_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::transformed_range_p<ForwardIt, UnaryOp>& r
        , strf::tag<CharT> = strf::tag<CharT>{} )
        : fp_(fp)
        , begin_(r.begin)
        , end_(r.end)
        , op_(r.op)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharT)>& ob) const override;

private:

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    UnaryOp op_;
};

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
template <typename Preview, typename >
STRF_HD void transformed_range_printer<CharT, FPack, ForwardIt, UnaryOp>
    ::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, preview, op_(*it));
    }
}

template <typename CharT, typename FPack, typename ForwardIt, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, ForwardIt, UnaryOp>::print_to
    ( strf::underlying_outbuf<sizeof(CharT)>& ob ) const
{
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{},fp_, no_preview, op_(*it))
            .print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer: public strf::printer<sizeof(CharT)>
{
public:

    using iterator = It;
    using value_type = typename std::iterator_traits<It>::value_type;
    template <typename Preview>
    STRF_HD sep_transformed_range_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::separated_transformed_range_p<It, CharT, UnaryOp>& r
        , strf::tag<CharT> = strf::tag<CharT>{} )
        : fp_(fp)
        , begin_(r.begin)
        , end_(r.end)
        , sep_begin_(r.sep_begin)
        , sep_len_(r.sep_len)
        , op_(r.op)
    {
        preview_(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<sizeof(CharT)>& ob) const override;

private:

    STRF_HD void preview_
        ( strf::print_preview<strf::preview_size::no, strf::preview_width::no>& ) const
    {
    }

    template < typename Preview
             , typename = std::enable_if_t< Preview::size_required
                                         || Preview::width_required > >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;
    UnaryOp op_;

    template <typename Category>
    STRF_HD decltype(auto) get_facet_(const FPack& fp) const
    {
        using sep_tag = strf::range_separator_input_tag<CharT>;
        return fp.template get_facet<Category, sep_tag>();
    }
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template <typename Preview, typename>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, preview, op_(*it));
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
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(fp_);
        using uchar = strf::underlying_char_type<sizeof(CharT)>;
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , reinterpret_cast<const uchar*>(sep_begin_)
                                 , sep_len_
                                 , get_facet_<strf::surrogate_policy_c>(fp_) );
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
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::underlying_outbuf<sizeof(CharT)>& ob ) const
{
    strf::print_preview<strf::preview_size::no, strf::preview_width::no> no_preview;
    using uchar = strf::underlying_char_type<sizeof(CharT)>;
    auto it = begin_;
    if (it != end_) {
        make_printer<CharT, FPack>(strf::rank<5>{}, fp_, no_preview, op_(*it))
            .print_to(ob);
        while (++it != end_) {
            strf::write(ob, reinterpret_cast<const uchar*>(sep_begin_), sep_len_);
            make_printer<CharT, FPack>(strf::rank<5>{}, fp_, no_preview, op_(*it))
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
    return {fp, preview, r};
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
    return {fp, preview, r};
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
    return {fp, preview, r};
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
    return {fp, preview, r};
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
    std::size_t sep_len = strf::detail::str_length(sep);
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
    std::size_t sep_len = strf::detail::str_length(sep);
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
    std::size_t sep_len = strf::detail::str_length(sep);
    return strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
        { &array[0], &array[0] + N, sep, sep_len, op };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_RANGE_HPP

