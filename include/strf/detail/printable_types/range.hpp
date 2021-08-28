#ifndef STRF_DETAIL_INPUT_TYPES_RANGE_HPP
#define STRF_DETAIL_INPUT_TYPES_RANGE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <strf/printer.hpp>

namespace strf {

template <typename It>
struct range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
};

template <typename It, typename CharIn>
struct separated_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
    const CharIn* sep_begin;
    std::size_t sep_len;
};

template <typename It, typename UnaryOp>
struct transformed_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
    UnaryOp op;
};

template <typename It, typename CharIn, typename UnaryOp>
struct separated_transformed_range_p
{
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    It begin;
    It end;
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
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using range_with_formatters
    = strf::detail::mp_replace_front
        < VF, strf::print_traits<strf::range_p<Iterator>> >;

template < typename Iterator
         , typename CharT
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using sep_range_with_formatters
    = strf::detail::mp_replace_front
        < VF, strf::print_traits<strf::separated_range_p<Iterator, CharT>> >;

namespace detail {

template <typename CharT, typename FPack, typename It>
class range_printer;

template <typename CharT, typename FPack, typename It>
class separated_range_printer;

template <typename CharT, typename FPack, typename It, typename... Fmts>
class fmt_range_printer;

template <typename CharT, typename FPack, typename It, typename... Fmts>
class fmt_separated_range_printer;

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class transformed_range_printer;

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer;

} // namespace detail

template <typename It>
struct print_traits<strf::range_p<It>>
{
    using forwarded_type = strf::range_p<It>;
    using formatters = strf::formatters_of<decltype(*std::declval<It>())>;

    template <typename CharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , forwarded_type x)
        -> strf::usual_printer_input
            < CharT, Preview, FPack, forwarded_type
            , strf::detail::range_printer<CharT, FPack, It> >
    {
        return {preview, fp, x};
    }

    template <typename CharT, typename Preview, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<strf::print_traits<strf::range_p<It>>, Fmts...> x )
        ->  strf::usual_printer_input
            < CharT
            , Preview, FPack
            , strf::value_with_formatters<strf::print_traits<strf::range_p<It>>, Fmts ...>
            , strf::detail::fmt_range_printer<CharT, FPack, It, Fmts...> >
    {
        return {preview, fp, x};
    }
};

template <typename It, typename SepCharT>
struct print_traits<strf::separated_range_p<It, SepCharT>>
{
    using forwarded_type = strf::separated_range_p<It, SepCharT>;
    using formatters = strf::formatters_of<decltype(*std::declval<It>())>;

    template <typename DestCharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<DestCharT>
        , Preview& preview
        , const FPack& fp
        ,  forwarded_type x)
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, forwarded_type
            , strf::detail::separated_range_printer<DestCharT, FPack, It> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }

    template <typename DestCharT, typename Preview, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<DestCharT>
        , Preview& preview
        , const FPack& fp
        , strf::value_with_formatters
            < strf::print_traits<strf::separated_range_p<It, SepCharT>>, Fmts... > x )
        ->  strf::usual_printer_input
            < DestCharT
            , Preview, FPack
            , strf::value_with_formatters
                < strf::print_traits<strf::separated_range_p<It, SepCharT>>, Fmts... >
            , strf::detail::fmt_separated_range_printer<DestCharT, FPack, It, Fmts...> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }
};

template <typename It, typename UnaryOp>
struct print_traits<strf::transformed_range_p<It, UnaryOp>>
{
    using forwarded_type = strf::transformed_range_p<It, UnaryOp>;

    template <typename CharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<CharT>
        , Preview& preview
        , const FPack& fp
        , forwarded_type x)
        -> strf::usual_printer_input
            < CharT, Preview, FPack, forwarded_type
            , strf::detail::transformed_range_printer<CharT, FPack, It, UnaryOp> >
    {
        return {preview, fp, x};
    }
};

template <typename It, typename SepCharT, typename UnaryOp>
struct print_traits<strf::separated_transformed_range_p<It, SepCharT, UnaryOp>>
{
    using forwarded_type = strf::separated_transformed_range_p<It, SepCharT, UnaryOp>;

    template <typename DestCharT, typename Preview, typename FPack>
    STRF_HD constexpr static auto make_printer_input
        ( strf::tag<DestCharT>
        , Preview& preview
        , const FPack& fp
        , forwarded_type x )
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, forwarded_type
            , strf::detail::sep_transformed_range_printer<DestCharT, FPack, It, UnaryOp> >
    {
        static_assert( std::is_same<SepCharT, DestCharT>::value
                     , "Character type of range separator string is different." );
        return {preview, fp, x};
    }
};

namespace detail {

template <typename CharT, typename FPack, typename It>
class range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, strf::detail::remove_cv_t<value_type> >;

    STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template < typename Preview
             , strf::detail::enable_if_t<Preview::something_required, int> = 0 >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
};

template <typename CharT, typename FPack, typename It>
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void range_printer<CharT, FPack, It>::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, *it) );
    }
}

template <typename CharT, typename FPack, typename It>
STRF_HD void range_printer<CharT, FPack, It>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    strf::no_print_preview no_preview;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<strf::no_print_preview>
            ( strf::make_printer_input<CharT>(no_preview, fp_, *it) ).print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It>
class separated_range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, strf::detail::remove_cv_t<value_type> >;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template < typename Preview
             , strf::detail::enable_if_t<Preview::something_required, int> = 0 >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;

    template < typename Category
             , typename Tag = strf::range_separator_input_tag<CharT> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template <typename CharT, typename FPack, typename It>
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void separated_range_printer<CharT, FPack, It>::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>(strf::make_printer_input<CharT>(preview, fp_, *it));
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
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , sep_begin_
                                 , sep_len_
                                 , use_facet_<strf::surrogate_policy_c>(fp_) );
        auto acc_seps_width = checked_mul(dw, static_cast<std::uint32_t>(count - 1));
        preview.subtract_width(acc_seps_width);
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename It>
STRF_HD void separated_range_printer<CharT, FPack, It>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    strf::no_print_preview no_preview;
    auto it = begin_;
    if (it != end_) {
        printer_type_<strf::no_print_preview>
            ( strf::make_printer_input<CharT>(no_preview, fp_, *it) )
            .print_to(ob);
        while (++it != end_) {
            ob.write(sep_begin_, sep_len_);
            printer_type_<strf::no_print_preview>
                ( strf::make_printer_input<CharT>(no_preview, fp_, *it) )
                .print_to(ob);
        }
    }
}

template < typename CharT
         , typename FPack
         , typename It
         , typename ... Fmts >
class fmt_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<It>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::print_traits<strf::range_p<It>> >;

public:

    template <typename... T>
    STRF_HD fmt_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , fmt_(input.arg)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, value_fmt_type_adapted_ >;

    STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template < typename Preview
             , strf::detail::enable_if_t<Preview::something_required, int> = 0 >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;
};


template < typename CharT
         , typename FPack
         , typename It
         , typename ... Fmts >
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void fmt_range_printer<CharT, FPack, It, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>
                ( preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharT, FPack, It, Fmts ...>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    strf::no_print_preview no_preview;
    auto r = fmt_.value();
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<strf::no_print_preview>
            ( strf::make_printer_input<CharT>
                ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(ob);
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
class fmt_separated_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<It>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::print_traits<strf::separated_range_p<It, CharT>> >;

public:

    template <typename... T>
    STRF_HD fmt_separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , fmt_(input.arg)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview>
    using printer_type_ = strf::printer_type
        < CharT, Preview, FPack, value_fmt_type_adapted_ >;

    STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template < typename Preview
             , strf::detail::enable_if_t<Preview::something_required, int> = 0 >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    fmt_type_adapted_ fmt_;

    template < typename Category
             , typename Tag = strf::range_separator_input_tag<CharT>>
    static inline STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(*(const FPack*)0)))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void fmt_separated_range_printer<CharT, FPack, It, Fmts ...>::preview_
    ( Preview& preview ) const
{
    auto r = fmt_.value();
    std::size_t count = 0;
    for(auto it = r.begin; it != r.end; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>
                ( preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
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
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , r.sep_begin
                                 , r.sep_len
                                 , use_facet_<strf::surrogate_policy_c>(fp_) );
        preview.subtract_width(checked_mul(dw, static_cast<std::uint32_t>(count - 1)));
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * r.sep_len);
    }
}

template< typename CharT
        , typename FPack
        , typename It
        , typename ... Fmts >
STRF_HD void fmt_separated_range_printer<CharT, FPack, It, Fmts ...>
::print_to( strf::basic_outbuff<CharT>& ob ) const
{
    strf::no_print_preview no_preview;
    auto r = fmt_.value();
    auto it = r.begin;
    if (it != r.end) {
        printer_type_<strf::no_print_preview>
            ( strf::make_printer_input<CharT>
                ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(ob);
        while(++it != r.end) {
            ob.write(r.sep_begin, r.sep_len);
            printer_type_<strf::no_print_preview>
                ( strf::make_printer_input<CharT>
                    ( no_preview, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
                .print_to(ob);
        }
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class transformed_range_printer: public strf::printer<CharT>
{
public:

    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , op_(input.arg.op)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview, typename Op = UnaryOp>
    using printer_type_ = strf::printer_type
        < CharT
        , Preview, FPack
        , strf::detail::remove_reference_t
            < decltype(std::declval<Op>()(*std::declval<iterator>())) > >;

    STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template < typename Preview
             , strf::detail::enable_if_t<Preview::something_required, int> = 0 >
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    UnaryOp op_;
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::preview_(Preview& preview) const
{
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, op_(*it)) );
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    strf::no_print_preview no_preview;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<strf::no_print_preview>
            ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
            .print_to(ob);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
class sep_transformed_range_printer: public strf::printer<CharT>
{
public:
    using iterator = It;
    using value_type = strf::detail::iterator_value_type<It>;

    template <typename... T>
    STRF_HD sep_transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
        , op_(input.arg.op)
    {
        preview_(input.preview);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    template <typename Preview, typename Op = UnaryOp>
    using printer_type_ = strf::printer_type
        < CharT
        , Preview, FPack
        , strf::detail::remove_reference_t
            < decltype(std::declval<Op>()(*std::declval<iterator>())) > >;

    STRF_HD void preview_(strf::no_print_preview&) const
    {
    }

    template <typename Preview, strf::detail::enable_if_t<Preview::something_required, int> = 0>
    STRF_HD void preview_(Preview& preview) const;

    const FPack& fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::size_t sep_len_;
    UnaryOp op_;

    template < typename Category
             , typename Tag = strf::range_separator_input_tag<CharT> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template <typename CharT, typename FPack, typename It, typename UnaryOp>
template < typename Preview
         , strf::detail::enable_if_t<Preview::something_required, int> >
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>
    ::preview_(Preview& preview) const
{
    std::size_t count = 0;
    for(auto it = begin_; it != end_; ++it) {
        printer_type_<Preview>
            ( strf::make_printer_input<CharT>(preview, fp_, op_(*it)) );
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
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , preview.remaining_width()
                                 , sep_begin_
                                 , sep_len_
                                 , use_facet_<strf::surrogate_policy_c>(fp_) );
        preview.subtract_width(checked_mul(dw, static_cast<std::uint32_t>(count - 1)));
    }
    if (Preview::size_required) {
        preview.add_size((count - 1) * sep_len_);
    }
}

template <typename CharT, typename FPack, typename It, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, It, UnaryOp>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    using preview_type = strf::print_preview
        < strf::preview_size::no, strf::preview_width::no >;
    preview_type no_preview;
    auto it = begin_;
    if (it != end_) {
        printer_type_<preview_type>
            ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
            .print_to(ob);
        while (++it != end_) {
            ob.write(sep_begin_, sep_len_);
            printer_type_<preview_type>
                ( strf::make_printer_input<CharT>(no_preview, fp_, op_(*it)) )
                .print_to(ob);
        }
    }
}

} // namespace detail

template <typename It>
inline STRF_HD strf::range_p<It> range(It begin, It end)
{
    return {begin, end};
}

template <typename It, typename CharT>
inline STRF_HD auto separated_range(It begin, It end, const CharT* sep)
    -> strf::separated_range_p<It, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return {begin, end, sep, sep_len};
}

template < typename Range
         , typename It = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto range(const Range& r) -> strf::range_p<It>
{
    return {r.begin(), r.end()};
}

template <typename T>
inline STRF_HD auto range(std::initializer_list<T> r)
    -> strf::range_p<const T*>
{
    return {r.begin(), r.end()};
}

template <typename T, std::size_t N>
inline STRF_HD auto range(T (&array)[N]) -> strf::range_p<const T*>
{
    return {&array[0], &array[0] + N};
}

template < typename Range
         , typename CharT
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto separated_range(const Range& r, const CharT* sep)
    -> strf::separated_range_p
        <typename Range::const_iterator, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return {r.begin(), r.end(), sep, sep_len};
}

template < typename T, typename CharT >
inline STRF_HD auto separated_range(std::initializer_list<T> r, const CharT* sep)
    -> strf::separated_range_p<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return {r.begin(), r.end(), sep, sep_len};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep)
    -> strf::separated_range_p<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return {&array[0], &array[0] + N, sep, sep_len};
}

template <typename It>
inline STRF_HD auto fmt_range(It begin, It end)
    -> strf::range_with_formatters<It>
{
    return strf::range_with_formatters<It>{{begin, end}};
}

template <typename It, typename CharT>
inline STRF_HD auto fmt_separated_range(It begin, It end, const CharT* sep)
    -> strf::sep_range_with_formatters<It, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::sep_range_with_formatters<It, CharT>
        {{begin, end, sep, sep_len}};
}

template < typename Range
         , typename It = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD
strf::range_with_formatters<It> fmt_range(const Range& r)
{
    strf::range_p<It> rr{r.begin(), r.end()};
    return strf::range_with_formatters<It>{rr};
}

template <typename T>
constexpr STRF_HD strf::range_with_formatters<const T*>
fmt_range(std::initializer_list<T> r) noexcept
{
    return strf::range_with_formatters<const T*>
        { strf::range_p<const T*>{r.begin(), r.end()} };
}

template <typename T, std::size_t N>
inline STRF_HD auto fmt_range(T (&array)[N])
    -> strf::range_with_formatters<const T*>
{
    return strf::range_with_formatters<const T*>{{&array[0], &array[0] + N}};
}

template < typename Range
         , typename CharT
         , typename It = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto fmt_separated_range(const Range& r, const CharT* sep)
    -> strf::sep_range_with_formatters<It, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    strf::separated_range_p<It, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_formatters<It, CharT>{rr};
}


template <typename T, typename CharT>
inline STRF_HD auto fmt_separated_range(std::initializer_list<T> r, const CharT* sep) noexcept
    -> strf::sep_range_with_formatters<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    strf::separated_range_p<const T*, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_formatters<const T*, CharT>{rr};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto fmt_separated_range(T (&array)[N], const CharT* sep)
    -> strf::sep_range_with_formatters<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_length<CharT>(sep);
    return strf::sep_range_with_formatters<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

template < typename It
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto range(It begin, It end, UnaryOp op)
    -> strf::transformed_range_p<It, UnaryOp>
{
    return {begin, end, op};
}

template < typename Range
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>()))
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto range(const Range& r, UnaryOp op)
    -> strf::transformed_range_p<It, UnaryOp>
{
    return {r.begin(), r.end(), op};
}

template < typename T
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(std::declval<const T&>())) >
inline STRF_HD auto range(std::initializer_list<T> r, UnaryOp op)
    -> strf::transformed_range_p<const T*, UnaryOp>
{
    return {r.begin(), r.end(), op};
}



template < typename T
         , std::size_t N
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto range(T (&array)[N], UnaryOp op)
    -> strf::transformed_range_p<const T*, UnaryOp>
{
    return {&array[0], &array[0] + N, op};
}

template < typename It
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>())) >
inline STRF_HD auto separated_range(It begin, It end, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<It, CharT, UnaryOp>
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return { begin, end, sep, sep_len, op };
}

template < typename Range
         , typename CharT
         , typename UnaryOp
         , typename It = typename Range::const_iterator
         , typename = decltype(std::declval<const UnaryOp&>()(*std::declval<const It&>()))
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto separated_range(const Range& r, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<It, CharT, UnaryOp>
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return { r.begin(), r.end(), sep, sep_len, op };
}

template < typename T
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(std::declval<const T&>())) >
inline STRF_HD auto separated_range(std::initializer_list<T> r, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return { r.begin(), r.end(), sep, sep_len, op };
}

template < typename T
         , std::size_t N
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*(T*)0)) >
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
{
    std::size_t sep_len = strf::detail::str_length(sep);
    return { &array[0], &array[0] + N, sep, sep_len, op };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_RANGE_HPP

