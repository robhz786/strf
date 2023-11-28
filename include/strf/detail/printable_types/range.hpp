#ifndef STRF_DETAIL_PRINTABLE_TYPES_RANGE_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_RANGE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <initializer_list>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/charset.hpp>

namespace strf {

template <typename Iterator>
struct range_p
{
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    Iterator begin;
    Iterator end;
};

template <typename Iterator, typename CharIn>
struct separated_range_p
{
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    Iterator begin;
    Iterator end;
    const CharIn* sep_begin;
    std::ptrdiff_t sep_len;
};

template <typename Iterator, typename UnaryOp>
struct transformed_range_p
{
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    Iterator begin;
    Iterator end;
    UnaryOp op;
};

template <typename Iterator, typename CharIn, typename UnaryOp>
struct separated_transformed_range_p
{
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    Iterator begin;
    Iterator end;
    const CharIn* sep_begin;
    std::ptrdiff_t sep_len;
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
        < VF, strf::printable_traits<strf::range_p<Iterator>> >;

template < typename Iterator
         , typename CharT
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using sep_range_with_formatters
    = strf::detail::mp_replace_front
        < VF, strf::printable_traits<strf::separated_range_p<Iterator, CharT>> >;

namespace detail {

template <typename CharT, typename FPack, typename Iterator>
class range_printer;

template <typename CharT, typename FPack, typename Iterator>
class separated_range_printer;

template <typename CharT, typename FPack, typename Iterator, typename... Fmts>
class fmt_range_printer;

template <typename CharT, typename FPack, typename Iterator, typename... Fmts>
class fmt_separated_range_printer;

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class transformed_range_printer;

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class sep_transformed_range_printer;

} // namespace detail

template <typename Iterator>
struct printable_traits<strf::range_p<Iterator>>
{
    using representative_type = strf::range_p<Iterator>;
    using forwarded_type = strf::range_p<Iterator>;
    using formatters = strf::formatters_of<decltype(*std::declval<Iterator>())>;
    using is_overridable = std::false_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x)
        -> strf::usual_printer_input
            < CharT, PreMeasurements, FPack, forwarded_type
            , strf::detail::range_printer<CharT, FPack, Iterator> >
    {
        return {pre, fp, x};
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::printable_with_fmt<strf::printable_traits<strf::range_p<Iterator>>, Fmts...> x )
        ->  strf::usual_printer_input
            < CharT
            , PreMeasurements, FPack
            , strf::printable_with_fmt<strf::printable_traits<strf::range_p<Iterator>>, Fmts ...>
            , strf::detail::fmt_range_printer<CharT, FPack, Iterator, Fmts...> >
    {
        return {pre, fp, x};
    }
};

template <typename Iterator, typename SepCharT>
struct printable_traits<strf::separated_range_p<Iterator, SepCharT>>
{
    using representative_type = strf::separated_range_p<Iterator, SepCharT>;
    using forwarded_type = strf::separated_range_p<Iterator, SepCharT>;
    using formatters = strf::formatters_of<decltype(*std::declval<Iterator>())>;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        ,  forwarded_type x)
        -> strf::usual_printer_input
            < DstCharT, PreMeasurements, FPack, forwarded_type
            , strf::detail::separated_range_printer<DstCharT, FPack, Iterator> >
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }

    template <typename DstCharT, typename PreMeasurements, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_input
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::printable_with_fmt
            < strf::printable_traits<strf::separated_range_p<Iterator, SepCharT>>, Fmts... > x )
        ->  strf::usual_printer_input
            < DstCharT
            , PreMeasurements, FPack
            , strf::printable_with_fmt
                < strf::printable_traits<strf::separated_range_p<Iterator, SepCharT>>, Fmts... >
            , strf::detail::fmt_separated_range_printer<DstCharT, FPack, Iterator, Fmts...> >
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }
};

template <typename Iterator, typename UnaryOp>
struct printable_traits<strf::transformed_range_p<Iterator, UnaryOp>>
{
    using representative_type = strf::transformed_range_p<Iterator, UnaryOp>;
    using forwarded_type = strf::transformed_range_p<Iterator, UnaryOp>;
    using is_overridable = std::false_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x)
        -> strf::usual_printer_input
            < CharT, PreMeasurements, FPack, forwarded_type
            , strf::detail::transformed_range_printer<CharT, FPack, Iterator, UnaryOp> >
    {
        return {pre, fp, x};
    }
};

template <typename Iterator, typename SepCharT, typename UnaryOp>
struct printable_traits<strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>>
{
    using representative_type = strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>;
    using forwarded_type = strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x )
        -> strf::usual_printer_input
            < DstCharT, PreMeasurements, FPack, forwarded_type
            , strf::detail::sep_transformed_range_printer<DstCharT, FPack, Iterator, UnaryOp> >
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }
};

namespace detail {

template <typename CharT, typename FPack, typename Iterator>
class range_printer: public strf::printer<CharT>
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename... T>
    STRF_HD explicit range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template <typename PreMeasurements>
    using printer_type_ = strf::printer_type
        < CharT, PreMeasurements, FPack, strf::detail::remove_cv_t<value_type> >;

    STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    iterator begin_;
    iterator end_;
};

template <typename CharT, typename FPack, typename Iterator>
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void range_printer<CharT, FPack, Iterator>::do_premeasurements_
    ( PreMeasurements* pre ) const
{
    for( iterator it = begin_
       ; it != end_ && (pre->has_remaining_width() || PreMeasurements::size_demanded)
       ; ++it)
    {
        printer_type_<PreMeasurements>( strf::make_printer_input<CharT>(pre, fp_, *it) );
    }
}

template <typename CharT, typename FPack, typename Iterator>
STRF_HD void range_printer<CharT, FPack, Iterator>::print_to
    ( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    for(iterator it = begin_; it != end_; ++it) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>(&no_pre, fp_, *it) ).print_to(dst);
    }
}

template <typename CharT, typename FPack, typename Iterator>
class separated_range_printer: public strf::printer<CharT>
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename... T>
    STRF_HD explicit separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template <typename PreMeasurements>
    using printer_type_ = strf::printer_type
        < CharT, PreMeasurements, FPack, strf::detail::remove_cv_t<value_type> >;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::ptrdiff_t sep_len_;

    template <typename Category, typename Tag = strf::string_input_tag<CharT>>
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template <typename CharT, typename FPack, typename Iterator>
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void separated_range_printer<CharT, FPack, Iterator>::do_premeasurements_
    ( PreMeasurements* pre ) const
{
    std::size_t count = 0;
    for( iterator it = begin_
       ; it != end_ && (pre->has_remaining_width() || PreMeasurements::size_demanded)
       ; ++it)
    {
        printer_type_<PreMeasurements>(strf::make_printer_input<CharT>(pre, fp_, *it));
        ++ count;
    }
    if (count < 2 || (! PreMeasurements::size_demanded && ! pre->has_remaining_width())) {
        return;
    }
    {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , sep_begin_
                                 , sep_begin_ + sep_len_ );
        pre->checked_subtract_width(strf::sat_mul(dw, count - 1));
    }
    if (PreMeasurements::size_demanded) {
        pre->add_size((count - 1) * static_cast<std::size_t>(sep_len_));
    }
}

template <typename CharT, typename FPack, typename Iterator>
STRF_HD void separated_range_printer<CharT, FPack, Iterator>::print_to
    ( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    auto it = begin_; // NOLINT (llvm-qualified-auto)
    if (it != end_) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>(&no_pre, fp_, *it) )
            .print_to(dst);
        while (++it != end_) {
            dst.write(sep_begin_, sep_len_);
            printer_type_<strf::no_premeasurements>
                ( strf::make_printer_input<CharT>(&no_pre, fp_, *it) )
                .print_to(dst);
        }
    }
}

template < typename CharT
         , typename FPack
         , typename Iterator
         , typename ... Fmts >
class fmt_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<Iterator>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::printable_traits<strf::range_p<Iterator>> >;

public:

    template <typename... T>
    STRF_HD explicit fmt_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , fmt_(input.arg)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& des) const override;

private:

    template <typename PreMeasurements>
    using printer_type_ = strf::printer_type
        < CharT, PreMeasurements, FPack, value_fmt_type_adapted_ >;

    STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    fmt_type_adapted_ fmt_;
};


template < typename CharT
         , typename FPack
         , typename Iterator
         , typename ... Fmts >
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void fmt_range_printer<CharT, FPack, Iterator, Fmts ...>::do_premeasurements_
    ( PreMeasurements* pre ) const
{
    auto r = fmt_.value();
    for( Iterator it = r.begin
       ; it != r.end && (pre->has_remaining_width() || PreMeasurements::size_demanded)
       ; ++it)
    {
        printer_type_<PreMeasurements>
            ( strf::make_printer_input<CharT>
                ( pre, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
    }
}

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharT, FPack, Iterator, Fmts ...>::print_to
    ( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    auto r = fmt_.value();
    for(Iterator it = r.begin; it != r.end; ++it) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>
                ( &no_pre, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(dst);
    }
}

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
class fmt_separated_range_printer: public strf::printer<CharT>
{
    using value_type_ = strf::detail::iterator_value_type<Iterator>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::printable_traits<strf::separated_range_p<Iterator, CharT>> >;

public:

    template <typename... T>
    STRF_HD explicit fmt_separated_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , fmt_(input.arg)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template <typename PreMeasurements>
    using printer_type_ = strf::printer_type
        < CharT, PreMeasurements, FPack, value_fmt_type_adapted_ >;

    STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    fmt_type_adapted_ fmt_;

    template <typename Category, typename Tag = strf::string_input_tag<CharT>>
    static inline STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(*(const FPack*)0)))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void fmt_separated_range_printer<CharT, FPack, Iterator, Fmts ...>::do_premeasurements_
    ( PreMeasurements* pre ) const
{
    auto r = fmt_.value();
    std::size_t count = 0;
    for ( Iterator it = r.begin  // NOLINT(llvm-qualified-auto)
        ; it != r.end && (pre->has_remaining_width() || PreMeasurements::size_demanded)
        ; ++it)
    {
        printer_type_<PreMeasurements>
            ( strf::make_printer_input<CharT>
                ( pre, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) );
        ++ count;
    }
    if (count < 2) {
        return;
    }
    if (pre->has_remaining_width()) {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , r.sep_begin
                                 , r.sep_begin + r.sep_len );
        pre->checked_subtract_width(strf::sat_mul(dw, (count - 1)));
    }
    if (PreMeasurements::size_demanded) {
        pre->add_size((count - 1) * static_cast<std::size_t>(r.sep_len));
    }
}

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
STRF_HD void fmt_separated_range_printer<CharT, FPack, Iterator, Fmts ...>
::print_to( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    auto r = fmt_.value();
    Iterator it = r.begin;
    if (it != r.end) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>
                ( &no_pre, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
            .print_to(dst);
        while(++it != r.end) {
            dst.write(r.sep_begin, r.sep_len);
            printer_type_<strf::no_premeasurements>
                ( strf::make_printer_input<CharT>
                    ( &no_pre, fp_, value_fmt_type_adapted_{{*it}, fmt_} ) )
                .print_to(dst);
        }
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class transformed_range_printer: public strf::printer<CharT>
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename... T>
    STRF_HD explicit transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , op_(input.arg.op)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template <typename PreMeasurements, typename Op = UnaryOp>
    using printer_type_ = strf::printer_type
        < CharT
        , PreMeasurements, FPack
        , strf::detail::remove_reference_t
            < decltype(std::declval<Op>()(*std::declval<iterator>())) > >;

    STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    iterator begin_;
    iterator end_;
    UnaryOp op_;
};

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void transformed_range_printer<CharT, FPack, Iterator, UnaryOp>
    ::do_premeasurements_(PreMeasurements* pre) const
{

    for( iterator it = begin_
       ; it != end_ && (pre->has_remaining_width() || PreMeasurements::size_demanded)
       ; ++it)
    {
        printer_type_<PreMeasurements>(strf::make_printer_input<CharT>(pre, fp_, op_(*it)));
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, Iterator, UnaryOp>::print_to
    ( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    for(iterator it = begin_; it != end_; ++it) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>(&no_pre, fp_, op_(*it)) )
            .print_to(dst);
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class sep_transformed_range_printer: public strf::printer<CharT>
{
public:
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename... T>
    STRF_HD explicit sep_transformed_range_printer
        ( const strf::usual_printer_input<T...>& input )
        : fp_(input.facets)
        , begin_(input.arg.begin)
        , end_(input.arg.end)
        , sep_begin_(input.arg.sep_begin)
        , sep_len_(input.arg.sep_len)
        , op_(input.arg.op)
    {
        do_premeasurements_(input.pre);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    template <typename PreMeasurements, typename Op = UnaryOp>
    using printer_type_ = strf::printer_type
        < CharT
        , PreMeasurements, FPack
        , strf::detail::remove_reference_t
            < decltype(std::declval<Op>()(*std::declval<iterator>())) > >;

    STRF_HD void do_premeasurements_(strf::no_premeasurements*) const
    {
    }

    template < typename PreMeasurements
             , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> = 0 >
    STRF_HD void do_premeasurements_(PreMeasurements* pre) const;

    FPack fp_;
    iterator begin_;
    iterator end_;
    const CharT* sep_begin_;
    std::ptrdiff_t sep_len_;
    UnaryOp op_;

    template <typename Category, typename Tag = strf::string_input_tag<CharT>>
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, Tag>();
    }
};

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
template < typename PreMeasurements
         , strf::detail::enable_if_t<PreMeasurements::something_demanded, int> >
STRF_HD void sep_transformed_range_printer<CharT, FPack, Iterator, UnaryOp>
    ::do_premeasurements_(PreMeasurements* pre) const
{
    std::size_t count = 0;
    for( iterator it = begin_
       ; it != end_ && (pre->has_remaining_width() || PreMeasurements::size_demanded)
       ; ++it)
    {
        printer_type_<PreMeasurements>
            ( strf::make_printer_input<CharT>(pre, fp_, op_(*it)) );
        ++ count;
    }
    if (count < 2) {
        return;
    }
    if (pre->has_remaining_width()) {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( use_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , sep_begin_
                                 , sep_begin_ + sep_len_ );
        pre->checked_subtract_width(strf::sat_mul(dw, (count - 1)));
    }
    if (PreMeasurements::size_demanded) {
        pre->add_size((count - 1) * static_cast<std::size_t>(sep_len_));
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, Iterator, UnaryOp>::print_to
    ( strf::destination<CharT>& dst ) const
{
    strf::no_premeasurements no_pre;
    auto it = begin_;
    if (it != end_) {
        printer_type_<strf::no_premeasurements>
            ( strf::make_printer_input<CharT>(&no_pre, fp_, op_(*it)) )
            .print_to(dst);
        while (++it != end_) {
            dst.write(sep_begin_, sep_len_);
            printer_type_<strf::no_premeasurements>
                ( strf::make_printer_input<CharT>(&no_pre, fp_, op_(*it)) )
                .print_to(dst);
        }
    }
}

} // namespace detail

template <typename Iterator>
inline STRF_HD strf::range_p<Iterator> range(Iterator begin, Iterator end)
{
    return {begin, end};
}

template <typename Iterator, typename CharT>
inline STRF_HD auto separated_range(Iterator begin, Iterator end, const CharT* sep)
    -> strf::separated_range_p<Iterator, CharT>
{
    return {begin, end, sep, strf::detail::str_ssize<CharT>(sep)};
}

template < typename Range
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto range(const Range& r) -> strf::range_p<Iterator>
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
    return {r.begin(), r.end(), sep, strf::detail::str_ssize<CharT>(sep)};
}

template < typename T, typename CharT >
inline STRF_HD auto separated_range(std::initializer_list<T> r, const CharT* sep)
    -> strf::separated_range_p<const T*, CharT>
{
    return {r.begin(), r.end(), sep, strf::detail::str_ssize<CharT>(sep)};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto separated_range(T (&array)[N], const CharT* sep)
    -> strf::separated_range_p<const T*, CharT>
{
    return {&array[0], &array[0] + N, sep, strf::detail::str_ssize<CharT>(sep)};
}

template <typename Iterator>
inline STRF_HD auto fmt_range(Iterator begin, Iterator end)
    -> strf::range_with_formatters<Iterator>
{
    return strf::range_with_formatters<Iterator>{{begin, end}};
}

template <typename Iterator, typename CharT>
inline STRF_HD auto fmt_separated_range(Iterator begin, Iterator end, const CharT* sep)
    -> strf::sep_range_with_formatters<Iterator, CharT>
{
    return strf::sep_range_with_formatters<Iterator, CharT>
        {{begin, end, sep, strf::detail::str_ssize<CharT>(sep)}};
}

template < typename Range
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD
strf::range_with_formatters<Iterator> fmt_range(const Range& r)
{
    const strf::range_p<Iterator> rr{r.begin(), r.end()};
    return strf::range_with_formatters<Iterator>{rr};
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
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto fmt_separated_range(const Range& r, const CharT* sep)
    -> strf::sep_range_with_formatters<Iterator, CharT>
{
    const auto sep_len = strf::detail::str_ssize<CharT>(sep);
    const strf::separated_range_p<Iterator, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_formatters<Iterator, CharT>{rr};
}


template <typename T, typename CharT>
inline STRF_HD auto fmt_separated_range(std::initializer_list<T> r, const CharT* sep) noexcept
    -> strf::sep_range_with_formatters<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_ssize<CharT>(sep);
    strf::separated_range_p<const T*, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_formatters<const T*, CharT>{rr};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto fmt_separated_range(T (&array)[N], const CharT* sep)
    -> strf::sep_range_with_formatters<const T*, CharT>
{
    const auto sep_len = strf::detail::str_ssize<CharT>(sep);
    return strf::sep_range_with_formatters<const T*, CharT>
        { {&array[0], &array[0] + N, sep, sep_len} };
}

template < typename Iterator
         , typename UnaryOp
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const Iterator&>())) >
inline STRF_HD auto range(Iterator begin, Iterator end, UnaryOp op)
    -> strf::transformed_range_p<Iterator, UnaryOp>
{
    return {begin, end, op};
}

template < typename Range
         , typename UnaryOp
         , typename Iterator = typename Range::const_iterator
         , typename
           = decltype(std::declval<const UnaryOp&>()(*std::declval<const Iterator&>()))
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto range(const Range& r, UnaryOp op)
    -> strf::transformed_range_p<Iterator, UnaryOp>
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

template < typename Iterator
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(*std::declval<const Iterator&>())) >
inline STRF_HD auto separated_range(Iterator begin, Iterator end, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<Iterator, CharT, UnaryOp>
{
    const auto sep_len = strf::detail::str_ssize(sep);
    return { begin, end, sep, sep_len, op };
}

template < typename Range
         , typename CharT
         , typename UnaryOp
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const UnaryOp&>()(*std::declval<const Iterator&>()))
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto separated_range(const Range& r, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<Iterator, CharT, UnaryOp>
{
    const auto sep_len = strf::detail::str_ssize(sep);
    return { r.begin(), r.end(), sep, sep_len, op };
}

template < typename T
         , typename CharT
         , typename UnaryOp
         , typename = decltype(std::declval<const UnaryOp&>()(std::declval<const T&>())) >
inline STRF_HD auto separated_range(std::initializer_list<T> r, const CharT* sep, UnaryOp op)
    -> strf::separated_transformed_range_p<const T*, CharT, UnaryOp>
{
    const auto sep_len = strf::detail::str_ssize(sep);
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
    const auto sep_len = strf::detail::str_ssize(sep);
    return { &array[0], &array[0] + N, sep, sep_len, op };
}

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_RANGE_HPP

