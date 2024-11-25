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
using range_with_fmt
    = strf::detail::mp_replace_front
        < VF, strf::printable_def<strf::range_p<Iterator>> >;

template < typename Iterator
         , typename CharT
         , typename V  = strf::detail::iterator_value_type<Iterator>
         , typename VF = strf::fmt_type<V> >
using sep_range_with_fmt
    = strf::detail::mp_replace_front
        < VF, strf::printable_def<strf::separated_range_p<Iterator, CharT>> >;


namespace detail {

template <typename CharT, typename FPack, typename Iterator>
class range_printer
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename PreMeasurements>
    STRF_HD range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , strf::range_p<Iterator> arg )
        : fp_(facets)
        , begin_(arg.begin)
        , end_(arg.end)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

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
        (void) strf::make_printer<CharT>(pre, fp_, *it);
    }
}

template <typename CharT, typename FPack, typename Iterator>
STRF_HD void range_printer<CharT, FPack, Iterator>::operator()
    ( strf::destination<CharT>& dst ) const
{
    for(iterator it = begin_; it != end_ && dst.good(); ++it) {
        detail::print_one_printable(dst, fp_, *it);
    }
}

template <typename CharT, typename FPack, typename Iterator>
class separated_range_printer
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename PreMeasurements>
    STRF_HD separated_range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , strf::separated_range_p<Iterator, CharT> arg )
        : fp_(facets)
        , begin_(arg.begin)
        , end_(arg.end)
        , sep_begin_(arg.sep_begin)
        , sep_len_(arg.sep_len)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

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
    STRF_DECLTYPE_AUTO((strf::get_facet<Category, Tag>(std::declval<FPack>())))
    get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, Tag>();
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
        (void) strf::make_printer<CharT>(pre, fp_, *it);
        ++ count;
    }
    if (count < 2 || (! PreMeasurements::size_demanded && ! pre->has_remaining_width())) {
        return;
    }
    {
        auto&& wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , sep_begin_
                                 , sep_begin_ + sep_len_ );
        pre->checked_add_width(strf::sat_mul(dw, count - 1));
    }
    if (PreMeasurements::size_demanded) {
        pre->add_size((count - 1) * static_cast<std::size_t>(sep_len_));
    }
}

template <typename CharT, typename FPack, typename Iterator>
STRF_HD void separated_range_printer<CharT, FPack, Iterator>::operator()
    ( strf::destination<CharT>& dst ) const
{
    auto it = begin_; // NOLINT (llvm-qualified-auto)
    if (it != end_) {
        detail::print_one_printable(dst, fp_, *it);
        while (++it != end_ && dst.good()) {
            dst.write(sep_begin_, sep_len_);
            detail::print_one_printable(dst, fp_, *it);
        }
    }
}

template < typename CharT
         , typename FPack
         , typename Iterator
         , typename ... Fmts >
class fmt_range_printer
{
    using value_type_ = strf::detail::iterator_value_type<Iterator>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::printable_def<strf::range_p<Iterator>> >;

public:

    template <typename PreMeasurements>
    STRF_HD fmt_range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const strf::value_and_format
            <strf::printable_def<strf::range_p<Iterator>>, Fmts...>& arg )
        : fp_(facets)
        , fmt_(arg)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& des) const;

private:

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
        (void) strf::make_printer<CharT>(pre, fp_, value_fmt_type_adapted_{{*it}, fmt_});
    }
}

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
STRF_HD void fmt_range_printer<CharT, FPack, Iterator, Fmts ...>::operator()
    ( strf::destination<CharT>& dst ) const
{
    auto r = fmt_.value();
    for(Iterator it = r.begin; it != r.end && dst.good(); ++it) {
        detail::print_one_printable(dst, fp_, value_fmt_type_adapted_{{*it}, fmt_});
    }
}

template< typename CharT
        , typename FPack
        , typename Iterator
        , typename ... Fmts >
class fmt_separated_range_printer
{
    using value_type_ = strf::detail::iterator_value_type<Iterator>;
    using value_fmt_type_ = strf::fmt_type<value_type_>;
    using value_fmt_type_adapted_
        = typename value_fmt_type_::template replace_fmts<Fmts...>;

    using fmt_type_adapted_ = detail::mp_replace_front
        < value_fmt_type_adapted_
        , strf::printable_def<strf::separated_range_p<Iterator, CharT>> >;

public:

    template <typename PreMeasurements>
    STRF_HD fmt_separated_range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const strf::value_and_format
            < strf::printable_def<strf::separated_range_p<Iterator, CharT>>
            , Fmts... >& arg )
        : fp_(facets)
        , fmt_(arg)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

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
    STRF_DECLTYPE_AUTO((strf::get_facet<Category, Tag>(*(const FPack*)0)))
    get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, Tag>();
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
        (void) strf::make_printer<CharT>(pre, fp_, value_fmt_type_adapted_{{*it}, fmt_});
        ++ count;
    }
    if (count < 2) {
        return;
    }
    if (pre->has_remaining_width()) {
        auto&& wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , r.sep_begin
                                 , r.sep_begin + r.sep_len );
        pre->checked_add_width(strf::sat_mul(dw, (count - 1)));
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
::operator()( strf::destination<CharT>& dst ) const
{
    auto r = fmt_.value();
    Iterator it = r.begin;
    if (it != r.end) {
        detail::print_one_printable(dst, fp_, value_fmt_type_adapted_{{*it}, fmt_});
        while(++it != r.end && dst.good()) {
            dst.write(r.sep_begin, r.sep_len);
            detail::print_one_printable(dst, fp_, value_fmt_type_adapted_{{*it}, fmt_});
        }
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class transformed_range_printer
{
public:

    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename PreMeasurements>
    STRF_HD transformed_range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const strf::transformed_range_p<Iterator, UnaryOp>& arg )
        : fp_(facets)
        , begin_(arg.begin)
        , end_(arg.end)
        , op_(arg.op)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

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
        (void) strf::make_printer<CharT>(pre, fp_, op_(*it));
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
STRF_HD void transformed_range_printer<CharT, FPack, Iterator, UnaryOp>::operator()
    ( strf::destination<CharT>& dst ) const
{
    for(iterator it = begin_; it != end_ && dst.good(); ++it) {
        detail::print_one_printable(dst, fp_, op_(*it));
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
class sep_transformed_range_printer
{
public:
    using iterator = Iterator;
    using value_type = strf::detail::iterator_value_type<Iterator>;

    template <typename PreMeasurements>
    STRF_HD sep_transformed_range_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const strf::separated_transformed_range_p<Iterator, CharT, UnaryOp>& arg )
        : fp_(facets)
        , begin_(arg.begin)
        , end_(arg.end)
        , sep_begin_(arg.sep_begin)
        , sep_len_(arg.sep_len)
        , op_(arg.op)
    {
        do_premeasurements_(pre);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

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
    STRF_DECLTYPE_AUTO((strf::get_facet<Category, Tag>(std::declval<FPack>())))
    get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, Tag>();
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
        (void) strf::make_printer<CharT>(pre, fp_, op_(*it));
        ++ count;
    }
    if (count < 2) {
        return;
    }
    if (pre->has_remaining_width()) {
        auto&& wcalc = get_facet_<strf::width_calculator_c>(fp_);
        auto dw = wcalc.str_width( get_facet_<strf::charset_c<CharT>>(fp_)
                                 , pre->remaining_width()
                                 , sep_begin_
                                 , sep_begin_ + sep_len_ );
        pre->checked_add_width(strf::sat_mul(dw, (count - 1)));
    }
    if (PreMeasurements::size_demanded) {
        pre->add_size((count - 1) * static_cast<std::size_t>(sep_len_));
    }
}

template <typename CharT, typename FPack, typename Iterator, typename UnaryOp>
STRF_HD void sep_transformed_range_printer<CharT, FPack, Iterator, UnaryOp>::operator()
    ( strf::destination<CharT>& dst ) const
{
    auto it = begin_;
    if (it != end_) {
        detail::print_one_printable(dst, fp_, op_(*it));
        while (++it != end_ && dst.good()) {
            dst.write(sep_begin_, sep_len_);
            detail::print_one_printable(dst, fp_, op_(*it));
        }
    }
}

} // namespace detail

template <typename Iterator>
struct printable_def<strf::range_p<Iterator>>
{
    using representative = strf::range_p<Iterator>;
    using forwarded_type = strf::range_p<Iterator>;
    using format_specifiers = strf::format_specifiers_of<decltype(*std::declval<Iterator>())>;
    using is_overridable = std::false_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x)
        -> strf::detail::range_printer<CharT, FPack, Iterator>
    {
        return {pre, fp, x};
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::value_and_format<strf::printable_def<strf::range_p<Iterator>>, Fmts...> x )
        -> strf::detail::fmt_range_printer<CharT, FPack, Iterator, Fmts...>
    {
        return {pre, fp, x};
    }
};

template <typename Iterator, typename SepCharT>
struct printable_def<strf::separated_range_p<Iterator, SepCharT>>
{
    using representative = strf::separated_range_p<Iterator, SepCharT>;
    using forwarded_type = strf::separated_range_p<Iterator, SepCharT>;
    using format_specifiers = strf::format_specifiers_of<decltype(*std::declval<Iterator>())>;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        ,  forwarded_type x)
        -> detail::separated_range_printer<DstCharT, FPack, Iterator>
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }

    template <typename DstCharT, typename PreMeasurements, typename FPack, typename... Fmts>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::value_and_format
            < strf::printable_def<strf::separated_range_p<Iterator, SepCharT>>, Fmts... > x )
        -> detail::fmt_separated_range_printer<DstCharT, FPack, Iterator, Fmts...>
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }
};

template <typename Iterator, typename UnaryOp>
struct printable_def<strf::transformed_range_p<Iterator, UnaryOp>>
{
    using representative = strf::transformed_range_p<Iterator, UnaryOp>;
    using forwarded_type = strf::transformed_range_p<Iterator, UnaryOp>;
    using is_overridable = std::false_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x)
        -> detail::transformed_range_printer<CharT, FPack, Iterator, UnaryOp>
    {
        return {pre, fp, x};
    }
};

template <typename Iterator, typename SepCharT, typename UnaryOp>
struct printable_def<strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>>
{
    using representative = strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>;
    using forwarded_type = strf::separated_transformed_range_p<Iterator, SepCharT, UnaryOp>;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        , forwarded_type x )
        -> detail::sep_transformed_range_printer<DstCharT, FPack, Iterator, UnaryOp>
    {
        static_assert( std::is_same<SepCharT, DstCharT>::value
                     , "Character type of range separator string is different." );
        return {pre, fp, x};
    }
};


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
    -> strf::range_with_fmt<Iterator>
{
    return strf::range_with_fmt<Iterator>{{begin, end}};
}

template <typename Iterator, typename CharT>
inline STRF_HD auto fmt_separated_range(Iterator begin, Iterator end, const CharT* sep)
    -> strf::sep_range_with_fmt<Iterator, CharT>
{
    return strf::sep_range_with_fmt<Iterator, CharT>
        {{begin, end, sep, strf::detail::str_ssize<CharT>(sep)}};
}

template < typename Range
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD
strf::range_with_fmt<Iterator> fmt_range(const Range& r)
{
    const strf::range_p<Iterator> rr{r.begin(), r.end()};
    return strf::range_with_fmt<Iterator>{rr};
}

template <typename T>
constexpr STRF_HD strf::range_with_fmt<const T*>
fmt_range(std::initializer_list<T> r) noexcept
{
    return strf::range_with_fmt<const T*>
        { strf::range_p<const T*>{r.begin(), r.end()} };
}

template <typename T, std::size_t N>
inline STRF_HD auto fmt_range(T (&array)[N])
    -> strf::range_with_fmt<const T*>
{
    return strf::range_with_fmt<const T*>{{&array[0], &array[0] + N}};
}

template < typename Range
         , typename CharT
         , typename Iterator = typename Range::const_iterator
         , typename = decltype(std::declval<const Range&>().begin())
         , typename = decltype(std::declval<const Range&>().end()) >
inline STRF_HD auto fmt_separated_range(const Range& r, const CharT* sep)
    -> strf::sep_range_with_fmt<Iterator, CharT>
{
    const auto sep_len = strf::detail::str_ssize<CharT>(sep);
    const strf::separated_range_p<Iterator, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_fmt<Iterator, CharT>{rr};
}


template <typename T, typename CharT>
inline STRF_HD auto fmt_separated_range(std::initializer_list<T> r, const CharT* sep) noexcept
    -> strf::sep_range_with_fmt<const T*, CharT>
{
    std::size_t sep_len = strf::detail::str_ssize<CharT>(sep);
    strf::separated_range_p<const T*, CharT> rr
    { r.begin(), r.end(), sep, sep_len };
    return strf::sep_range_with_fmt<const T*, CharT>{rr};
}

template <typename T, std::size_t N, typename CharT>
inline STRF_HD auto fmt_separated_range(T (&array)[N], const CharT* sep)
    -> strf::sep_range_with_fmt<const T*, CharT>
{
    const auto sep_len = strf::detail::str_ssize<CharT>(sep);
    return strf::sep_range_with_fmt<const T*, CharT>
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

