#ifndef STRF_DETAIL_PRINTABLE_WITH_FMT_HPP
#define STRF_DETAIL_PRINTABLE_WITH_FMT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {

template <typename PrintingTraits, class... Fmts>
class printable_with_fmt;

namespace detail {

template <typename SeparatorFmt, typename FmtsPairListBefore, typename FmtsPairListAfter>
struct separate_fmtpair_list_impl;

template <
    typename SeparatorFmt,
    typename... PairsBefore,
    typename    FirstFmtInFirstPairAfter,
    typename... OtherPairsAfter >
struct separate_fmtpair_list_impl
    < SeparatorFmt
    , mp_type_list<PairsBefore...>
    , mp_type_list
        < mp_type_pair<FirstFmtInFirstPairAfter, SeparatorFmt>
        , OtherPairsAfter... > >
{
    using pairs_before = mp_type_list<PairsBefore...>;
    using pairs_after  = mp_type_list<OtherPairsAfter...>;
    using separator_pair = mp_type_pair<FirstFmtInFirstPairAfter, SeparatorFmt>;
    using separator_fmt = SeparatorFmt;
    constexpr static bool found = true;
};

template <typename SeparatorFmt, typename... PairsBefore>
struct separate_fmtpair_list_impl
    < SeparatorFmt
    , mp_type_list<PairsBefore...>
    , mp_type_list<> >
{
    using pairs_before = mp_type_list<PairsBefore...>;
    using pairs_after  = mp_type_list<>;
    using separator_pair = void;
    using separator_fmt = void;
    constexpr static bool found = false;
};

template
    < typename SeparatorFmt
    , typename... PairsBefore
    , typename    FirstFmtInFirstPairAfter
    , typename    SecondFmtInFirstPairAfter
    , typename... OtherPairsAfter >
struct separate_fmtpair_list_impl
    < SeparatorFmt
    , mp_type_list<PairsBefore...>
    , mp_type_list
        < mp_type_pair<FirstFmtInFirstPairAfter, SecondFmtInFirstPairAfter>
        , OtherPairsAfter... > >
{
    using this_pair = mp_type_pair<FirstFmtInFirstPairAfter, SecondFmtInFirstPairAfter>;

    using next_pairs_list_before =
        typename mp_type_list<PairsBefore...>::template add_back <this_pair>;

    using next_pairs_list_after = mp_type_list<OtherPairsAfter...>;

    using next_impl = separate_fmtpair_list_impl
        < SeparatorFmt
        , next_pairs_list_before
        , next_pairs_list_after >;

    using pairs_before   = typename next_impl::pairs_before;
    using pairs_after    = typename next_impl::pairs_after;
    using separator_pair = typename next_impl::separator_pair;
    using separator_fmt  = typename next_impl::separator_fmt;
    constexpr static bool found = next_impl::found;
};

template <typename SeparatorFmt, typename FmtsPairsList>
struct separate_fmtpair_list: separate_fmtpair_list_impl
    < SeparatorFmt, mp_type_list<>, FmtsPairsList >
{
};

template <typename PairsList>
struct transpose_fmt_list;

template <typename... Pairs>
struct transpose_fmt_list<mp_type_list<Pairs...>>
{
    using list_of_firsts  = mp_type_list<typename Pairs::first ...>;
    using list_of_seconds = mp_type_list<typename Pairs::second ...>;
};

template <typename PrintableWithFmt, typename FmtList>
struct get_list_of_fmtfn_bases_impl;

template <typename PrintableWithFmt, typename... Fmts>
struct get_list_of_fmtfn_bases_impl<PrintableWithFmt, mp_type_list<Fmts...>>
{
    using type = mp_type_list<typename Fmts::template fn<PrintableWithFmt>...>;
};

template <typename PrintableWithFmt, typename FmtList>
using get_list_of_fmtfn_bases =
    typename get_list_of_fmtfn_bases_impl<PrintableWithFmt, FmtList>::type;

template <typename SrcList, typename DstList>
struct each_dst_is_contructible_from_cref_of_each_src_impl;

template <typename SrcList, typename DstList>
struct each_dst_is_contructible_from_rval_ref_of_each_src_impl;

template <typename... Src, typename... Dst>
struct each_dst_is_contructible_from_cref_of_each_src_impl
    < mp_type_list<Src...>, mp_type_list<Dst...> >
{
    static constexpr bool value =
        fold_and< std::is_constructible<Src, const Dst&>::value... >::value;
};

template <typename... Src, typename... Dst>
struct each_dst_is_contructible_from_rval_ref_of_each_src_impl
    < mp_type_list<Src...>, mp_type_list<Dst...> >
{
    static constexpr bool value =
        fold_and< std::is_constructible<Src, detail::remove_cvref_t<Dst> >::value... >::value;
};

template <typename SrcList, typename DstList>
constexpr bool each_dst_is_contructible_from_cref_of_each_src =
    each_dst_is_contructible_from_cref_of_each_src_impl<SrcList, DstList>::value;

template <typename SrcList, typename DstList>
constexpr bool each_dst_is_contructible_from_rval_ref_of_each_src =
    each_dst_is_contructible_from_rval_ref_of_each_src_impl<SrcList, DstList>::value;

template
    < bool Enabled
    , typename SeparatorDstFmt
    , typename SrcPrintableWithFmt
    , typename DstPrintableWithFmt >
struct separate_fmts_of_two_printable_with_fmt;

template
    < typename SeparatorDstFmt
    , typename SrcPrintableTraits
    , typename... SrcFmts
    , typename DstPrintableTraits
    , typename... DstFmts >
struct separate_fmts_of_two_printable_with_fmt
    < true
    , SeparatorDstFmt
    , printable_with_fmt<SrcPrintableTraits, SrcFmts...>
    , printable_with_fmt<DstPrintableTraits, DstFmts...> >
{
    static_assert(sizeof...(SrcFmts) == sizeof...(DstFmts), "");

    using pairs_list = mp_type_list<mp_type_pair<SrcFmts, DstFmts>...>;
    using impl = separate_fmtpair_list<SeparatorDstFmt, pairs_list>;

    using before = transpose_fmt_list< typename impl::pairs_before >;
    using after  = transpose_fmt_list< typename impl::pairs_after >;

    using src_fmts_before = typename before::list_of_firsts;
    using dst_fmts_before = typename before::list_of_seconds;

    using src_fmts_after = typename after::list_of_firsts;
    using dst_fmts_after = typename after::list_of_seconds;

    constexpr static bool found_selected_fmt_in_dst_list = impl::found;

    using src_printable_with_fmt = printable_with_fmt<SrcPrintableTraits, SrcFmts...>;
    using dst_printable_with_fmt = printable_with_fmt<DstPrintableTraits, DstFmts...>;

    using src_fmtfn_bases_before = get_list_of_fmtfn_bases
        < src_printable_with_fmt, src_fmts_before >;

    using dst_fmtfn_bases_before = get_list_of_fmtfn_bases
        < dst_printable_with_fmt, dst_fmts_before >;

    using src_fmtfn_bases_after = get_list_of_fmtfn_bases
        < src_printable_with_fmt, src_fmts_after>;

    using dst_fmtfn_bases_after = get_list_of_fmtfn_bases
        < dst_printable_with_fmt, dst_fmts_after>;

    using separator_fmtfn_base = typename
        SeparatorDstFmt::template fn<dst_printable_with_fmt>;
};

template
    < class From
    , class To
    , template <class...> class List
    , class... T >
struct fmt_replace_impl2
{
    template <class U>
    using f = strf::detail::conditional_t<std::is_same<From, U>::value, To, U>;

    using type = List<f<T>...>;
};

template <class From, class List>
struct fmt_replace_impl;

template
    < class From
    , template <class...> class List
    , class... T>
struct fmt_replace_impl<From, List<T...> >
{
    template <class To>
    using type_tmpl =
        typename strf::detail::fmt_replace_impl2
            < From, To, List, T...>::type;
};

} // namespace detail

template <typename List, typename From, typename To>
using fmt_replace
    = typename strf::detail::fmt_replace_impl<From, List>
    ::template type_tmpl<To>;

template <typename PrintingTraits, class... Fmts>
using value_with_formatters
STRF_DEPRECATED_MSG("value_with_formatters renamed to printable_with_fmt")
= printable_with_fmt<PrintingTraits, Fmts...>;

namespace detail {

template <typename T>
struct is_printable_with_fmt : std::false_type
{ };

template <typename... T>
struct is_printable_with_fmt<strf::printable_with_fmt<T...>>: std::true_type
{ };

template <typename T>
struct is_printable_with_fmt<const T> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<volatile T> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<T&> : is_printable_with_fmt<T>
{ };

template <typename T>
struct is_printable_with_fmt<T&&> : is_printable_with_fmt<T>
{ };


template <typename... T>
struct are_empty;

template <>
struct are_empty<> : std::true_type {};

template <typename First, typename... Others>
struct are_empty<First, Others...>
    : std::integral_constant
        < bool
        , std::is_empty<First>::value
       && are_empty<Others...>::value >
{
};

template <typename PrintableWithFmt>
struct all_base_fmtfn_classes_are_empty;

template <typename PrintingTraits, typename... Fmts>
struct all_base_fmtfn_classes_are_empty< printable_with_fmt<PrintingTraits, Fmts...> >
    : are_empty<typename Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>> ...>
{
};

} // namespace detail

template <typename PrintingTraits, class... Fmts>
class printable_with_fmt
    : public Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>> ...
{
public:
    using traits = PrintingTraits;
    using value_type = typename PrintingTraits::forwarded_type;

    template <typename... OtherFmts>
    using replace_fmts = strf::printable_with_fmt<PrintingTraits, OtherFmts ...>;

    explicit constexpr STRF_HD printable_with_fmt(const value_type& v)
        : value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , const strf::printable_with_fmt<OtherPrintingTraits, Fmts...>& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<printable_with_fmt<OtherPrintingTraits, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherPrintingTraits>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , strf::printable_with_fmt<OtherPrintingTraits, Fmts...>&& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<printable_with_fmt<OtherPrintingTraits, Fmts...>> &&>(std::move(f)) )
        ...
        , value_(static_cast<value_type&&>(v))
    {
    }

    template <typename... F, typename... FInit>
    constexpr STRF_HD printable_with_fmt
        ( const value_type& v
        , strf::tag<F...>
        , FInit&&... finit )
        : F::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename... OtherFmts>
    constexpr STRF_HD explicit printable_with_fmt
        ( const strf::printable_with_fmt<PrintingTraits, OtherFmts...>& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<printable_with_fmt<PrintingTraits, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD explicit printable_with_fmt
        ( strf::printable_with_fmt<PrintingTraits, OtherFmts...>&& f )
        : Fmts::template fn<printable_with_fmt<PrintingTraits, Fmts...>>
            ( static_cast
              < typename OtherFmts
              :: template fn<printable_with_fmt<PrintingTraits, OtherFmts ...>>&& >(std::move(f)) )
        ...
        , value_(static_cast<value_type&&>(f.value()))
    {
    }

    template
        < typename SrcPrintingTraits
        , typename... SrcFmts
        , typename SelectedFmt
        , typename... SelectedFmtInitArgs
        , typename Helper = detail::separate_fmts_of_two_printable_with_fmt
             < sizeof...(SrcFmts) == sizeof...(Fmts)
             , SelectedFmt
             , printable_with_fmt<SrcPrintingTraits, SrcFmts...>
             , printable_with_fmt<PrintingTraits, Fmts...> >

        , detail::enable_if_t<Helper::found_selected_fmt_in_dst_list, int> = 0

        , detail::enable_if_t
            < std::is_constructible
                < value_type, const typename SrcPrintingTraits::forwarded_type&>::value
            , int > = 0

        , detail::enable_if_t
            < detail::each_dst_is_contructible_from_cref_of_each_src
                < typename Helper::src_fmtfn_bases_before
                , typename Helper::dst_fmtfn_bases_before >
            , int > = 0

        , detail::enable_if_t
            < detail::each_dst_is_contructible_from_cref_of_each_src
                 < typename Helper::src_fmtfn_bases_after
                 , typename Helper::dst_fmtfn_bases_after >
            , int > = 0

        , detail::enable_if_t
            < std::is_constructible
                  < typename Helper::separator_fmtfn_base
                  , SelectedFmtInitArgs... > ::value
            , int > = 0 >

    constexpr STRF_HD printable_with_fmt
        ( const strf::printable_with_fmt<SrcPrintingTraits, SrcFmts...>& src
        , strf::tag<SelectedFmt>
        , SelectedFmtInitArgs&&... args )
        : printable_with_fmt
            ( detail::mp_type_list
                < typename Helper::src_fmtfn_bases_before
                , typename Helper::dst_fmtfn_bases_before
                , typename Helper::src_fmtfn_bases_after
                , typename Helper::dst_fmtfn_bases_after
                , typename Helper::separator_fmtfn_base > ()
            , src
            , std::forward<SelectedFmtInitArgs&&>(args)... )
    {
    }

private:

    template
        < typename... SrcBasesBefore
        , typename... DstBasesBefore
        , typename... SrcBasesAfter
        , typename... DstBasesAfter
        , typename SrcPrintableWithFmt
        , typename SelectedBase
        , typename... Args >
    constexpr STRF_HD printable_with_fmt
        ( detail::mp_type_list
            < detail::mp_type_list<SrcBasesBefore...>
            , detail::mp_type_list<DstBasesBefore...>
            , detail::mp_type_list<SrcBasesAfter...>
            , detail::mp_type_list<DstBasesAfter...>
            , SelectedBase >
        , const SrcPrintableWithFmt& src
        , Args&&... args )
        : DstBasesBefore(static_cast<const SrcBasesBefore&>(src))...
        , SelectedBase(std::forward<Args>(args)...)
        , DstBasesAfter(static_cast<const SrcBasesAfter&>(src))...
        , value_(src.value())
    {
    }

public:

    constexpr STRF_HD const value_type& value() const
    {
        return value_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD value_type& value()
    {
        return value_;
    }

private:

    value_type value_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

} // namespace strf

#endif  // STRF_DETAIL_PRINTABLE_WITH_FMT_HPP

