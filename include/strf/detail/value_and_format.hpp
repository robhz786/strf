#ifndef STRF_DETAIL_VALUE_AND_FORMAT_HPP
#define STRF_DETAIL_VALUE_AND_FORMAT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {

template <typename PrintableDef, class... Fmts>
class value_and_format;

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

template <typename ValueAndFormat, typename FmtList>
struct get_list_of_fmtfn_bases_impl;

template <typename ValueAndFormat, typename... Fmts>
struct get_list_of_fmtfn_bases_impl<ValueAndFormat, mp_type_list<Fmts...>>
{
    using type = mp_type_list<typename Fmts::template fn<ValueAndFormat>...>;
};

template <typename ValueAndFormat, typename FmtList>
using get_list_of_fmtfn_bases =
    typename get_list_of_fmtfn_bases_impl<ValueAndFormat, FmtList>::type;

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
    , typename SrcValueAndFormat
    , typename DstValueAndFormat >
struct separate_fmts_of_two_value_and_format;

template
    < typename SeparatorDstFmt
    , typename SrcPrintableDef
    , typename... SrcFmts
    , typename DstPrintableDef
    , typename... DstFmts >
struct separate_fmts_of_two_value_and_format
    < true
    , SeparatorDstFmt
    , value_and_format<SrcPrintableDef, SrcFmts...>
    , value_and_format<DstPrintableDef, DstFmts...> >
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

    using src_value_and_format = value_and_format<SrcPrintableDef, SrcFmts...>;
    using dst_value_and_format = value_and_format<DstPrintableDef, DstFmts...>;

    using src_fmtfn_bases_before = get_list_of_fmtfn_bases
        < src_value_and_format, src_fmts_before >;

    using dst_fmtfn_bases_before = get_list_of_fmtfn_bases
        < dst_value_and_format, dst_fmts_before >;

    using src_fmtfn_bases_after = get_list_of_fmtfn_bases
        < src_value_and_format, src_fmts_after>;

    using dst_fmtfn_bases_after = get_list_of_fmtfn_bases
        < dst_value_and_format, dst_fmts_after>;

    using separator_fmtfn_base = typename
        SeparatorDstFmt::template fn<dst_value_and_format>;
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

template <typename PrintableDef, class... Fmts>
using value_with_formatters
STRF_DEPRECATED_MSG("value_with_formatters renamed to value_and_format")
= value_and_format<PrintableDef, Fmts...>;

namespace detail {

template <typename T>
struct is_value_and_format : std::false_type
{ };

template <typename... T>
struct is_value_and_format<strf::value_and_format<T...>>: std::true_type
{ };

template <typename T>
struct is_value_and_format<const T> : is_value_and_format<T>
{ };

template <typename T>
struct is_value_and_format<volatile T> : is_value_and_format<T>
{ };

template <typename T>
struct is_value_and_format<T&> : is_value_and_format<T>
{ };

template <typename T>
struct is_value_and_format<T&&> : is_value_and_format<T>
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

template <typename ValueAndFormat>
struct all_base_fmtfn_classes_are_empty;

template <typename PrintableDef, typename... Fmts>
struct all_base_fmtfn_classes_are_empty< value_and_format<PrintableDef, Fmts...> >
    : are_empty<typename Fmts::template fn<value_and_format<PrintableDef, Fmts...>> ...>
{
};

} // namespace detail

template <typename PrintableDef, class... Fmts>
class value_and_format
    : public Fmts::template fn<value_and_format<PrintableDef, Fmts...>> ...
{
public:
    using traits = PrintableDef;
    using value_type = typename PrintableDef::forwarded_type;

    template <typename... OtherFmts>
    using replace_fmts = strf::value_and_format<PrintableDef, OtherFmts ...>;

    explicit constexpr STRF_HD value_and_format(const value_type& v)
        : value_(v)
    {
    }

    template <typename OtherPrintableDef>
    constexpr STRF_HD value_and_format
        ( const value_type& v
        , const strf::value_and_format<OtherPrintableDef, Fmts...>& f )
        : Fmts::template fn<value_and_format<PrintableDef, Fmts...>>
            ( static_cast
              < const typename Fmts
             :: template fn<value_and_format<OtherPrintableDef, Fmts...>>& >(f) )
        ...
        , value_(v)
    {
    }

    template <typename OtherPrintableDef>
    constexpr STRF_HD value_and_format
        ( const value_type& v
        , strf::value_and_format<OtherPrintableDef, Fmts...>&& f )
        : Fmts::template fn<value_and_format<PrintableDef, Fmts...>>
            ( static_cast
              < typename Fmts
             :: template fn<value_and_format<OtherPrintableDef, Fmts...>> &&>(std::move(f)) )
        ...
        , value_(static_cast<value_type&&>(v))
    {
    }

    template <typename... F, typename... FInit>
    constexpr STRF_HD value_and_format
        ( const value_type& v
        , strf::tag<F...>
        , FInit&&... finit )
        : F::template fn<value_and_format<PrintableDef, Fmts...>>
            (std::forward<FInit>(finit))
        ...
        , value_(v)
    {
    }

    template <typename... OtherFmts>
    constexpr STRF_HD explicit value_and_format
        ( const strf::value_and_format<PrintableDef, OtherFmts...>& f )
        : Fmts::template fn<value_and_format<PrintableDef, Fmts...>>
            ( static_cast
              < const typename OtherFmts
             :: template fn<value_and_format<PrintableDef, OtherFmts ...>>& >(f) )
        ...
        , value_(f.value())
    {
    }

    template <typename ... OtherFmts>
    constexpr STRF_HD explicit value_and_format
        ( strf::value_and_format<PrintableDef, OtherFmts...>&& f )
        : Fmts::template fn<value_and_format<PrintableDef, Fmts...>>
            ( static_cast
              < typename OtherFmts
              :: template fn<value_and_format<PrintableDef, OtherFmts ...>>&& >(std::move(f)) )
        ...
        , value_(static_cast<value_type&&>(f.value()))
    {
    }

    template
        < typename SrcPrintableDef
        , typename... SrcFmts
        , typename SelectedFmt
        , typename... SelectedFmtInitArgs
        , typename Helper = detail::separate_fmts_of_two_value_and_format
             < sizeof...(SrcFmts) == sizeof...(Fmts)
             , SelectedFmt
             , value_and_format<SrcPrintableDef, SrcFmts...>
             , value_and_format<PrintableDef, Fmts...> >

        , detail::enable_if_t<Helper::found_selected_fmt_in_dst_list, int> = 0

        , detail::enable_if_t
            < std::is_constructible
                < value_type, const typename SrcPrintableDef::forwarded_type&>::value
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

    constexpr STRF_HD value_and_format
        ( const strf::value_and_format<SrcPrintableDef, SrcFmts...>& src
        , strf::tag<SelectedFmt>
        , SelectedFmtInitArgs&&... args )
        : value_and_format
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
        , typename SrcValueAndFormat
        , typename SelectedBase
        , typename... Args >
    constexpr STRF_HD value_and_format
        ( detail::mp_type_list
            < detail::mp_type_list<SrcBasesBefore...>
            , detail::mp_type_list<DstBasesBefore...>
            , detail::mp_type_list<SrcBasesAfter...>
            , detail::mp_type_list<DstBasesAfter...>
            , SelectedBase >
        , const SrcValueAndFormat& src
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

#endif  // STRF_DETAIL_VALUE_AND_FORMAT_HPP

