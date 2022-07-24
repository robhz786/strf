#ifndef STRF_DETAIL_DO_TR_PRINT_HPP
#define STRF_DETAIL_DO_TR_PRINT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/simple_string_view.hpp>
#include <strf/detail/printable_traits.hpp>
#include <strf/detail/tr_printer.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

namespace do_tr_print_ {

using strf::detail::mp_type_list;

template <typename Arg>
struct assert_is_printable
{
    static_assert(strf::detail::is_printable<Arg>::value, "Type is not Printable");
    using type = Arg;
};

template <typename Arg>
using assert_is_printable_t = typename assert_is_printable<Arg>::type;

template <typename... Args>
struct first_is_fpe;

template <>
struct first_is_fpe<>
{
    static constexpr bool value = false;
};

template <typename Arg0, typename... OtherArgs>
struct first_is_fpe<Arg0, OtherArgs...>
{
    static constexpr bool value = strf::detail::is_fpe<Arg0>::value;
};

template <bool FirstIsFpe, typename... Args>
struct separate_args_2;

template <typename Arg0, typename... OtherArgs>
struct separate_args_2<true, Arg0, OtherArgs...>
{
private:
    static constexpr bool next_is_fpe = first_is_fpe<OtherArgs...>::value;
    static_assert(sizeof...(OtherArgs) > (unsigned)next_is_fpe, "Missing tr-string");

    using continuation_ = separate_args_2<next_is_fpe, OtherArgs...>;
    using continuation_fpes_ = typename continuation_::fpes;

public:
    using fpes = typename continuation_fpes_::template add_front<Arg0>;
    using printables = typename continuation_::printables;
    using tr_string = typename continuation_::tr_string;
};

template <typename Arg0, typename... Args>
struct separate_args_2<false, Arg0, Args...>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<Args...>;
    using tr_string = Arg0;
};

template <typename... Args>
struct separate_args;

template <>
struct separate_args<>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename... OtherArgs>
struct separate_args<Arg0, OtherArgs...>
{
    using arg0_rmref = strf::detail::remove_cvref_t<Arg0>;
    static constexpr bool arg0_is_fpe = strf::detail::is_fpe<arg0_rmref>::value;

    using helper = separate_args_2<arg0_is_fpe, Arg0, OtherArgs...>;

    using fpes = typename helper::fpes;
    using printables = typename helper::printables;
    using tr_string = typename helper::tr_string;
};


template <typename FpesList, typename PrintablesList, typename IndexSeq>
class tr_print_impl_;

template <typename FpesList, typename PrintablesList>
using tr_print_impl = tr_print_impl_
    < FpesList
    , PrintablesList
    , strf::detail::make_index_sequence<PrintablesList::size> >;


template <typename... Fpes, typename... Printables, std::size_t... I>
class tr_print_impl_
    < mp_type_list<Fpes...>
    , mp_type_list<Printables...>
    , strf::detail::index_sequence<I...> >
{
public:

    template <typename ReservePolicy, typename DestCreator>
    using return_type =
        typename ReservePolicy::template return_type<DestCreator>;

private:

    template <typename CharT>
    static inline STRF_HD const strf::stringifier<CharT>&
    as_stringifier_cref_(const strf::stringifier<CharT>& p) noexcept
    {
        return p;
    }

    template <typename CharT>
    using strf_inilist_ = std::initializer_list<const strf::stringifier<CharT>*>;

    template < bool Ln
             , typename ReservePolicy
             , typename DestCreator
             , typename Charset
             , typename ErrHandler >
    static STRF_HD return_type<ReservePolicy, DestCreator> print_3_
        ( ReservePolicy reserve_policy
        , const DestCreator& dest_creator
        , Charset charset
        , const ErrHandler& err_handler
        , const typename ReservePolicy::preprinting_type* preprinting_arr
        , strf::detail::simple_string_view<typename DestCreator::char_type> tr_string
        , strf_inilist_<typename DestCreator::char_type> stringifiers )
    {
        typename ReservePolicy::preprinting_type pre;
        strf::detail::tr_string_printer<Charset, ErrHandler> tr_printer
            ( pre, preprinting_arr, stringifiers
            , tr_string.begin(), tr_string.end()
            , charset, err_handler );
        return reserve_policy.template print<Ln>(dest_creator, pre, tr_printer);
    }

    template < bool Ln
             , typename ReservePolicy
             , typename DestCreator
             , typename Charset
             , typename ErrHandler
             , typename... Stringifiers >
    static STRF_HD return_type<ReservePolicy, DestCreator> print_2_
        ( ReservePolicy reserve_policy
        , const DestCreator& dest_creator
        , Charset charset
        , const ErrHandler& err_handler
        , const typename ReservePolicy::preprinting_type* preprinting_arr
        , strf::detail::simple_string_view<typename DestCreator::char_type> tr_string
        , const Stringifiers&... stringifiers )
    {
        return print_3_<Ln>( reserve_policy, dest_creator, charset, err_handler
                           , preprinting_arr, tr_string, {&stringifiers...} );
    }

public:

    template <bool Ln, typename ReservePolicy, typename DestCreator>
    static STRF_HD return_type<ReservePolicy, DestCreator> print
        ( ReservePolicy reserve_policy
        , const DestCreator& dest_creator
        , Fpes... fpes
        , strf::detail::simple_string_view<typename DestCreator::char_type> tr_string
        , Printables... printables )
    {
        using preprinting_t = typename ReservePolicy::preprinting_type;
        using char_type = typename DestCreator::char_type;
        auto fp = strf::pack((Fpes&&)fpes...);

        using charset_cat = strf::charset_c<char_type>;
        auto charset = strf::use_facet<charset_cat, void>(fp);

        using err_handler_cat = strf::tr_error_notifier_c;
        auto&& err_handler = strf::use_facet<err_handler_cat, void>(fp);

        constexpr std::size_t printables_count = sizeof...(printables);
        preprinting_t preprinting_arr[printables_count ? printables_count : 1];

        return print_2_<Ln>
            ( reserve_policy, dest_creator, charset, err_handler, preprinting_arr, tr_string
            , as_stringifier_cref_<char_type>
                ( strf::stringifier_type
                  < char_type, preprinting_t, decltype(fp), Printables >
                    ( strf::make_stringifier_input<char_type>
                      ( preprinting_arr[I], fp, (Printables&&)printables ) ) )... );
    }
};

template <typename CharT, typename TrString>
struct check_tr_string_type
{
    constexpr static bool passed = std::is_constructible
        < strf::detail::simple_string_view<CharT>, TrString >
        :: value;
    static_assert(passed, "Invalid type to be used as tr-string");
};


template <bool Ln, typename ReservePolicy, typename DestCreator, typename... Args>
inline STRF_HD typename ReservePolicy::template return_type<DestCreator> do_tr_print
    ( ReservePolicy reserve_policy
    , const DestCreator& dest_creator
    , Args&&... args )
{
    using separator = strf::do_tr_print_::separate_args<Args...>;
    using fpes_list = typename separator::fpes;
    using printables_list = typename separator::printables;
    using tr_string_t = typename separator::tr_string;

    using impl = strf::do_tr_print_::tr_print_impl<fpes_list, printables_list>;
    using char_type = typename DestCreator::char_type;
    using tr_checker = strf::do_tr_print_::check_tr_string_type<char_type, tr_string_t>;
    static_assert(tr_checker::passed, "");

    return impl::template print<Ln>(reserve_policy, dest_creator, (Args&&)args...);
}

} // namespace do_tr_print_
} // namespace strf

#endif  // STRF_DETAIL_DO_TR_PRINT_HPP

