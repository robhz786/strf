#ifndef STRF_DETAIL_PRINTABLE_TYPES_TR_STRING_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_TR_STRING_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/tr_printer.hpp>

namespace strf {

template <typename CharT>
struct tr_string_tag: string_input_tag<CharT>
{
};

namespace detail {

template <typename CharT, typename IdxSeq, typename... Printers>
class tr_printers_container;

template <typename CharT, std::size_t... I, typename... Printers >
class tr_printers_container<CharT, detail::index_sequence<I...>, Printers...>
{
    using printers_tuple_t_ = detail::printers_tuple_impl
        <CharT, detail::index_sequence<I...>, detail::printer_wrapper<CharT, Printers>...>;

    printers_tuple_t_ tuple;

    static constexpr ptrdiff_t num_printers= sizeof...(Printers);

public:

    template < typename... Args
             , typename... FPElems
             , strf::size_presence SizePresence
             , strf::width_presence WidthPresence >
    STRF_HD tr_printers_container
        ( const detail::simple_tuple<Args...>& args
        , const strf::facets_pack<FPElems...>& fp
        , strf::premeasurements<SizePresence, WidthPresence>* pp_array )
        : tuple{args, fp, pp_array}
    {
    }

    template < typename... Args, typename... FPElems >
    STRF_HD tr_printers_container
        ( const detail::simple_tuple<Args...>& args
        , strf::no_premeasurements* pre
        , const strf::facets_pack<FPElems...>& fp )
        : tuple{args, pre, fp}
    {
    }

    using const_polymophic_printer_ptr = const detail::polymorphic_printer<CharT>*;

    struct array_of_pointers_t {
        const_polymophic_printer_ptr arr[num_printers];
    };

    STRF_HD STRF_HD array_of_pointers_t make_array_of_pointers() const
    {
        return {& tuple.template get<I>()...};
    }

    STRF_HD constexpr static std::size_t size()
    {
        return num_printers;
    }
};

template <typename CharT, typename IdxSeq, typename... Args>
struct tr_string_arg
{
    using char_type = CharT;
    detail::simple_string_view<CharT> tr_string;
    detail::simple_tuple<strf::forwarded_printable_type<Args>...> args;
};

template <typename CharT, typename Charset, typename ErrHandler>
struct tr_printer_no_args
{
    using err_handler_t = detail::conditional_t
        < std::is_copy_constructible<ErrHandler>::value
        , ErrHandler
        , const ErrHandler& >;

    detail::simple_string_view<CharT> tr_string_;
    err_handler_t err_handler_;
    Charset charset_;

    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        detail::tr_string_write
            ( tr_string_.begin(), tr_string_.end(), nullptr
            , 0, dst, charset_, err_handler_ );
    }
};

template < typename CharT, typename Charset, typename ErrHandler, typename PrintersContainer >
class tr_printer
{
    static_assert(std::is_same<CharT, typename Charset::code_unit>::value, "");

    using charset_cat_ = strf::charset_c<CharT>;
    using err_handler_cat_ = strf::tr_error_notifier_c;
    using facet_tag_ =  strf::tr_string_tag<CharT>;
    using err_handler_t = detail::conditional_t
        < std::is_copy_constructible<ErrHandler>::value
        , ErrHandler
        , const ErrHandler& >;

    detail::simple_string_view<CharT> tr_string_;
    Charset charset_;
    err_handler_t err_handler_;
    PrintersContainer printers_container_;

public:

    template < typename Pre, typename FPack, typename IdxSeq, typename... Args
             , detail::enable_if_t<Pre::something_demanded, int> = 0 >
    STRF_HD tr_printer
        ( Pre (&pre_arr)[sizeof...(Args)]
        , const FPack& facets
        , const detail::tr_string_arg<CharT, IdxSeq, Args...>& arg )
        : tr_string_{arg.tr_string}
        , charset_{strf::use_facet<charset_cat_, facet_tag_>(facets)}
        , err_handler_{strf::use_facet<err_handler_cat_, facet_tag_>(facets)}
        , printers_container_(arg.args, facets, pre_arr)
    {
    }

    template <typename FPack, typename IdxSeq, typename... Args >
    STRF_HD tr_printer
        ( strf::no_premeasurements* pre
        , const FPack& facets
        , const detail::tr_string_arg<CharT, IdxSeq, Args...>& arg )
        : tr_string_{arg.tr_string}
        , charset_{strf::use_facet<charset_cat_, facet_tag_>(facets)}
        , err_handler_{strf::use_facet<err_handler_cat_, facet_tag_>(facets)}
        , printers_container_(arg.args, facets, pre)
    {
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        (void)dst;
        detail::tr_string_write
            ( tr_string_.begin(), tr_string_.end()
            , printers_container_.make_array_of_pointers().arr
            , PrintersContainer::size(), dst, charset_, err_handler_ );
    }
};

template <typename CharT, typename Pre, typename FPack, typename... Args>
struct tr_printer_alias_helper;

template <typename CharT, typename Pre, typename FPack>
struct tr_printer_alias_helper<CharT, Pre, FPack>
{
    using charset_t = strf::facet_type_in_pack
        < strf::charset_c<CharT>
        , strf::tr_string_tag<CharT>
        , FPack >;

    using tr_error_notifier_t = strf::facet_type_in_pack
        < strf::tr_error_notifier_c
        , strf::tr_string_tag<CharT>
        , FPack >;

    using type = tr_printer_no_args<CharT, charset_t, tr_error_notifier_t>;
};

template <typename CharT, typename Pre, typename FPack, typename... Args>
struct tr_printer_alias_helper
{
    static_assert(sizeof...(Args) != 0, "");

    using charset_t = strf::facet_type_in_pack
        < strf::charset_c<CharT>
        , strf::tr_string_tag<CharT>
        , FPack >;

    using tr_error_notifier_t = strf::facet_type_in_pack
        < strf::tr_error_notifier_c
        , strf::tr_string_tag<CharT>
        , FPack >;

    using printers_container_type =
        tr_printers_container
        < CharT
        , detail::make_index_sequence<sizeof...(Args)>
        , strf::printer_type<CharT, Pre, FPack, Args>... >;

    using type = tr_printer
        < CharT
        , charset_t
        , tr_error_notifier_t
        , printers_container_type >;
};

template <typename CharT, typename Pre, typename FPack, typename... Args>
using tr_printer_alias = typename
    tr_printer_alias_helper<CharT, Pre, FPack, Args...>::type;

} // namespace detail

template < typename StringT
         , typename SimpleStringViewT
             = decltype(detail::to_simple_string_view(std::declval<StringT>()))
         , typename CharT = typename SimpleStringViewT::char_type
         , typename... Args >
constexpr STRF_HD auto tr(const StringT& tr_string, Args&&... args)
    -> detail::tr_string_arg
        < CharT
        , detail::make_index_sequence<sizeof...(Args)>
        , detail::remove_cvref_t<Args>... >
{
    using tuple_t = detail::simple_tuple<strf::forwarded_printable_type<Args>...>;

    return { detail::to_simple_string_view(tr_string)
           , tuple_t{ detail::simple_tuple_from_args{}, (Args&&)args...} };
}

template <typename CharT>
struct printable_def<detail::tr_string_arg<CharT, detail::index_sequence<>>>
{
    using representative = void;
    using forwarded_type = detail::tr_string_arg<CharT, detail::index_sequence<>>;

    template <typename CharIn, typename Pre, typename FPack>
    static STRF_HD auto make_printer
        ( strf::tag<CharIn>
        , Pre* pre
        , const FPack& facets
        , const forwarded_type& arg )
    {
        STRF_MAYBE_UNUSED(pre);

        using charset_cat = strf::charset_c<CharT>;
        using err_handler_cat = strf::tr_error_notifier_c;
        using facet_tag =  strf::tr_string_tag<CharT>;
        using charset_t = strf::facet_type_in_pack<charset_cat, facet_tag, FPack>;
        using wcalc_t = strf::facet_type_in_pack<strf::width_calculator_c, facet_tag, FPack>;
        using tr_error_notifier_t = strf::facet_type_in_pack
            < strf::tr_error_notifier_c, facet_tag, FPack >;

        const auto charset = strf::use_facet<charset_cat, facet_tag>(facets);
        STRF_IF_CONSTEXPR (Pre::size_and_width_demanded) {
            detail::tr_pre_size_and_width<CharT, charset_t, wcalc_t> tr_pre
                ( nullptr, nullptr, 0, pre->remaining_width()
                , strf::use_facet<strf::width_calculator_c, facet_tag>(facets)
                , charset );
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_size(tr_pre.accumulated_ssize());
            pre->add_width(tr_pre.accumulated_width());
        }
        else STRF_IF_CONSTEXPR (Pre::size_demanded) {
            detail::tr_pre_size<CharT> tr_pre
                (nullptr, 0, charset.replacement_char_size());
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_size(tr_pre.accumulated_ssize());
        }
        else STRF_IF_CONSTEXPR (Pre::width_demanded) {
            detail::tr_pre_width<CharT, charset_t, wcalc_t> tr_pre
                ( nullptr, 0, pre->remaining_width()
                , strf::use_facet<strf::width_calculator_c, facet_tag>(facets)
                , charset );
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_width(tr_pre.accumulated_width());
        }

        return detail::tr_printer_no_args<CharT, charset_t, tr_error_notifier_t>
            { arg.tr_string
            , strf::use_facet<err_handler_cat, facet_tag>(facets)
            , charset };
    }
};


template <typename CharT, std::size_t... I, typename... Args>
struct printable_def<detail::tr_string_arg<CharT, detail::index_sequence<I...>, Args...>>
{
    using representative = void;
    using forwarded_type = detail::tr_string_arg<CharT, detail::index_sequence<I...>, Args...>;

    template < typename CharIn, typename FPack >
    static STRF_HD auto make_printer
        ( strf::tag<CharIn>
        , strf::no_premeasurements* pre
        , const FPack& facets
        , const forwarded_type& arg )
        -> detail::tr_printer_alias<CharT, strf::no_premeasurements, FPack, Args...>
    {
        return {pre, facets, arg};
    }

    template < typename CharIn, typename Pre, typename FPack
             , detail::enable_if_t<Pre::something_demanded, int> = 0 >
    static STRF_HD auto make_printer
        ( strf::tag<CharIn>
        , Pre* pre
        , const FPack& facets
        , const forwarded_type& arg )
        -> detail::tr_printer_alias<CharT, Pre, FPack, Args...>
    {
        static_assert( std::is_same<CharT, CharIn>::value
                     , "tr-string character type mismatch" );

        using charset_cat = strf::charset_c<CharT>;
        using facet_tag =  strf::tr_string_tag<CharT>;
        using charset_t = strf::facet_type_in_pack<charset_cat, facet_tag, FPack>;
        using wcalc_t = strf::facet_type_in_pack<strf::width_calculator_c, facet_tag, FPack>;

        Pre pre_arr[sizeof...(Args)] = {};
        detail::tr_printer_alias<CharT, Pre, FPack, Args...> printer{pre_arr, facets, arg};

        constexpr auto num_printers = sizeof...(Args);
        const auto charset = strf::use_facet<charset_cat, facet_tag>(facets);
        STRF_MAYBE_UNUSED(charset);
        STRF_MAYBE_UNUSED(num_printers);

        STRF_IF_CONSTEXPR (Pre::size_and_width_demanded) {
            std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
            strf::width_t width_arr[num_printers] = {pre_arr[I].accumulated_width()...};
            detail::tr_pre_size_and_width<CharT, charset_t, wcalc_t> tr_pre
                ( size_arr, width_arr, num_printers, pre->remaining_width()
                , strf::use_facet<strf::width_calculator_c, facet_tag>(facets)
                , charset );
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_size(tr_pre.accumulated_ssize());
            pre->add_width(tr_pre.accumulated_width());
        }
        else STRF_IF_CONSTEXPR (Pre::size_demanded) {
            std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
            detail::tr_pre_size<CharT> tr_pre
                (size_arr, num_printers, charset.replacement_char_size());
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_size(tr_pre.accumulated_ssize());
        }
        else STRF_IF_CONSTEXPR (Pre::width_demanded) {
            strf::width_t width_arr[num_printers] = {pre_arr[I].accumulated_width()...};
            detail::tr_pre_width<CharT, charset_t, wcalc_t> tr_pre
                ( width_arr, num_printers, pre->remaining_width()
                , strf::use_facet<strf::width_calculator_c, facet_tag>(facets)
                , charset );
            detail::tr_do_premeasurements(tr_pre, arg.tr_string.begin(), arg.tr_string.end());
            pre->add_width(tr_pre.accumulated_width());
        }

        return printer;
    }
};

} // namespace strf

#endif  // STRF_DETAIL_PRINTABLE_TYPES_TR_STRING_HPP

