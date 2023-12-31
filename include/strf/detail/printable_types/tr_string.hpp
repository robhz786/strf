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


struct destroyable_base
{
    virtual STRF_HD ~destroyable_base() STRF_DEFAULT_IMPL;

    destroyable_base() = default;
    destroyable_base(const destroyable_base&) = default;
    destroyable_base(destroyable_base&&) = default;
    destroyable_base& operator=(const destroyable_base&) = default;
    destroyable_base& operator=(destroyable_base&&) = default;
};

template <typename CharT, typename IdxSeq, typename... Printers>
struct tr_printers_container;

template <typename CharT, std::size_t... I, typename... Printers >
struct tr_printers_container<CharT, strf::detail::index_sequence<I...>, Printers...>
    : destroyable_base
{
    using printers_tuple_t_ = strf::detail::printers_tuple_impl
        <CharT, strf::detail::index_sequence<I...>, Printers...>;

    static constexpr ptrdiff_t num_printers_= sizeof...(Printers);
    printers_tuple_t_ tuple;
    const strf::printer<CharT>* array_of_pointers[num_printers_];

    template < typename... Args
             , typename... FPElems
             , strf::size_presence SizePresence
             , strf::width_presence WidthPresence >
    STRF_HD tr_printers_container
        ( const strf::detail::simple_tuple<Args...>& args
        , const strf::facets_pack<FPElems...>& fp
        , strf::premeasurements<SizePresence, WidthPresence>* pp_array
        , const strf::printer<CharT>**& array_of_pointers_ref )
        : tuple{args, fp, pp_array}
        , array_of_pointers{ & tuple.template get<I>()...}
    {
        array_of_pointers_ref = array_of_pointers;
    }

    template < typename... Args, typename... FPElems >
    STRF_HD tr_printers_container
        ( const strf::detail::simple_tuple<Args...>& args
        , strf::no_premeasurements* pre
        , const strf::facets_pack<FPElems...>& fp
        , const strf::printer<CharT>**& array_of_pointers_ref )
        : tuple{args, pre, fp}
        , array_of_pointers{ & tuple.template get<I>()...}
    {
        array_of_pointers_ref = array_of_pointers;
    }
};

// template <typename CharT>
// class printers_array_deferred_init_impl<CharT, 0, 0>
// {
// public:
//     template < typename... FPElems
//              , strf::size_presence SizePresence
//              , strf::width_presence WidthPresence >
//     void construct
//         ( strf::detail::simple_tuple<> args
//         , const strf::facets_pack<FPElems...>&
//         , strf::premeasurements<SizePresence, WidthPresence>*
//         , const strf::printer<CharT>**& array_of_pointers_ref )
//     {
//         using pre_t = strf::premeasurements<SizePresence, WidthPresence>;
//         using fp_t = strf::facets_pack<FPElems...>;

//         using printer_tuple_t =
//             strf::detail::printers_tuple_from_args<CharT, pre_t, fp_t, Args...>;

//         static_assert(sizeof(printer_tuple_t) <= MemSize, "");
//         static_assert(alignof(printer_tuple_t) <= Alignment, "");

//         new (&storage_) tr_printers_container<printer_tuple_t>
//             (args, fp, pre_array, array_of_pointers_ref);
//     }
// };




template <typename CharT, std::size_t MemSize, std::size_t Alignment>
class printers_array_deferred_init_impl
{
    using storage_type_ = typename std::aligned_storage<MemSize, Alignment>:: type;

    // template < typename... Args
    //          , typename... FPElems
    //          , strf::size_presence SizePresence
    //          , strf::width_presence WidthPresence >
    // using printer_tuple_t_ = strf::detail::printers_tuple_from_args
    //     < CharT
    //     , strf::premeasurements<SizePresence, WidthPresence>
    //     , strf::facets_pack<FPElems...>
    //     , Args...>;

    storage_type_ storage_;

public:

    printers_array_deferred_init_impl() = default;
    ~printers_array_deferred_init_impl() = default;

    printers_array_deferred_init_impl(const printers_array_deferred_init_impl&) = delete;
    printers_array_deferred_init_impl(printers_array_deferred_init_impl&&) = delete;

    printers_array_deferred_init_impl&
    operator=(const printers_array_deferred_init_impl&) = delete;

    printers_array_deferred_init_impl&
    operator=(printers_array_deferred_init_impl&&) = delete;

    template < typename... Args
             , typename... FPElems
             , strf::size_presence SizePresence
             , strf::width_presence WidthPresence >
    STRF_HD void construct
        ( const strf::detail::simple_tuple<Args...>& args
        , const strf::facets_pack<FPElems...>& fp
        , strf::premeasurements<SizePresence, WidthPresence>* pre_array
        , const strf::printer<CharT>**& array_of_pointers_ref )
    {
        using pre_t = strf::premeasurements<SizePresence, WidthPresence>;
        using fp_t = strf::facets_pack<FPElems...>;

        using printers_container = strf::detail::tr_printers_container
			< CharT
			, strf::detail::make_index_sequence<sizeof...(Args)>
			, strf::printer_type<CharT, pre_t, fp_t, Args>... >;

        static_assert(sizeof(printers_container) <= MemSize, "");
        static_assert(alignof(printers_container) <= Alignment, "");

        new (&storage_) printers_container(args, fp, pre_array, array_of_pointers_ref);
    }

    template <typename... Args, typename... FPElems>
    STRF_HD void construct
        ( const strf::detail::simple_tuple<Args...>& args
        , const strf::facets_pack<FPElems...>& fp
        , strf::no_premeasurements* pre
        , const strf::printer<CharT>**& array_of_pointers_ref )
    {
        using pre_t = strf::no_premeasurements;
        using fp_t = strf::facets_pack<FPElems...>;

        using printers_container = strf::detail::tr_printers_container
			< CharT
			, strf::detail::make_index_sequence<sizeof...(Args)>
			, strf::printer_type<CharT, pre_t, fp_t, Args>... >;

        static_assert(sizeof(printers_container) <= MemSize, "");
        static_assert(alignof(printers_container) <= Alignment, "");

        new (&storage_) printers_container(args, pre, fp, array_of_pointers_ref);
    }


    STRF_HD void destroy()
    {
#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
        reinterpret_cast<destroyable_base*>(&storage_)->~destroyable_base();

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif
    }
};

template <typename CharT, typename Charset, typename ErrHandler>
class tr_printer_no_args;

template <typename CharT, typename Charset, typename ErrHandler, typename PrintersStorage>
class tr_printer;

template <typename CharT, typename Pre, typename FPack, typename... Args>
struct printers_array_deferred_init_alias_helper
{
    using data_t = tr_printers_container
		< CharT
		, strf::detail::make_index_sequence<sizeof...(Args)>
		, strf::printer_type<CharT, Pre, FPack, Args>... >;

    using type = printers_array_deferred_init_impl
        <CharT, sizeof(data_t), alignof(destroyable_base) >;
};

template <typename CharT, typename Pre, typename FPack, typename... Args>
using printers_array_deferred_init = typename
    printers_array_deferred_init_alias_helper<CharT, Pre, FPack, Args...>::type;

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

    using type = tr_printer
        < CharT
        , charset_t
        , tr_error_notifier_t
        , printers_array_deferred_init<CharT, Pre, FPack, Args...> >;
};

template <typename CharT, typename Pre, typename FPack, typename... Args>
using tr_printer_alias = typename
    tr_printer_alias_helper<CharT, Pre, FPack, Args...>::type;

template <typename CharT, typename... Args>
struct tr_string_arg
{
    using char_type = CharT;
    strf::detail::simple_string_view<CharT> tr_string;
    strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...> args;
};

template < typename CharT
         , typename Pre
         , typename FPack
         , typename... Args >
struct tr_printer_input
{
    Pre* pre;
    FPack facets;
    detail::tr_string_arg<CharT, Args...> arg;
    //Pre[sizeof...(Args)] pre_arr;

    using printer_type = tr_printer_alias<CharT, Pre, FPack, Args...>;
};


template <typename CharT, typename Charset, typename ErrHandler>
class tr_printer_no_args: public strf::printer<CharT>
{
    using charset_cat_ = strf::charset_c<CharT>;
    using err_handler_cat_ = strf::tr_error_notifier_c;
    using facet_tag_ =  strf::tr_string_tag<CharT>;
    using err_handler_t = strf::detail::conditional_t
        < std::is_copy_constructible<ErrHandler>::value
        , ErrHandler
        , const ErrHandler& >;

    strf::detail::simple_string_view<CharT> tr_string_;
    Charset charset_;
    err_handler_t err_handler_;

    template <typename FPack>
    void STRF_HD init_
        ( const tr_printer_input<CharT, strf::no_premeasurements, FPack>& )
    {
    }

    template <typename FPack>
    void STRF_HD init_
        ( const tr_printer_input<CharT, strf::full_premeasurements, FPack>& i)
    {
        using wcalc_t = strf::facet_type_in_pack
             <strf::width_calculator_c, strf::tr_string_tag<CharT>, FPack>;

        strf::detail::tr_pre_size_and_width<CharT, Charset, wcalc_t> tr_pre
              ( nullptr, nullptr, 0, i.pre->remaining_width()
              , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
              , charset_ );
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->add_size(tr_pre.accumulated_ssize());
        i.pre->reset_remaining_width(tr_pre.remaining_width());
    }

    template <typename FPack>
    void STRF_HD init_
        ( const tr_printer_input
            < CharT
            , strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>
            , FPack >& i )
    {
        strf::detail::tr_pre_size<CharT> tr_pre
            (nullptr, 0, charset_.replacement_char_size());
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->add_size(tr_pre.accumulated_ssize());
    }

    template <std::size_t... I, typename FPack>
    void STRF_HD init_
        ( const tr_printer_input
            < CharT
            , strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>
            , FPack >& i )
    {
        using wcalc_t = strf::facet_type_in_pack
             <strf::width_calculator_c, strf::tr_string_tag<CharT>, FPack>;
        strf::detail::tr_pre_width<CharT, Charset, wcalc_t> tr_pre
              ( nullptr, 0, i.pre->remaining_width()
              , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
              , charset_ );
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->reset_remaining_width(tr_pre.remaining_width());
    }

public:

    template <typename Pre, typename FPack>
    explicit STRF_HD tr_printer_no_args
        ( tr_printer_input<CharT, Pre, FPack> i)
        : tr_string_{i.arg.tr_string}
        , charset_{strf::use_facet<charset_cat_, facet_tag_>(i.facets)}
        , err_handler_{strf::use_facet<err_handler_cat_, facet_tag_>(i.facets)}

    {
        init_(i);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        (void)dst;
        strf::detail::tr_string_write
            ( tr_string_.begin(), tr_string_.end(), nullptr
            , 0, dst, charset_, err_handler_ );
    }
};


template <typename CharT, typename Charset, typename ErrHandler, typename PrintersStorage>
class tr_printer: public strf::printer<CharT>
{
    static_assert(std::is_same<CharT, typename Charset::code_unit>::value, "");

    using charset_cat_ = strf::charset_c<CharT>;
    using err_handler_cat_ = strf::tr_error_notifier_c;
    using facet_tag_ =  strf::tr_string_tag<CharT>;

    using err_handler_t = strf::detail::conditional_t
        < std::is_copy_constructible<ErrHandler>::value
        , ErrHandler
        , const ErrHandler& >;

    strf::detail::simple_string_view<CharT> tr_string_;
    Charset charset_;
    err_handler_t err_handler_;
    std::ptrdiff_t num_printers_;
    const strf::printer<CharT>** printers_;
    PrintersStorage storage_;

    // template <std::size_t... I, typename Pre, typename FPack, typename... Args>
    // void STRF_HD init_
    //     ( strf::detail::index_sequence<I...>
    //     , const tr_printer_input<CharT, Pre, FPack, Args...>& i)
    // {
    //     using wcalc_t = strf::facet_type_in_pack
    //         <strf::width_calculator_c, strf::tr_string_tag<CharT>, FPack>;

    //     constexpr std::size_t num_printers = sizeof...(I);
    //     STRF_IF_CONSTEXPR (Pre::size_and_width_demanded) {
    //         Pre pre_arr[num_printers];
    //         storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
    //         std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
    //         strf::width_t width_arr[num_printers] =
    //             {(strf::width_max - pre_arr[I].remaining_width())...};
    //         strf::detail::tr_pre_size_and_width<CharT, Charset, wcalc_t> tr_pre
    //             ( size_arr, width_arr, num_printers, i.pre->remaining_width()
    //             , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
    //             , charset_ );
    //         strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
    //         i.pre->add_size(tr_pre.accumulated_ssize());
    //         i.pre->reset_remaining_width(tr_pre.remaining_width());

    //     } else STRF_IF_CONSTEXPR (Pre::size_demanded) {
    //         Pre pre_arr[num_printers];
    //         storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
    //         std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
    //         strf::detail::tr_pre_size<CharT> tr_pre
    //             (size_arr, num_printers, charset_.replacement_char_size());
    //         strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin, tr_string_.end());
    //         i.pre->add_size(tr_pre.accumulated_ssize());

    //     } else if (input.pre->has_remaining_width()) {
    //         Pre pre_arr[num_printers];
    //         storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
    //         strf::width_t width_arr[num_printers] =
    //             {(strf::width_max - pre_arr[I].remaining_width())...};
    //         strf::detail::tr_pre_width<CharT, Charset, wcalc_t> tr_pre
    //             ( width_arr, num_printers, i.pre->remaining_width()
    //             , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
    //             , charset_ );
    //         strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
    //         i.pre->reset_remaining_width(tr_pre.remaining_width());

    //     } else {
    //         storage_.construct(i.arg.args, i.facets, i.pre, printers_);
    //     }
    // }

    template <std::size_t... I, typename FPack, typename... Args>
    void STRF_HD init_
        ( strf::detail::index_sequence<I...>
        , const tr_printer_input<CharT, strf::no_premeasurements, FPack, Args...>& i)
    {
        storage_.construct(i.arg.args, i.facets, i.pre, printers_);
    }

    template <std::size_t... I, typename FPack, typename... Args>
    void STRF_HD init_
        ( strf::detail::index_sequence<I...>
        , const tr_printer_input<CharT, strf::full_premeasurements, FPack, Args...>& i)
    {
        constexpr std::size_t num_printers = sizeof...(I);
        using wcalc_t = strf::facet_type_in_pack
             <strf::width_calculator_c, strf::tr_string_tag<CharT>, FPack>;
        strf::full_premeasurements pre_arr[num_printers];
        storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
        std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
        strf::width_t width_arr[num_printers] =
            {(strf::width_max - pre_arr[I].remaining_width())...};
        strf::detail::tr_pre_size_and_width<CharT, Charset, wcalc_t> tr_pre
            ( size_arr, width_arr, num_printers, i.pre->remaining_width()
              , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
              , charset_ );
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->add_size(tr_pre.accumulated_ssize());
        i.pre->reset_remaining_width(tr_pre.remaining_width());
    }

    template <std::size_t... I, typename FPack, typename... Args>
    void STRF_HD init_
        ( strf::detail::index_sequence<I...>
        , const tr_printer_input
            < CharT
            , strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>
            , FPack
            , Args...>& i )
    {
        constexpr std::size_t num_printers = sizeof...(I);
        strf::premeasurements<strf::size_presence::yes, strf::width_presence::no> pre_arr[num_printers];
        storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
        std::ptrdiff_t size_arr[num_printers] = {pre_arr[I].accumulated_ssize()...};
        strf::detail::tr_pre_size<CharT> tr_pre
            (size_arr, num_printers, charset_.replacement_char_size());
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->add_size(tr_pre.accumulated_ssize());
    }

    template <std::size_t... I, typename FPack, typename... Args>
    void STRF_HD init_
        ( strf::detail::index_sequence<I...>
        , const tr_printer_input
            < CharT
            , strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>
            , FPack
            , Args...>& i )
    {
        constexpr std::size_t num_printers = sizeof...(I);
        using wcalc_t = strf::facet_type_in_pack
             <strf::width_calculator_c, strf::tr_string_tag<CharT>, FPack>;
        strf::premeasurements<strf::size_presence::no, strf::width_presence::yes>
            pre_arr[num_printers];
        storage_.construct(i.arg.args, i.facets, pre_arr, printers_);
        strf::width_t width_arr[num_printers] =
            {(strf::width_max - pre_arr[I].remaining_width())...};
        strf::detail::tr_pre_width<CharT, Charset, wcalc_t> tr_pre
            ( width_arr, num_printers, i.pre->remaining_width()
            , strf::use_facet<strf::width_calculator_c, facet_tag_>(i.facets)
            , charset_ );
        strf::detail::tr_do_premeasurements(tr_pre, tr_string_.begin(), tr_string_.end());
        i.pre->reset_remaining_width(tr_pre.remaining_width());
    }

public:

    template <typename Pre, typename FPack, typename... Args>
    explicit STRF_HD tr_printer
        ( tr_printer_input<CharT, Pre, FPack, Args...> i)
        : tr_string_{i.arg.tr_string}
        , charset_{strf::use_facet<charset_cat_, facet_tag_>(i.facets)}
        , err_handler_{strf::use_facet<err_handler_cat_, facet_tag_>(i.facets)}
        , num_printers_(sizeof...(Args))

    {
        init_(strf::detail::make_index_sequence<sizeof...(Args)>{}, i);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        (void)dst;
        strf::detail::tr_string_write
            ( tr_string_.begin(), tr_string_.end(), printers_
            , num_printers_, dst, charset_, err_handler_ );
    }

    tr_printer(const tr_printer&) = delete;
    tr_printer(tr_printer&&) = delete;
    tr_printer& operator=(const tr_printer&) = delete;
    tr_printer& operator=(tr_printer&&) = delete;

    STRF_HD ~tr_printer() override
    {
        storage_.destroy();
    }
};

} // namespace detail


template < typename StringT
         , typename SimpleStringViewT
             = decltype(strf::detail::to_simple_string_view(std::declval<StringT>()))
         , typename CharT = typename SimpleStringViewT::char_type
         , typename... Args >
constexpr STRF_HD detail::tr_string_arg<CharT, strf::detail::remove_cvref_t<Args>...>
tr(const StringT& tr_string, Args&&... args)
{
    using tuple_t = strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>;

    return { strf::detail::to_simple_string_view(tr_string)
           , tuple_t{ detail::simple_tuple_from_args{}, (Args&&)args...} };
}

template <typename CharT, typename... Args>
struct printable_traits<detail::tr_string_arg<CharT, Args...>>
{
    using representative_type = void;
    using forwarded_type = detail::tr_string_arg<CharT, Args...>;

    template <typename CharIn, typename Pre, typename FPack>
    static STRF_HD auto make_printer
        ( strf::tag<CharIn>
        , Pre* pre
        , const FPack& facets
        , const forwarded_type& arg )
        -> strf::detail::tr_printer_input<CharT, Pre, FPack, Args...>
    {
        static_assert( std::is_same<CharT, CharIn>::value
                     , "tr-string character type mismatch" );
        return {pre, facets, arg};
    }
};

// template <typename CharT, typename... Args>
// constexpr STRF_HD auto tag_invoke
//     (strf::printable_tag, const detail::tr_string_arg<CharT, Args...>&) noexcept
//     -> strf::printable_traits<detail::tr_string_arg<CharT, Args...>>
//     { return {}; }

// namespace detail {

// template < typename CharT, typename Charset, typename ErrHandler
//          , std::size_t... I, typename... Printers >
// class tr_printer
//     < CharT, Charset, ErrHandler
//     , strf::detail::printers_tuple_impl
//         <CharT, strf::detail::index_sequence<I...>, Printers...> >

//     : public strf::printer<typename Charset::code_unit>
// {
//     static_assert(std::is_same<CharT, typename Charset::code_unit>::value, "");
//     using charset_cat_ = strf::charset_c<CharT>;
//     using err_handler_cat_ = strf::tr_error_notifier_c;
//     using printers_tuple_t_ = strf::detail::printers_tuple_impl
//         <CharT, strf::detail::index_sequence<I...>, Printers...>;
//     using facet_tag_ =  strf::tr_string_tag<CharT>;

//     static constexpr std::size_t num_printers_ = printers_tuple_t_::size;

// public:

//     template <typename Ch, typename Pre, typename... T>
//     explicit STRF_HD tr_printer(strf::usual_printer_input<Ch, Pre, T...> i)
//         : printers_tuple_{i.arg.args, i.pre, i.facets}
//         , printer_ptrs_array_{&printers_tuple_.template get<I>()...}
//         , tr_string_{i.arg.tr_string}
//         , charset_{strf::use_facet<charset_cat_, facet_tag_>(i.facets)}
//         , err_handler_{strf::use_facet<err_handler_cat_, facet_tag_>(i.facets)}

//     {
//         STRF_IF_CONSTEXPR (Pre::something_demanded) {
//             Pre pre[num_printers_];

//             auto invalid_arg_size = charset_.replacement_char_size();
//             std::ptrdiff_t s = strf::detail::tr_string_size
//                 ( printer_ptrs_array_, num_printers_, tr_string_.begin(), tr_string_.end()
//                 , invalid_arg_size );
//             i.pre->add_size(s);
//         }
//     }

//     STRF_HD void print_to(strf::destination<CharT>& dst) const override
//     {
//         (void)dst;
//         // strf::detail::tr_string_write
//         //     ( tr_string_.begin(), tr_string_.end(), printer_ptrs_array_
//         //     , num_printers_, dst, charset_, err_handler_ );
//     }

// private:

//     printers_tuple_t_ printers_tuple_;
//     const strf::printer<CharT>* printer_ptrs_array_[num_printers_];
//     strf::detail::simple_string_view<CharT> tr_string_;
//     Charset charset_;

//     using err_handler_t = strf::detail::conditional_t
//         < std::is_copy_constructible<ErrHandler>::value
//         , ErrHandler
//         , const ErrHandler& >;
//     err_handler_t err_handler_;
// };

// } // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_PRINTABLE_TYPES_TR_STRING_HPP

