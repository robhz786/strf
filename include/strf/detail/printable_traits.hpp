#ifndef STRF_DETAIL_PRINTABLE_TRAITS_HPP
#define STRF_DETAIL_PRINTABLE_TRAITS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_with_fmt.hpp>
#include <strf/detail/preprinting.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

template<typename T>
struct printable_traits;

template<typename PrintingTraits, typename... Fmts>
struct printable_traits<strf::printable_with_fmt<PrintingTraits, Fmts...>> : PrintingTraits
{
};

namespace detail {

template <typename T>
struct printable_traits_finder;

} // namespace detail

template <typename T>
using printable_traits_of = typename
    detail::printable_traits_finder<strf::detail::remove_cvref_t<T>>
    ::traits;

struct printable_tag
{
private:
    static const printable_tag& tag_();

public:

    template < typename Arg >
    constexpr STRF_HD auto operator()(Arg&&) const -> strf::printable_traits_of<Arg>
    {
        return {};
    }
};

using print_traits_tag
STRF_DEPRECATED_MSG("print_traits_tag type renamed to printable_tag")
=  printable_tag;

template <typename T>
using print_traits
STRF_DEPRECATED_MSG("print_traits renamed to printable_traits")
= printable_traits<T>;

template <typename T>
using print_traits_of
STRF_DEPRECATED_MSG("print_traits_of renamed to printable_traits_of")
= printable_traits_of<T>;

namespace detail {

template <typename T>
struct has_tag_invoke_with_printable_tag_tester
{
    template < typename U
             , typename = decltype(strf::detail::tag_invoke(strf::printable_tag{}, std::declval<U>())) >
    static STRF_HD std::true_type test_(const U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using result = decltype(test_<T>((T*){}));
};

template <typename T>
using  has_tag_invoke_with_printable_tag =
    typename has_tag_invoke_with_printable_tag_tester<strf::detail::remove_cvref_t<T>>::result;


template <typename T>
struct has_printable_traits_specialization
{
    template <typename U, typename = typename strf::printable_traits<U>::forwarded_type>
    static STRF_HD std::true_type test(const U*);

    template <typename U>
    static STRF_HD std::false_type test(...);

    using T_ = strf::detail::remove_cvref_t<T>;
    using result = decltype(test<T_>((const T_*){}));

    constexpr static bool value = result::value;
};

template <bool HasPrintableTraits, typename T>
struct is_printable_tester_2;

template <typename T>
struct is_printable_tester_2<true, T> : std::true_type
{
};

template <typename T>
struct is_printable_tester_2<false, T>: has_tag_invoke_with_printable_tag<T>
{
};

template <typename T>
struct is_printable_tester
    : is_printable_tester_2<strf::detail::has_printable_traits_specialization<T>::value, T>
{
};

template <typename T>
using is_printable = is_printable_tester< strf::detail::remove_cvref_t<T> >;

struct select_printable_traits_specialization
{
    template <typename T>
    using select = strf::printable_traits<T>;
};

struct select_printable_traits_from_tag_invoke
{
    template <typename T>
    using select = decltype
        ( strf::detail::tag_invoke(strf::printable_tag{}, std::declval<T>() ));
};

template <typename T>
struct printable_traits_finder
{
    using selector_ = strf::detail::conditional_t
        < strf::detail::has_printable_traits_specialization<T>::value
        , strf::detail::select_printable_traits_specialization
        , strf::detail::select_printable_traits_from_tag_invoke >;

    using traits = typename selector_::template select<T>;
    using forwarded_type = typename traits::forwarded_type;
};

template <typename Traits, typename... F>
struct printable_traits_finder<strf::printable_with_fmt<Traits, F...>>
{
    using traits = Traits;
    using forwarded_type = strf::printable_with_fmt<Traits, F...>;
};

template <typename T>
struct printable_traits_finder<T&> : printable_traits_finder<T>
{
};

template <typename T>
struct printable_traits_finder<T&&> : printable_traits_finder<T>
{
};

template <typename T>
struct printable_traits_finder<const T> : printable_traits_finder<T>
{
};

template <typename T>
struct printable_traits_finder<volatile T> : printable_traits_finder<T>
{
};

template <typename PrintingTraits, typename Formatters>
struct mp_define_printable_with_fmt;

template < typename PrintingTraits
         , template <class...> class List
         , typename... Fmts >
struct mp_define_printable_with_fmt<PrintingTraits, List<Fmts...>>
{
    using type = strf::printable_with_fmt<PrintingTraits, Fmts...>;
};

template <typename PrintingTraits>
struct extract_formatters_from_printable_traits_impl
{
private:
    template <typename U, typename Fmts = typename U::formatters>
    static Fmts get_formatters_(U*);

    template <typename U>
    static strf::tag<> get_formatters_(...);

public:

    using type = decltype(get_formatters_<PrintingTraits>(nullptr));
};

template <typename PrintingTraits>
using extract_formatters_from_printable_traits =
    typename extract_formatters_from_printable_traits_impl<PrintingTraits>::type;

template <typename PrintingTraits>
using default_value_with_formatter_of_printable_traits = typename
    strf::detail::mp_define_printable_with_fmt
        < PrintingTraits
        , extract_formatters_from_printable_traits<PrintingTraits> >
    :: type;

template <typename T>
struct formatters_finder
{
    using traits = typename printable_traits_finder<T>::traits;
    using formatters = extract_formatters_from_printable_traits<traits>;
    using fmt_type = typename
        strf::detail::mp_define_printable_with_fmt<traits, formatters>::type;
};

template <typename PrintingTraits, typename... Fmts>
struct formatters_finder<strf::printable_with_fmt<PrintingTraits, Fmts...>>
{
    using traits = PrintingTraits;
    using formatters = strf::tag<Fmts...>;
    using fmt_type = strf::printable_with_fmt<PrintingTraits, Fmts...>;
};

} // namespace detail

template <typename T>
using forwarded_printable_type = typename
    detail::printable_traits_finder<strf::detail::remove_cvref_t<T>>
    ::forwarded_type;

template <typename T>
using fmt_type = typename
    detail::formatters_finder<strf::detail::remove_cvref_t<T>>
    ::fmt_type;

template <typename T>
using fmt_value_type = typename fmt_type<T>::value_type;

template <typename T>
using formatters_of = typename strf::detail::formatters_finder<T>::formatters;

inline namespace format_functions {

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename T>
constexpr STRF_HD fmt_type<T> fmt(T&& value)
    noexcept(noexcept(fmt_type<T>{fmt_value_type<T>{value}}))
{
    return fmt_type<T>{fmt_value_type<T>{value}};
}

#else //defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

namespace detail_format_functions {

struct fmt_fn
{
    template < typename T
             , bool IsVWF = detail::is_printable_with_fmt<T>::value
             , strf::detail::enable_if_t<!IsVWF, int> = 0 >
    constexpr STRF_HD fmt_type<T> operator()(T&& value) const
        noexcept(noexcept(fmt_type<T>{fmt_value_type<T>{(T&&)value}}))
    {
        return fmt_type<T>{fmt_value_type<T>{(T&&)value}};
    }

    template < typename T
             , bool IsVWF = detail::is_printable_with_fmt<T>::value
             , strf::detail::enable_if_t<IsVWF, int> = 0 >
    constexpr STRF_HD T&& operator()(T&& value) const
    {
        return static_cast<T&&>(value);
    }
};

} // namespace detail_format_functions

constexpr detail_format_functions::fmt_fn fmt {};

#endif

}  // namespace format_functions

struct printable_overrider_c;
struct dont_override;

using print_override_c
STRF_DEPRECATED_MSG("print_override_c was renamed printable_overrider_c")
= printable_overrider_c;

using no_print_override
STRF_DEPRECATED_MSG("no_print_override was renamed dont_override")
= dont_override;

namespace detail {

template <typename PrintableTraits>
struct get_is_overridable_helper {
    template <typename U>
    static STRF_HD typename U::is_overridable test_(const U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using result = decltype(test_<PrintableTraits>((PrintableTraits*){}));
};

template <typename PrintableTraits>
using get_is_overridable = typename
    get_is_overridable_helper<PrintableTraits>::result;

namespace mk_pr_in {

template < typename UnadaptedMaker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg >
struct can_make_printer_input_impl
{
    template <typename P>
    static STRF_HD auto test_(PrePrinting& pre, const FPack& facets, const Arg& arg)
        -> decltype( std::declval<const P&>()
                         .make_input
                             ( strf::tag<CharT>{}, pre, facets, arg )
                   , std::true_type{} );

    template <typename P>
    static STRF_HD std::false_type test_(...);

    using result = decltype
        ( test_<UnadaptedMaker>
            ( std::declval<PrePrinting&>()
            , std::declval<FPack>()
            , std::declval<Arg>() ));
};

template < typename UnadaptedMaker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg >
using can_make_printer_input = typename
    can_make_printer_input_impl<UnadaptedMaker, CharT, PrePrinting, FPack, Arg>
    ::result;

struct arg_adapter_rm_fmt
{
    template <typename PrintingTraits, typename... Fmts>
    static constexpr STRF_HD const typename PrintingTraits::forwarded_type&
    adapt_arg(const strf::printable_with_fmt<PrintingTraits, Fmts...>& x)
    {
        return x.value();
    }
};

template <typename To>
struct arg_adapter_cast
{
    template <typename From>
    static constexpr STRF_HD To adapt_arg(const From& x)
    {
        return static_cast<To>(x);
    }
};

template < typename PrintingTraits
         , typename Maker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Vwf
         , typename DefaultVwf >
struct adapter_selector_3
{
    static_assert( ! std::is_same<Vwf, DefaultVwf>::value, "");
    using adapter_type = arg_adapter_cast<const Vwf&>;
};

template < typename PrintingTraits
         , typename Maker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename DefaultVwf >
struct adapter_selector_3<PrintingTraits, Maker, CharT, PrePrinting, FPack, DefaultVwf, DefaultVwf>
{
    static constexpr bool can_pass_directly =
        can_make_printer_input<Maker, CharT, PrePrinting, FPack, DefaultVwf>
        ::value;

    using adapter_type = typename std::conditional
        < can_pass_directly
        , arg_adapter_cast<const DefaultVwf&>
        , arg_adapter_rm_fmt >
        ::type;
};

template < typename PrintingTraits
         , typename Maker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , typename DefaultVwf >
struct adapter_selector_2
{
    static constexpr bool can_pass_directly =
        can_make_printer_input<Maker, CharT, PrePrinting, FPack, Arg>
        ::value;
    static constexpr bool can_pass_as_fmt =
        can_make_printer_input<Maker, CharT, PrePrinting, FPack, DefaultVwf>
        ::value;
    static constexpr bool shall_adapt = !can_pass_directly && can_pass_as_fmt;

    using destination_type = typename std::conditional
        < shall_adapt, DefaultVwf, typename PrintingTraits::forwarded_type>
        :: type;
    using adapter_type = arg_adapter_cast<destination_type>;
};

template < typename PrintingTraits
         , typename Maker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename DefaultVwf
         , typename... Fmts >
struct adapter_selector_2
    < PrintingTraits, Maker, CharT, PrePrinting, FPack
    , strf::printable_with_fmt<PrintingTraits, Fmts...>, DefaultVwf >
{
    using vwf = strf::printable_with_fmt<PrintingTraits, Fmts...>;
    using other = adapter_selector_3
        < PrintingTraits, Maker, CharT, PrePrinting, FPack, vwf, DefaultVwf >;
    using adapter_type = typename other::adapter_type;
};

template < typename PrintingTraits
         , typename Maker
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg >
struct adapter_selector
{
    using vwf = default_value_with_formatter_of_printable_traits<PrintingTraits>;
    using other = adapter_selector_2<PrintingTraits, Maker, CharT, PrePrinting, FPack, Arg, vwf>;
    using adapter_type = typename other::adapter_type;
};

template < typename PrintingTraits, typename Maker, typename CharT
         , typename PrePrinting, typename FPack, typename Arg >
using select_adapter = typename
    adapter_selector<PrintingTraits, Maker, CharT, PrePrinting, FPack, Arg>
    ::adapter_type;

template <typename Overrider, typename OverrideTag>
struct maker_getter_overrider
{
    using return_maker_type = const Overrider&;
    using maker_type = Overrider;

    template <typename FPack>
    static constexpr STRF_HD return_maker_type get_maker(const FPack& fp)
    {
        return strf::use_facet<strf::printable_overrider_c, OverrideTag>(fp);
    }
};

template <typename PrintingTraits>
struct maker_getter_printable_traits
{
    using return_maker_type = PrintingTraits;
    using maker_type = PrintingTraits;

    template <typename FPack>
    static constexpr STRF_HD maker_type get_maker(const FPack&)
    {
        return maker_type{};
    }
};

template < typename PrintingTraits
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , bool Overridable >
struct maker_getter_selector_2
{
    static_assert(Overridable, "");
    using representative_type = typename PrintingTraits::representative_type;
    using overrider_ = decltype
        ( strf::use_facet<strf::printable_overrider_c, representative_type>(std::declval<FPack>()) );
    using overrider = strf::detail::remove_cvref_t<overrider_>;
    using maker_getter_type = typename std::conditional
        < std::is_same<overrider, strf::dont_override>::value
        , maker_getter_printable_traits<PrintingTraits>
        , maker_getter_overrider<overrider, representative_type> >
        ::type;
};

template < typename PrintingTraits
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg >
struct maker_getter_selector_2<PrintingTraits, CharT, PrePrinting, FPack, Arg, false>
{
    using maker_getter_type = maker_getter_printable_traits<PrintingTraits>;
};

template < typename PrintingTraits
         , typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg >
struct maker_getter_selector
{
    using other = maker_getter_selector_2
        < PrintingTraits, CharT, PrePrinting, FPack, Arg
        , get_is_overridable<PrintingTraits>::value >;
    using maker_getter_type = typename other::maker_getter_type;
};

template < typename PrintingTraits, typename CharT, typename PrePrinting
         , typename FPack, typename Arg >
using select_maker_getter = typename maker_getter_selector
    <PrintingTraits, CharT, PrePrinting, FPack, Arg>
    :: maker_getter_type;

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
struct selector
{
    using traits = strf::printable_traits_of<Arg>;
    using maker_getter_type = select_maker_getter<traits, CharT, PrePrinting, FPack, Arg>;
    using maker_type = typename maker_getter_type::maker_type;
    using adapter_type = select_adapter<traits, maker_type, CharT, PrePrinting, FPack, Arg>;
};

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
struct selector_no_override
{
    using traits = strf::printable_traits_of<Arg>;
    using maker_getter_type = maker_getter_printable_traits<traits>;
    using adapter_type = select_adapter<traits, traits, CharT, PrePrinting, FPack, Arg>;
};

template < typename CharT, typename PrePrinting, typename FPack, typename Arg
         , typename Selector = selector<CharT, PrePrinting, FPack, Arg> >
struct helper: Selector::maker_getter_type, Selector::adapter_type
{
};

template < typename CharT, typename PrePrinting, typename FPack, typename Arg
         , typename Selector = selector_no_override<CharT, PrePrinting, FPack, Arg> >
struct helper_no_override: Selector::maker_getter_type, Selector::adapter_type
{
};

} // namespace mk_pr_in
} // namespace detail

template < typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , typename Helper
             = strf::detail::mk_pr_in::helper_no_override<CharT, PrePrinting, FPack, Arg>
         , typename Maker = typename Helper::maker_type
         , typename ChTag = strf::tag<CharT> >
STRF_DEPRECATED_MSG("make_default_arg_printer_input was renamed to make_default_printer_input")
constexpr STRF_HD auto make_default_arg_printer_input(PrePrinting& p, const FPack& fp, const Arg& arg)
    noexcept(noexcept(Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg))))
    -> decltype(Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg)))
{
    return Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg));
}

template < typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , typename Helper = strf::detail::mk_pr_in::helper<CharT, PrePrinting, FPack, Arg>
         , typename Maker = typename Helper::maker_type
         , typename ChTag = strf::tag<CharT> >
STRF_DEPRECATED_MSG("make_arg_printer_input was renamed to make_printer_input")
constexpr STRF_HD auto make_arg_printer_input(PrePrinting& p, const FPack& fp, const Arg& arg)
    -> decltype(((const Maker*)0)->make_input(ChTag{}, p, fp, Helper::adapt_arg(arg)))
{
    return Helper::get_maker(fp)
        .make_input(strf::tag<CharT>{}, p, fp, Helper::adapt_arg(arg));
}

template < typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , typename Helper
             = strf::detail::mk_pr_in::helper_no_override<CharT, PrePrinting, FPack, Arg>
         , typename Maker = typename Helper::maker_type
         , typename ChTag = strf::tag<CharT> >
constexpr STRF_HD auto make_default_printer_input(PrePrinting& p, const FPack& fp, const Arg& arg)
    noexcept(noexcept(Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg))))
    -> decltype(Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg)))
{
    return Maker::make_input(ChTag{}, p, fp, Helper::adapt_arg(arg));
}

template < typename CharT
         , typename PrePrinting
         , typename FPack
         , typename Arg
         , typename Helper = strf::detail::mk_pr_in::helper<CharT, PrePrinting, FPack, Arg>
         , typename Maker = typename Helper::maker_type
         , typename ChTag = strf::tag<CharT> >
constexpr STRF_HD auto make_printer_input(PrePrinting& p, const FPack& fp, const Arg& arg)
    -> decltype(((const Maker*){})->make_input(ChTag{}, p, fp, Helper::adapt_arg(arg)))
{
    return Helper::get_maker(fp)
        .make_input(strf::tag<CharT>{}, p, fp, Helper::adapt_arg(arg));
}

struct dont_override
{
    using category = printable_overrider_c;
    template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
    constexpr static STRF_HD auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& facets
        , Arg&& arg )
        noexcept(noexcept(strf::make_default_printer_input<CharT>(pre, facets, arg)))
        -> decltype(strf::make_default_printer_input<CharT>(pre, facets, arg))
    {
        return strf::make_default_printer_input<CharT>(pre, facets, arg);
    }
};

struct printable_overrider_c
{
    static constexpr bool constrainable = true;

    constexpr static STRF_HD dont_override get_default() noexcept
    {
        return {};
    }
};

namespace detail {

template <typename T>
struct is_printable_and_overridable_helper {

    template <typename U>
    static STRF_HD typename printable_traits_of<U>::is_overridable test_(const U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using result = decltype(test_<T>((T*){}));
};

} // namespace detail

template <typename T>
using is_printable_and_overridable = typename
    strf::detail::is_printable_and_overridable_helper<T>::result;

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename T>
constexpr bool is_printable_and_overridable_v = is_printable_and_overridable<T>::value;

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)

template <typename T>
using representative_of_printable = typename
    strf::printable_traits_of<T>::representative_type;

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
using printer_input_type = decltype
    ( strf::make_printer_input<CharT>
        ( std::declval<PrePrinting&>()
        , std::declval<const FPack&>()
        , std::declval<Arg>() ) );

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
using printer_type = typename printer_input_type<CharT, PrePrinting, FPack, Arg>::printer_type;

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
using arg_printer_type
STRF_DEPRECATED_MSG("arg_printer_type was renamed to printer_type")
= printer_type<CharT, PrePrinting, FPack, Arg>;

template <typename CharT, typename PrePrinting, typename FPack, typename Arg, typename Printer>
struct usual_printer_input;

template <typename CharT, typename PrePrinting, typename FPack, typename Arg, typename Printer>
using usual_arg_printer_input
STRF_DEPRECATED_MSG("usual_arg_printer_input was renamed to usual_printer_input")
= usual_printer_input<CharT, PrePrinting, FPack, Arg, Printer>;

template< typename CharT
        , strf::precalc_size PrecalcSize
        , strf::precalc_width PrecalcWidth
        , typename FPack
        , typename Arg
        , typename Printer >
struct usual_printer_input
    < CharT, strf::preprinting<PrecalcSize, PrecalcWidth>, FPack, Arg, Printer >
{
    using char_type = CharT;
    using arg_type = Arg;
    using preprinting_type = strf::preprinting<PrecalcSize, PrecalcWidth>;
    using fpack_type = FPack;
    using printer_type = Printer;

    preprinting_type& pre;
    FPack facets;
    Arg arg;
};

} // namespace strf

#endif  // STRF_DETAIL_PRINTABLE_TRAITS_HPP

