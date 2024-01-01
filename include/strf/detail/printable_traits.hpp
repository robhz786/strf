#ifndef STRF_DETAIL_PRINTABLE_TRAITS_HPP
#define STRF_DETAIL_PRINTABLE_TRAITS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_with_fmt.hpp>
#include <strf/detail/premeasurements.hpp>
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

    using result = decltype(test_<T>((T*)nullptr));
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
    using result = decltype(test<T_>((const T_*)nullptr));

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

    using result = decltype(test_<PrintableTraits>((PrintableTraits*)nullptr));
};

template <typename PrintableTraits>
using get_is_overridable = typename
    get_is_overridable_helper<PrintableTraits>::result;

namespace mk_pr_in {

template < typename PTraits
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg >
struct can_call_make_printer_L0
{
    template < typename P
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , std::declval<const Arg&>() ) ) >
    static STRF_HD std::true_type test_
        ( PreMeasurements* pre, const FPack& facets, const Arg& arg );

    template <typename P>
    static STRF_HD std::false_type test_(...);

    using result = decltype
        ( test_<PTraits>
            ( std::declval<PreMeasurements*>()
            , std::declval<FPack>()
            , std::declval<Arg>() ));
};

template < typename PTraits
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg >
using can_call_make_printer = typename
    can_call_make_printer_L0<PTraits, CharT, PreMeasurements, FPack, Arg>
    ::result;

struct printable_arg_fmt_remover
{
    template <typename PrintingTraits, typename... Fmts>
    static constexpr STRF_HD const typename PrintingTraits::forwarded_type&
    convert_printable_arg(const strf::printable_with_fmt<PrintingTraits, Fmts...>& x)
    {
        return x.value();
    }
};

template <typename To>
struct printable_arg_caster
{
    template <typename From>
    static constexpr STRF_HD To convert_printable_arg(const From& x)
    {
        return static_cast<To>(x);
    }
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Vwf
         , typename DefaultVwf >
struct printable_arg_converter_selector_3
{
    static_assert( ! std::is_same<Vwf, DefaultVwf>::value, "");
    using type = printable_arg_caster<const Vwf&>;
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename DefaultVwf >
struct printable_arg_converter_selector_3
    < PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack, DefaultVwf, DefaultVwf >
{
    static constexpr bool can_pass_directly =
        can_call_make_printer<TraitsOrFacet, CharT, PreMeasurements, FPack, DefaultVwf>
        ::value;

    using type = typename std::conditional
        < can_pass_directly
        , printable_arg_caster<const DefaultVwf&>
        , printable_arg_fmt_remover >
        ::type;
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename DefaultVwf >
struct printable_arg_converter_selector_2
{
    static constexpr bool can_pass_directly =
        can_call_make_printer<TraitsOrFacet, CharT, PreMeasurements, FPack, Arg>
        ::value;
    static constexpr bool can_pass_as_fmt =
        can_call_make_printer<TraitsOrFacet, CharT, PreMeasurements, FPack, DefaultVwf>
        ::value;
    static constexpr bool shall_adapt = !can_pass_directly && can_pass_as_fmt;

    using destination_type = typename std::conditional
        < shall_adapt, DefaultVwf, typename PrintingTraits::forwarded_type>
        :: type;
    using type = printable_arg_caster<destination_type>;
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename DefaultVwf
         , typename... Fmts >
struct printable_arg_converter_selector_2
    < PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack
    , strf::printable_with_fmt<PrintingTraits, Fmts...>, DefaultVwf >
{
    using vwf = strf::printable_with_fmt<PrintingTraits, Fmts...>;
    using other = printable_arg_converter_selector_3
        < PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack, vwf, DefaultVwf >;
    using type = typename other::type;
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg >
struct printable_arg_converter_selector
{
    using vwf = default_value_with_formatter_of_printable_traits<PrintingTraits>;
    using other = printable_arg_converter_selector_2
        < PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack, Arg, vwf >;
    using type = typename other::type;
};

template < typename PrintingTraits, typename TraitsOrFacet, typename CharT
         , typename PreMeasurements, typename FPack, typename Arg >
using select_printable_arg_converter = typename
    printable_arg_converter_selector<PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack, Arg>
    ::type;

template <typename Overrider, typename OverrideTag>
struct overrider_getter
{
    using traits_of_facet_type = Overrider;

    template <typename FPack>
    static constexpr STRF_HD const Overrider& get_traits_or_facet(const FPack& fp)
    {
        return strf::use_facet<strf::printable_overrider_c, OverrideTag>(fp);
    }
};

template <typename PrintingTraits>
struct printable_traits_getter
{
    using traits_of_facet_type = PrintingTraits;
    using traits_type = PrintingTraits;

    template <typename FPack>
    static constexpr STRF_HD traits_of_facet_type get_traits_or_facet(const FPack&)
    {
        return traits_of_facet_type{};
    }
};

template < typename PrintingTraits
         , typename CharT
         , typename FPack
         , typename Arg
         , bool Overridable >
struct traits_or_facet_getter_selector_2
{
    static_assert(Overridable, "");
    using representative_type = typename PrintingTraits::representative_type;
    using overrider_ = decltype
        ( strf::use_facet<strf::printable_overrider_c, representative_type>
          (std::declval<FPack>()) );

    using overrider = strf::detail::remove_cvref_t<overrider_>;
    using traits_or_facet_getter_type = typename std::conditional
        < std::is_same<overrider, strf::dont_override>::value
        , printable_traits_getter<PrintingTraits>
        , overrider_getter<overrider, representative_type> >
        ::type;
};

template < typename PrintingTraits
         , typename CharT
         , typename FPack
         , typename Arg >
struct traits_or_facet_getter_selector_2<PrintingTraits, CharT, FPack, Arg, false>
{
    using traits_or_facet_getter_type = printable_traits_getter<PrintingTraits>;
};

template < typename PrintingTraits
         , typename CharT
         , typename FPack
         , typename Arg >
struct traits_or_facet_getter_selector
{
    using other = traits_or_facet_getter_selector_2
        < PrintingTraits, CharT, FPack, Arg
        , get_is_overridable<PrintingTraits>::value >;
    using traits_or_facet_getter_type = typename other::traits_or_facet_getter_type;
};

template < typename PrintingTraits, typename CharT, typename FPack, typename Arg >
using select_traits_or_facet_getter = typename traits_or_facet_getter_selector
    <PrintingTraits, CharT, FPack, Arg>
    :: traits_or_facet_getter_type;

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
struct selector_override_allowed
{
    using traits = strf::printable_traits_of<Arg>;
    using traits_or_facet_getter =
        select_traits_or_facet_getter< traits, CharT, FPack, Arg >;
    using traits_of_facet_type = typename traits_or_facet_getter::traits_of_facet_type;
    using printable_arg_converter =
        select_printable_arg_converter
        <traits, traits_of_facet_type, CharT, PreMeasurements, FPack, Arg>;
};

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
struct selector_override_forbidden
{
    using traits = strf::printable_traits_of<Arg>;
    using traits_or_facet_getter = printable_traits_getter<traits>;
    using printable_arg_converter =
        select_printable_arg_converter<traits, traits, CharT, PreMeasurements, FPack, Arg>;
};

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename TraitsOrFacetGetter, typename PrintableArgConverter >
class printer_type_finder
{
    template
        < typename Printer
        , typename Dst = strf::destination<CharT>
        , typename = decltype(std::declval<Printer>().print_to(std::declval<Dst&>())) >
    static STRF_HD strf::tag<Printer> test_(const Printer&);

    template
        < typename PrinterInputT
        , typename Printer = typename PrinterInputT::printer_type >
    static STRF_HD strf::tag<Printer> test_(const PrinterInputT&);

    using tag_type_ = decltype
        (test_(TraitsOrFacetGetter::get_traits_or_facet(std::declval<const FPack&>()).make_printer
                  ( strf::tag<CharT>{}
                  , std::declval<PreMeasurements*>()
                  , std::declval<const FPack&>()
                  , PrintableArgConverter::convert_printable_arg(std::declval<const Arg&>()))));
public:
    using printer_type = typename tag_type_::type;
};

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename Selector >
struct info_base
    : Selector::traits_or_facet_getter
    , Selector::printable_arg_converter
    , printer_type_finder< CharT, PreMeasurements, FPack, Arg
                         , typename Selector::traits_or_facet_getter
                         , typename Selector::printable_arg_converter>
{
};

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg >
struct info_override_allowed:
    info_base< CharT, PreMeasurements, FPack, Arg
              , selector_override_allowed<CharT, PreMeasurements, FPack, Arg> >
{
};

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg>
struct info_override_forbidden:
    info_base< CharT, PreMeasurements, FPack, Arg
             , selector_override_forbidden<CharT, PreMeasurements, FPack, Arg> >
{
};

} // namespace mk_pr_in

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
using printing_info_override_allowed = mk_pr_in::info_override_allowed
    <CharT, PreMeasurements, FPack, detail::remove_cvref_t<Arg> >;

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
using printing_info_override_forbidden = mk_pr_in::info_override_forbidden
    <CharT, PreMeasurements, FPack, detail::remove_cvref_t<Arg> >;

} // namespace detail

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper =
             detail::printing_info_override_forbidden<CharT, PreMeasurements, FPack, Arg>
         , typename PTraits = typename Helper::traits_type
         , typename ChTag = strf::tag<CharT> >
STRF_DEPRECATED_MSG("make_default_arg_printer_input was renamed to make_default_printer")
constexpr STRF_HD decltype(auto) make_default_arg_printer_input
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
    noexcept(noexcept(PTraits::make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg))))
{
    return PTraits::make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg));
}

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = detail::printing_info_override_allowed<CharT, PreMeasurements, FPack, Arg>
         , typename TraitsOrFacet = typename Helper::traits_of_facet_type
         , typename ChTag = strf::tag<CharT> >
STRF_DEPRECATED_MSG("make_arg_printer_input was renamed to make_printer")
constexpr STRF_HD decltype(auto) make_arg_printer_input
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
{
    return Helper::get_traits_or_facet(fp)
        .make_printer(strf::tag<CharT>{}, p, fp, Helper::convert_printable_arg(arg));
}

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = strf::detail::printing_info_override_forbidden<CharT, PreMeasurements, FPack, Arg>
         , typename PTraits = typename Helper::traits_type
         , typename ChTag = strf::tag<CharT> >
constexpr STRF_HD decltype(auto) make_default_printer
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
    noexcept(noexcept(PTraits::make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg))))
{
    return PTraits::make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg));
}

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = detail::printing_info_override_allowed<CharT, PreMeasurements, FPack, Arg>
         , typename ChTag = strf::tag<CharT> >
constexpr STRF_HD decltype(auto) make_printer
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
{
    return Helper::get_traits_or_facet(fp)
        .make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg));
}

struct dont_override
{
    using category = printable_overrider_c;
    template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
    constexpr static STRF_HD decltype(auto) make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , Arg&& arg )
        noexcept(noexcept(strf::make_default_printer<CharT>(pre, facets, arg)))
    {
        return strf::make_default_printer<CharT>(pre, facets, arg);
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

    using result = decltype(test_<T>((T*)nullptr));
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

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
using printer_type = typename
    detail::printing_info_override_allowed
    < CharT, PreMeasurements, FPack, Arg >
    ::printer_type;

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
using arg_printer_type
STRF_DEPRECATED_MSG("arg_printer_type was renamed to printer_type")
= printer_type<CharT, PreMeasurements, FPack, Arg>;

template < typename CharT, typename PreMeasurements, typename FPack
         , typename Arg, typename Printer >
struct usual_printer_input;

template < typename CharT, typename PreMeasurements, typename FPack
         , typename Arg, typename Printer >
using usual_arg_printer_input
STRF_DEPRECATED_MSG("usual_arg_printer_input was renamed to usual_printer_input")
= usual_printer_input<CharT, PreMeasurements, FPack, Arg, Printer>;

template< typename CharT
        , strf::size_presence SizePresence
        , strf::width_presence WidthPresence
        , typename FPack
        , typename Arg
        , typename Printer >
struct usual_printer_input
    < CharT, strf::premeasurements<SizePresence, WidthPresence>, FPack, Arg, Printer >
{
    using char_type = CharT;
    using arg_type = Arg;
    using premeasurements_type = strf::premeasurements<SizePresence, WidthPresence>;
    using fpack_type = FPack;
    using printer_type = Printer;

    premeasurements_type* pre;
    FPack facets;
    Arg arg;
};

} // namespace strf

#endif  // STRF_DETAIL_PRINTABLE_TRAITS_HPP

