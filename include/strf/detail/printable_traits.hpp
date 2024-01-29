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

namespace detail {

// forward declarations of things defined in <strf/detail/printing_helpers.hpp>

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
struct helper_for_printing_with_premeasurements;

template <typename CharT, typename FPack, typename Arg>
struct helper_for_printing_without_premeasurements;

template <typename CharT, typename FPack, typename Arg>
struct helper_for_tr_printing_without_premeasurements;

template <typename CharT, typename FPack, typename Printable, typename... Printables>
inline STRF_HD void print_one_printable
    ( strf::destination<CharT>& dst
    , const FPack& fp
    , const Printable& printable )
{
    using helper = helper_for_printing_without_premeasurements
        <CharT, FPack, Printable>;
    helper::print(helper::get_traits_or_facet(fp), dst, fp, printable);
}

template <typename CharT>
inline STRF_HD void call_printers(strf::destination<CharT>&)
{
}

template <typename CharT, typename Printer, typename... Printers>
inline STRF_HD void call_printers
    ( strf::destination<CharT>& dst
    , const Printer& printer0
    , const Printers&... printers )
{
    printer0(dst);
    if (dst.good()) {
        call_printers<CharT>(dst, printers...);
    }
}

template <typename CharT, typename FPack>
inline STRF_HD void print_printables(strf::destination<CharT>&, const FPack&)
{
}

template <typename CharT, typename FPack, typename Printable, typename... Printables>
inline STRF_HD void print_printables
    ( strf::destination<CharT>& dst
    , const FPack& fp
    , const Printable& printable
    , const Printables&... printables )
{
    print_one_printable(dst, fp, printable);
    if (dst.good()) {
        print_printables<CharT>(dst, fp, printables...);
    }
}

} // namespace detail

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
struct extract_format_specifiers_from_printable_traits_impl
{
private:
    template <typename U, typename Fmts = typename U::format_specifiers>
    static Fmts get_format_specifiers_(U*);

    template <typename U>
    static strf::tag<> get_format_specifiers_(...);

public:

    using type = decltype(get_format_specifiers_<PrintingTraits>(nullptr));
};

template <typename PrintingTraits>
using extract_format_specifiers_from_printable_traits =
    typename extract_format_specifiers_from_printable_traits_impl<PrintingTraits>::type;

template <typename PrintingTraits>
using default_printable_with_fmt_of_printable_traits = typename
    strf::detail::mp_define_printable_with_fmt
        < PrintingTraits
        , extract_format_specifiers_from_printable_traits<PrintingTraits> >
    :: type;

template <typename T>
struct format_specifiers_finder
{
    using traits = typename printable_traits_finder<T>::traits;
    using format_specifiers = extract_format_specifiers_from_printable_traits<traits>;
    using fmt_type = typename
        strf::detail::mp_define_printable_with_fmt<traits, format_specifiers>::type;
};

template <typename PrintingTraits, typename... Fmts>
struct format_specifiers_finder<strf::printable_with_fmt<PrintingTraits, Fmts...>>
{
    using traits = PrintingTraits;
    using format_specifiers = strf::tag<Fmts...>;
    using fmt_type = strf::printable_with_fmt<PrintingTraits, Fmts...>;
};

} // namespace detail

template <typename T>
using forwarded_printable_type = typename
    detail::printable_traits_finder<strf::detail::remove_cvref_t<T>>
    ::forwarded_type;

template <typename T>
using fmt_type = typename
    detail::format_specifiers_finder<strf::detail::remove_cvref_t<T>>
    ::fmt_type;

template <typename T>
using fmt_value_type = typename fmt_type<T>::value_type;

template <typename T>
using format_specifiers_of = typename strf::detail::format_specifiers_finder<T>::format_specifiers;

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
             , strf::detail::enable_if_t<!IsVWF, int> = 0
             , typename FmtType = fmt_type<T>
             , typename FmtValueType = typename FmtType::value_type >
    constexpr STRF_HD fmt_type<T> operator()(T&& value) const
        noexcept(noexcept(FmtType{FmtValueType{(T&&)value}}))
    {
        return FmtType{FmtValueType{(T&&)value}};
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

} // namespace detail

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = detail::helper_for_printing_with_premeasurements
                 < CharT, PreMeasurements, FPack, Arg >
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
             = detail::helper_for_printing_with_premeasurements
                 < CharT, PreMeasurements, FPack, Arg >
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
    detail::helper_for_printing_with_premeasurements
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

