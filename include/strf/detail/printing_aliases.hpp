#ifndef STRF_DETAIL_PRINTING_ALIASES_HPP
#define STRF_DETAIL_PRINTING_ALIASES_HPP

#include <strf/detail/printable_info.hpp>

namespace strf {

using print_traits_tag
STRF_DEPRECATED_MSG("print_traits_tag type renamed to printable_tag")
=  printable_tag;

template <typename T>
using print_traits
STRF_DEPRECATED_MSG("print_traits renamed to printable_def")
= printable_def<T>;

template <typename T>
using print_traits_of
STRF_DEPRECATED_MSG("print_traits_of renamed to printable_def_of")
= printable_def_of<T>;

namespace detail {

template <typename CharT, typename PreMeasurements, typename FPack, typename PrintableInfo>
struct helper_for_printing_with_premeasurements;

} //

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = detail::helper_for_printing_with_premeasurements
                 < CharT, PreMeasurements, FPack, detail::get_printable_info<Arg> >
         , typename ChTag = strf::tag<CharT> >
STRF_DEPRECATED_MSG("make_arg_printer_input was renamed to make_printer")
constexpr STRF_HD decltype(auto) make_arg_printer_input
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
{
    return Helper::get_printable_def_or_facet(fp)
        .make_printer(strf::tag<CharT>{}, p, fp, Helper::convert_printable_arg(arg));
}

template < typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg
         , typename Helper
             = detail::helper_for_printing_with_premeasurements
           < CharT, PreMeasurements, FPack, detail::get_printable_info<Arg> >
         , typename ChTag = strf::tag<CharT> >
constexpr STRF_HD decltype(auto) make_printer
    ( PreMeasurements* p, const FPack& fp, const Arg& arg )
{
    return Helper::get_printable_def_or_facet(fp)
        .make_printer(ChTag{}, p, fp, Helper::convert_printable_arg(arg));
}

template <typename Representative>
struct printable_overrider_c;

template <typename Representative>
struct dont_override
{
    using category = printable_overrider_c<Representative>;
};

template <typename Representative>
struct printable_overrider_c
{
    static constexpr bool constrainable = true;

    constexpr static STRF_HD dont_override<Representative> get_default() noexcept
    {
        return {};
    }
};

namespace detail {

template <typename T>
struct is_printable_and_overridable_helper {

    template <typename U>
    static STRF_HD typename printable_def_of<U>::is_overridable test_(const U*);

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

namespace detail {

template <typename PrintableDef>
struct representative_from_printable_def
{
    using type = PrintableDef;
};

template <typename P>
struct representative_from_printable_def<printable_def<P>>
{
    template <typename U>
    static STRF_HD strf::tag<typename U::representative> test_(const U&);

    template <typename U>
    static STRF_HD strf::tag<P> test_(...);

    using pdef = printable_def<P>;
    using tag_type = decltype(test_<pdef>(std::declval<pdef>()));
    using type = typename tag_type::type;
};

template <typename PrintableDef>
using extract_representative = typename representative_from_printable_def<PrintableDef>::type;

template <typename CharT, typename PreMeasurements, typename FPack, typename PrintableInfo>
using printer_type_pi = typename
    detail::helper_for_printing_with_premeasurements<CharT, PreMeasurements, FPack, PrintableInfo>
    ::printer_type;

} // namespace detail

template <typename T>
using representative_of_printable = detail::extract_representative<strf::printable_def_of<T>>;

template <typename Printable>
using printable_overrider_c_of =
    printable_overrider_c< representative_of_printable<Printable> >;

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
using printer_type =
    detail::printer_type_pi<CharT, PreMeasurements, FPack, detail::get_printable_info<Arg>>;

namespace detail {

template <typename PrintableDef, typename Formatters>
struct mp_define_value_and_format;

template < typename PrintableDef
         , template <class...> class List
         , typename... Fmts >
struct mp_define_value_and_format<PrintableDef, List<Fmts...>>
{
    using type = strf::value_and_format<PrintableDef, Fmts...>;
};

template <typename T>
struct format_specifiers_finder
{
    using printable_def = typename get_printable_info<T>::printable_def;
    using format_specifiers = extract_format_specifiers_from_printable_def<printable_def>;
    using fmt_type = typename
        strf::detail::mp_define_value_and_format<printable_def, format_specifiers>::type;
};

template <typename PrintableDef, typename... Fmts>
struct format_specifiers_finder<strf::value_and_format<PrintableDef, Fmts...>>
{
    using printable_def = PrintableDef;
    using format_specifiers = strf::tag<Fmts...>;
    using fmt_type = strf::value_and_format<PrintableDef, Fmts...>;
};

} // namespace detail

template <typename T>
using fmt_type = typename
    detail::format_specifiers_finder<strf::detail::remove_cvref_t<T>>
    ::fmt_type;

template <typename T>
using fmt_value_type = typename fmt_type<T>::value_type;

template <typename T>
using format_specifiers_of = typename strf::detail::format_specifiers_finder<T>::format_specifiers;

template <typename T>
using forwarded_printable_type = typename detail::get_printable_info<T>::forwarded_type;

} // namespace strf


#endif  // STRF_DETAIL_PRINTING_ALIASES_HPP
