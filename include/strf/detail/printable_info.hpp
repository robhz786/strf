#ifndef STRF_DETAIL_PRINTABLE_INFO_HPP
#define STRF_DETAIL_PRINTABLE_INFO_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {

template <typename PrintableDef, class... Fmts>
class value_and_format;

template<typename T>
struct printable_def;

template<typename PrintableDef, typename... Fmts>
struct printable_def<strf::value_and_format<PrintableDef, Fmts...>> : PrintableDef
{
};

struct printable_tag
{
};

namespace detail {

template <typename T>
struct printable_def_not_found
{
    static_assert(std::is_void<T>::value, "Type is not printable");

    using representative = T;
    using forwarded_type = strf::reference_wrapper<const T>;
};

template <typename T>
struct has_printable_def_specialization
{
    template <typename U, typename = typename strf::printable_def<U>::forwarded_type>
    static STRF_HD std::true_type test(const U*);

    template <typename U>
    static STRF_HD std::false_type test(...);

    using T_ = strf::detail::remove_cvref_t<T>;
    using result = decltype(test<T_>((const T_*)nullptr));

    constexpr static bool value = result::value;
};

template <bool HasPrintableDefSpecialization, typename Printable>
struct printable_def_finder_2;

template <typename Printable>
struct printable_def_finder_2<false, Printable>
{
    template < typename U
             , typename PrintableDef =
                   decltype(get_printable_def(strf::printable_tag{}, std::declval<U>())) >
    static STRF_HD PrintableDef test_(const U*);

    template <typename U>
    static STRF_HD printable_def_not_found<U> test_(...);

    using type = decltype(test_<Printable>((Printable*)0));
};

template <typename Printable>
struct printable_def_finder_2<true, Printable>
{
    using type = strf::printable_def<Printable>;
};

template <typename Printable>
struct printable_def_finder
{
    constexpr static bool has_specialization =
        strf::detail::has_printable_def_specialization<Printable>::value;

    using type = typename printable_def_finder_2<has_specialization, Printable>::type;
};

template <typename PrintableDef, typename... Fmts>
struct printable_def_finder<strf::value_and_format<PrintableDef, Fmts...>>
{
    using type = PrintableDef;
};

template <typename T>
struct printable_def_finder<T&> : printable_def_finder<T>
{
};

template <typename T>
struct printable_def_finder<T&&> : printable_def_finder<T>
{
};

template <typename T>
struct printable_def_finder<const T> : printable_def_finder<T>
{
};

template <typename T>
struct printable_def_finder<volatile T> : printable_def_finder<T>
{
};

} // namespace detail

template <typename Printable>
using printable_def_of = typename detail::printable_def_finder<Printable>::type;

namespace detail {

template <typename PrintableDef>
struct extract_format_specifiers_from_printable_def_impl
{
private:
    template <typename U, typename Fmts = typename U::format_specifiers>
    static Fmts get_format_specifiers_(U*);

    template <typename U>
    static strf::tag<> get_format_specifiers_(...);

public:

    using type = decltype(get_format_specifiers_<PrintableDef>(nullptr));
};

template <typename PrintableDef>
using extract_format_specifiers_from_printable_def =
    typename extract_format_specifiers_from_printable_def_impl<PrintableDef>::type;

template <typename... T>
struct are_empty;

template <>
struct are_empty<> : std::true_type {};

template <typename First, typename... Others>
struct are_empty<First, Others...>
    : std::integral_constant
        < bool
        , std::is_empty<First>::value && are_empty<Others...>::value >
{
};

template <typename PrintableDef, typename... Fmts>
struct all_base_fmtfn_classes_are_empty_2
    : are_empty<typename Fmts::template fn<value_and_format<PrintableDef, Fmts...>> ...>
{
};

template <typename ValueAndFormat>
struct all_base_fmtfn_classes_are_empty;

template <typename PrintableDef, typename... Fmts>
struct all_base_fmtfn_classes_are_empty< value_and_format<PrintableDef, Fmts...> >
    : all_base_fmtfn_classes_are_empty_2<PrintableDef, Fmts...>
{
};

template <bool, typename PrintableDef, typename... Fmts>
struct and_fmtfn_are_empty;

template <typename PrintableDef, typename... Fmts>
struct and_fmtfn_are_empty<false, PrintableDef, Fmts...> : std::false_type
{
};

template <typename PrintableDef, typename... Fmts>
struct and_fmtfn_are_empty<true, PrintableDef, Fmts...>
    : all_base_fmtfn_classes_are_empty_2<PrintableDef, Fmts...>
{
};

template <typename FmtListA, typename FmtListB>
struct same_formatters;

template <
    template <class...> class MpListA,
    template <class...> class MpListB,
    typename... FormattersA,
    typename... FormattersB>
struct same_formatters<MpListA<FormattersA...>, MpListB<FormattersB...>>
    : std::is_same<strf::tag<FormattersA...>, strf::tag<FormattersB...>>
{
};

template <typename PrintableDef, class... Fmts>
struct can_remove_fmts
{
    using default_fmts_ = extract_format_specifiers_from_printable_def<PrintableDef>;
    static constexpr bool v1 = same_formatters<strf::tag<Fmts...>, default_fmts_>::value;
    static constexpr bool value = and_fmtfn_are_empty<v1, PrintableDef, Fmts...>::value;
};

template <typename T>
struct is_value_with_default_formatting: std::false_type
{
};

template <typename PrintableDef, class... Fmts>
struct is_value_with_default_formatting<strf::value_and_format<PrintableDef, Fmts...> >
{
    using default_fmts_ = extract_format_specifiers_from_printable_def<PrintableDef>;
    static constexpr bool v1 = same_formatters<strf::tag<Fmts...>, default_fmts_>::value;
    static constexpr bool value = and_fmtfn_are_empty<v1, PrintableDef>::value;
};

template <typename T>
constexpr bool is_value_with_default_formatting_v = is_value_with_default_formatting<T>::value;

template <typename T>
struct sanitize_printable_forwarded_type_impl
{
    using type = T;
};

template <typename T>
struct sanitize_printable_forwarded_type_impl<const T>
{
    using type = T;
};

template <typename T>
struct sanitize_printable_forwarded_type_impl<T&>
{
    using type = strf::reference_wrapper<T>;
};

template <typename T>
struct sanitize_printable_forwarded_type_impl<const T&>
{
    using type = strf::reference_wrapper<const T>;
};

template <typename T>
using sanitize_printable_forwarded_type = typename sanitize_printable_forwarded_type_impl<T>::type;

template <typename PrintableDef>
struct printable_forwarded_type_extractor
{
    using type = sanitize_printable_forwarded_type<typename PrintableDef::forwarded_type>;
};

template <typename T>
struct printable_forwarded_type_extractor<strf::printable_def<T>>
{
    template <typename PD, typename ForwardedType = typename PD::forwarded_type>
    static STRF_HD auto test_forwarded_type(const PD*)
        -> sanitize_printable_forwarded_type<ForwardedType>;

    template <typename PD>
    static STRF_HD auto test_forwarded_type(...) -> strf::reference_wrapper<const T>;

    using pd_type = strf::printable_def<T>;
    using type = decltype(test_forwarded_type<pd_type>(nullptr));
};

template <typename PrintableDef>
using extract_printable_forwarded_type =
    typename printable_forwarded_type_extractor<PrintableDef>::type;

template <bool CanRemoveFmt, typename PrintableDef, typename... Fmts>
struct printable_info_fmts_2;

template <typename PrintableDef, typename... Fmts>
struct printable_info_fmts_2<true, PrintableDef, Fmts...>
{
    static constexpr bool can_remove_fmt = true;
    using forwarded_type = strf::value_and_format<PrintableDef, Fmts...>;
    using printable_def = PrintableDef;
    using result_type = extract_printable_forwarded_type<PrintableDef>;
};

template <typename PrintableDef, typename... Fmts>
struct printable_info_fmts_2<false, PrintableDef, Fmts...>
{
    static constexpr bool can_remove_fmt = false;
    using forwarded_type = strf::value_and_format<PrintableDef, Fmts...>;
    using printable_def = PrintableDef;
    using result_type = forwarded_type;
};

template <typename PrintableDef, typename... Fmts>
struct printable_info_fmt:
    printable_info_fmts_2<can_remove_fmts<PrintableDef, Fmts...>::value, PrintableDef, Fmts...>
{
};

template <typename PrintableDef>
struct printable_info
{
    static constexpr bool can_remove_fmt = false;
    using printable_def = PrintableDef;
    using forwarded_type = extract_printable_forwarded_type<PrintableDef>;
    using result_type = forwarded_type;
};

template <typename Printable>
struct printable_info_finder
{
    using type = printable_info<strf::printable_def_of<Printable> >;
};

template <typename PrintableDef, typename... Fmts>
struct printable_info_finder<strf::value_and_format<PrintableDef, Fmts...> >
{
    using type = printable_info_fmt<PrintableDef, Fmts...>;
};

template <typename Printable>
struct printable_info_finder<Printable&> : printable_info_finder<Printable>
{
};

template <typename Printable>
struct printable_info_finder<const Printable> : printable_info_finder<Printable>
{
};

template <typename Printable>
struct printable_info_finder<const Printable&> : printable_info_finder<Printable>
{
};

template <typename Printable>
struct printable_info_finder<volatile Printable> : printable_info_finder<Printable>
{
};

template <typename Printable>
using get_printable_info = typename printable_info_finder<Printable>::type;

} // namespace detail
} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_INFO_HPP
