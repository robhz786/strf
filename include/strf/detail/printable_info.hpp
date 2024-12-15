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


template <typename Printable>
struct usual_printable;

template <typename PrintableDef>
struct printable_of;

template <typename T>
struct printable_def_not_found;

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

template <typename PrintableDef, class... Fmts>
struct can_remove_fmts
{
    using default_fmts_ = extract_format_specifiers_from_printable_def<PrintableDef>;
    static constexpr bool v1 = same_formatters<strf::tag<Fmts...>, default_fmts_>::value;
    static constexpr bool value = and_fmtfn_are_empty<v1, PrintableDef, Fmts...>::value;
};

template <typename ForwardedType>
struct printable_as_member_type_impl
{
    using type = ForwardedType;
};

template <typename ForwardedType>
struct printable_as_member_type_impl<const ForwardedType>
{
    using type = ForwardedType;
};

template <typename ForwardedType>
struct printable_as_member_type_impl<ForwardedType&>
{
    using type = strf::reference_wrapper<ForwardedType>;
};

template <typename ForwardedType>
struct printable_as_member_type_impl<const ForwardedType&>
{
    using type = strf::reference_wrapper<const ForwardedType>;
};

template <typename ForwardedType>
using printable_as_member_type = typename printable_as_member_type_impl<ForwardedType>::type;

template <typename PrintableDef>
struct printable_forwarded_type_extractor
{
    using type = typename PrintableDef::forwarded_type;
};

template <typename T>
struct printable_forwarded_type_extractor<strf::printable_def<T>>
{
    template <typename PD, typename ForwardedType = typename PD::forwarded_type>
    static STRF_HD ForwardedType test_forwarded_type(const PD*);

    template <typename PD>
    static STRF_HD const T& test_forwarded_type(...);

    using pd_type = strf::printable_def<T>;
    using type = decltype(test_forwarded_type<pd_type>(nullptr));
};

template <typename PrintableDef>
using extract_printable_forwarded_type =
    typename printable_forwarded_type_extractor<PrintableDef>::type;


template <typename PrintableDef>
struct rm_fmt_info_finder
{
    using type = printable_of<PrintableDef>;
};

template <typename P>
struct rm_fmt_info_finder<strf::printable_def<P>>
{
    using type = usual_printable<P>;
};

template <bool CanRemoveFmt, typename PrintableDef, typename... Fmts>
struct printable_fmt_2;

template <typename PrintableDef, typename... Fmts>
struct printable_fmt_2<true, PrintableDef, Fmts...>
{
    using printable_def = PrintableDef;
    using forwarded_type = strf::value_and_format<PrintableDef, Fmts...>;
    using as_member_type = forwarded_type;

    using rm_fmt_info = typename rm_fmt_info_finder<printable_def>::type;
    using remove_fmt_if_possible = typename rm_fmt_info::forwarded_type;
};

template <typename PrintableDef, typename... Fmts>
struct printable_fmt_2<false, PrintableDef, Fmts...>
{
    using printable_def = PrintableDef;
    using forwarded_type = strf::value_and_format<PrintableDef, Fmts...>;
    using as_member_type = forwarded_type;
};

template <typename PrintableDef, typename... Fmts>
struct printable_fmt:
    printable_fmt_2<can_remove_fmts<PrintableDef, Fmts...>::value, PrintableDef, Fmts...>
{
};

template <typename PrintableDef>
struct printable_of
{
    using printable_def = PrintableDef;
    using forwarded_type = typename printable_def::forwarded_type;
    using as_member_type = printable_as_member_type<forwarded_type>;
};

template <typename Printable>
struct usual_printable
{
    using printable_def = strf::printable_def<Printable>;
    using forwarded_type = extract_printable_forwarded_type<printable_def>;
    using as_member_type = printable_as_member_type<forwarded_type>;
};

template <typename T>
struct printable_def_not_found
{
    static_assert(std::is_void<T>::value, "Argument is not printable");
};

template <bool PrintableDefSpecializationExists, typename Printable>
struct printable_info_finder_2;

template <typename Printable>
struct printable_info_finder_2<false, Printable>
{
    template < typename U
             , typename PrintableDef =
                   decltype(get_printable_def(strf::printable_tag{}, std::declval<U>())) >
    static STRF_HD printable_of<PrintableDef> test_(const U*);

    template <typename U>
    static STRF_HD printable_def_not_found<U> test_(...);

    using type = decltype(test_<Printable>(nullptr));
};

template <typename Printable>
struct printable_info_finder_2<true, Printable>
{
    using type = usual_printable<Printable>;
};

template <typename Printable>
struct printable_info_finder
{
    template <typename U, std::size_t = sizeof(strf::printable_def<U>)>
    static STRF_HD std::true_type test_(const U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using printable_def_specialization_exists = decltype(test_<Printable>(nullptr));

    using type = typename printable_info_finder_2
        < printable_def_specialization_exists::value, Printable >
        ::type;
};

template <typename PrintableDef, typename... Fmts>
struct printable_info_finder<strf::value_and_format<PrintableDef, Fmts...> >
{
    using type = printable_fmt<PrintableDef, Fmts...>;
};

template <typename Printable>
struct printable_info_finder<Printable&> : printable_info_finder<Printable>
{
};

template <typename Printable>
struct printable_info_finder<Printable&&> : printable_info_finder<Printable>
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

template <typename Printable>
using printable_def_of = typename detail::get_printable_info<Printable>::printable_def;

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_INFO_HPP
