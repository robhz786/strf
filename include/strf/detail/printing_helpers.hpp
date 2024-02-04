#ifndef STRF_DETAIL_PRINTING_HELPERS_HPP
#define STRF_DETAIL_PRINTING_HELPERS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>

namespace strf {
namespace detail {
namespace printing_helpers {

template <typename DefaultPrintableWithFmt>
struct printable_arg_fmt_remover
{
    static constexpr STRF_HD decltype(auto) convert_printable_arg
        ( const DefaultPrintableWithFmt& x )
    {
        return x.value();
    }
};

template <typename DefaultPrintableWithFmt>
struct printable_arg_fmt_adder
{
    template <typename Printable>
    static constexpr STRF_HD DefaultPrintableWithFmt convert_printable_arg(const Printable& x)
    {
        return DefaultPrintableWithFmt{typename DefaultPrintableWithFmt::value_type{x}};
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

struct invalid_arg{};

struct printable_arg_invalid
{
    template <typename From>
    static constexpr STRF_HD auto convert_printable_arg(const From&)
    {
        return invalid_arg{};
    }
};


template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack
         , typename Arg >
class printable_arg_converter_selector_for_printing_with_premeasurements
{
    using default_printable_with_fmt =
        default_printable_with_fmt_of_printable_traits<PrintingTraits>;
    using fmt_value_type = typename default_printable_with_fmt::value_type;
    using fwd_type = typename PrintingTraits::forwarded_type;

    template < typename P, typename A
             , detail::enable_if_t
                 < std::is_same<A, default_printable_with_fmt>::value
                && detail::all_base_fmtfn_classes_are_empty<A>::value
                 , int > = 0
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>().value() ))  >
    static STRF_HD auto test_(strf::rank<4>*, const A& arg)
        -> printable_arg_fmt_remover<default_printable_with_fmt>;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>() ) ) >
    static STRF_HD auto test_(strf::rank<3>*, const A& arg)
        -> printable_arg_caster<const A&>;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , static_cast<fwd_type>(std::declval<const A&>()) ) ) >
    static STRF_HD auto test_(strf::rank<2>*, const A& arg)
        -> printable_arg_caster<fwd_type>;


    template < typename P, typename A
             , detail::enable_if_t< !detail::is_printable_with_fmt<A>::value, int > = 0
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , default_printable_with_fmt{std::declval<const A&>()} ))  >
    static STRF_HD auto test_(strf::rank<1>*, const A& arg)
        -> printable_arg_fmt_adder<default_printable_with_fmt>;

public:
    using type =
        decltype(test_<TraitsOrFacet, Arg>( std::declval<strf::rank<4>*>()
                                          , std::declval<const Arg&>() ));
};

template < typename PrintingTraits, typename TraitsOrFacet, typename CharT
         , typename PreMeasurements, typename FPack, typename Arg >
using select_printable_arg_converter_for_printing_with_premeasurements = typename
    printable_arg_converter_selector_for_printing_with_premeasurements
    <PrintingTraits, TraitsOrFacet, CharT, PreMeasurements, FPack, Arg>
    ::type;

template <typename Overrider, typename OverrideTag>
struct overrider_getter
{
    using traits_or_facet_type = Overrider;

    template <typename FPack>
    static constexpr STRF_HD const Overrider& get_traits_or_facet(const FPack& fp)
    {
        return strf::use_facet<strf::printable_overrider_c, OverrideTag>(fp);
    }
};

template <typename PrintingTraits>
struct printable_traits_getter
{
    using traits_or_facet_type = PrintingTraits;
    using traits_type = PrintingTraits;

    template <typename FPack>
    static constexpr STRF_HD traits_or_facet_type get_traits_or_facet(const FPack&)
    {
        return traits_or_facet_type{};
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
struct selector_for_printing_with_premeasurements
{
    using traits = strf::printable_traits_of<Arg>;
    using traits_or_facet_getter =
        select_traits_or_facet_getter< traits, CharT, FPack, Arg >;
    using traits_or_facet_type = typename traits_or_facet_getter::traits_or_facet_type;
    using printable_arg_converter =
        select_printable_arg_converter_for_printing_with_premeasurements
        <traits, traits_or_facet_type, CharT, PreMeasurements, FPack, Arg>;
};

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename TraitsOrFacet, typename PrintableArgConverter >
using find_printer_type =
    decltype( std::declval<TraitsOrFacet>().make_printer
                ( strf::tag<CharT>{}
                , std::declval<PreMeasurements*>()
                , std::declval<const FPack&>()
                , PrintableArgConverter::convert_printable_arg(std::declval<const Arg&>())));

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename Selector >
struct helper_for_printing_with_premeasurements_impl
    : Selector::traits_or_facet_getter
    , Selector::printable_arg_converter
{
    using traits_or_facet_type = typename Selector::traits_or_facet_getter::traits_or_facet_type;
    using printer_type = find_printer_type
        < CharT, PreMeasurements, FPack, Arg
        , traits_or_facet_type, typename Selector::printable_arg_converter>;
};

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg >
struct helper_for_printing_with_premeasurements
    : helper_for_printing_with_premeasurements_impl
        < CharT, PreMeasurements, FPack, Arg
        , selector_for_printing_with_premeasurements
            < CharT, PreMeasurements, FPack, Arg > >
{
};

template <int R, typename PrintableArgConverter>
struct directly_call_print
{
    template <typename TraitsOrFacet, typename CharT, typename FPack, typename Arg>
    STRF_HD static void print
        ( const TraitsOrFacet& tof
        , strf::destination<CharT>& dst
        , const FPack& fp
        , const Arg& arg )
    {
        tof.print(dst, fp, PrintableArgConverter::convert_printable_arg(arg));
    }
};

template <int R, typename PrintableArgConverter>
struct print_using_make_printer
{
    template <typename TraitsOrFacet, typename CharT, typename FPack, typename Arg >
    STRF_HD static void print
        ( const TraitsOrFacet& tof
        , strf::destination<CharT>& dst
        , const FPack& fp
        , const Arg& arg )
    {
        using premeasurements_type = strf::no_premeasurements;
        using printer_type = find_printer_type
            < CharT, premeasurements_type, FPack, Arg, TraitsOrFacet, PrintableArgConverter>;

        premeasurements_type pre;
        printer_type
            ( tof.make_printer
                ( strf::tag<CharT>{}, &pre, fp
                  , PrintableArgConverter::convert_printable_arg(arg) ) )
            (dst);
    }
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename FPack
         , typename Arg >
class printer_selector_for_printing_without_premeasurements
{
    using default_printable_with_fmt =
        default_printable_with_fmt_of_printable_traits<PrintingTraits>;
    using fmt_value_type = typename default_printable_with_fmt::value_type;
    using fwd_type = typename PrintingTraits::forwarded_type;
    using premeasurements_type = strf::no_premeasurements;

    template < typename P, typename A
             , detail::enable_if_t
                 < std::is_same<A, default_printable_with_fmt>::value
                && detail::all_base_fmtfn_classes_are_empty<A>::value
                 , int > = 0
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>().value() ) ) >
    static STRF_HD auto test_(strf::rank<8>*, const A& arg)
        -> directly_call_print<8, printable_arg_fmt_remover<default_printable_with_fmt> >;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>() ) ) >
    static STRF_HD auto test_(strf::rank<7>*, const A& arg)
        -> directly_call_print<7, printable_arg_caster<const A&> >;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , static_cast<fwd_type>(std::declval<const A&>()) ) ) >
    static STRF_HD auto test_(strf::rank<6>*, const A& arg)
        -> directly_call_print<6, printable_arg_caster<fwd_type> >;


    template < typename P, typename A
             , detail::enable_if_t< !detail::is_printable_with_fmt<A>::value, int > = 0
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , default_printable_with_fmt{std::declval<const A&>()} ) ) >
    static STRF_HD auto test_(strf::rank<5>*, const A& arg)
        -> directly_call_print<5, printable_arg_fmt_adder<default_printable_with_fmt> >;


    template < typename P, typename A
             , detail::enable_if_t
                 < std::is_same<A, default_printable_with_fmt>::value
                && detail::all_base_fmtfn_classes_are_empty<A>::value
                 , int > = 0
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>().value() ))  >
    static STRF_HD auto test_(strf::rank<4>*, const A& arg)
        -> print_using_make_printer<4, printable_arg_fmt_remover<default_printable_with_fmt> >;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>() ) ) >
    static STRF_HD auto test_(strf::rank<3>*, const A& arg)
        -> print_using_make_printer<3, printable_arg_caster<const A&> >;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , static_cast<fwd_type>(std::declval<const A&>()) ) ) >
    static STRF_HD auto test_(strf::rank<2>*, const A& arg)
        -> print_using_make_printer<2, printable_arg_caster<fwd_type> >;


    template < typename P, typename A
             , detail::enable_if_t< !detail::is_printable_with_fmt<A>::value, int > = 0
             , typename = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , default_printable_with_fmt{std::declval<const A&>()} ))  >
    static STRF_HD auto test_(strf::rank<1>*, const A& arg)
        -> print_using_make_printer<1, printable_arg_fmt_adder<default_printable_with_fmt> >;

public:
    using type =
        decltype(test_<TraitsOrFacet, Arg>( std::declval<strf::rank<8>*>()
                                          , std::declval<const Arg&>() ));
};


template <typename CharT, typename FPack, typename Arg >
struct selector_for_printing_without_premeasurements
{
    using traits = strf::printable_traits_of<Arg>;
    using traits_or_facet_getter =
        select_traits_or_facet_getter< traits, CharT, FPack, Arg >;
    using traits_or_facet_type = typename traits_or_facet_getter::traits_or_facet_type;

    using print_caller = typename
        printer_selector_for_printing_without_premeasurements
        < traits, traits_or_facet_type, CharT, FPack, Arg >
        ::type;
};

template < typename CharT, typename FPack, typename Arg
         , typename Selector =
               selector_for_printing_without_premeasurements
               <CharT, FPack, Arg> >
struct helper_for_printing_without_premeasurements
    : Selector::traits_or_facet_getter
    , Selector::print_caller
{
};

// tr_printing without_premeasurements

template <typename CharT, typename MakePrinterReturnType, typename PrintableArgConverter>
struct printer_wrapper_maker_without_premeasurements
{
    using wrapped_type = MakePrinterReturnType;

    using polymorphic_printer_type = detail::printer_wrapper<CharT, wrapped_type>;

    template <typename FPack, typename Arg, typename TraitsOrFacet>
    static STRF_HD auto make_polymorphic_printer
        ( const TraitsOrFacet& tof
        , const FPack& fp
        , const Arg& arg )
   {
       strf::no_premeasurements no_pre;
       return tof.make_printer(strf::tag<CharT>{}, &no_pre, fp, arg);
   }
};


template <typename FPack, typename Printable>
struct printer_adapter_input
{
    FPack fpack;
    Printable printable;
};

template <typename CharT, typename FPack, typename PrintingTraits, typename Printable>
class polymorphic_printer_that_calls_print_from_facet
    : public detail::polymorphic_printer<CharT>
    , private FPack
{
public:

    STRF_HD explicit polymorphic_printer_that_calls_print_from_facet
        ( const printer_adapter_input<FPack, Printable>& i )
        : FPack(i.fpack)
        , printable_(i.printable)
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        strf::use_facet
            < strf::printable_overrider_c
            , typename PrintingTraits::representative_type > (facets_())
            .print(dst, facets_(), printable_);
    }

private:

    STRF_HD const FPack& facets_() const
    {
        return *this;
    }

    Printable printable_;
};

template <typename CharT, typename FPack, typename PrintingTraits, typename Printable>
class polymorphic_printer_that_calls_print_from_traits
    : public detail::polymorphic_printer<CharT>
    , private FPack
{
public:

    STRF_HD explicit polymorphic_printer_that_calls_print_from_traits
        ( const printer_adapter_input<FPack, Printable>& i )
        : FPack(i.fpack)
        , printable_(i.printable)
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        PrintingTraits::print(dst, facets_(), printable_);
    }

private:

    STRF_HD const FPack& facets_() const
    {
        return *this;
    }

    Printable printable_;
};

template < typename CharT
         , typename PrintingTraits
         , typename TraitsOrFacet
         , typename FPack
         , typename Arg
         , typename PrintableArgConverter >
struct print_caller_adapter_maker
{
    using converted_printable_type =
        decltype(PrintableArgConverter::convert_printable_arg(std::declval<const Arg&>()));

    using polymorphic_printer_type =
        detail::conditional_t
            < std::is_same<PrintingTraits, TraitsOrFacet>::value
            , polymorphic_printer_that_calls_print_from_traits
                < CharT, FPack, PrintingTraits, converted_printable_type>
            , polymorphic_printer_that_calls_print_from_facet
                < CharT, FPack, PrintingTraits, converted_printable_type> >;

    static STRF_HD auto make_polymorphic_printer
        ( const TraitsOrFacet&
        , const FPack& fp
        , const Arg& arg )
        -> printer_adapter_input<FPack, converted_printable_type>
    {
        return {fp, PrintableArgConverter::convert_printable_arg(arg)};
    }
};

template < typename PrintingTraits
         , typename TraitsOrFacet
         , typename CharT
         , typename FPack
         , typename Arg >
class polymorphic_printer_maker_selector_for_printing_without_premeasurements
{
    using default_printable_with_fmt =
        default_printable_with_fmt_of_printable_traits<PrintingTraits>;
    using fmt_value_type = typename default_printable_with_fmt::value_type;
    using fwd_type = typename PrintingTraits::forwarded_type;
    using premeasurements_type = strf::no_premeasurements;

    // printer_wrapper_maker_without_premeasurements

    template < typename P, typename A
             , detail::enable_if_t
                 < std::is_same<A, default_printable_with_fmt>::value
                && detail::all_base_fmtfn_classes_are_empty<A>::value
                 , int > = 0
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>().value() ))  >
    static STRF_HD auto test_(strf::rank<8>*, const A& arg)
        -> printer_wrapper_maker_without_premeasurements
            < CharT
            , MakePrinterReturnType
            , printable_arg_fmt_remover<default_printable_with_fmt> >;


    template < typename P, typename A
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>() ) ) >
    static STRF_HD auto test_(strf::rank<7>*, const A& arg)
        -> printer_wrapper_maker_without_premeasurements
            < CharT, MakePrinterReturnType, printable_arg_caster<const A&> >;


    template < typename P, typename A
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , static_cast<fwd_type>(std::declval<const A&>()) ) ) >
    static STRF_HD auto test_(strf::rank<6>*, const A& arg)
        -> printer_wrapper_maker_without_premeasurements
            < CharT, MakePrinterReturnType, printable_arg_caster<fwd_type> >;


    template < typename P, typename A
             , detail::enable_if_t< !detail::is_printable_with_fmt<A>::value, int > = 0
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const P&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , default_printable_with_fmt{std::declval<const A&>()} ))  >
    static STRF_HD auto test_(strf::rank<5>*, const A& arg)
        -> printer_wrapper_maker_without_premeasurements
            < CharT, MakePrinterReturnType, printable_arg_fmt_adder<default_printable_with_fmt> >;


    // print_caller_adapter_maker

    template < typename P, typename A
             , detail::enable_if_t
                 < std::is_same<A, default_printable_with_fmt>::value
                && detail::all_base_fmtfn_classes_are_empty<A>::value
                 , int > = 0
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>().value() ) ) >
    static STRF_HD auto test_(strf::rank<4>*, const A& arg)
        -> print_caller_adapter_maker
            < CharT, PrintingTraits, TraitsOrFacet, FPack, Arg
            , printable_arg_fmt_remover<default_printable_with_fmt> >;


    template < typename P, typename A
             , typename = detail::enable_if_t<detail::is_printable_with_fmt<A>::value>
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<const A&>() ) ) >
    static STRF_HD auto test_(strf::rank<3>*, const A& arg)
        -> print_caller_adapter_maker
            < CharT, PrintingTraits, TraitsOrFacet, FPack, Arg
            , printable_arg_caster<const A&> >;


    template < typename P, typename A
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , static_cast<fwd_type>(std::declval<const A&>()) ) ) >
    static STRF_HD auto test_(strf::rank<2>*, const A& arg)
        -> print_caller_adapter_maker
            < CharT, PrintingTraits, TraitsOrFacet, FPack, Arg
            , printable_arg_caster<fwd_type> >;


    template < typename P, typename A
             , typename = detail::enable_if_t<std::is_convertible<A, fwd_type>::value>
             , typename = decltype
                 ( std::declval<const P&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , default_printable_with_fmt{std::declval<fwd_type>()} ) ) >
    static STRF_HD auto test_(strf::rank<1>*, const A& arg)
        -> print_caller_adapter_maker
            < CharT, PrintingTraits, TraitsOrFacet, FPack, Arg
            , printable_arg_fmt_adder<default_printable_with_fmt> >;

public:
    using type =
        decltype(test_<TraitsOrFacet, Arg>( std::declval<strf::rank<8>*>()
                                          , std::declval<const Arg&>() ));
};

template <typename CharT, typename FPack, typename Arg >
struct selector_for_tr_printing_without_premeasurements
{
    using traits = strf::printable_traits_of<Arg>;
    using traits_or_facet_getter =
        select_traits_or_facet_getter< traits, CharT, FPack, Arg >;
    using traits_or_facet_type = typename traits_or_facet_getter::traits_or_facet_type;
    using polymorphic_printer_maker = typename
        polymorphic_printer_maker_selector_for_printing_without_premeasurements
        < traits, traits_or_facet_type, CharT, FPack, Arg >
        ::type;
};

template < typename CharT, typename FPack, typename Arg
         , typename Selector =
               selector_for_tr_printing_without_premeasurements
               < CharT, FPack, Arg > >
struct helper_for_tr_printing_without_premeasurements
    : Selector::polymorphic_printer_maker
    , Selector::traits_or_facet_getter
{
};

} // namespace printing_helpers

template <typename CharT, typename PreMeasurements, typename FPack, typename Arg>
struct helper_for_printing_with_premeasurements
    : printing_helpers::helper_for_printing_with_premeasurements
        <CharT, PreMeasurements, FPack, detail::remove_cvref_t<Arg> >
{
};

template <typename CharT, typename FPack, typename Arg>
struct helper_for_printing_without_premeasurements
    : printing_helpers::helper_for_printing_without_premeasurements
        <CharT, FPack, detail::remove_cvref_t<Arg> >
{
};

template <typename CharT, typename FPack, typename Arg>
struct helper_for_tr_printing_without_premeasurements
    : printing_helpers::helper_for_tr_printing_without_premeasurements
        <CharT, FPack, detail::remove_cvref_t<Arg> >
{
};

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_PRINTING_HELPERS_HPP

