#ifndef STRF_DETAIL_PRINTING_HELPERS_HPP
#define STRF_DETAIL_PRINTING_HELPERS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printing_aliases.hpp>
#include <strf/detail/polymorphic_printer.hpp>
#include <strf/facets_pack.hpp>

namespace strf {
namespace detail {
namespace printing_helpers {

template <typename ValueAndFormat, typename ValueFwdT>
struct printable_arg_fmt_remover
{
    // using fwd_type = ValueFwdT;
    // using output_type = ValueFwdT;
    using input_type = const ValueAndFormat&;

    static constexpr STRF_HD ValueFwdT convert_printable_arg
        ( const ValueAndFormat& x )
    {
        return x.value();
    }
};

template <typename FwdType>
struct printable_arg_forwarder
{
    // using fwd_type = FwdType;
    // using output_type = FwdType;
    using input_type = FwdType;

    static constexpr STRF_HD FwdType convert_printable_arg(FwdType x)
    {
        return x;
    }
};

template < typename PrintableInfo
         , typename DefOrFacet
         , typename CharT
         , typename PreMeasurements
         , typename FPack >
class printable_arg_converter_selector_for_printing_with_premeasurements
{
    using fwd_type = typename PrintableInfo::forwarded_type;

    template < typename PDoF, typename PInfo, typename FwdT
             , typename ArgFmtRemoved = typename PInfo::remove_fmt_if_possible
             , typename = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , std::declval<ArgFmtRemoved>() ))  >
    static STRF_HD auto test_(strf::rank<1>*, FwdT)
        -> printable_arg_fmt_remover<FwdT, ArgFmtRemoved>;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<PreMeasurements*>()
                     , std::declval<const FPack&>()
                     , std::declval<FwdT>() ) ) >
    static STRF_HD auto test_(strf::rank<0>*, FwdT)
        -> printable_arg_forwarder<fwd_type>;

public:
    using type =
        decltype(test_<DefOrFacet, PrintableInfo, fwd_type>(
                     std::declval<strf::rank<1>*>(), std::declval<fwd_type>() ));
};

template < typename PrintableInfo, typename DefOrFacet, typename CharT
         , typename PreMeasurements, typename FPack >
using select_printable_arg_converter_for_printing_with_premeasurements = typename
    printable_arg_converter_selector_for_printing_with_premeasurements
    <PrintableInfo, DefOrFacet, CharT, PreMeasurements, FPack>
    ::type;

template <typename Overrider, typename Representative>
struct use_overrider
{
    using printable_def_or_facet_type = Overrider;

    template <typename FPack>
    static constexpr STRF_HD const Overrider& get_printable_def_or_facet(const FPack& fp)
    {
        return strf::get_facet<strf::printable_overrider_c<Representative>, Representative>(fp);
    }
};

template <typename PrintableDef>
struct use_printable_def
{
    using printable_def_or_facet_type = PrintableDef;
    using printable_def_type = PrintableDef;

    template <typename FPack>
    static constexpr STRF_HD printable_def_or_facet_type get_printable_def_or_facet(const FPack&)
    {
        return printable_def_or_facet_type{};
    }
};

template < typename PrintableDef
         , typename CharT
         , typename FPack
         , bool Overridable >
struct printable_def_or_facet_getter_selector_2
{
    static_assert(Overridable, "");
    using representative = detail::extract_representative<PrintableDef>;
    using overrider_ = decltype
        ( strf::get_facet<strf::printable_overrider_c<representative>, representative>
          (std::declval<FPack>()) );

    using overrider = strf::detail::remove_cvref_t<overrider_>;
    using printable_def_or_facet_getter_type = typename std::conditional
        < std::is_same<overrider, strf::dont_override<representative>>::value
        , use_printable_def<PrintableDef>
        , use_overrider<overrider, representative> >
        ::type;
};

template < typename PrintableDef
         , typename CharT
         , typename FPack >
struct printable_def_or_facet_getter_selector_2<PrintableDef, CharT, FPack, false>
{
    using printable_def_or_facet_getter_type = use_printable_def<PrintableDef>;
};

template < typename PrintableDef
         , typename CharT
         , typename FPack >
struct printable_def_or_facet_getter_selector
{
    template <typename U>
    static STRF_HD typename U::is_overridable test_(const U*);

    template <typename U>
    static STRF_HD std::false_type test_(...);

    using is_overridable = decltype(test_<PrintableDef>((PrintableDef*)nullptr));

    using other = printable_def_or_facet_getter_selector_2
        < PrintableDef, CharT, FPack, is_overridable::value >;
    using printable_def_or_facet_getter_type = typename other::printable_def_or_facet_getter_type;
};

template < typename PrintableDef, typename CharT, typename FPack >
using select_printable_def_or_facet_getter = typename printable_def_or_facet_getter_selector
    <PrintableDef, CharT, FPack>
    :: printable_def_or_facet_getter_type;

template <typename CharT, typename PreMeasurements, typename FPack, typename PrintableInfo>
struct selector_for_printing_with_premeasurements
{
    using printable_def_or_facet_getter =
        select_printable_def_or_facet_getter<typename PrintableInfo::printable_def, CharT, FPack>;
    using printable_def_or_facet_type =
        typename printable_def_or_facet_getter::printable_def_or_facet_type;

    using printable_arg_converter =
        select_printable_arg_converter_for_printing_with_premeasurements
        <PrintableInfo, printable_def_or_facet_type, CharT, PreMeasurements, FPack>;
};

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename DefOrFacet, typename PrintableArgConverter >
using find_printer_type =
    decltype( std::declval<DefOrFacet>().make_printer
                ( strf::tag<CharT>{}
                , std::declval<PreMeasurements*>()
                , std::declval<const FPack&>()
                , PrintableArgConverter::convert_printable_arg(std::declval<const Arg&>())));

template < typename CharT, typename PreMeasurements, typename FPack, typename Arg
         , typename Selector >
struct helper_for_printing_with_premeasurements_impl
    : Selector::printable_def_or_facet_getter
    , Selector::printable_arg_converter
{
    using printable_def_or_facet_type =
        typename Selector::printable_def_or_facet_getter::printable_def_or_facet_type;
    using printer_type = find_printer_type
        < CharT, PreMeasurements, FPack, Arg
        , printable_def_or_facet_type, typename Selector::printable_arg_converter>;
};

template <typename CharT, typename PreMeasurements, typename FPack, typename PrintingInfo>
struct helper_for_printing_with_premeasurements
    : helper_for_printing_with_premeasurements_impl
        < CharT, PreMeasurements, FPack, typename PrintingInfo::forwarded_type
        , selector_for_printing_with_premeasurements
            < CharT, PreMeasurements, FPack, PrintingInfo > >
{
};

template <int R, typename PrintableArgConverter>
struct directly_call_print
{
    template <typename DefOrFacet, typename CharT, typename FPack, typename Arg>
    STRF_HD static void print
        ( const DefOrFacet& tof
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
    template <typename DefOrFacet, typename CharT, typename FPack, typename Arg >
    STRF_HD static void print
        ( const DefOrFacet& tof
        , strf::destination<CharT>& dst
        , const FPack& fp
        , const Arg& arg )
    {
        using premeasurements_type = strf::no_premeasurements;
        using printer_type = find_printer_type
            < CharT, premeasurements_type, FPack, Arg, DefOrFacet, PrintableArgConverter>;

        premeasurements_type pre;
        printer_type
            ( tof.make_printer
                ( strf::tag<CharT>{}, &pre, fp
                  , PrintableArgConverter::convert_printable_arg(arg) ) )
            (dst);
    }
};

template < typename PrintableInfo
         , typename DefOrFacet
         , typename CharT
         , typename FPack >
class printer_selector_for_printing_without_premeasurements
{
    using fwd_type = typename PrintableInfo::forwarded_type;
    using premeasurements_type = strf::no_premeasurements;

    template < typename PDoF, typename PInfo, typename FwdT
             , typename ArgFmtRemoved = typename PInfo::remove_fmt_if_possible
             , typename = decltype
                 ( std::declval<const PDoF&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<ArgFmtRemoved>() )) >
    static STRF_HD auto test_(strf::rank<3>*, FwdT)
        -> directly_call_print<3, printable_arg_fmt_remover<FwdT, ArgFmtRemoved> >;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename = decltype
                 ( std::declval<const PDoF&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<FwdT>() ) ) >
    static STRF_HD auto test_(strf::rank<2>*, FwdT)
        -> directly_call_print<2, printable_arg_forwarder<fwd_type> >;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename ArgFmtRemoved = typename PInfo::remove_fmt_if_possible
             , typename = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<ArgFmtRemoved>() ))  >
    static STRF_HD auto test_(strf::rank<1>*, FwdT)
        -> print_using_make_printer<1, printable_arg_fmt_remover<FwdT, ArgFmtRemoved> >;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<FwdT>() ) ) >
    static STRF_HD auto test_(strf::rank<0>*, FwdT)
        -> print_using_make_printer<0, printable_arg_forwarder<fwd_type> >;

public:
    using type =
        decltype(test_<DefOrFacet, PrintableInfo, fwd_type>
                 ( std::declval<strf::rank<3>*>(), std::declval<fwd_type>() ));
};


template <typename CharT, typename FPack, typename PrintableInfo >
struct selector_for_printing_without_premeasurements
{
    using printable_def_or_facet_getter =
        select_printable_def_or_facet_getter<typename PrintableInfo::printable_def, CharT, FPack>;

    using printable_def_or_facet_type =
        typename printable_def_or_facet_getter::printable_def_or_facet_type;

    using print_caller = typename
        printer_selector_for_printing_without_premeasurements
        < PrintableInfo, printable_def_or_facet_type, CharT, FPack >
        ::type;
};

template < typename CharT, typename FPack, typename PrintableInfo
         , typename Selector =
               selector_for_printing_without_premeasurements<CharT, FPack, PrintableInfo> >
struct helper_for_printing_without_premeasurements
    : Selector::printable_def_or_facet_getter
    , Selector::print_caller
{
};

// tr_printing without_premeasurements

template <typename CharT, typename MakePrinterReturnType, typename PrintableArgConverter>
struct printer_wrapper_maker_without_premeasurements
{
    using wrapped_type = MakePrinterReturnType;

    using polymorphic_printer_type = detail::printer_wrapper<CharT, wrapped_type>;

    template <typename FPack, typename DefOrFacet>
    static STRF_HD auto make_polymorphic_printer_input
        ( const DefOrFacet& dof
        , const FPack& fp
        , typename PrintableArgConverter::input_type arg )
   {
       strf::no_premeasurements no_pre;
       return dof.make_printer( strf::tag<CharT>{}, &no_pre, fp
                              , PrintableArgConverter::convert_printable_arg(arg) );
   }
};


template <typename FPack, typename PrintableInfo>
struct printer_adapter_input
{
    FPack fpack;
    typename PrintableInfo::as_member_type printable;
};

template <typename CharT, typename FPack, typename PrintableInfo>
class polymorphic_printer_that_calls_print_from_facet
    : public detail::polymorphic_printer<CharT>
    , private FPack
{
public:

    STRF_HD explicit polymorphic_printer_that_calls_print_from_facet
        ( const printer_adapter_input<FPack, PrintableInfo>& i )
        : FPack(i.fpack)
        , printable_(i.printable)
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        using printable_def = typename PrintableInfo::printable_def;
        using representative = detail::extract_representative<printable_def>;
        strf::get_facet
            < strf::printable_overrider_c<representative>, representative > (facets_())
            .print(dst, facets_(), printable_);
    }

private:

    STRF_HD const FPack& facets_() const
    {
        return *this;
    }

    typename PrintableInfo::as_member_type printable_;
};

template <typename CharT, typename FPack, typename PrintableInfo>
class polymorphic_printer_that_calls_print_from_pritable_def
    : public detail::polymorphic_printer<CharT>
    , private FPack
{
public:
    using printable_def = typename PrintableInfo::printable_def;

    STRF_HD explicit polymorphic_printer_that_calls_print_from_pritable_def
        ( const printer_adapter_input<FPack, PrintableInfo>& i )
        : FPack(i.fpack)
        , printable_(i.printable)
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        printable_def::print(dst, facets_(), printable_);
    }

private:

    STRF_HD const FPack& facets_() const
    {
        return *this;
    }

    typename PrintableInfo::as_member_type printable_;
};

template < typename CharT
         , typename PrintableInfo
         , typename DefOrFacet
         , typename FPack
         , typename PrintableArgConverter >
struct print_caller_adapter_maker
{
    using polymorphic_printer_type =
        detail::conditional_t
            < std::is_same<DefOrFacet, typename PrintableInfo::printable_def>::value
            , polymorphic_printer_that_calls_print_from_pritable_def
                < CharT, FPack, PrintableInfo>
            , polymorphic_printer_that_calls_print_from_facet
                < CharT, FPack, PrintableInfo> >;

    static STRF_HD auto make_polymorphic_printer_input
        ( const DefOrFacet&
        , const FPack& fp
        , typename PrintableArgConverter::input_type arg )
        -> printer_adapter_input<FPack, PrintableInfo>
    {
        return {fp, PrintableArgConverter::convert_printable_arg(arg)};
    }
};

template < typename PrintableInfo
         , typename DefOrFacet
         , typename CharT
         , typename FPack >
class polymorphic_printer_maker_selector_for_printing_without_premeasurements
{
    using fwd_type = typename PrintableInfo::forwarded_type;
    using premeasurements_type = strf::no_premeasurements;
    using printable_def = typename PrintableInfo::printable_def;

    // printer_wrapper_maker_without_premeasurements

    template < typename PDoF, typename PInfo, typename FwdT
             , typename ArgFmtRemoved = typename PInfo::remove_fmt_if_possible
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<ArgFmtRemoved>() ))  >
    static STRF_HD auto test_(strf::rank<3>*, FwdT)
        -> printer_wrapper_maker_without_premeasurements
            < CharT
            , MakePrinterReturnType
            , printable_arg_fmt_remover<FwdT, ArgFmtRemoved> >;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename MakePrinterReturnType = decltype
                 ( std::declval<const PDoF&>().make_printer
                     ( strf::tag<CharT>{}
                     , std::declval<premeasurements_type*>()
                     , std::declval<const FPack&>()
                     , std::declval<FwdT>() ) ) >
    static STRF_HD auto test_(strf::rank<2>*, FwdT)
        -> printer_wrapper_maker_without_premeasurements
            < CharT, MakePrinterReturnType, printable_arg_forwarder<fwd_type> >;

    // print_caller_adapter_maker

    template < typename PDoF, typename PInfo, typename FwdT
             , typename ArgFmtRemoved = typename PInfo::remove_fmt_if_possible
             , typename = decltype
                 ( std::declval<const PDoF&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<ArgFmtRemoved>() )) >
    static STRF_HD auto test_(strf::rank<1>*, FwdT)
        -> print_caller_adapter_maker
            < CharT, typename PInfo::info_of_without_fmt, DefOrFacet
            , FPack, printable_arg_fmt_remover<FwdT, ArgFmtRemoved> >;


    template < typename PDoF, typename PInfo, typename FwdT
             , typename = decltype
                 ( std::declval<const PDoF&>().print
                     ( std::declval<strf::destination<CharT>&>()
                     , std::declval<const FPack&>()
                     , std::declval<FwdT>() ) ) >
    static STRF_HD auto test_(strf::rank<0>*, FwdT)
        -> print_caller_adapter_maker
            < CharT, PrintableInfo, DefOrFacet, FPack
            , printable_arg_forwarder<fwd_type> >;

public:
    using type =
        decltype(test_<DefOrFacet, PrintableInfo, fwd_type>
                 ( std::declval<strf::rank<3>*>(), std::declval<fwd_type>() ));
};

template <typename CharT, typename FPack, typename PrintableInfo >
struct selector_for_tr_printing_without_premeasurements
{
    using printable_def = typename PrintableInfo::printable_def;
    using printable_def_or_facet_getter =
        select_printable_def_or_facet_getter<printable_def, CharT, FPack>;
    using printable_def_or_facet_type =
        typename printable_def_or_facet_getter::printable_def_or_facet_type;
    using polymorphic_printer_maker = typename
        polymorphic_printer_maker_selector_for_printing_without_premeasurements
        < PrintableInfo, printable_def_or_facet_type, CharT, FPack >
        ::type;
};

template < typename CharT, typename FPack, typename PrintableInfo
         , typename Selector =
               selector_for_tr_printing_without_premeasurements<CharT, FPack, PrintableInfo> >
struct helper_for_tr_printing_without_premeasurements
    : Selector::polymorphic_printer_maker
    , Selector::printable_def_or_facet_getter
{
};

} // namespace printing_helpers

template <typename CharT, typename PreMeasurements, typename FPack, typename PrintableInfo>
struct helper_for_printing_with_premeasurements
    : printing_helpers::helper_for_printing_with_premeasurements
        <CharT, PreMeasurements, FPack, PrintableInfo >
{
};

template <typename CharT, typename FPack, typename PrintableInfo>
struct helper_for_printing_without_premeasurements
    : printing_helpers::helper_for_printing_without_premeasurements
        <CharT, FPack, PrintableInfo >
{
};

template <typename CharT, typename FPack, typename PrintableInfo>
struct helper_for_tr_printing_without_premeasurements
    : printing_helpers::helper_for_tr_printing_without_premeasurements
        <CharT, FPack, PrintableInfo >
{
};


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

template <typename... PrintablesInfo>
struct args_printer;

template <>
struct args_printer<>
{
    template <typename CharT, typename FPack>
    inline static STRF_HD void print(strf::destination<CharT>&, const FPack&)
    {
    }
};

template <typename PrintableInfo0, typename... PrintablesInfo>
struct args_printer<PrintableInfo0, PrintablesInfo...>
{

    template <typename CharT, typename FPack>
    static STRF_HD void print(
        strf::destination<CharT>& dst,
        const FPack& fp,
        typename PrintableInfo0::forwarded_type arg0,
        typename PrintablesInfo::forwarded_type... args )
    {
        using helper = helper_for_printing_without_premeasurements<CharT, FPack, PrintableInfo0>;
        helper::print(helper::get_printable_def_or_facet(fp), dst, fp, arg0);

        if (dst.good()) {
            args_printer<PrintablesInfo...>::print(dst, fp, args...);
        }
    }
};

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_PRINTING_HELPERS_HPP

