#ifndef STRF_DETAIL_RESERVE_POLICIES_HPP
#define STRF_DETAIL_RESERVE_POLICIES_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>

namespace strf {

namespace detail {

template <typename Dst>
inline STRF_HD decltype(std::declval<Dst&>().finish())
    finish(strf::rank<2>, Dst& dst)
{
    return dst.finish();
}

template <typename Dst>
inline STRF_HD void finish(strf::rank<1>, Dst&)
{
}

template <typename DestinationCreator, bool Sized>
struct destination_creator_traits;

template <typename DestinationCreator>
struct destination_creator_traits<DestinationCreator, false>
{
    using destination_type = typename DestinationCreator::destination_type;
    using finish_return_type =
        decltype(strf::detail::finish(strf::rank<2>(), std::declval<destination_type&>()));
};

template <typename DestinationCreator>
struct destination_creator_traits<DestinationCreator, true>
{
    using destination_type = typename DestinationCreator::sized_destination_type;
    using finish_return_type =
        decltype(strf::detail::finish(strf::rank<2>(), std::declval<destination_type&>()));
};

template <typename DestinationCreator, bool Sized>
using destination_finish_return_type = typename
    destination_creator_traits<DestinationCreator, Sized>::finish_return_type;

} // namespace detail

struct no_reserve
{
    template <typename DestCreator>
    using return_type = strf::detail::destination_finish_return_type<DestCreator, false>;

    using premeasurements_type = strf::premeasurements
        <strf::size_presence::no, strf::width_presence::no>;
};


struct reserve_given_space
{
public:
    std::size_t space = 0;

    STRF_HD constexpr explicit reserve_given_space(std::size_t s)
        : space(s)
    {
    }

    template <typename DestCreator>
    using return_type = strf::detail::destination_finish_return_type<DestCreator, true>;

    using premeasurements_type = strf::premeasurements
        <strf::size_presence::no, strf::width_presence::no>;
};

struct reserve_calc
{
public:
    template <typename DestCreator>
    using return_type = strf::detail::destination_finish_return_type<DestCreator, true>;

    using premeasurements_type = strf::premeasurements
        <strf::size_presence::yes, strf::width_presence::no>;
};

namespace detail {

template <typename ReservePolicy, typename CharT, typename FpeList, typename PrintablesList>
struct printing_without_tr_string;

template <typename CharT, typename... Fpes, typename... Printables>
struct printing_without_tr_string
    < strf::no_reserve
    , CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...> >
{
    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::no_reserve
        , DestCreator&& dest_creator
        , Fpes... fpes
        , Printables... printables )
    {
        using dest_creator_t = detail::remove_cvref_t<DestCreator>;
        using dest_type = typename dest_creator_t::destination_type;

        dest_type dst{((DestCreator&&)dest_creator).create()};
        detail::print_printables(dst, strf::pack((Fpes&&)fpes...), printables...);

        STRF_IF_CONSTEXPR (AddEndOfLine) {
            strf::put<CharT>(dst, static_cast<CharT>('\n'));
        }
        return strf::detail::finish(strf::rank<2>(), dst);
    }
};

template <typename CharT, typename... Fpes, typename... Printables>
struct printing_without_tr_string
    < strf::reserve_given_space
    , CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...> >
{
    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::reserve_given_space given_space
        , DestCreator&& dest_creator
        , Fpes... fpes
        , Printables... printables )
    {
        using dest_creator_t = detail::remove_cvref_t<DestCreator>;
        using dest_type = typename dest_creator_t::sized_destination_type;

        dest_type dst{((DestCreator&&)dest_creator).create(given_space.space)};
        detail::print_printables(dst, strf::pack((Fpes&&)fpes...), printables...);

        STRF_IF_CONSTEXPR (AddEndOfLine) {
            strf::put<CharT>(dst, static_cast<CharT>('\n'));
        }
        return strf::detail::finish(strf::rank<2>(), dst);
    }
};

template
    < bool AddEndOfLine
    , typename ReturnType
    , typename DestCreator
    , typename... Printers >
STRF_HD ReturnType reserve_calculated_size_and_call_printables
    ( DestCreator&& dest_creator
    , const strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>* pre
    , const Printers&... printers )
{
    using dest_creator_t = detail::remove_cvref_t<DestCreator>;
    using dest_type = typename dest_creator_t::sized_destination_type;

    dest_type dst{((DestCreator&&)dest_creator).create(pre->accumulated_size() + AddEndOfLine)};
    strf::detail::call_printers(dst, printers...);

    STRF_IF_CONSTEXPR (AddEndOfLine) {
        using char_type = typename dest_creator_t::char_type;
        strf::put<char_type>(dst, static_cast<char_type>('\n'));
    }
    return strf::detail::finish(strf::rank<2>(), dst);
}

template
    < typename CharT
    , typename FpesList
    , typename PrintablesList
    , typename PrintingHelperList >
struct reserve_calc_printer_base;

template
    < typename CharT
    , typename... Fpes
    , typename... Printables
    , typename... Helpers >
struct reserve_calc_printer_base
    < CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...>
    , mp_type_list<Helpers...> >
{
    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::reserve_calc
        , DestCreator&& dest_creator
        , Fpes... fpes
        , Printables... printables )
    {
        strf::premeasurements<strf::size_presence::yes, strf::width_presence::no> pre;

        auto fp = strf::pack((Fpes&&)fpes...);
        return detail::reserve_calculated_size_and_call_printables<AddEndOfLine, ReturnType>
            ( dest_creator
            , &pre
            , Helpers::get_traits_or_facet(fp).make_printer
                ( strf::tag<CharT>{}, &pre, fp
                , Helpers::convert_printable_arg((Printables&&)printables))... );
    }
};

template <typename CharT, typename... Fpes, typename... Printables>
struct printing_without_tr_string
    < strf::reserve_calc
    , CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...> >

    : reserve_calc_printer_base
        < CharT
        , mp_type_list<Fpes...>
        , mp_type_list<Printables...>
        , mp_type_list
            < detail::helper_for_printing_with_premeasurements
                < CharT
                , strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>
                , decltype(strf::pack(std::declval<Fpes>()...))
                , Printables >... > >
{
};

template <typename CharT, typename FpeList, typename PrintableList, typename HelperList>
struct printing_with_tr_string_no_premeasurements;

template <typename CharT, typename... Fpes, typename... Printables, typename... Helpers>
struct printing_with_tr_string_no_premeasurements
    < CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...>
    , mp_type_list<Helpers...> >
{
    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::no_reserve
        , DestCreator&& dest_creator
        , Fpes... fpes
        , detail::simple_string_view<CharT> tr_string
        , Printables... printables )
    {
        using dest_creator_t = detail::remove_cvref_t<DestCreator>;
        using dest_type = typename dest_creator_t::destination_type;

        dest_type dst{dest_creator.create()};
        print_<AddEndOfLine>(dst, fpes..., tr_string, printables...);

        return strf::detail::finish(strf::rank<2>(), dst);
    }

    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::reserve_given_space given_space
        , DestCreator&& dest_creator
        , Fpes... fpes
        , detail::simple_string_view<CharT> tr_string
        , Printables... printables )
    {
        using dest_creator_t = detail::remove_cvref_t<DestCreator>;
        using dest_type = typename dest_creator_t::sized_destination_type;

        dest_type dst{dest_creator.create(given_space.space)};
        print_<AddEndOfLine>(dst, fpes..., tr_string, printables...);

        return strf::detail::finish(strf::rank<2>(), dst);
    }

private:

    template <bool AddEndOfLine>
    static STRF_HD void print_
        ( strf::destination<CharT>& dst
        , Fpes... fpes
        , detail::simple_string_view<CharT> tr_string
        , Printables... printables )
    {
        auto fp = strf::pack((Fpes&&)fpes...);

        using charset_cat = strf::charset_c<CharT>;
        auto charset = strf::use_facet<charset_cat, void>(fp);

        using err_handler_cat = strf::tr_error_notifier_c;
        auto&& err_handler = strf::use_facet<err_handler_cat, void>(fp);

        detail::tr_string_write
            ( dst, charset, err_handler, tr_string.begin(), tr_string.end()
            , { & static_cast< const detail::polymorphic_printer<CharT>& >
                  ( typename Helpers::polymorphic_printer_type
                      ( Helpers::make_polymorphic_printer
                          ( Helpers::get_traits_or_facet(fp), fp, printables) ) )... } );

        STRF_IF_CONSTEXPR (AddEndOfLine) {
            strf::put<CharT>(dst, static_cast<CharT>('\n'));
        }
    }
};


template < typename CharT, typename FpeList, typename PrintableList, typename HelperList
         , typename PrintablesIndexSequence >
struct printing_with_tr_string_reserve_calc;

template < typename CharT
         , typename... Fpes
         , typename... Printables
         , typename... Helpers
         , std::size_t... I >
struct printing_with_tr_string_reserve_calc
    < CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...>
    , mp_type_list<Helpers...>
    , strf::detail::index_sequence<I...> >
{
    using premeasurements_type =
        strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>;

    template <bool AddEndOfLine, typename ReturnType, typename DestCreator>
    static STRF_HD ReturnType create_destination_and_print
        ( strf::reserve_calc
        , DestCreator&& dest_creator
        , Fpes... fpes
        , detail::simple_string_view<CharT> tr_string
        , Printables... printables )
    {
        auto fp = strf::pack((Fpes&&)fpes...);
        premeasurements_type pre_array[sizeof...(Printables) + 1];

        return do_create_destination_and_print_<AddEndOfLine, ReturnType>
            ( (DestCreator&&) dest_creator
            , tr_string.begin(), tr_string.end()
            , fp
            , pre_array
            , { & static_cast< const detail::polymorphic_printer<CharT>& >
                  ( detail::printer_wrapper<CharT, typename Helpers::printer_type>
                      ( Helpers::get_traits_or_facet(fp).make_printer
                          ( strf::tag<CharT>{}
                          , &pre_array[I]
                          , fp
                          , Helpers::convert_printable_arg((Printables&&)printables))))... } );
    }

private:

    template < bool AddEndOfLine, typename ReturnType, typename DestCreator, typename FPack >
    static STRF_HD ReturnType do_create_destination_and_print_
        ( DestCreator&& dest_creator
        , const CharT* tr_string
        , const CharT* tr_string_end
        , const FPack& fp
        , const premeasurements_type* args_pre
        , std::initializer_list<const detail::polymorphic_printer<CharT>*> printers )
    {
        using dest_creator_t = detail::remove_cvref_t<DestCreator>;
        using dest_type = typename dest_creator_t::destination_type;

        using charset_cat = strf::charset_c<CharT>;
        auto charset = strf::use_facet<charset_cat, void>(fp);

        using err_handler_cat = strf::tr_error_notifier_c;
        auto&& err_handler = strf::use_facet<err_handler_cat, void>(fp);

        auto invalid_arg_size = charset.replacement_char_size();
        const std::ptrdiff_t size = strf::detail::tr_string_size
            ( args_pre, static_cast<std::ptrdiff_t>(printers.size())
            , tr_string, tr_string_end, invalid_arg_size );
        dest_type dst{dest_creator.create(static_cast<std::size_t>(size + AddEndOfLine))};

        strf::detail::tr_string_write
            ( tr_string, tr_string_end, printers.begin()
            , static_cast<std::ptrdiff_t>(printers.size())
            , dst, charset, err_handler );

        STRF_IF_CONSTEXPR (AddEndOfLine) {
            strf::put<CharT>(dst, static_cast<CharT>('\n'));
        }
        return strf::detail::finish(strf::rank<2>(), dst);
    }
};

template <typename ReservePolicy, typename CharT, typename FpeList, typename PrintablesList>
struct printing_with_tr_string;

template <typename ReservePolicy, typename CharT, typename... Fpes, typename... Printables>
struct printing_with_tr_string
    < ReservePolicy
    , CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...> >
    : printing_with_tr_string_no_premeasurements
        < CharT
        , mp_type_list<Fpes...>
        , mp_type_list<Printables...>
        , mp_type_list
            < detail::helper_for_tr_printing_without_premeasurements
                < CharT
                , decltype(strf::pack(std::declval<Fpes>()...))
                , Printables >... > >
{
};

template <typename CharT, typename... Fpes, typename... Printables>
struct printing_with_tr_string
    < strf::reserve_calc
    , CharT
    , mp_type_list<Fpes...>
    , mp_type_list<Printables...> >
    : printing_with_tr_string_reserve_calc
        < CharT
        , mp_type_list<Fpes...>
        , mp_type_list<Printables...>
        , mp_type_list
            < detail::helper_for_printing_with_premeasurements
                < CharT
                , strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>
                , decltype(strf::pack(std::declval<Fpes>()...))
                , Printables >... >
        , strf::detail::make_index_sequence<sizeof...(Printables)> >
{
};

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_RESERVE_POLICIES_HPP

