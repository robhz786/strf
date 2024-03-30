#ifndef STRF_DETAIL_SEPARATE_ARGS_AND_PRINT_HPP
#define STRF_DETAIL_SEPARATE_ARGS_AND_PRINT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/reserve_policies.hpp>

namespace strf {
namespace detail {

template <typename Arg>
struct assert_is_printable
{
    static_assert(strf::detail::is_printable<Arg>::value, "Type is not Printable");
    using type = Arg;
};

template <typename Arg>
using assert_is_printable_t = typename assert_is_printable<Arg>::type;

namespace args_without_tr {

template <typename Arg>
struct print_arg_validator
{
    static constexpr bool is_fpe       = strf::detail::is_fpe<Arg>::value;
    static constexpr bool is_printable = strf::detail::is_printable<Arg>::value;

    static_assert( ! (is_fpe && is_printable)
                 , "type is both Printable and FacetPackElement");

    static_assert( ! (! is_fpe && ! is_printable)
                 , "type is not Printable nor FacetPackElement");
};

template <bool FirstArgIsFpe, typename... Args>
struct separate_args_2;

template <typename... Args>
struct separate_args_2<false, Args...>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<assert_is_printable_t<Args>...>;
};

template <typename Arg>
struct separate_args_2<true, Arg>
{
    using fpes = mp_type_list<Arg>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename Arg1, typename... OtherArgs>
struct separate_args_2<true, Arg0, Arg1, OtherArgs...>
{
    using validator = print_arg_validator<strf::detail::remove_cvref_t<Arg1>>;
    static constexpr bool arg1_is_fpe = validator::is_fpe;

    using continuation = separate_args_2<arg1_is_fpe, Arg1, OtherArgs...>;

    using continuation_fpes = typename continuation::fpes;
    using fpes = typename continuation_fpes::template add_front<Arg0>;

    using printables = typename continuation::printables;
};


template <typename... Args>
struct separate_args;

template <>
struct separate_args<>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename... OtherArgs>
struct separate_args<Arg0, OtherArgs...>
{
    using validator = print_arg_validator<strf::detail::remove_cvref_t<Arg0>>;
    static constexpr bool arg0_is_fpe = validator::is_fpe;

    using helper = separate_args_2<arg0_is_fpe, Arg0, OtherArgs...>;

    using fpes = typename helper::fpes;
    using printables = typename helper::printables;
};

} // namespace args_without_tr


namespace args_with_tr {

template <typename... Args>
struct first_is_fpe;

template <>
struct first_is_fpe<>
{
    static constexpr bool value = false;
};

template <typename Arg0, typename... OtherArgs>
struct first_is_fpe<Arg0, OtherArgs...>
{
    static constexpr bool value = strf::detail::is_fpe<Arg0>::value;
};

template <bool FirstIsFpe, typename... Args>
struct separate_args_2;

template <typename Arg0, typename... OtherArgs>
struct separate_args_2<true, Arg0, OtherArgs...>
{
private:
    static constexpr bool next_is_fpe = first_is_fpe<OtherArgs...>::value;
    static_assert(sizeof...(OtherArgs) > (unsigned)next_is_fpe, "Missing tr-string");

    using continuation_ = separate_args_2<next_is_fpe, OtherArgs...>;
    using continuation_fpes_ = typename continuation_::fpes;

public:
    using fpes = typename continuation_fpes_::template add_front<Arg0>;
    using printables = typename continuation_::printables;
    using tr_string = typename continuation_::tr_string;
};

template <typename Arg0, typename... Args>
struct separate_args_2<false, Arg0, Args...>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<Args...>;
    using tr_string = Arg0;
};

template <typename... Args>
struct separate_args;

template <>
struct separate_args<>
{
    using fpes = mp_type_list<>;
    using printables = mp_type_list<>;
};

template <typename Arg0, typename... OtherArgs>
struct separate_args<Arg0, OtherArgs...>
{
    using arg0_rmref = strf::detail::remove_cvref_t<Arg0>;
    static constexpr bool arg0_is_fpe = strf::detail::is_fpe<arg0_rmref>::value;

    using helper = separate_args_2<arg0_is_fpe, Arg0, OtherArgs...>;

    using fpes = typename helper::fpes;
    using printables = typename helper::printables;
    using tr_string = typename helper::tr_string;
};

template <typename CharT, typename TrString>
struct check_tr_string_type
{
    constexpr static bool passed = std::is_constructible
        < strf::detail::simple_string_view<CharT>, TrString >
        :: value;
    static_assert(passed, "Invalid type to be used as tr-string");
};

} // namespace args_with_tr


template < bool AddEndOfLine
         , typename ReservePolicy
         , typename DestCreatorArg
         , typename... Args >
STRF_HD auto separate_args_and_print
    ( ReservePolicy reserve_policy
    , DestCreatorArg&& dest_creator
    , Args&&... args )
{
    using dest_creator_t = detail::remove_cvref_t<DestCreatorArg>;
    using return_type = typename ReservePolicy::template return_type<dest_creator_t>;
    using args_separator_t = args_without_tr::separate_args<Args...>;
    using impl = detail::printing_without_tr_string
            < ReservePolicy
            , typename dest_creator_t::char_type
            , typename args_separator_t::fpes
            , typename args_separator_t::printables >;

    return impl::template create_destination_and_print<AddEndOfLine, return_type>
        (reserve_policy, (DestCreatorArg&&)dest_creator, (Args&&)args...);
}

template < bool AddEndOfLine
         , typename ReservePolicy
         , typename DestCreatorArg
         , typename... Args >
STRF_HD auto separate_tr_args_and_print
    ( ReservePolicy reserve_policy
    , DestCreatorArg&& dest_creator
    , Args&&... args )
{
    using dest_creator_t = detail::remove_cvref_t<DestCreatorArg>;
    using char_type = typename dest_creator_t::char_type;
    using return_type = typename ReservePolicy::template return_type<dest_creator_t>;
    using args_separator_t = args_with_tr::separate_args<Args...>;

    using tr_string_t = typename args_separator_t::tr_string;
    using tr_checker = args_with_tr::check_tr_string_type<char_type, tr_string_t>;
    static_assert(tr_checker::passed, "");

    using impl = detail::printing_with_tr_string
            < ReservePolicy
            , char_type
            , typename args_separator_t::fpes
            , typename args_separator_t::printables >;

    return impl::template create_destination_and_print<AddEndOfLine, return_type>
        ( reserve_policy, (DestCreatorArg&&)dest_creator, (Args&&)args...);
}

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_SEPARATE_ARGS_AND_PRINT_HPP

