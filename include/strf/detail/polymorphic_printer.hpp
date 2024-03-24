#ifndef STRF_DETAIL_POLYMORPHIC_PRINTER_HPP
#define STRF_DETAIL_POLYMORPHIC_PRINTER_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/destination.hpp>

namespace strf {
namespace detail {

template <typename CharT>
class polymorphic_printer
{
public:
    using char_type = CharT;

    constexpr polymorphic_printer() noexcept = default;
    constexpr polymorphic_printer(const polymorphic_printer&) = default;
    constexpr polymorphic_printer(polymorphic_printer&&) = default;
    constexpr polymorphic_printer& operator=(const polymorphic_printer&) = default;
    constexpr polymorphic_printer& operator=(polymorphic_printer&&) = default;

    virtual STRF_HD ~polymorphic_printer() STRF_DEFAULT_IMPL;

    virtual STRF_HD void print_to(strf::destination<CharT>& dst) const = 0;
};

template <typename CharT, typename Printer>
class printer_wrapper: public detail::polymorphic_printer<CharT>
{
public:
    template < typename... IniArgs
             , strf::detail::enable_if_t
                   < ! std::is_same
                       < strf::tag<printer_wrapper>
                       , strf::tag<detail::remove_cvref_t<IniArgs>...> >
                       :: value
                  , int > = 0 >
    STRF_HD explicit printer_wrapper(IniArgs&&... ini_args)
        : printer_((IniArgs&&)ini_args...)
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        printer_(dst);
    }

private:
    Printer printer_;
};


template <typename CharT>
struct clonable_polymorphic_printer: polymorphic_printer<CharT>
{
    virtual STRF_HD clonable_polymorphic_printer*
    copy_to(void* dst) const = 0;

    virtual STRF_HD clonable_polymorphic_printer*
    move_to(void* dst) && = 0;
};

template <typename CharT, typename Printer>
class clonable_printer_wrapper: public detail::clonable_polymorphic_printer<CharT>
{
public:
    template < typename... IniArgs
             , strf::detail::enable_if_t
                   < ! std::is_same
                       < strf::tag<clonable_printer_wrapper>
                       , strf::tag<detail::remove_cvref_t<IniArgs>...> >
                       :: value
                  , int > = 0 >
    STRF_HD explicit clonable_printer_wrapper(IniArgs&&... ini_args)
        : printer_((IniArgs&&)ini_args...)
    {
    }

    STRF_HD clonable_printer_wrapper* copy_to(void* dst) const override
    {
        return new (dst) clonable_printer_wrapper(printer_);
    }

    STRF_HD clonable_printer_wrapper* move_to(void* dst) && override
    {
        return new (dst) clonable_printer_wrapper(std::move(printer_));
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override
    {
        printer_(dst);
    }

private:
    Printer printer_;
};

} // namespace detail

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

// template <typename CharIn>
// struct tr_string_input_tag: strf::string_input_tag<CharIn>
// {
// };

// template <typename CharIn>
// struct is_tr_string_of
// {
//     template <typename T>
//     using fn = std::is_same<strf::tr_string_input_tag<CharIn>, T>;
// };

// template <typename T>
// struct is_tr_string: std::false_type
// {
// };

// template <typename CharIn>
// struct is_tr_string<strf::is_tr_string_of<CharIn>> : std::true_type
// {
// };

} // namespace strf

#endif  // STRF_DETAIL_POLYMORPHIC_PRINTER_HPP

