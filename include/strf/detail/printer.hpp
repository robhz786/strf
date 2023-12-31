#ifndef STRF_DETAIL_PRINTER_HPP
#define STRF_DETAIL_PRINTER_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/destination.hpp>

namespace strf {

template <typename CharT>
class printer
{
public:
    using char_type = CharT;

    printer() = default;
    printer(const printer&) = default;
    printer(printer&&) noexcept = default;
    printer& operator=(const printer&) = default;
    printer& operator=(printer&&) noexcept = default;

    virtual STRF_HD ~printer() STRF_DEFAULT_IMPL;

    virtual STRF_HD void print_to(strf::destination<CharT>& dst) const = 0;
};

template <typename CharT>
using arg_printer
STRF_DEPRECATED_MSG("arg_printer was renamed to printer")
= printer<CharT>;

namespace detail {

// template <typename CharT>
// class printer_wrapper
// {
//     template <typename Printer>
//     static STRF_HD void invoke(void* printer_ptr, strf::destination<CharT>& dst)
//     {
//         std::move(*reinterpret_cast<Printer*>(printer_ptr)).print_to(dst);
//     }

//     typename void (*invoke_fptr) (void* printer_ptr, strf::destination<CharT>& dst);

// public:

//     template <typename Printer>
//     STRF_HD printer_wrapper(Printer&& p)
//         : invoker_(invoke<strf::remove_cv_ref_t<Printer>>)
//         , printer_ptr_(&p)
//     {
//     }

//     STRF_HD void print_to(strf::destination<CharT>& dst) &&
//     {
//         invoker_(printer_ptr_, dst);
//     }

//     invoke_fptr invoker_;
//     void* printer_ptr_
// };





template <typename CharT, typename Printer>
class printer_wrapper: public printer<CharT>
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

    STRF_HD void print_to(strf::destination<CharT>& dst) const
    {
        printer_.print_to(dst);
    }

private:
    Printer printer_;
};

#if defined(__cpp_fold_expressions)

template <typename CharT, typename... Printers>
inline STRF_HD void write_args( strf::destination<CharT>& dst
                              , const Printers&... printers )
{
    (... , printers.print_to(dst));
}

#else // defined(__cpp_fold_expressions)

template <typename CharT>
inline STRF_HD void write_args(strf::destination<CharT>&)
{
}

template <typename CharT, typename Printer, typename... Printers>
inline STRF_HD void write_args
    ( strf::destination<CharT>& dst
    , const Printer& printer0
    , const Printers&... printers )
{
    printer0.print_to(dst);
    if (dst.good()) {
        write_args<CharT>(dst, printers...);
    }
}

#endif // defined(__cpp_fold_expressions)

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

#endif  // STRF_DETAIL_PRINTER_HPP

