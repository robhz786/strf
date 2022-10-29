#ifndef STRF_JOIN_HPP
#define STRF_JOIN_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/charset.hpp>

#if defined(_MSC_VER)
#include <tuple>
#endif

namespace strf {

namespace detail {

template <typename CharT, typename PrePrinting, typename FPack, typename... FwdArgs>
class join_printer;

template <typename CharT, typename PrePrinting, typename FPack, typename... FwdArgs>
class aligned_join_printer;

template <typename... FwdArgs>
struct join_printing;

template <typename... FwdArgs>
struct join_t
{
    strf::detail::simple_tuple<FwdArgs...> args;
};

template< typename CharT, typename PrePrinting, typename FPack
        , bool HasAlignment, typename... FwdArgs>
class join_printer_input
{
public:
    using printer_type = strf::detail::conditional_t
        < HasAlignment
        , strf::detail::aligned_join_printer<CharT, PrePrinting, FPack, FwdArgs...>
        , strf::detail::join_printer<CharT, PrePrinting, FPack, FwdArgs...> >;

    PrePrinting& pre;
    FPack facets;
    strf::printable_with_fmt
        < strf::detail::join_printing<FwdArgs...>
        , strf::alignment_formatter_q<HasAlignment> > arg;
};

template <typename... FwdArgs>
struct join_printing
{
    using forwarded_type = strf::detail::join_t<FwdArgs...>;

    template <bool HasAlignment>
    using fmt_tmpl = strf::printable_with_fmt
        < join_printing<FwdArgs...>
        , strf::alignment_formatter_q<HasAlignment> >;

    using formatters = strf::tag<strf::alignment_formatter>;

    template< typename CharT, typename PrePrinting, typename FPack, bool HasAlignment >
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& facets
        , fmt_tmpl<HasAlignment> x )
        -> join_printer_input
            < CharT, PrePrinting, FPack, HasAlignment, FwdArgs... >
    {
        return {pre, facets, x};
    }
};

} // namespace detail

template <typename... FwdArgs>
struct printable_traits<strf::detail::join_t<FwdArgs...>>
    : strf::detail::join_printing<FwdArgs...>
{
};

struct aligned_join_maker
{
    constexpr STRF_HD explicit aligned_join_maker
        ( strf::width_t width_
        , strf::text_alignment align_ = strf::text_alignment::right ) noexcept
        : width(width_)
        , align(align_)
    {
    }

    template <typename CharT>
    constexpr STRF_HD aligned_join_maker
        ( strf::width_t width_
        , strf::text_alignment align_
        , CharT fillchar_ ) noexcept
        : width(width_)
        , align(align_)
        , fillchar(fillchar_)
    {
        static_assert( strf::is_char<CharT>::value // issue 19
                     , "Refusing non-char argument to set the fill character, "
                       "since one may pass 0 instead of '0' by accident." );
    }

    strf::width_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';

    template<typename... Args>
    constexpr STRF_HD strf::printable_with_fmt
        < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
        , strf::alignment_formatter_q<true> >
    operator()(const Args&... args) const
    {
        return { { strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                     { strf::detail::simple_tuple_from_args{}
                     , static_cast<strf::forwarded_printable_type<Args>>(args)... } }
               , strf::tag< strf::alignment_formatter_q<true> > {}
               , strf::alignment_format {fillchar, width, align}};
    }
};

namespace detail {

template<typename CharT, typename... Printers>
class aligned_join_printer_impl: public printer<CharT>
{
    using printers_tuple_ = strf::detail::printers_tuple<CharT, Printers...>;

public:

    template <strf::precalc_size PrecalcSize, typename FPack, typename... FwdArgs>
    STRF_HD explicit aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , strf::preprinting<PrecalcSize, strf::precalc_width::no>
              , FPack
              , true, FwdArgs...>& input )
        : afmt_(input.arg.get_alignment_format())
    {
        auto charset = use_facet_<strf::charset_c<CharT>>(input.facets);
        encode_fill_func_ = charset.encode_fill_func();
        strf::preprinting<PrecalcSize, strf::precalc_width::yes> pre { afmt_.width };
        new (printers_ptr_()) printers_tuple_{input.arg.value().args, pre, input.facets};
        fillcount_ = pre.remaining_width().round();
        STRF_IF_CONSTEXPR (static_cast<bool>(PrecalcSize)) {
            input.pre.add_size(pre.accumulated_size());
            if (fillcount_ > 0) {
                auto fcharsize = charset.encoded_char_size(afmt_.fill);
                input.pre.add_size(fillcount_ * fcharsize);
            }
        }
    }

    template <strf::precalc_size PrecalcSize, typename FPack, typename... FwdArgs>
    STRF_HD explicit aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , strf::preprinting<PrecalcSize, strf::precalc_width::yes>
              , FPack
              , true, FwdArgs...>& input )
        : afmt_(input.arg.get_alignment_format())
    {
        auto charset = use_facet_<strf::charset_c<CharT>>(input.facets);
        encode_fill_func_ = charset.encode_fill_func();
        strf::width_t wmax = afmt_.width;
        strf::width_t diff = 0;
        if (input.pre.remaining_width() > afmt_.width) {
            wmax = input.pre.remaining_width();
            diff = input.pre.remaining_width() - afmt_.width;
        }
        strf::preprinting<PrecalcSize, strf::precalc_width::yes> pre{wmax};
        // to-do: what if the line below throws ?
        new (printers_ptr_()) printers_tuple_{input.arg.value().args, pre, input.facets};
        if (pre.remaining_width() > diff) {
            fillcount_ = (pre.remaining_width() - diff).round();
        }
        width_t width = fillcount_ + wmax - pre.remaining_width();
        input.pre.subtract_width(width);
        STRF_IF_CONSTEXPR (static_cast<bool>(PrecalcSize)) {
            input.pre.add_size(pre.accumulated_size());
            if (fillcount_ > 0) {
                auto fcharsize = charset.encoded_char_size(afmt_.fill);
                input.pre.add_size( fillcount_ * fcharsize);
            }
        }
    }
    aligned_join_printer_impl(const aligned_join_printer_impl&) = delete;
    aligned_join_printer_impl(aligned_join_printer_impl&&) = delete;
    aligned_join_printer_impl& operator=(const aligned_join_printer_impl&) = delete;
    aligned_join_printer_impl& operator=(aligned_join_printer_impl&&) = delete;

    STRF_HD ~aligned_join_printer_impl()
    {
        printers_ptr_()->~printers_tuple_();
    }

    STRF_HD void print_to(strf::destination<CharT>& dest) const override
    {
        switch (afmt_.alignment) {
            case strf::text_alignment::left: {
                strf::detail::write(dest, printers_());
                write_fill_(dest, fillcount_);
                break;
            }
            case strf::text_alignment::right: {
                write_fill_(dest, fillcount_);
                strf::detail::write(dest, printers_());
                break;
            }
            default: {
                STRF_ASSERT(afmt_.alignment == strf::text_alignment::center);
                auto half_fillcount = fillcount_ >> 1;
                write_fill_(dest, half_fillcount);
                strf::detail::write(dest, printers_());
                write_fill_(dest, fillcount_ - half_fillcount);
                break;
            }
        }
    }

private:

    using printers_tuple_storage_ = typename std::aligned_storage
#if defined(_MSC_VER)
        <sizeof(std::tuple<Printers...>), alignof(strf::printer<CharT>)>
#else
        <sizeof(printers_tuple_), alignof(printers_tuple_)>
#endif
        :: type;
    printers_tuple_storage_ pool_;
    strf::alignment_format afmt_;
    strf::encode_fill_f<CharT> encode_fill_func_;
    strf::width_t width_;
    std::uint16_t fillcount_ = 0;

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD printers_tuple_ * printers_ptr_()
    {
        return reinterpret_cast<printers_tuple_*>(&pool_);
    }
    STRF_HD const printers_tuple_& printers_() const
    {
        return *reinterpret_cast<const printers_tuple_*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

    template < typename Category, typename FPack
             , typename Tag = strf::aligned_join_maker>
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }

    STRF_HD void write_fill_(strf::destination<CharT>& dest, int count) const
    {
        encode_fill_func_(dest, count, afmt_.fill);
    }
};

template <typename CharT, typename PrePrinting, typename FPack, typename Arg>
struct print_impl_with_width_precalc_;

template < typename CharT, strf::precalc_size PrecalcSize, strf::precalc_width PrecalcWidth
         , typename FPack, typename Arg >
struct print_impl_with_width_precalc_<CharT, preprinting<PrecalcSize, PrecalcWidth>, FPack, Arg>
{
    using type = strf::printer_type
        < CharT, strf::preprinting<PrecalcSize, strf::precalc_width::yes>, FPack, Arg >;
};

template<typename CharT, typename PrePrinting, typename FPack, typename... Args>
using aligned_join_printer_impl_of = strf::detail::aligned_join_printer_impl
    < CharT
    , typename print_impl_with_width_precalc_<CharT, PrePrinting, FPack, Args>::type... >;

template<typename CharT, typename PrePrinting, typename FPack, typename... FwdArgs>
class aligned_join_printer
    : public strf::detail::aligned_join_printer_impl_of
        < CharT, PrePrinting, FPack, FwdArgs... >
{
public:

    template <typename FPack2>
    constexpr STRF_HD explicit aligned_join_printer
        ( const strf::detail::join_printer_input
            < CharT, PrePrinting, FPack2, true, FwdArgs... >& input )
        : strf::detail::aligned_join_printer_impl_of
            < CharT, PrePrinting, FPack, FwdArgs... > (input)
    {
    }
};

template<typename CharT, typename... Printers>
class join_printer_impl: public printer<CharT> {
public:

    template<typename PrePrinting, typename FPack, typename... FwdArgs>
    STRF_HD join_printer_impl
        ( const strf::detail::simple_tuple<FwdArgs...>& args
        , PrePrinting& pre
        , const FPack& facets )
        : printers_{args, pre, facets}
    {
    }

    STRF_HD void print_to(strf::destination<CharT>& dest) const override
    {
        strf::detail::write(dest, printers_);
    }

private:

    strf::detail::printers_tuple<CharT, Printers...> printers_;
};

template <typename CharT, typename PrePrinting, typename FPack, typename... FwdArgs>
class join_printer
    : public strf::detail::join_printer_impl
        < CharT, strf::printer_type<CharT, PrePrinting, FPack, FwdArgs>... >
{
public:

    template <typename FPack2>
    STRF_HD explicit join_printer
        ( const strf::detail::join_printer_input
              < CharT, PrePrinting, FPack2, false, FwdArgs... >& input )
        : strf::detail::join_printer_impl
            < CharT, strf::printer_type<CharT, PrePrinting, FPack, FwdArgs>... >
            ( input.arg.value().args, input.pre, input.facets )
    {
    }
};

} // namespace detail

template<typename... Args>
constexpr STRF_HD strf::printable_with_fmt
    < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
    , strf::alignment_formatter_q<false> >
join(const Args&... args)
{
    return strf::printable_with_fmt
        < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
        , strf::alignment_formatter_q<false> >
        { strf::detail::join_t<strf::forwarded_printable_type<Args>...>
            { strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                { strf::detail::simple_tuple_from_args{}
                , static_cast<strf::forwarded_printable_type<Args>>(args)... } } };
}

constexpr STRF_HD strf::aligned_join_maker join_align
    ( strf::width_t width
    , strf::text_alignment align ) noexcept
{
    return {width, align, U' '};
}

template <typename CharT>
constexpr STRF_HD strf::aligned_join_maker join_align
    ( strf::width_t width
    , strf::text_alignment align
    , CharT fillchar ) noexcept
{
    return {width, align, fillchar};
}

constexpr STRF_HD strf::aligned_join_maker join_center(strf::width_t width) noexcept
{
    return {width, strf::text_alignment::center, U' '};
}
template <typename CharT>
constexpr STRF_HD strf::aligned_join_maker join_center(strf::width_t width, CharT fillchar) noexcept
{
    return {width, strf::text_alignment::center, fillchar};
}
constexpr STRF_HD strf::aligned_join_maker join_left(strf::width_t width) noexcept
{
    return {width, strf::text_alignment::left, U' '};
}
template <typename CharT>
constexpr STRF_HD strf::aligned_join_maker join_left(strf::width_t width, CharT fillchar) noexcept
{
    return {width, strf::text_alignment::left, fillchar};
}
constexpr STRF_HD strf::aligned_join_maker join_right(strf::width_t width) noexcept
{
    return {width, strf::text_alignment::right, U' '};
}
template <typename CharT>
constexpr STRF_HD strf::aligned_join_maker join_right(strf::width_t width, CharT fillchar) noexcept
{
    return {width, strf::text_alignment::right, fillchar};
}

} // namespace strf

#endif  // STRF_JOIN_HPP

