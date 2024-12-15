#ifndef STRF_DETAIL_PRINTABLE_TYPES_JOIN_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_JOIN_HPP

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

template <typename... PrintablesInfo>
using tuple_of_printables_pi = simple_tuple<typename PrintablesInfo::as_member_type ...>;

template <typename... PrintablesInfo>
struct join_t
{
    STRF_HD join_t(typename PrintablesInfo::forwarded_type... a)
        : args(strf::detail::simple_tuple_from_args{}, a...)
    {
    }

    tuple_of_printables_pi<PrintablesInfo...> args;
};

template <typename... PrintablesInfo>
using join_printable_def = strf::printable_def<join_t<PrintablesInfo...>>;

} // namespace detail

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
        , fillchar(detail::cast_unsigned(fillchar_))
    {
        static_assert( strf::is_char<CharT>::value // issue 19
                     , "Refusing non-char argument to set the fill character, "
                       "since one may pass 0 instead of '0' by accident." );
    }

    strf::width_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';

    template<typename... Args>
    constexpr STRF_HD auto operator()(const Args&... args) const
    {
        using joint_t_ = detail::join_t<detail::get_printable_info<Args>...>;
        using printable_def_ = strf::printable_def<joint_t_>;
        using afmt_t = strf::alignment_format_specifier_q<true>;

        return strf::value_and_format<printable_def_, afmt_t>
            { joint_t_{args... }
            , strf::tag< strf::alignment_format_specifier_q<true> > {}
            , strf::alignment_format {fillchar, width, align} };
    }
};

namespace detail {

template<typename CharT, typename... Printers>
struct aligned_join_printer
{
    using printers_tuple_t_ = strf::detail::printers_tuple<CharT, Printers...>;

    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        if (fillcount_ <= 0) {
            printers_tuple_(dst);
        } else {
            switch (afmt_.alignment) {
                case strf::text_alignment::left: {
                    printers_tuple_(dst);
                    encode_fill_func_(dst, fillcount_, afmt_.fill);
                    break;
                }
                case strf::text_alignment::right: {
                    encode_fill_func_(dst, fillcount_, afmt_.fill);
                    printers_tuple_(dst);
                    break;
                }
                default: {
                    STRF_ASSERT(afmt_.alignment == strf::text_alignment::center);
                    auto half_fillcount = fillcount_ >> 1;
                    encode_fill_func_(dst, half_fillcount, afmt_.fill);
                    printers_tuple_(dst);
                    encode_fill_func_(dst, fillcount_ - half_fillcount, afmt_.fill);
                    break;
                }
            }
        }
    }

    printers_tuple_t_ printers_tuple_;
    strf::alignment_format afmt_;
    strf::encode_fill_f<CharT> encode_fill_func_;
    int fillcount_ = 0;
};

template < typename CharT, strf::size_presence SizePresence, typename FPack
         , typename... PrintableInfos >
using aligned_join_printer_of = strf::detail::aligned_join_printer
    < CharT
    , detail::printer_type_pi
        < CharT
        , strf::premeasurements<SizePresence, strf::width_presence::yes>
        , FPack
        , PrintableInfos > ... >;

template <typename CharT, typename PreMeasurements, typename FPack, typename... PrintablesInfo>
class join_printer;

template <typename CharT, typename FPack, typename... PrintablesInfo>
class join_printer<CharT, strf::no_premeasurements, FPack, PrintablesInfo...>
{
    using tuple_t_ = tuple_of_printables_pi<PrintablesInfo...>;

public:
    STRF_HD join_printer
        ( strf::no_premeasurements*
        , const FPack& facets
        , const tuple_of_printables_pi<PrintablesInfo...>& printables )
        : printables_{printables}
        , facets_(facets)
    {
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        do_print_(dst, printables_);
    }

private:

    template <std::size_t... I>
    STRF_HD void do_print_
        ( strf::destination<CharT>& dst
        , const strf::detail::simple_tuple_impl
              < strf::detail::index_sequence<I...>
              , typename PrintablesInfo::as_member_type... >& args_tuple ) const
    {
        args_printer<PrintablesInfo...>::print(dst, facets_, args_tuple.template get<I>()...);
    }

    tuple_t_ printables_;
    FPack facets_;
};

template <typename CharT, typename PreMeasurements, typename FPack, typename... PrintablesInfo>
class join_printer
{
public:
    STRF_HD join_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , const tuple_of_printables_pi<PrintablesInfo...>& printables )
        : printers_{printables, pre, facets}
    {
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        printers_(dst);
    }

    strf::detail::printers_tuple
        < CharT, detail::printer_type_pi<CharT, PreMeasurements, FPack, PrintablesInfo>... >
        printers_;
};

} // namespace detail

template <typename... PrintablesInfo>
struct printable_def<detail::join_t<PrintablesInfo...>>
{
    using join_t_ = detail::join_t<PrintablesInfo...>;
    using forwarded_type = join_t_;

    template <bool HasAlignment>
    using fmt_tmpl = strf::value_and_format
        < printable_def
        , strf::alignment_format_specifier_q<HasAlignment> >;

    using format_specifiers = strf::tag<strf::alignment_format_specifier>;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& facets
        , fmt_tmpl<false> x )
        -> detail::join_printer<CharT, PreMeasurements, FPack, PrintablesInfo...>
    {
        return {pre, facets, x.value().args};
    }

    template <typename CharT, strf::size_presence SizePresence, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , strf::premeasurements<SizePresence, strf::width_presence::no>* pre
        , const FPack& facets
        , fmt_tmpl<true> arg )
    {
        using sub_pre_t = strf::premeasurements<SizePresence, strf::width_presence::yes>;
        using pf_type = detail::aligned_join_printer_of
            <CharT, SizePresence, FPack, PrintablesInfo...>;

        auto charset = get_facet<strf::charset_c<CharT>, void>(facets);
        const auto afmt = arg.get_alignment_format();
        sub_pre_t sub_pre{afmt.width};
        pf_type pf
            { {arg.value().args, &sub_pre, facets}
            , afmt
            , charset.encode_fill_func() };
        pf.fillcount_ = sub_pre.remaining_width().round();
        STRF_MAYBE_UNUSED(pre);
        STRF_IF_CONSTEXPR (static_cast<bool>(SizePresence)) {
            pre->add_size(sub_pre.accumulated_ssize());
            if (pf.fillcount_ > 0) {
                auto fcharsize = charset.encoded_char_size(afmt.fill);
                pre->add_size(pf.fillcount_ * fcharsize);
            }
        }
        return pf;
    }

    template <typename CharT, strf::size_presence SizePresence, typename FPack>
    STRF_HD constexpr static auto make_printer
        ( strf::tag<CharT>
        , strf::premeasurements<SizePresence, strf::width_presence::yes>* pre
        , const FPack& facets
        , fmt_tmpl<true> arg )
    {
        using sub_pre_t = strf::premeasurements<SizePresence, strf::width_presence::yes>;
        using pf_type = detail::aligned_join_printer_of
            <CharT, SizePresence, FPack, PrintablesInfo...>;

        const auto charset = get_facet<strf::charset_c<CharT>, void>(facets);
        const auto afmt = arg.get_alignment_format();
        strf::width_t wmax = afmt.width;
        strf::width_t wdiff = 0;
        if (pre->remaining_width_greater_than(afmt.width)) {
            wmax = pre->remaining_width();
            wdiff = wmax - afmt.width;
        }
        sub_pre_t sub_pre{wmax};
        pf_type pf
            { {arg.value().args, &sub_pre, facets}
            , afmt
            , charset.encode_fill_func() };
        pf.fillcount_ =
            ( sub_pre.remaining_width_greater_than(wdiff)
            ? (sub_pre.remaining_width() - wdiff).round()
            : 0 );
        pre->add_width
            ( strf::sat_sub(strf::sat_add(wmax, pf.fillcount_), sub_pre.remaining_width()) );
        STRF_IF_CONSTEXPR (static_cast<bool>(SizePresence)) {
            pre->add_size(sub_pre.accumulated_ssize());
            if (pf.fillcount_ > 0) {
                auto fill_char_size = charset.encoded_char_size(afmt.fill);
                pre->add_size(pf.fillcount_ * fill_char_size);
            }
        }
        return pf;
    }

    template <typename FPack, typename CharT>
    STRF_HD static void print
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const strf::detail::join_t<PrintablesInfo...>& j )
    {
        do_print_(dst, fp, j.args);
    }

private:

    template<typename CharT, typename FPack, std::size_t... I>
    STRF_HD static void do_print_
        ( strf::destination<CharT>& dst
        , const FPack& fp
        , const strf::detail::simple_tuple_impl
            < strf::detail::index_sequence<I...>
            , typename PrintablesInfo::as_member_type... >& args_tuple )
    {
        detail::args_printer<PrintablesInfo...>::print(dst, fp, args_tuple.template get<I>()...);
    }
};


template<typename... Args>
constexpr STRF_HD auto join(const Args&... args)
{
    using joint_t_ = detail::join_t<detail::get_printable_info<Args>...>;
    using printable_def_ = strf::printable_def<joint_t_>;
    using afmt_t = strf::alignment_format_specifier_q<false>;

    return strf::value_and_format<printable_def_, afmt_t>{joint_t_{args...}};
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

#endif // STRF_DETAIL_PRINTABLE_TYPES_JOIN_HPP

