#ifndef STRF_JOIN_HPP
#define STRF_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/printers_tuple.hpp>

#if defined(_MSC_VER)
#include <tuple>
#endif

namespace strf {

template<bool HasSplitPos>
struct split_pos_format;

template<typename T, bool HasSplitPos>
class split_pos_format_fn;

template<typename T>
class split_pos_format_fn<T, true> {
public:

    constexpr STRF_HD split_pos_format_fn() noexcept
    {
    }

    constexpr STRF_HD explicit split_pos_format_fn(std::ptrdiff_t pos) noexcept
        : pos_(pos)
    {
    }

    template <typename U, bool B>
    constexpr STRF_HD explicit split_pos_format_fn
        ( const split_pos_format_fn<U, B>& r ) noexcept
        : pos_(r.split_pos())
    {
    }

    constexpr STRF_HD T&& split_pos(std::ptrdiff_t pos) && noexcept
    {
        pos_ = pos;
        return static_cast<T&&>(*this);
    }

    constexpr STRF_HD std::ptrdiff_t split_pos() const noexcept
    {
        return pos_;
    }

private:

    std::ptrdiff_t pos_ = 0;
};

template<typename T>
class split_pos_format_fn<T, false>
{
    using adapted_derived_type_ = strf::fmt_replace
        < T
        , strf::split_pos_format<false>
        , strf::split_pos_format<true> >;
public:

    constexpr STRF_HD split_pos_format_fn() noexcept
    {
    }

    template<typename U>
    constexpr STRF_HD explicit split_pos_format_fn(const strf::split_pos_format_fn<U, false>&) noexcept
    {
    }

    constexpr STRF_HD adapted_derived_type_ split_pos(std::ptrdiff_t pos) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<strf::split_pos_format<true>> {}
               , pos};
    }

    constexpr STRF_HD std::ptrdiff_t split_pos() const noexcept
    {
        return 0;
    }
};

template<bool HasSplitPos>
struct split_pos_format
{
    template<typename T>
    using fn = strf::split_pos_format_fn<T, HasSplitPos>;
};

namespace detail {

template <typename CharT, typename Preview, typename FPack, typename... FwdArgs>
class join_printer;

template <typename CharT, typename Preview, typename FPack, typename... FwdArgs>
class aligned_join_printer;

template <typename... FwdArgs>
struct join_printing;

template <typename... FwdArgs>
struct join_t
{
    strf::detail::simple_tuple<FwdArgs...> args;
};

template< typename CharT, typename Preview, typename FPack
        , bool HasSplitPos, bool HasAlignment, typename... FwdArgs>
class join_printer_input
{
public:
    using printer_type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_join_printer<CharT, Preview, FPack, FwdArgs...>
        , strf::detail::join_printer<CharT, Preview, FPack, FwdArgs...> >;

    Preview& preview;
    FPack fp;
    strf::value_with_format
        < strf::detail::join_printing<FwdArgs...>
        , strf::split_pos_format<HasSplitPos>
        , strf::alignment_format_q<HasAlignment> > arg;
};

template <typename... FwdArgs>
struct join_printing
{
    using facet_tag = void;
    using forwarded_type = strf::detail::join_t<FwdArgs...>;

    template <bool HasSplitPos, bool HasAlignment>
    using fmt_tmpl = strf::value_with_format
        < join_printing<FwdArgs...>
        , strf::split_pos_format<HasSplitPos>
        , strf::alignment_format_q<HasAlignment> >;

    using fmt_type = fmt_tmpl<false, false>;

    template< typename CharT, typename Preview, typename FPack
            , bool HasSplitPos, bool HasAlignment >
    STRF_HD constexpr static auto make_printer_input
        ( Preview& preview, const FPack& fp, fmt_tmpl<HasSplitPos, HasAlignment> x)
        -> join_printer_input
            < CharT, Preview, FPack, HasSplitPos, HasAlignment, FwdArgs... >
    {
        return {preview, fp, x};
    }
};

} // namespace detail

template <typename... FwdArgs>
struct print_traits<strf::detail::join_t<FwdArgs...>>
    : strf::detail::join_printing<FwdArgs...>
{
};

struct aligned_join_maker
{
    std::int16_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';
    std::ptrdiff_t split_pos = 1;

    template<typename... Args>
    constexpr STRF_HD strf::value_with_format
        < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
        , strf::split_pos_format<true>
        , strf::alignment_format_q<true> >
    operator()(const Args&... args) const
    {
        return { { strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                     { strf::detail::simple_tuple_from_args{}
                     , static_cast<strf::forwarded_printable_type<Args>>(args)... } }
               , strf::tag< strf::split_pos_format<true>
                          , strf::alignment_format_q<true> > {}
               , split_pos
               , strf::alignment_format_data {fillchar, width, align}};
    }
};

namespace detail {

template<typename CharT>
STRF_HD void print_split
    ( strf::basic_outbuff<CharT>& ob
    , strf::encode_fill_f<CharT> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos )
{
    (void) split_pos;
    encode_fill(ob, fillcount, fillchar);
}

template<typename CharT, typename Printer, typename... Printers>
STRF_HD void print_split
    ( strf::basic_outbuff<CharT>& ob
    , strf::encode_fill_f<CharT> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , const Printer& first_printer
    , const Printers&... printers )
{
    if (split_pos > 0) {
        first_printer.print_to(ob);
        print_split
            ( ob, encode_fill, fillcount, fillchar, split_pos - 1, printers... );
    } else {
        encode_fill(ob, fillcount, fillchar);
        strf::detail::write_args(ob, first_printer, printers...);
    }
}

template<typename CharT, std::size_t... I, typename... Printers>
STRF_HD void print_split
    ( const strf::detail::printers_tuple_impl
        < CharT, std::index_sequence<I...>, Printers... >& printers
    , strf::basic_outbuff<CharT>& ob
    , strf::encode_fill_f<CharT> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos )
{
    strf::detail::print_split
        ( ob, encode_fill, fillcount, fillchar, split_pos
        , printers.template get<I>()... );
}

template<typename CharT, typename... Printers>
class aligned_join_printer_impl: public printer<CharT>
{
    using printers_tuple_ = strf::detail::printers_tuple<CharT, Printers...>;

public:

    template < strf::preview_size ReqSize, bool HasSplitPos, typename FPack
             , typename... FwdArgs >
    STRF_HD aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , strf::print_preview<ReqSize, strf::preview_width::no>
              , FPack
              , HasSplitPos
              , true, FwdArgs...>& input )
        : split_pos_(input.arg.split_pos())
        , afmt_(input.arg.get_alignment_format_data())
    {
        auto enc = get_facet_<strf::char_encoding_c<CharT>>(input.fp);
        encode_fill_func_ = enc.encode_fill_func();
        strf::print_preview<ReqSize, strf::preview_width::yes> preview { afmt_.width };
        new (printers_ptr_()) printers_tuple_{input.arg.value().args, preview, input.fp};
        if (preview.remaining_width() > 0) {
            fillcount_ = preview.remaining_width().round();
        }
        STRF_IF_CONSTEXPR (static_cast<bool>(ReqSize)) {
            input.preview.add_size(preview.get_size());
            if (fillcount_ > 0) {
                auto fcharsize = enc.encoded_char_size(afmt_.fill);
                input.preview.add_size(fillcount_ * fcharsize);
            }
        }
    }

    template < strf::preview_size ReqSize, bool HasSplitPos, typename FPack
             , typename... FwdArgs >
    STRF_HD aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , strf::print_preview<ReqSize, strf::preview_width::yes>
              , FPack
              , HasSplitPos
              , true, FwdArgs...>& input )
        : split_pos_(input.arg.split_pos())
        , afmt_(input.arg.get_alignment_format_data())
    {
        auto enc = get_facet_<strf::char_encoding_c<CharT>>(input.fp);
        encode_fill_func_ = enc.encode_fill_func();
        if (afmt_.width < 0) {
            afmt_.width = 0;
        }
        strf::width_t wmax = afmt_.width;
        strf::width_t diff = 0;
        if (input.preview.remaining_width() > afmt_.width) {
            wmax = input.preview.remaining_width();
            diff = input.preview.remaining_width() - afmt_.width;
        }
        strf::print_preview<ReqSize, strf::preview_width::yes> preview{wmax};
        // to-do: what if the line below throws ?
        new (printers_ptr_()) printers_tuple_{input.arg.value().args, preview, input.fp};
        if (preview.remaining_width() > diff) {
            fillcount_ = (preview.remaining_width() - diff).round();
        }
        width_t width = fillcount_ + wmax - preview.remaining_width();
        input.preview.subtract_width(width);
        STRF_IF_CONSTEXPR (static_cast<bool>(ReqSize)) {
            input.preview.add_size(preview.get_size());
            if (fillcount_ > 0) {
                auto fcharsize = enc.encoded_char_size(afmt_.fill);
                input.preview.add_size( fillcount_ * fcharsize);
            }
        }
    }

    STRF_HD ~aligned_join_printer_impl()
    {
        printers_ptr_()->~printers_tuple_();
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override
    {
        switch (afmt_.alignment) {
            case strf::text_alignment::left: {
                strf::detail::write(ob, printers_());
                write_fill_(ob, fillcount_);
                break;
            }
            case strf::text_alignment::right: {
                write_fill_(ob, fillcount_);
                strf::detail::write(ob, printers_());
                break;
            }
            case strf::text_alignment::split: {
                print_split_(ob);
                break;
            }
            default: {
                STRF_ASSERT(afmt_.alignment == strf::text_alignment::center);
                auto half_fillcount = fillcount_ >> 1;
                write_fill_(ob, half_fillcount);
                strf::detail::write(ob, printers_());
                write_fill_(ob, fillcount_ - half_fillcount);
                break;
            }
        }
    }

private:

    using printers_tuple_storage_ = typename std::aligned_storage_t
#if defined(_MSC_VER)
    <sizeof(std::tuple<Printers...>), alignof(strf::printer<CharT>)>;
#else
    <sizeof(printers_tuple_), alignof(printers_tuple_)>;
#endif
    printers_tuple_storage_ pool_;
    std::ptrdiff_t split_pos_;
    strf::alignment_format_data afmt_;
    strf::encode_fill_f<CharT> encode_fill_func_;
    strf::width_t width_;
    std::int16_t fillcount_ = 0;

#if defined(__GNUC__) && (__GNUC__ == 6)
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

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic pop
#endif

    template <typename Category, typename FPack>
    static decltype(auto) STRF_HD get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, strf::aligned_join_maker>();
    }

    STRF_HD void write_fill_(strf::basic_outbuff<CharT>& ob, int count) const
    {
        encode_fill_func_(ob, count, afmt_.fill);
    }

    STRF_HD void print_split_(strf::basic_outbuff<CharT>& ob) const;
};

template<typename CharT, typename... Printers>
STRF_HD void aligned_join_printer_impl<CharT, Printers...>::print_split_
    ( strf::basic_outbuff<CharT>& ob ) const
{
    strf::detail::print_split
        ( printers_(), ob, encode_fill_func_, fillcount_, afmt_.fill, split_pos_ );
}

template <typename CharT, typename Preview, typename FPack, typename Arg>
struct print_impl_with_width_preview_;

template < typename CharT, strf::preview_size PrevSize, strf::preview_width PrevWidth
         , typename FPack, typename Arg >
struct print_impl_with_width_preview_<CharT, print_preview<PrevSize, PrevWidth>, FPack, Arg>
{
    using type = strf::printer_type
        < CharT, strf::print_preview <PrevSize, strf::preview_width::yes>, FPack, Arg >;
};

template<typename CharT, typename Preview, typename FPack, typename... Args>
using aligned_join_printer_impl_of = strf::detail::aligned_join_printer_impl
    < CharT
    , typename print_impl_with_width_preview_<CharT, Preview, FPack, Args>::type... >;

template<typename CharT, typename Preview, typename FPack, typename... FwdArgs>
class aligned_join_printer
    : public strf::detail::aligned_join_printer_impl_of
        < CharT, Preview, FPack, FwdArgs... >
{
public:

    template <typename FPack2, bool HasSplitPos>
    constexpr STRF_HD aligned_join_printer
        ( const strf::detail::join_printer_input
            < CharT, Preview, FPack2, HasSplitPos, true, FwdArgs... >& input )
        : strf::detail::aligned_join_printer_impl_of
            < CharT, Preview, FPack, FwdArgs... > (input)
    {
    }

    virtual STRF_HD ~aligned_join_printer()
    {
    }
};

template<typename CharT, typename... Printers>
class join_printer_impl: public printer<CharT> {
public:

    template<typename Preview, typename FPack, typename... FwdArgs>
    STRF_HD join_printer_impl
        ( const strf::detail::simple_tuple<FwdArgs...>& args
        , Preview& preview
        , const FPack& fp )
        : printers_{args, preview, fp}
    {
    }

    STRF_HD ~join_printer_impl()
    {
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override
    {
        strf::detail::write(ob, printers_);
    }

private:

    strf::detail::printers_tuple<CharT, Printers...> printers_;
};

template <typename CharT, typename Preview, typename FPack, typename... FwdArgs>
class join_printer
    : public strf::detail::join_printer_impl
        < CharT, strf::printer_type<CharT, Preview, FPack, FwdArgs>... >
{
public:

    template <typename FPack2, bool HasSplitPos>
    STRF_HD join_printer
        ( const strf::detail::join_printer_input
              < CharT, Preview, FPack2, HasSplitPos, false, FwdArgs... >& input )
        : strf::detail::join_printer_impl
            < CharT, strf::printer_type<CharT, Preview, FPack, FwdArgs>... >
            ( input.arg.value().args, input.preview, input.fp )
    {
    }

    virtual STRF_HD ~join_printer()
    {
    }
};

} // namespace detail

template<typename... Args>
constexpr STRF_HD strf::value_with_format
    < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
    , strf::split_pos_format<false>
    , strf::alignment_format_q<false> >
join(const Args&... args)
{
    return strf::value_with_format
        < strf::detail::join_printing<strf::forwarded_printable_type<Args>...>
        , strf::split_pos_format<false>
        , strf::alignment_format_q<false> >
        { strf::detail::join_t<strf::forwarded_printable_type<Args>...>
            { strf::detail::simple_tuple<strf::forwarded_printable_type<Args>...>
                { strf::detail::simple_tuple_from_args{}
                , static_cast<strf::forwarded_printable_type<Args>>(args)... } } };
}

constexpr STRF_HD strf::aligned_join_maker join_align
    ( std::int16_t width
    , strf::text_alignment align
    , char32_t fillchar = U' '
    , int split_pos = 0 )
{
    return {width, align, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_maker join_center(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::center, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_maker join_left(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::left, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_maker join_right(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::right, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_maker join_split(std::int16_t width, char32_t fillchar,
    std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_maker join_split(std::int16_t width, std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, U' ', split_pos};
}

} // namespace strf

#endif  // STRF_JOIN_HPP

