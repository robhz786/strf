#ifndef STRF_JOIN_HPP
#define STRF_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/printers_tuple.hpp>
#include <strf/detail/format_functions.hpp>

#if defined(_MSC_VER)
#include <tuple>
#endif

namespace strf {

template<bool Active>
struct split_pos_format;

template<bool Active, typename T>
class split_pos_format_fn;

template<typename T>
class split_pos_format_fn<true, T> {
public:

    constexpr STRF_HD split_pos_format_fn() noexcept
    {
    }

    constexpr STRF_HD explicit split_pos_format_fn(std::ptrdiff_t pos) noexcept
        : pos_(pos)
    {
    }

    template <bool B, typename U>
    constexpr STRF_HD explicit split_pos_format_fn
        ( const split_pos_format_fn<B,U>& r ) noexcept
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
class split_pos_format_fn<false, T>
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
    constexpr STRF_HD explicit split_pos_format_fn(const strf::split_pos_format_fn<false, U>&) noexcept
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

template<bool Active>
struct split_pos_format
{
    template<typename T>
    using fn = strf::split_pos_format_fn<Active, T>;
};

struct aligned_join_t
{
    std::int16_t width = 0;
    strf::text_alignment align = strf::text_alignment::right;
    char32_t fillchar = U' ';
    std::ptrdiff_t split_pos = 1;

    template<typename ... Args>
    constexpr STRF_HD strf::value_with_format
        < strf::detail::simple_tuple< strf::detail::opt_val_or_cref<Args>...>
                                    , strf::split_pos_format<true>
                                    , strf::alignment_format_q<true> >
    operator()(const Args& ... args) const
    {
        return { strf::detail::make_simple_tuple<Args...>(args...)
               , strf::tag< strf::split_pos_format<true>
               , strf::alignment_format_q<true> > {}
               , split_pos
               , strf::alignment_format_data {fillchar, width, align}};
    }
};

namespace detail {

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class join_printer;

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class aligned_join_printer;

template< typename CharT, typename FPack, typename Preview
        , bool HasSplitPos, bool HasAlignment, typename ... Args>
class join_printer_input
{
public:
    using printer_type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_join_printer<CharT, FPack, Preview, Args ...>
        , strf::detail::join_printer<CharT, FPack, Preview, Args ...> >;

    FPack fp;
    Preview& preview;
    strf::value_with_format
        < strf::detail::simple_tuple<Args...>
        , strf::split_pos_format<HasSplitPos>
        , strf::alignment_format_q<HasAlignment> > arg;
};

} // namespace detail

template< typename CharT, typename FPack, typename Preview
        , bool HasSplitPos, bool HasAlignment, typename ... Args>
struct printable_traits
    < CharT, FPack, Preview
    , strf::value_with_format
        < strf::detail::simple_tuple<Args...>
        , strf::split_pos_format<HasSplitPos>
        , strf::alignment_format_q<HasAlignment> > >
{
    template<typename Arg>
    constexpr static STRF_HD strf::detail::join_printer_input
        < CharT, FPack, Preview, HasSplitPos, HasAlignment, Args... >
    make_input(const FPack& fp, Preview& preview, const Arg& arg)
    {
        return {fp, preview, arg};
    }
};

namespace detail {

template<std::size_t CharSize>
STRF_HD void print_split
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_f<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli)
{
    (void) split_pos;
    encode_fill(ob, fillcount, fillchar, inv_seq_poli, surr_poli);
}

template<std::size_t CharSize, typename Printer, typename ... Printers>
STRF_HD void print_split
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_f<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli
    , const Printer& p
    , const Printers& ... printers )
{
    if (split_pos > 0) {
        p.print_to(ob);
        print_split( ob, encode_fill, fillcount, fillchar, split_pos - 1
                   , inv_seq_poli, surr_poli, printers... );
    } else {
        encode_fill(ob, fillcount, fillchar, inv_seq_poli, surr_poli);
        strf::detail::write_args(ob, p, printers...);
    }
}

template<std::size_t CharSize, std::size_t ... I, typename ... Printers>
STRF_HD void print_split
    ( const strf::detail::printers_tuple_impl
        < CharSize, std::index_sequence<I...>, Printers... >& printers
    , strf::underlying_outbuf<CharSize>& ob
    , strf::encode_fill_f<CharSize> encode_fill
    , unsigned fillcount
    , char32_t fillchar
    , std::ptrdiff_t split_pos
    , strf::invalid_seq_policy inv_seq_poli
    , strf::surrogate_policy surr_poli )
{
    strf::detail::print_split( ob, encode_fill, fillcount, fillchar, split_pos, inv_seq_poli
                             , surr_poli, printers.template get<I>()... );
}

template<std::size_t CharSize, typename ... Printers>
class aligned_join_printer_impl: public printer<CharSize>
{
    using printers_tuple_ = strf::detail::printers_tuple<CharSize, Printers...>;

public:

    template < typename FPack, strf::preview_size ReqSize, bool HasSplitPos
             , typename CharT, typename ... Args >
    STRF_HD aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , FPack
              , strf::print_preview<ReqSize, strf::preview_width::no>
              , HasSplitPos
              , true, Args ...>& input )
        : split_pos_(input.arg.split_pos())
        , afmt_(input.arg.get_alignment_format_data())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c>(input.fp))
    {
        auto enc = get_facet_<strf::char_encoding_c<CharT>>(input.fp);
        encode_fill_func_ = enc.encode_fill_func();
        strf::print_preview<ReqSize, strf::preview_width::yes> p { afmt_.width };
        new (printers_ptr_()) printers_tuple_
            { input.fp, p, input.arg.value(), strf::tag<CharT>{} };
        if (p.remaining_width() > 0) {
            fillcount_ = p.remaining_width().round();
        }
        STRF_IF_CONSTEXPR (static_cast<bool>(ReqSize)) {
            input.preview.add_size(p.get_size());
            if (fillcount_ > 0) {
                auto fcharsize = enc.encoded_char_size(afmt_.fill);
                input.preview.add_size(fillcount_ * fcharsize);
            }
        }
    }

    template < typename FPack, strf::preview_size ReqSize, bool HasSplitPos
             , typename CharT, typename ... Args >
    STRF_HD aligned_join_printer_impl
        ( const strf::detail::join_printer_input
              < CharT
              , FPack
              , strf::print_preview<ReqSize, strf::preview_width::yes>
              , HasSplitPos
              , true, Args ...>& input )
        : split_pos_(input.arg.split_pos())
        , afmt_(input.arg.get_alignment_format_data())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c>(input.fp))
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
        strf::print_preview<ReqSize, strf::preview_width::yes> p{wmax};
        // to-do: what if the line below throws ?
        new (printers_ptr_()) printers_tuple_
            { input.fp, p, input.arg.value(), strf::tag<CharT>() };
        if (p.remaining_width() > diff) {
            fillcount_ = (p.remaining_width() - diff).round();
        }
        width_t width = fillcount_ + wmax - p.remaining_width();
        input.preview.subtract_width(width);
        STRF_IF_CONSTEXPR (static_cast<bool>(ReqSize)) {
            input.preview.add_size(p.get_size());
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

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override
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
    <sizeof(std::tuple<Printers...>), alignof(strf::printer<CharSize>)>;
#else
    <sizeof(printers_tuple_), alignof(printers_tuple_)>;
#endif
    printers_tuple_storage_ pool_;
    std::ptrdiff_t split_pos_;
    strf::alignment_format_data afmt_;
    strf::encode_fill_f<CharSize> encode_fill_func_;
    strf::width_t width_;
    std::int16_t fillcount_ = 0;
    strf::invalid_seq_policy inv_seq_poli_;
    strf::surrogate_policy surr_poli_;

    STRF_HD printers_tuple_ * printers_ptr_()
    {
        return reinterpret_cast<printers_tuple_*>(&pool_);
    }
    STRF_HD const printers_tuple_& printers_() const
    {
        return *reinterpret_cast<const printers_tuple_*>(&pool_);
    }

    template <typename Category, typename FPack>
    static decltype(auto) STRF_HD get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, strf::aligned_join_t>();
    }

    STRF_HD void write_fill_(strf::underlying_outbuf<CharSize>& ob, int count) const
    {
        encode_fill_func_(ob, count, afmt_.fill, inv_seq_poli_, surr_poli_);
    }

    STRF_HD void print_split_(strf::underlying_outbuf<CharSize>& ob) const;
};

template<std::size_t CharSize, typename ... Printers>
STRF_HD void aligned_join_printer_impl<CharSize, Printers...>::print_split_
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    strf::detail::print_split( printers_(), ob, encode_fill_func_, fillcount_
                             , afmt_.fill, split_pos_, inv_seq_poli_, surr_poli_ );
}

template<typename CharT, typename FPack, typename Preview, typename ... Args>
using aligned_join_printer_impl_of = strf::detail::aligned_join_printer_impl
    < sizeof(CharT)
    , strf::printer_impl
        < CharT
        , FPack
        , strf::print_preview
            < static_cast<strf::preview_size>(Preview::size_required)
            , strf::preview_width::yes >
        , Args >... >;

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class aligned_join_printer
    : public strf::detail::aligned_join_printer_impl_of
        < CharT, FPack, Preview, Args... >
{
public:

    template <typename FPack2, bool HasSplitPos>
    constexpr STRF_HD aligned_join_printer
        ( const strf::detail::join_printer_input
            < CharT, FPack2, Preview, HasSplitPos, true, Args... >& input )
        : strf::detail::aligned_join_printer_impl_of
            < CharT, FPack, Preview, Args... > (input)
    {
    }

    virtual STRF_HD ~aligned_join_printer()
    {
    }
};

template<std::size_t CharSize, typename ... Printers>
class join_printer_impl: public printer<CharSize> {
public:

    template<typename FPack, typename Preview, typename CharT, typename ... Args>
    STRF_HD join_printer_impl
        ( const FPack& fp
        , Preview& preview
        , const strf::detail::simple_tuple<Args...>& args
        , strf::tag<CharT> tag_char )
        : printers_{fp, preview, args, tag_char}
    {
    }

    STRF_HD ~join_printer_impl()
    {
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override
    {
        strf::detail::write(ob, printers_);
    }

private:

    strf::detail::printers_tuple<CharSize, Printers...> printers_;
};

template<typename CharT, typename FPack, typename Preview, typename ... Args>
class join_printer
    : public strf::detail::join_printer_impl
        < sizeof(CharT), strf::printer_impl<CharT, FPack, Preview, Args> ... >
{
public:

    template <typename FPack2, bool HasSplitPos>
    STRF_HD join_printer
        ( const strf::detail::join_printer_input
              < CharT, FPack2, Preview, HasSplitPos, false, Args... >& input )
        : strf::detail::join_printer_impl
            < sizeof(CharT), strf::printer_impl<CharT, FPack, Preview, Args>... >
            ( input.fp, input.preview, input.arg.value(), strf::tag<CharT>() )
    {
    }

    virtual STRF_HD ~join_printer()
    {
    }
};

} // namespace detail

template<typename ... Args>
constexpr STRF_HD strf::value_with_format
    < strf::detail::simple_tuple<strf::detail::opt_val_or_cref<Args>...>
    , strf::split_pos_format<false>
    , strf::alignment_format_q<false> >
join(const Args& ... args)
{
    return strf::value_with_format
        < strf::detail::simple_tuple<strf::detail::opt_val_or_cref<Args>...>
        , strf::split_pos_format<false>
        , strf::alignment_format_q<false> >
        { strf::detail::make_simple_tuple(args...) };
}

constexpr STRF_HD strf::aligned_join_t join_align
    ( std::int16_t width
    , strf::text_alignment align
    , char32_t fillchar = U' '
    , int split_pos = 0 )
{
    return {width, align, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_t join_center(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::center, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_left(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::left, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_right(std::int16_t width, char32_t fillchar = U' ') noexcept
{
    return {width, strf::text_alignment::right, fillchar, 0};
}

constexpr STRF_HD strf::aligned_join_t join_split(std::int16_t width, char32_t fillchar,
    std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, fillchar, split_pos};
}

constexpr STRF_HD strf::aligned_join_t join_split(std::int16_t width, std::ptrdiff_t split_pos) noexcept
{
    return {width, strf::text_alignment::split, U' ', split_pos};
}

} // namespace strf

#endif  // STRF_JOIN_HPP

