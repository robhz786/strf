#ifndef STRF_DETAIL_INPUT_TYPES_BOOL_HPP
#define STRF_DETAIL_INPUT_TYPES_BOOL_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/facets_pack.hpp>
#include <strf/printer.hpp>
#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/facets/lettercase.hpp>

namespace strf {
namespace detail {

template <typename CharT> class bool_printer;
template <typename CharT> class fmt_bool_printer;

} // namespace detail

template <>
struct print_traits<bool>
{
    using override_tag = bool;
    using forwarded_type = bool;
    using formatters = strf::tag<strf::alignment_formatter>;

    template <typename CharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& fp, bool x ) noexcept
        -> strf::usual_printer_input
            < CharT, Preview, FPack, bool, strf::detail::bool_printer<CharT> >
    {
        return {preview, fp, x};
    }

    template <typename CharT, typename Preview, typename FPack, typename... T>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& fp
        , strf::value_with_formatters<T...> x ) noexcept
        -> strf::usual_printer_input
            < CharT, Preview, FPack
            , strf::value_with_formatters<T...>
            , strf::detail::fmt_bool_printer<CharT> >
    {
        return {preview, fp, x};
    }
};

constexpr STRF_HD strf::print_traits<bool>
tag_invoke(strf::print_traits_tag, bool) noexcept { return {}; }

namespace detail {

template <typename CharT>
class bool_printer: public printer<CharT>
{
public:

    template <typename... T>
    constexpr STRF_HD bool_printer
        ( const strf::usual_printer_input<T...>& input )
        : value_(input.arg)
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.facets))
    {
        input.preview.subtract_width(5u - (int)input.arg);
        input.preview.add_size(5 - (int)input.arg);
    }

    void STRF_HD print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    bool value_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void STRF_HD bool_printer<CharT>::print_to(strf::basic_outbuff<CharT>& ob) const
{
    auto size = 5 - (int)value_;
    ob.ensure(size);
    auto p = ob.pointer();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<CharT>('T' | mask_first_char);
        p[1] = static_cast<CharT>('R' | mask_others_chars);
        p[2] = static_cast<CharT>('U' | mask_others_chars);
        p[3] = static_cast<CharT>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<CharT>('F' | mask_first_char);
        p[1] = static_cast<CharT>('A' | mask_others_chars);
        p[2] = static_cast<CharT>('L' | mask_others_chars);
        p[3] = static_cast<CharT>('S' | mask_others_chars);
        p[4] = static_cast<CharT>('E' | mask_others_chars);
    }
    ob.advance(size);
}

template <typename CharT>
class fmt_bool_printer: public printer<CharT>
{
    using this_type_ = fmt_bool_printer<CharT>;

public:

    template <typename... T>
    STRF_HD fmt_bool_printer
        ( const strf::usual_printer_input<CharT, T...>& input )
        : value_(input.arg.value())
        , afmt_(input.arg.get_alignment_format())
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.facets))
    {
        auto enc = strf::get_facet<charset_c<CharT>, bool>(input.facets);
        std::uint16_t w = 5 - (int)input.arg.value();
        auto fmt_width = afmt_.width.round();
        if (fmt_width > w) {
            encode_fill_ = enc.encode_fill_func();
            fillcount_ = static_cast<std::uint16_t>(fmt_width - w);
            input.preview.subtract_width(fmt_width);
            input.preview.add_size(w + fillcount_ * enc.encoded_char_size(afmt_.fill));
        } else {
            fillcount_ = 0;
            input.preview.subtract_width(w);
            input.preview.add_size(w);
        }
    }

    void STRF_HD print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_;
    std::uint16_t fillcount_;
    bool value_;
    strf::alignment_format afmt_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void fmt_bool_printer<CharT>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    decltype(fillcount_) right_fillcount = 0;
    if (fillcount_ > 0) {
        decltype(fillcount_) left_fillcount;
        switch (afmt_.alignment) {
            case strf::text_alignment::left:
                right_fillcount = fillcount_;
                goto print_value;
            case strf::text_alignment::right:
                left_fillcount = fillcount_;
                break;
            default:
                left_fillcount = fillcount_ >> 1;
                right_fillcount = fillcount_ - left_fillcount;
        }
        encode_fill_(ob, left_fillcount, afmt_.fill);
    }
    print_value:
    auto size = 5 - (int)value_;
    ob.ensure(size);
    auto p = ob.pointer();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<CharT>('T' | mask_first_char);
        p[1] = static_cast<CharT>('R' | mask_others_chars);
        p[2] = static_cast<CharT>('U' | mask_others_chars);
        p[3] = static_cast<CharT>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<CharT>('F' | mask_first_char);
        p[1] = static_cast<CharT>('A' | mask_others_chars);
        p[2] = static_cast<CharT>('L' | mask_others_chars);
        p[3] = static_cast<CharT>('S' | mask_others_chars);
        p[4] = static_cast<CharT>('E' | mask_others_chars);
    }
    ob.advance(size);

    if (right_fillcount != 0) {
        encode_fill_(ob, right_fillcount, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class bool_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class  fmt_bool_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class bool_printer<char>;
STRF_EXPLICIT_TEMPLATE class bool_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class bool_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class bool_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_BOOL_HPP

