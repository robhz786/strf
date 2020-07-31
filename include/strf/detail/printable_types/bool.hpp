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
    using facet_tag = bool;
    using forwarded_type = bool;
    using fmt_type = strf::value_with_format
        < strf::print_traits<bool>
        , strf::alignment_format >;

    template <typename CharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_input
        ( bool x, Preview& preview, const FPack& fp ) noexcept
        -> strf::usual_printer_input
            < CharT, bool, Preview, FPack, strf::detail::bool_printer<CharT> >
    {
        return {x, preview, fp};
    }

    template <typename CharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_input
        ( fmt_type x, Preview& preview, const FPack& fp ) noexcept
        -> strf::usual_printer_input
            < CharT, fmt_type, Preview, FPack, strf::detail::fmt_bool_printer<CharT> >
    {
        return {x, preview, fp};
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
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.fp))
    {
        input.preview.subtract_width(5 - (int)input.arg);
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
        , afmt_(input.arg.get_alignment_format_data())
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.fp))
    {
        auto enc = strf::get_facet<char_encoding_c<CharT>, bool>(input.fp);
        std::uint16_t w = 5 - (int)input.arg.value();
        if (afmt_.width > w) {
            encode_fill_ = enc.encode_fill_func();
            fillcount_ = static_cast<std::uint16_t>(afmt_.width - w);
            input.preview.subtract_width(afmt_.width);
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
    strf::alignment_format_data afmt_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void fmt_bool_printer<CharT>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    if (afmt_.alignment != strf::text_alignment::left) {
        std::uint16_t s = afmt_.alignment == strf::text_alignment::center;
        std::uint16_t count = fillcount_ >> s;
        encode_fill_(ob, count, afmt_.fill);
    }

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

    if ( afmt_.alignment == strf::text_alignment::left) {
        encode_fill_(ob, fillcount_, afmt_.fill);
    }
    else if ( afmt_.alignment == strf::text_alignment::center) {
        std::uint16_t half_count = fillcount_ >> 1;
        std::uint16_t count = fillcount_ - half_count;
        encode_fill_(ob, count, afmt_.fill);
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

