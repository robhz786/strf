#ifndef STRF_DETAIL_INPUT_TYPES_BOOL_HPP
#define STRF_DETAIL_INPUT_TYPES_BOOL_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/facets_pack.hpp>
#include <strf/printer.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/char_encoding.hpp>
#include <strf/detail/facets/lettercase.hpp>

namespace strf {
namespace detail {

template <std::size_t CharSize> class bool_printer;
template <std::size_t CharSize> class fmt_bool_printer;

} // namespace detail

inline STRF_HD auto make_fmt(strf::rank<1>, bool b)
{
    return strf::value_with_format<bool, strf::alignment_format>(b);
}

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, bool>
    : strf::usual_printable_traits
        < CharT, FPack, strf::detail::bool_printer<sizeof(CharT)> >
{
};

template <typename CharT, typename FPack, typename Preview>
struct printable_traits
    < CharT, FPack, Preview
    , strf::value_with_format<bool, strf::alignment_format> >
    : strf::usual_printable_traits
        < CharT, FPack, strf::detail::fmt_bool_printer<sizeof(CharT)> >
{
};

namespace detail {

template <std::size_t CharSize>
class bool_printer: public printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename... T>
    constexpr STRF_HD bool_printer
        ( const strf::usual_printer_input<T...>& input )
        : value_(input.arg)
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.fp))
    {
        input.preview.subtract_width(5 - (int)input.arg);
        input.preview.add_size(5 - (int)input.arg);
    }

    void STRF_HD print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    bool value_;
    strf::lettercase lettercase_;
};

template <std::size_t CharSize>
void STRF_HD bool_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    auto size = 5 - (int)value_;
    ob.ensure(size);
    auto p = ob.pointer();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<char_type>('T' | mask_first_char);
        p[1] = static_cast<char_type>('R' | mask_others_chars);
        p[2] = static_cast<char_type>('U' | mask_others_chars);
        p[3] = static_cast<char_type>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<char_type>('F' | mask_first_char);
        p[1] = static_cast<char_type>('A' | mask_others_chars);
        p[2] = static_cast<char_type>('L' | mask_others_chars);
        p[3] = static_cast<char_type>('S' | mask_others_chars);
        p[4] = static_cast<char_type>('E' | mask_others_chars);
    }
    ob.advance(size);
}

template <std::size_t CharSize>
class fmt_bool_printer: public printer<CharSize>
{
    using arg_type_ = strf::value_with_format<bool, strf::alignment_format>;
    using this_type_ = fmt_bool_printer<CharSize>;

public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename... T>
    STRF_HD fmt_bool_printer
        ( const strf::usual_printer_input<CharT, T...>& input )
        : value_(input.arg.value())
        , inv_seq_poli_(strf::get_facet<strf::invalid_seq_policy_c, bool>(input.fp))
        , surr_poli_(get_facet<strf::surrogate_policy_c, bool>(input.fp))
        , afmt_(input.arg.get_alignment_format_data())
        , lettercase_(strf::get_facet<strf::lettercase_c, bool>(input.fp))
    {
        decltype(auto) enc = strf::get_facet<char_encoding_c<CharT>, bool>(input.fp);
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

    void STRF_HD print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::encode_fill_f<CharSize> encode_fill_;
    std::uint16_t fillcount_;
    bool value_;
    strf::invalid_seq_policy inv_seq_poli_;
    strf::surrogate_policy surr_poli_;
    strf::alignment_format_data afmt_;
    strf::lettercase lettercase_;
};

template <std::size_t CharSize>
void fmt_bool_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (afmt_.alignment != strf::text_alignment::left) {
        std::uint16_t s = afmt_.alignment == strf::text_alignment::center;
        std::uint16_t count = fillcount_ >> s;
        encode_fill_(ob, count, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }

    auto size = 5 - (int)value_;
    ob.ensure(size);
    auto p = ob.pointer();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<char_type>('T' | mask_first_char);
        p[1] = static_cast<char_type>('R' | mask_others_chars);
        p[2] = static_cast<char_type>('U' | mask_others_chars);
        p[3] = static_cast<char_type>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<char_type>('F' | mask_first_char);
        p[1] = static_cast<char_type>('A' | mask_others_chars);
        p[2] = static_cast<char_type>('L' | mask_others_chars);
        p[3] = static_cast<char_type>('S' | mask_others_chars);
        p[4] = static_cast<char_type>('E' | mask_others_chars);
    }
    ob.advance(size);

    if ( afmt_.alignment == strf::text_alignment::left) {
        encode_fill_(ob, fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
    else if ( afmt_.alignment == strf::text_alignment::center) {
        std::uint16_t half_count = fillcount_ >> 1;
        std::uint16_t count = fillcount_ - half_count;
        encode_fill_(ob, count, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
}

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_BOOL_HPP

