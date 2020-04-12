#ifndef STRF_DETAIL_INPUT_TYPES_BOOL_HPP
#define STRF_DETAIL_INPUT_TYPES_BOOL_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/facets_pack.hpp>
#include <strf/printer.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/charset.hpp>

namespace strf {

namespace detail {

template <std::size_t CharSize>
class bool_printer: public printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD bool_printer(const FPack&, Preview& preview, bool value, strf::tag<CharT>)
        : value_(value)
    {
        preview.subtract_width(5 - (int)value);
        preview.add_size(5 - (int)value);
    }

    void STRF_HD print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    bool value_;
};


template <std::size_t CharSize>
void STRF_HD bool_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    auto size = 5 - (int)value_;
    ob.ensure(size);
    auto p = ob.pointer();
    if (value_) {
        p[0] = static_cast<char_type>('t');
        p[1] = static_cast<char_type>('r');
        p[2] = static_cast<char_type>('u');
        p[3] = static_cast<char_type>('e');
    } else {
        p[0] = static_cast<char_type>('f');
        p[1] = static_cast<char_type>('a');
        p[2] = static_cast<char_type>('l');
        p[3] = static_cast<char_type>('s');
        p[4] = static_cast<char_type>('e');
    }
    ob.advance(size);
}

template <std::size_t CharSize>
class fmt_bool_printer: public printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD fmt_bool_printer
        ( FPack& fp
        , Preview& preview
        , strf::value_with_format<bool, strf::alignment_format> input
        , strf::tag<CharT> )
        : value_(input.value())
        , inv_seq_poli_(strf::get_facet<strf::invalid_seq_policy_c, bool>(fp))
        , surr_poli_(get_facet<strf::surrogate_policy_c, bool>(fp))
        , afmt_(input.get_alignment_format_data())
    {
        decltype(auto) cs = strf::get_facet<charset_c<CharT>, bool>(fp);
        std::uint16_t w = 5 - (int)input.value();
        if (afmt_.width > w) {
            encode_fill_ = cs.encode_fill_func();
            fillcount_ = static_cast<std::uint16_t>(afmt_.width - w);
            preview.subtract_width(afmt_.width);
            preview.add_size(w + fillcount_ * cs.encoded_char_size(afmt_.fill));
        } else {
            fillcount_ = 0;
            preview.subtract_width(w);
            preview.add_size(w);
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
    if (value_) {
        p[0] = static_cast<char_type>('t');
        p[1] = static_cast<char_type>('r');
        p[2] = static_cast<char_type>('u');
        p[3] = static_cast<char_type>('e');
    } else {
        p[0] = static_cast<char_type>('f');
        p[1] = static_cast<char_type>('a');
        p[2] = static_cast<char_type>('l');
        p[3] = static_cast<char_type>('s');
        p[4] = static_cast<char_type>('e');
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

inline STRF_HD auto make_fmt(strf::rank<1>, bool b)
{
    return strf::value_with_format<bool, strf::alignment_format>(b);
}

template < typename CharT, typename FPack, typename Preview >
inline STRF_HD strf::detail::bool_printer<sizeof(CharT)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, bool b)
{
    return {fp, preview, b, strf::tag<CharT>{}};
}

template < typename CharT, typename FPack, typename Preview >
inline STRF_HD strf::detail::fmt_bool_printer<sizeof(CharT)> make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format<bool, strf::alignment_format> x )
{
    return {fp, preview, x, strf::tag<CharT>{}};
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_BOOL_HPP

