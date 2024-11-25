#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>

namespace strf {

struct width_calculator_c;

template <typename CharT>
struct width_and_ptr {
    strf::width_t width;
    const CharT* ptr = nullptr;
};

class fast_width_t final
{
public:
    using category = width_calculator_c;

    template <typename Charset>
    constexpr STRF_HD strf::width_t char_width
        ( Charset
        , typename Charset::code_unit ) const noexcept
    {
        return 1;
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t str_width
        ( Charset
        , strf::width_t limit
        , const typename Charset::code_unit* begin
        , const typename Charset::code_unit* end ) const noexcept
    {
        STRF_ASSERT(begin <= end);
        const auto str_len = detail::safe_cast_size_t(end - begin);
        const auto nn_limit = limit >= 0 ? limit : 0;
        const auto nn_limit_floor = static_cast<unsigned> (nn_limit.underlying_value() >> 16);

        return ( str_len <= nn_limit_floor
               ? static_cast<strf::width_t>(str_len)
               : nn_limit );
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD auto str_width_and_pos
        ( Charset
        , strf::width_t limit
        , const typename Charset::code_unit* begin
        , const typename Charset::code_unit* end ) const noexcept
        -> strf::width_and_ptr<typename Charset::code_unit>
    {
        STRF_ASSERT(begin <= end);
        const auto str_len = detail::safe_cast_size_t(end - begin);
        const auto nn_limit_floor = limit.non_negative_floor();

        if (str_len <= nn_limit_floor) {
            return { static_cast<strf::width_t>(str_len), end };
        }
        return { static_cast<strf::width_t>(nn_limit_floor), begin + nn_limit_floor };
    }
};

class width_as_fast_u32len_t final
{
public:
    using category = width_calculator_c;

    template <typename Charset>
    constexpr STRF_HD strf::width_t char_width
        ( Charset
        , typename Charset::code_unit ) const noexcept
    {
        return 1;
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t str_width
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end ) const
    {
        STRF_ASSERT(str <= str_end);
        auto lim = limit.non_negative_ceil();
        auto res = charset.count_codepoints_fast(str, str_end, lim);
        STRF_ASSERT(res.count <= strf::width_max.ceil());
        return strf::width_t::sat_cast(res.count);
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD auto str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end ) const
        -> strf::width_and_ptr<typename Charset::code_unit>
    {
        STRF_ASSERT(str <= str_end);
        auto lim = limit.non_negative_floor();
        auto res = charset.count_codepoints_fast(str, str_end, lim);
        STRF_ASSERT(res.count <= lim);
        return {res.count, res.ptr};
    }
};


class width_as_u32len_t final
{
public:
    using category = width_calculator_c;

    template <typename Charset>
    constexpr STRF_HD strf::width_t char_width
        ( Charset
        , typename Charset::code_unit ) const noexcept
    {
        return 1;
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t str_width
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end ) const
    {
        STRF_ASSERT(str <= str_end);
        auto lim = limit.non_negative_ceil();
        auto res = charset.count_codepoints(str, str_end, lim);
        STRF_ASSERT(res.count <= strf::width_max.floor());
        return strf::width_t::sat_cast(res.count);
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD auto str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end ) const
        -> strf::width_and_ptr<typename Charset::code_unit>
    {
        STRF_ASSERT(str <= str_end);
        auto lim = limit.non_negative_floor();
        auto res = charset.count_codepoints(str, str_end, lim);
        STRF_ASSERT(static_cast<std::ptrdiff_t>(res.count) <= lim);
        return {res.count, res.ptr};
    }
};

namespace detail {

struct std_width_calc_func_return {
    STRF_HD std_width_calc_func_return
        ( strf::width_t remaining_width_
        , unsigned state_
        , const char32_t* ptr_ ) noexcept
        : remaining_width(remaining_width_)
        , state(state_)
        , ptr(ptr_)
    {
    }

    strf::width_t remaining_width;
    unsigned state;
    const char32_t* ptr;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers,google-readability-function-size,hicpp-function-size)
STRF_FUNC_IMPL STRF_HD std_width_calc_func_return std_width_calc_func
    ( const char32_t* str
    , const char32_t* end
    , strf::width_t width
    , unsigned state
    , bool return_pos ) noexcept
{
    // following http://www.unicode.org/reports/tr29/tr29-37.html#Grapheme_Cluster_Boundaries

    using namespace strf::width_literal;

    using state_t = unsigned;
    constexpr state_t initial          = 0;
    constexpr state_t after_prepend    = 1;
    constexpr state_t after_core       = 1 << 1;
    constexpr state_t after_ri         = after_core | (1 << 2);
    constexpr state_t after_xpic       = after_core | (1 << 3);
    constexpr state_t after_xpic_zwj   = after_core | (1 << 4);
    constexpr state_t after_hangul     = after_core | (1 << 5);
    constexpr state_t after_hangul_l   = after_hangul | (1 << 6);
    constexpr state_t after_hangul_v   = after_hangul | (1 << 7);
    constexpr state_t after_hangul_t   = after_hangul | (1 << 8);
    constexpr state_t after_hangul_lv  = after_hangul | (1 << 9);
    constexpr state_t after_hangul_lvt = after_hangul | (1 << 10);
    constexpr state_t after_poscore    = 1 << 11;
    constexpr state_t after_cr         = 1 << 12;

    strf::width_t ch_width;
    char32_t ch = 0;
    goto next_codepoint;

    handle_other:
    if (state == after_prepend) {
        state = after_core;
        goto next_codepoint;
    }
    state = after_core;

    decrement_width:
    // should come here after the first codepoint of every grapheme cluster
    if (ch_width >= width) {
        if (! return_pos) {
            return {0, 0, nullptr};
        }
        if (ch_width > width) {
            return {0, 0, str -1};
        }
        width = 0;
        goto next_codepoint; // because there might be more codepoints in this grapheme cluster
    }
    width -= ch_width;

    next_codepoint:
    if (str == end) {
        return {width, state, str};
    }
    ch = *str;
    ++str;
    ch_width = 1_w;
    if (ch <= 0x007E) {
        if (0x20 <= ch) {
            ch_width = 1;
            goto handle_other;
        }
        if (0x000D == ch) { // CR
            goto handle_cr;
        }
        if (0x000A == ch) { // LF
            goto handle_lf;
        }
        goto handle_control;
    }

#include <strf/detail/ch32_width_and_gcb_prop>

    handle_zwj:
    if (state == after_xpic) {
        state = after_xpic_zwj;
        goto next_codepoint;
    }
    goto handle_spacing_mark; // because the code is the same

    handle_extend:
    handle_extend_and_control:
    if (state == after_xpic) {
        goto next_codepoint;
    }

    handle_spacing_mark:
    if (state & (after_prepend | after_core | after_poscore)) {
        state = after_poscore;
        goto next_codepoint;
    }
    state = after_poscore;
    goto decrement_width;

    handle_prepend:
    if (state == after_prepend) {
        goto next_codepoint;
    }
    state = after_prepend;
    goto decrement_width;

    handle_regional_indicator: {
        if (state == after_ri) {
            state = after_core;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_ri;
            goto decrement_width;
        }
        state = after_ri;
        goto next_codepoint;
    }
    handle_extended_picto: {
        if (state == after_xpic_zwj) {
            state = after_xpic;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_xpic;
            goto decrement_width;
        }
        state = after_xpic;
        goto next_codepoint;
    }
    handle_hangul_l: {
        if (state == after_hangul_l) {
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_hangul_l;
            goto decrement_width;
        }
        state = after_hangul_l;
        goto next_codepoint;
    }
    handle_hangul_v: {
        constexpr state_t mask = ~after_hangul &
            (after_hangul_l | after_hangul_v | after_hangul_lv);
        if (state & mask) {
            state = after_hangul_v;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_hangul_v;
            goto decrement_width;
        }
        state = after_hangul_v;
        goto next_codepoint;
    }
    handle_hangul_t: {
        constexpr state_t mask = ~after_hangul &
            (after_hangul_v | after_hangul_lv | after_hangul_lvt | after_hangul_t);
        if (state & mask) {
            state = after_hangul_t;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_hangul_t;
            goto decrement_width;
        }
        state = after_hangul_t;
        goto next_codepoint;
    }
    handle_hangul_lv_or_lvt:
    if ( ch <= 0xD788 // && 0xAC00 <= ch
         && 0 == (ch & 3)
         && 0 == ((ch - 0xAC00) >> 2) % 7)
    {   // LV
        if (state == after_hangul_l) {
            state = after_hangul_lv;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_hangul_lv;
            goto decrement_width;
        }
        state = after_hangul_lv;
        goto next_codepoint;

    } else { // LVT
        if (state == after_hangul_l) {
            state = after_hangul_lvt;
            goto next_codepoint;
        }
        if (state != after_prepend) {
            state = after_hangul_lvt;
            goto decrement_width;
        }
        state = after_hangul_lvt;
        goto next_codepoint;
    }

    handle_cr:
    state = after_cr;
    goto decrement_width;

    handle_lf:
    if (state == after_cr) {
        state = initial;
        goto next_codepoint;
    }
    handle_control:
    state = initial;
    goto decrement_width;
}

#else

STRF_HD std_width_calc_func_return std_width_calc_func
    ( const char32_t* begin
    , const char32_t* end
    , strf::width_t width
    , unsigned state
    , bool return_pos );

#endif // ! defined(STRF_OMIT_IMPL)

#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

} // namespace detail

class std_width_calc_t
{
public:
    using category = strf::width_calculator_c;

    static STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t char32_width(char32_t ch) noexcept
    {
        using namespace strf::width_literal;
        if (ch <= 0x115F)
            return ch < 0x1100 ? 1_w : 2_w;
        if (ch <= 0xFF60) {
            if (ch <= 0xD7A3) {
                if (ch <= 0x303E) {
                    if (ch <= 0x232A)
                        return (ch < 0x2329) ? 1_w : 2_w;
                    return (ch < 0x2E80) ? 1_w : 2_w;
                }
                if (ch <= 0xA4CF)
                    return (ch < 0x3040) ? 1_w : 2_w;
                return (ch < 0xAC00) ? 1_w : 2_w;
            }
            if (ch <= 0xFE19) {
                if (ch <= 0xFAFF)
                    return (ch < 0xF900) ? 1_w : 2_w;
                return (ch < 0xFE10) ? 1_w : 2_w;
            }
            if (ch <= 0xFE6F)
                return (ch < 0xFE30) ? 1_w : 2_w;
            return (ch < 0xFF00) ? 1_w : 2_w;
        }
        if (ch <= 0x3FFFD) {
            if (ch <= 0x1F9FF) {
                if (ch <= 0x1F64F) {
                    if (ch <= 0xFFE6)
                        return (ch < 0xFFE0) ? 1_w : 2_w;
                    return (ch < 0x1F300) ? 1_w : 2_w;
                }
                return (ch < 0x1F900) ? 1_w : 2_w;
            }
            if (ch <= 0x2FFFD)
                return (ch < 0x20000) ? 1_w : 2_w;
            return (ch < 0x30000) ? 1_w : 2_w;
        }
        return 1_w;
    }

    template < typename Charset
             , typename CodeUnit = typename Charset::code_unit
             , strf::detail::enable_if_t<sizeof(CodeUnit) == 4, int> = 0 >
    static constexpr STRF_HD strf::width_t char_width
        ( Charset
        , typename Charset::code_unit ch )
    {
        return char32_width(static_cast<char32_t>(ch));
    }

    template < typename Charset
             , typename CodeUnit = typename Charset::code_unit
             , strf::detail::enable_if_t<sizeof(CodeUnit) != 4, int> = 0 >
    static constexpr STRF_HD strf::width_t char_width
        ( Charset charset
        , typename Charset::code_unit ch )
    {
        return char32_width(charset.decode_unit(ch));
    }

    template <typename Charset>
    static STRF_HD strf::width_t str_width
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end )
    {
        str_end = str <= str_end ? str_end : str;
        const auto buff_size = 32;
        char32_t buff[buff_size];
        constexpr auto flags = strf::transcode_flags::none;
        const auto to_u32 = charset.to_u32();
        limit = limit <= 0 ? 0 : limit;
        detail::std_width_calc_func_return res{limit, 0, nullptr};
        while(1) {
            auto res_tr = to_u32.transcode(str, str_end, buff, buff + buff_size, nullptr, flags);
            str = res_tr.src_ptr;
            res = detail::std_width_calc_func
                ( buff, res_tr.dst_ptr, res.remaining_width, res.state, false );

            if ( res.remaining_width <= 0 ||
                 res_tr.stop_reason != transcode_stop_reason::insufficient_output_space)
            {
                return limit - res.remaining_width;
            }
        }
    }

    template <typename Charset>
    static STRF_HD auto str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , const typename Charset::code_unit* str_end )
        -> strf::width_and_ptr<typename Charset::code_unit>
    {
        str_end = str <= str_end ? str_end : str;
        const auto buff32_size = 32;
        char32_t buff32[buff32_size];
        constexpr auto flags = strf::transcode_flags::none;
        const auto to_u32 = charset.to_u32();
        detail::std_width_calc_func_return resw{limit, 0, nullptr};
        while(1) {
            auto res_tr = to_u32.transcode(str, str_end, buff32, buff32 + buff32_size, nullptr, flags);
            resw = detail::std_width_calc_func
                ( buff32, res_tr.dst_ptr, resw.remaining_width, resw.state, true );

            if ( resw.remaining_width <= 0 ||
                 res_tr.stop_reason != transcode_stop_reason::insufficient_output_space)
            {
                const auto width = limit - resw.remaining_width;
                if (resw.ptr != res_tr.dst_ptr) {
                    const auto u32dist = resw.ptr - buff32;
                    auto res_sz = to_u32.transcode_size(str, str_end, u32dist, flags);
                    return {width, res_sz.src_ptr};
                }
                return {width, res_tr.src_ptr};
            }
            str = res_tr.src_ptr;
        }
    }
};

#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic pop
#endif

#if !defined(__CUDACC__) || (__CUDA_VER_MAJOR__ >= 11 && __CUDA_VER_MINOR__ >= 3)

STRF_HD constexpr fast_width_t fast_width = fast_width_t{};
STRF_HD constexpr width_as_fast_u32len_t width_as_fast_u32len = width_as_fast_u32len_t{};
STRF_HD constexpr width_as_u32len_t width_as_u32len = width_as_u32len_t{};
STRF_HD constexpr std_width_calc_t std_width_calc = std_width_calc_t{};

#endif

struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static constexpr STRF_HD strf::std_width_calc_t get_default() noexcept
    {
        return {};
    }
};

} // namespace strf

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

