#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/charset.hpp>

namespace strf {

struct width_calculator_c;

struct width_and_pos {
    strf::width_t width;
    std::size_t pos;
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
    constexpr STRF_HD strf::width_t str_width
        ( Charset
        , strf::width_t limit
        , const typename Charset::code_unit*
        , std::size_t str_len
        , strf::surrogate_policy ) const noexcept
    {
        return ( str_len <= limit.floor()
               ? static_cast<std::uint16_t>(str_len)
               : limit );
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_and_pos str_width_and_pos
        ( Charset
        , strf::width_t limit
        , const typename Charset::code_unit*
        , std::size_t str_len
        , strf::surrogate_policy ) const noexcept
    {
        const auto limit_floor = static_cast<std::size_t>(limit.floor());
        if (str_len <= limit_floor) {
            return { static_cast<std::uint16_t>(str_len), str_len };
        }
        return { static_cast<std::uint16_t>(limit_floor), limit_floor };
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
        , std::size_t str_len
        , strf::surrogate_policy ) const
    {
        auto lim = limit.floor();
        auto ret = charset.codepoints_fast_count(str, str_len, lim);
        STRF_ASSERT(ret.count <= strf::width_max.floor());
        return static_cast<std::uint16_t>(ret.count);
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_and_pos str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , std::size_t str_len
        , strf::surrogate_policy ) const
    {
        auto lim = limit.floor();
        auto res = charset.codepoints_fast_count(str, str_len, lim);
        STRF_ASSERT(res.count <= lim);
        return { static_cast<std::uint16_t>(res.count), res.pos };
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
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        auto lim = limit.floor();
        auto ret = charset.codepoints_robust_count(str, str_len, lim, surr_poli);
        STRF_ASSERT(ret.count <= strf::width_max.floor());
        return static_cast<std::uint16_t>(ret.count);
    }

    template <typename Charset>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_and_pos str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        auto lim = limit.floor();
        auto res = charset.codepoints_robust_count(str, str_len, lim, surr_poli);
        STRF_ASSERT(res.count <= lim);
        return { static_cast<std::uint16_t>(res.count), res.pos };
    }

};

namespace detail {
template <typename WFunc>
class width_accumulator: public strf::transcode_dest<char32_t>
{
public:

    STRF_HD width_accumulator(strf::width_t limit, WFunc func)
        : strf::transcode_dest<char32_t>(buff_, buff_ + buff_size_)
        , limit_(limit)
        , func_(func)
    {
    }

    STRF_HD void recycle() override;

    struct result
    {
        strf::width_t width;
        bool whole_string_covered;
        std::size_t codepoints_count;
    };

    result STRF_HD get_result()
    {
        recycle();
        this->set_good(false);
        return {width_, whole_string_covered_, codepoints_count_};
    }

private:

    bool whole_string_covered_ = true;
    constexpr static std::size_t buff_size_ = 16;
    char32_t buff_[buff_size_];
    const strf::width_t limit_;
    strf::width_t width_ = 0;
    std::size_t codepoints_count_ = 0;
    WFunc func_;
};

template <typename WFunc>
void STRF_HD width_accumulator<WFunc>::recycle()
{
    auto end = this->buffer_ptr();
    this->set_buffer_ptr(buff_);
    if (this->good()) {
        auto it = buff_;
        for (; it != end; ++it)
        {
            auto w = width_ + func_(*it);
            if (w > limit_) {
                this->set_good(false);
                whole_string_covered_ = false;
                break;
            }
            width_ = w;
        }
        codepoints_count_ += (it - buff_);
    }
}

} // namespace detail


template <typename CharWidthFunc>
class width_by_func
{
public:
    using category = strf::width_calculator_c;

    width_by_func() = default;

    explicit STRF_HD width_by_func(CharWidthFunc f)
        : func_(f)
    {
    }

    template <typename Charset>
    strf::width_t STRF_HD char_width
        ( Charset charset
        , typename Charset::code_unit ch ) const
    {
        return func_(charset.decode_unit(ch));
    }

    template <typename Charset>
    STRF_HD strf::width_t str_width
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        strf::detail::width_accumulator<CharWidthFunc> acc(limit, func_);
        strf::invalid_seq_notifier inv_seq_notifier{};
        charset.to_u32().transcode(acc, str, str_len, inv_seq_notifier, surr_poli);
        return acc.get_result().width;
    }

    template <typename Charset>
    STRF_HD strf::width_and_pos str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        strf::detail::width_accumulator<CharWidthFunc> acc(limit, func_);
        strf::invalid_seq_notifier inv_seq_notifier{};
        charset.to_u32().transcode(acc, str, str_len, inv_seq_notifier, surr_poli);
        auto res = acc.get_result();
        if (res.whole_string_covered) {
            return {res.width, str_len};
        }
        auto res2 = charset.codepoints_robust_count
            (str, str_len, res.codepoints_count, surr_poli);
        return {res.width, res2.pos};
    }

private:

    CharWidthFunc func_;
};


template <typename CharWidthFunc>
width_by_func<CharWidthFunc> STRF_HD make_width_calculator(CharWidthFunc f)
{
    return width_by_func<CharWidthFunc>{f};
}

namespace detail {

struct std_width_calc_func_return {
    STRF_HD std_width_calc_func_return
        ( strf::width_t width_
        , unsigned state_
        , const char32_t* ptr_ ) noexcept
        : width(width_)
        , state(state_)
        , ptr(ptr_)
    {
    }
    std_width_calc_func_return(const std_width_calc_func_return&) = default;

    strf::width_t width;
    unsigned state;
    const char32_t* ptr;
};

#if ! defined(STRF_OMIT_IMPL)

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
    char32_t ch;
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

class std_width_decrementer: public strf::transcode_dest<char32_t> {
public:
    STRF_HD std_width_decrementer (strf::width_t initial_width)
        : strf::transcode_dest<char32_t>(buff_, buff_size_)
        , width_{initial_width}
    {
        this->set_good(initial_width != 0);
    }

    STRF_HD void recycle() noexcept override {
        if (this->good()) {
            auto res = detail::std_width_calc_func(buff_, this->buffer_ptr(), width_, state_, false);
            width_ = res.width;
            state_ = res.state;
            if (width_ == 0) {
                this->set_good(false);
            }
        }
        this->set_buffer_ptr(buff_);
    }

    STRF_HD strf::width_t get_remaining_width() {
        if (width_ != 0 && this->buffer_ptr() != buff_) {
            auto res = detail::std_width_calc_func(buff_, this->buffer_ptr(), width_, state_, false);
            return res.width;
        }
        return width_;
    }

private:
    strf::width_t width_;
    unsigned state_ = 0;
    static constexpr std::size_t buff_size_ = 16;
    char32_t buff_[buff_size_];
};

class std_width_decrementer_with_pos: public strf::transcode_dest<char32_t> {
public:
    STRF_HD std_width_decrementer_with_pos (strf::width_t initial_width)
        : strf::transcode_dest<char32_t>(buff_, buff_size_)
        , width_{initial_width}
    {
        this->set_good(initial_width != 0);
    }

    STRF_HD void recycle() noexcept override {
        if (this->good()) {
            auto res = detail::std_width_calc_func(buff_, this->buffer_ptr(), width_, state_, true);
            width_ = res.width;
            state_ = res.state;
            codepoints_count_ += (res.ptr - buff_);
            if (width_ == 0 && res.ptr != this->buffer_ptr()) {
                this->set_good(false);
            }
        }
        this->set_buffer_ptr(buff_);
    }

    struct result {
        strf::width_t remaining_width;
        bool whole_string_covered;
        std::size_t codepoints_count;
    };

    STRF_HD result get_remaining_width_and_codepoints_count() {
        if (! this->good()) {
            return {0, false, codepoints_count_};
        }
        auto res = detail::std_width_calc_func(buff_, this->buffer_ptr(), width_, state_, true);
        width_ = res.width;
        codepoints_count_ += (res.ptr - buff_);
        bool whole_string_covered = (res.ptr == this->buffer_ptr());
        return {width_, whole_string_covered, codepoints_count_};
    }

private:
    strf::width_t width_;
    unsigned state_ = 0;
    std::size_t codepoints_count_ = 0;
    static constexpr std::size_t buff_size_ = 16;
    char32_t buff_[buff_size_];
};

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
        , std::size_t str_len
        , strf::surrogate_policy surr_poli )
    {
        strf::detail::std_width_decrementer decr{limit};
        strf::invalid_seq_notifier inv_seq_notifier{};
        charset.to_u32().transcode(decr, str, str_len, inv_seq_notifier, surr_poli);
        return (limit - decr.get_remaining_width());
    }

    template <typename Charset>
    static STRF_HD strf::width_and_pos str_width_and_pos
        ( Charset charset
        , strf::width_t limit
        , const typename Charset::code_unit* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli )
    {
        strf::detail::std_width_decrementer_with_pos decr{limit};
        strf::invalid_seq_notifier inv_seq_notifier{};
        charset.to_u32().transcode(decr, str, str_len, inv_seq_notifier, surr_poli);
        auto res = decr.get_remaining_width_and_codepoints_count();

        strf::width_t width = limit - res.remaining_width;
        if (res.whole_string_covered) {
            return {width, str_len};
        }
        auto res2 = charset.codepoints_robust_count
            (str, str_len, res.codepoints_count, surr_poli);
        return {width, res2.pos};
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

