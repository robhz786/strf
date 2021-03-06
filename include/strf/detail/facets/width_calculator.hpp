#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/char_encoding.hpp>

namespace strf {

struct width_calculator_c;

struct width_and_pos {
    strf::width_t width;
    std::size_t pos;
};

class fast_width final
{
public:
    using category = width_calculator_c;

    template <typename CharEncoding>
    STRF_HD strf::width_t char_width
        ( CharEncoding
        , typename CharEncoding::char_type ) const noexcept
    {
        return 1;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t str_width
        ( CharEncoding
        , strf::width_t limit
        , const typename CharEncoding::char_type*
        , std::size_t str_len
        , strf::surrogate_policy ) const noexcept
    {
        if (str_len <= (std::size_t) limit.floor()) {
            return static_cast<std::int16_t>(str_len);
        }
        return limit;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
        ( CharEncoding
        , strf::width_t limit
        , const typename CharEncoding::char_type*
        , std::size_t str_len
        , strf::surrogate_policy ) const noexcept
    {
        if (limit > 0) {
            const auto limit_floor = static_cast<std::size_t>(limit.floor());
            if (str_len <= limit_floor) {
                return { static_cast<std::int16_t>(str_len), str_len };
            }
            return { static_cast<std::int16_t>(limit_floor), limit_floor };
        }
        return {0, 0};
    }
};

class width_as_fast_u32len final
{
public:
    using category = width_calculator_c;

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t char_width
        ( CharEncoding
        , typename CharEncoding::char_type ) const noexcept
    {
        return 1;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t str_width
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy ) const
    {
        if (limit > 0) {
            auto lim = limit.floor();
            auto ret = enc.codepoints_fast_count(str, str_len, lim);
            STRF_ASSERT((std::ptrdiff_t)ret.count <= strf::width_max.floor());
            return static_cast<std::int16_t>(ret.count);
        }
        return 0;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy ) const
    {
        if (limit > 0) {
            std::ptrdiff_t lim = limit.floor();
            auto res = enc.codepoints_fast_count(str, str_len, lim);
            STRF_ASSERT((std::ptrdiff_t)res.count <= lim);
            return { static_cast<std::int16_t>(res.count), res.pos };
        }
        return {0, 0};
    }
};


class width_as_u32len final
{
public:
    using category = width_calculator_c;

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t char_width
        ( CharEncoding
        , typename CharEncoding::char_type ) const noexcept
    {
        return 1;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t str_width
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        if (limit > 0) {
            auto lim = limit.floor();
            auto ret = enc.codepoints_robust_count(str, str_len, lim, surr_poli);
            STRF_ASSERT((std::ptrdiff_t)ret.count <= strf::width_max.floor());
            return static_cast<std::int16_t>(ret.count);
        }
        return 0;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        if (limit > 0) {
            std::ptrdiff_t lim = limit.floor();
            auto res = enc.codepoints_robust_count(str, str_len, lim, surr_poli);
            STRF_ASSERT((std::ptrdiff_t)res.count <= lim);
            return { static_cast<std::int16_t>(res.count), res.pos };
        }
        return {0, 0};
    }

};

namespace detail {
template <typename WFunc>
class width_accumulator: public strf::basic_outbuff<char32_t>
{
public:

    STRF_HD width_accumulator(strf::width_t limit, WFunc func)
        : strf::basic_outbuff<char32_t>(buff_, buff_ + buff_size_)
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
    auto end = this->pointer();
    this->set_pointer(buff_);
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

    template <typename CharEncoding>
    strf::width_t STRF_HD char_width
        ( CharEncoding enc
        , typename CharEncoding::char_type ch ) const
    {
        return func_(enc.decode_char(ch));
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_t str_width
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        strf::detail::width_accumulator<CharWidthFunc> acc(limit, func_);
        strf::invalid_seq_notifier inv_seq_notifier{};
        enc.to_u32().transcode(acc, str, str_len, inv_seq_notifier, surr_poli);
        return acc.get_result().width;
    }

    template <typename CharEncoding>
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
        ( CharEncoding enc
        , strf::width_t limit
        , const typename CharEncoding::char_type* str
        , std::size_t str_len
        , strf::surrogate_policy surr_poli ) const
    {
        strf::detail::width_accumulator<CharWidthFunc> acc(limit, func_);
        strf::invalid_seq_notifier inv_seq_notifier{};
        enc.to_u32().transcode(acc, str, str_len, inv_seq_notifier, surr_poli);
        auto res = acc.get_result();
        if (res.whole_string_covered) {
            return {res.width, str_len};
        }
        auto res2 = enc.codepoints_robust_count
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


struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static constexpr STRF_HD strf::fast_width get_default() noexcept
    {
        return {};
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace strf

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

