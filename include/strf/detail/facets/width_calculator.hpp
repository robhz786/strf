#ifndef STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP
#define STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

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
    STRF_HD strf::width_t char_width
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
        if (str_len <= limit.floor()) {
            return static_cast<std::uint16_t>(str_len);
        }
        return limit;
    }

    template <typename Charset>
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
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
    constexpr STRF_HD strf::width_t str_width
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
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
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
    constexpr STRF_HD strf::width_t str_width
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
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
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

    template <typename Charset>
    strf::width_t STRF_HD char_width
        ( Charset charset
        , typename Charset::code_unit ch ) const
    {
        return func_(charset.decode_char(ch));
    }

    template <typename Charset>
    constexpr STRF_HD strf::width_t str_width
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
    constexpr STRF_HD strf::width_and_pos str_width_and_pos
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

STRF_HD constexpr fast_width_t fast_width = fast_width_t{};
STRF_HD constexpr width_as_fast_u32len_t width_as_fast_u32len = width_as_fast_u32len_t{};
STRF_HD constexpr width_as_u32len_t width_as_u32len = width_as_u32len_t{};

struct width_calculator_c
{
    static constexpr bool constrainable = true;

    static constexpr STRF_HD strf::width_as_u32len_t get_default() noexcept
    {
        return {};
    }
};

#if defined(STRF_SEPARATE_COMPILATION)

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace strf

#endif  // STRF_DETAIL_FACETS_WIDTH_CALCULATOR_HPP

