#ifndef STRF_DETAIL_PREPRINTING_HPP
#define STRF_DETAIL_PREPRINTING_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/width_t.hpp>


namespace strf {

template <bool Active>
class width_decumulator;

template <>
class width_decumulator<true>
{
public:
    constexpr width_decumulator() = default;

    explicit constexpr STRF_HD width_decumulator(strf::width_t initial_width) noexcept
        : width_(initial_width)
    {}

    STRF_HD width_decumulator(const width_decumulator&) = delete;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void subtract_width(strf::width_t w) noexcept
    {
        if (w < width_) {
            width_ -= w;
        } else {
            width_ = 0;
        }
    }

    template <typename IntT>
    STRF_CONSTEXPR_IN_CXX14
    strf::detail::enable_if_t<std::is_integral<IntT>::value>
    STRF_HD subtract_width(IntT w) noexcept
    {
        subtract_int_(std::is_signed<IntT>{}, w);
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void clear_remaining_width() noexcept
    {
        width_ = 0;
    }
    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return width_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void reset_remaining_width(strf::width_t w) noexcept
    {
        width_ = w;
    }

private:

    template <typename W>
    void STRF_HD subtract_int_(std::true_type, W w) noexcept
    {
        if (w > 0) {
            if (w <= static_cast<int>(width_.floor())) {
                width_ -= static_cast<std::uint16_t>(w);
            } else {
                width_ = 0;
            }
        }
    }

    template <typename W>
    void STRF_HD subtract_int_(std::false_type, W w) noexcept
    {
        if (w <= width_.floor()) {
            width_ -= static_cast<std::uint16_t>(w);
        } else {
            width_ = 0;
        }
    }

    strf::width_t width_ = strf::width_max;
};

template <>
class width_decumulator<false>
{
public:

    constexpr STRF_HD width_decumulator() noexcept
    {
    }

    template <typename T>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void subtract_width(T) const noexcept
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void clear_remaining_width() noexcept
    {
    }

    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return 0;
    }
};

template <bool Active>
class size_accumulator;

template <>
class size_accumulator<true>
{
public:
    explicit constexpr STRF_HD size_accumulator(std::size_t initial_size = 0) noexcept
        : size_(initial_size)
    {
    }

    STRF_HD size_accumulator(const size_accumulator&) = delete;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(std::size_t s) noexcept
    {
        size_ += s;
    }

    constexpr STRF_HD std::size_t accumulated_size() const noexcept
    {
        return size_;
    }

private:

    std::size_t size_;
};

template <>
class size_accumulator<false>
{
public:

    constexpr STRF_HD size_accumulator() noexcept
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(std::size_t) noexcept
    {
    }

    constexpr STRF_HD std::size_t accumulated_size() const noexcept
    {
        return 0;
    }
};

template <bool Active>
using size_preview STRF_DEPRECATED_MSG("size_preview was renamed to size_accumulator")
    = size_accumulator<Active>;

enum class precalc_width: bool { no = false, yes = true };
enum class precalc_size : bool { no = false, yes = true };

using preview_size STRF_DEPRECATED_MSG("preview_size was renamed to precalc_size")
    = precalc_size;
using preview_width STRF_DEPRECATED_MSG("preview_width was renamed to precalc_width")
    = precalc_width;

template <strf::precalc_size SizeRequired, strf::precalc_width WidthRequired>
class preprinting
    : public strf::size_accumulator<static_cast<bool>(SizeRequired)>
    , public strf::width_decumulator<static_cast<bool>(WidthRequired)>
{
public:

    static constexpr bool size_required = static_cast<bool>(SizeRequired);
    static constexpr bool width_required = static_cast<bool>(WidthRequired);
    static constexpr bool nothing_required = ! size_required && ! width_required;
    static constexpr bool something_required = size_required || width_required;
    static constexpr bool all_required =       size_required && width_required;

    template <strf::precalc_width W = WidthRequired>
    STRF_HD constexpr explicit preprinting
        ( strf::detail::enable_if_t<static_cast<bool>(W), strf::width_t> initial_width ) noexcept
        : strf::width_decumulator<true>{initial_width}
    {
    }

    constexpr STRF_HD preprinting() noexcept
    {
    }
};


using no_preprinting
    = strf::preprinting<strf::precalc_size::no, strf::precalc_width::no>;
using full_preprinting
    = strf::preprinting<strf::precalc_size::yes, strf::precalc_width::yes>;

template <strf::precalc_size SizeRequired, strf::precalc_width WidthRequired>
using print_preview STRF_DEPRECATED_MSG("print_preview renamed to preprinting")
    = strf::preprinting<SizeRequired, WidthRequired>;

using no_print_preview
    STRF_DEPRECATED_MSG("no_print_preview renamed to no_preprinting")
    = strf::no_preprinting;

using print_size_and_width_preview
    STRF_DEPRECATED_MSG("print_size_and_width_preview to fulll_preprinting")
    = strf::full_preprinting;

} // namespace strf

#endif  // STRF_DETAIL_PREPRINTING_HPP

