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
    width_decumulator() = default;
    ~width_decumulator() = default;

    explicit constexpr STRF_HD width_decumulator(strf::width_t initial_width) noexcept
        : width_(initial_width)
    {}

    width_decumulator(const width_decumulator&) = delete;
    width_decumulator(width_decumulator&&) = delete;
    width_decumulator& operator=(const width_decumulator&) = delete;
    width_decumulator& operator=(width_decumulator&&) = delete;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void subtract_width(strf::width_t w) noexcept
    {
        width_ -= w;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void checked_subtract_width(strf::width_t w) noexcept
    {
        w = w.ge_zero() ? w : 0;
        width_ = (w < width_) ? (width_ - w) : 0;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void clear_remaining_width() noexcept
    {
        width_ = 0;
    }
    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        width_ = width_.ge_zero() ? width_ : 0;
        return width_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void reset_remaining_width(strf::width_t w) noexcept
    {
        width_ = w;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void zeroize_remaining_width_if_negative() noexcept
    {
        width_ = width_.gt_zero() ? width_ : 0;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool has_remaining_width() const noexcept
    {
        const bool is_positive = width_.gt_zero();
        width_ = is_positive ? width_ : 0;
        return is_positive;
    }

private:

    mutable strf::width_t width_ = strf::width_max;
};

template <>
class width_decumulator<false>
{
public:

    constexpr width_decumulator() noexcept = default;

    template <typename T>
    constexpr STRF_HD void subtract_width(T) const noexcept
    {
    }
    template <typename T>
    constexpr STRF_HD void checked_subtract_width(T) const noexcept
    {
    }
    constexpr STRF_HD void clear_remaining_width() noexcept
    {
    }
    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD void zeroize_remaining_width_if_negative() const noexcept
    {
    }
    constexpr STRF_HD bool has_remaining_width() noexcept
    {
        return false;
    }
};

template <bool Active>
class size_accumulator;

template <>
class size_accumulator<true>
{
public:
    size_accumulator() = default;

    explicit constexpr STRF_HD size_accumulator(std::ptrdiff_t initial_size) noexcept
        : size_(initial_size)
    {
    }

    ~size_accumulator() = default;
    size_accumulator(const size_accumulator&) = delete;
    size_accumulator(size_accumulator&&) = delete;
    size_accumulator& operator=(const size_accumulator&) = delete;
    size_accumulator& operator=(size_accumulator&&) = delete;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(std::ptrdiff_t s) noexcept
    {
        STRF_ASSERT(s >= 0);
        size_ += s;
    }

    constexpr STRF_HD std::size_t accumulated_usize() const noexcept
    {
        return size_;
    }
    constexpr STRF_HD std::ptrdiff_t accumulated_ssize() const noexcept
    {
        return size_;
    }

private:

    std::ptrdiff_t size_ = 0;
};

template <>
class size_accumulator<false>
{
public:

    constexpr size_accumulator() noexcept = default;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_usize(std::size_t) noexcept
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(std::ptrdiff_t) noexcept
    {
    }
    constexpr STRF_HD std::ptrdiff_t accumulated_ssize() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD std::size_t accumulated_usize() const noexcept
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

    constexpr preprinting() noexcept = default;

    preprinting(const preprinting&) = delete;
    preprinting(preprinting&&) = delete;
    preprinting& operator=(const preprinting&) = delete;
    preprinting& operator=(preprinting&&) = delete;

    ~preprinting() = default;
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

