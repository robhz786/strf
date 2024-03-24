#ifndef STRF_DETAIL_PREMEASUREMENTS_HPP
#define STRF_DETAIL_PREMEASUREMENTS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/width_t.hpp>


namespace strf {

template <bool Active>
class width_accumulator;

template <>
class width_accumulator<true>
{
public:
    constexpr width_accumulator() = default;
    ~width_accumulator() = default;
    constexpr width_accumulator(const width_accumulator&) = default;
    constexpr width_accumulator(width_accumulator&&) = default;
    constexpr width_accumulator& operator=(const width_accumulator&) = default;
    constexpr width_accumulator& operator=(width_accumulator&&) = default;

    explicit constexpr STRF_HD width_accumulator(strf::width_t limit) noexcept
        : limit_(limit)
    {}

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_width(strf::width_t w) noexcept
    {
        width_ += w;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void checked_add_width(strf::width_t w) noexcept
    {
        if (w > 0 && width_ < limit_) {
            width_ = ( w >= limit_ - width_
                     ? limit_
                     : width_ + w );
        }
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void saturate_width() noexcept
    {
        width_ = limit_;
    }
    constexpr STRF_HD strf::width_t accumulated_width() const noexcept
    {
        return width_ <= limit_ ? width_ : limit_;
    }
    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return width_ <= limit_ ? limit_ - width_ : 0;
    }
    constexpr STRF_HD bool remaining_width_greater_than(strf::width_t w) const noexcept
    {
        return width_ + w < limit_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool has_remaining_width() const noexcept
    {
        return width_ < limit_;
    }

private:

    strf::width_t limit_ = strf::width_max;
    strf::width_t width_ = 0;
};

template <>
class width_accumulator<false>
{
public:
    constexpr width_accumulator() noexcept = default;
    ~width_accumulator() noexcept = default;
    constexpr width_accumulator(const width_accumulator&) = default;
    constexpr width_accumulator(width_accumulator&&) = default;
    constexpr width_accumulator& operator=(const width_accumulator&) = default;
    constexpr width_accumulator& operator=(width_accumulator&&) = default;

    template <typename T>
    constexpr STRF_HD void add_width(T) const noexcept
    {
    }
    template <typename T>
    constexpr STRF_HD void checked_add_width(T) const noexcept
    {
    }
    constexpr STRF_HD void saturate_width() noexcept
    {
    }
    constexpr STRF_HD strf::width_t accumulated_width() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD bool remaining_width_greater_than(strf::width_t) const noexcept
    {
        return false;
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
    constexpr size_accumulator() = default;
    ~size_accumulator() = default;
    constexpr size_accumulator(const size_accumulator&) = default;
    constexpr size_accumulator(size_accumulator&&) = default;
    constexpr size_accumulator& operator=(const size_accumulator&) = default;
    constexpr size_accumulator& operator=(size_accumulator&&) = default;

    template < typename IntT
             , strf::detail::enable_if_t<std::is_integral<IntT>::value, int> =0>
    explicit constexpr STRF_HD size_accumulator(IntT initial_size) noexcept
        : size_(detail::safe_cast_size_t(initial_size))
    {
    }

    template < typename IntT
             , strf::detail::enable_if_t<std::is_integral<IntT>::value, int> =0>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(IntT s) noexcept
    {
        STRF_ASSERT(detail::ge_zero(s));
        size_ += detail::safe_cast_size_t(s);
    }

    constexpr STRF_HD std::size_t accumulated_size() const noexcept
    {
        return size_;
    }
    constexpr STRF_HD std::ptrdiff_t accumulated_ssize() const noexcept
    {
        return static_cast<std::ptrdiff_t>(size_);
    }

private:

    std::size_t size_ = 0;
};

template <>
class size_accumulator<false>
{
public:
    constexpr size_accumulator() noexcept = default;
    ~size_accumulator() noexcept = default;
    constexpr size_accumulator(const size_accumulator&) = default;
    constexpr size_accumulator(size_accumulator&&) = default;
    constexpr size_accumulator& operator=(const size_accumulator&) = default;
    constexpr size_accumulator& operator=(size_accumulator&&) = default;

    template <typename IntT>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void add_size(IntT) noexcept
    {
    }
    constexpr STRF_HD std::ptrdiff_t accumulated_ssize() const noexcept
    {
        return 0;
    }
    constexpr STRF_HD std::size_t accumulated_size() const noexcept
    {
        return 0;
    }
};

template <bool Active>
using size_preview STRF_DEPRECATED_MSG("size_preview was renamed to size_accumulator")
    = size_accumulator<Active>;

enum class width_presence: bool { no = false, yes = true };
enum class size_presence : bool { no = false, yes = true };

using preview_size STRF_DEPRECATED_MSG("preview_size was renamed to size_presence")
    = size_presence;
using preview_width STRF_DEPRECATED_MSG("preview_width was renamed to width_presence")
    = width_presence;

template <strf::size_presence SizePresence, strf::width_presence WidthPresence>
class premeasurements
    : public strf::size_accumulator<static_cast<bool>(SizePresence)>
    , public strf::width_accumulator<static_cast<bool>(WidthPresence)>
{
public:
    static constexpr strf::size_presence size_presence_v = SizePresence;
    static constexpr strf::width_presence width_presence_v = WidthPresence;

    static constexpr bool size_demanded = static_cast<bool>(SizePresence);
    static constexpr bool width_demanded = static_cast<bool>(WidthPresence);
    static constexpr bool no_demands = ! size_demanded && ! width_demanded;
    static constexpr bool something_demanded = size_demanded || width_demanded;
    static constexpr bool size_and_width_demanded = size_demanded && width_demanded;

    // for backwards compatibility:
    static constexpr bool size_required = size_demanded;
    static constexpr bool width_required = width_demanded;
    static constexpr bool nothing_required = no_demands;
    static constexpr bool something_required = something_demanded;
    static constexpr bool all_required = size_and_width_demanded;

    template <bool W = width_demanded>
    STRF_HD constexpr explicit premeasurements
        ( strf::detail::enable_if_t<W, strf::width_t> width_limit ) noexcept
        : strf::width_accumulator<W>{width_limit}
    {
    }

    constexpr premeasurements() noexcept = default;
    ~premeasurements() noexcept = default;
    constexpr premeasurements(const premeasurements&) = default;
    constexpr premeasurements(premeasurements&&) = default;
    constexpr premeasurements& operator=(const premeasurements&) = default;
    constexpr premeasurements& operator=(premeasurements&&) = default;
};


using no_premeasurements
    = strf::premeasurements<strf::size_presence::no, strf::width_presence::no>;
using full_premeasurements
    = strf::premeasurements<strf::size_presence::yes, strf::width_presence::yes>;

template <strf::size_presence SizePresence, strf::width_presence WidthPresence>
using print_preview STRF_DEPRECATED_MSG("print_preview renamed to premeasurements")
    = strf::premeasurements<SizePresence, WidthPresence>;

using no_print_preview
    STRF_DEPRECATED_MSG("no_print_preview renamed to no_premeasurements")
    = strf::no_premeasurements;

using print_size_and_width_preview
    STRF_DEPRECATED_MSG("print_size_and_width_preview to fulll_premeasurements")
    = strf::full_premeasurements;

} // namespace strf

#endif  // STRF_DETAIL_PREMEASUREMENTS_HPP

