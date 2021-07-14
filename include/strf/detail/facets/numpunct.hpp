#ifndef STRF_DETAIL_FACETS_NUMPUNCT_HPP
#define STRF_DETAIL_FACETS_NUMPUNCT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>
#include <limits.h>

namespace strf {
namespace detail {

struct digits_group_common
{
    // The groups are stored in an underlying_int_t_ value.
    // Each group occupies  grp_bits_count_ bits


    using underlying_int_t_ = std::uint32_t;
    constexpr static unsigned grp_bits_count_ = 5;
    constexpr static unsigned grps_count_max = sizeof(underlying_int_t_) * 8 / grp_bits_count_;
    constexpr static unsigned grp_max = (1 << grp_bits_count_) - 1;
    constexpr static underlying_int_t_ grp_bits_mask_ = (1 << grp_bits_count_) - 1;
    constexpr static unsigned repeat_last = 2;
    constexpr static unsigned dont_repeat_last = 3;

};

} // namespace detail

class digits_grouping;

class digits_grouping_iterator
{
    using common = strf::detail::digits_group_common;
    using underlying_int_t_ = common::underlying_int_t_;
    constexpr static auto grp_bits_count_ = common::grp_bits_count_;
    constexpr static auto grp_bits_mask_  = common::grp_bits_mask_;
    constexpr static auto repeat_last_      = common::repeat_last;
    constexpr static auto dont_repeat_last_ = common::dont_repeat_last;

public:

    digits_grouping_iterator() = delete;
    constexpr digits_grouping_iterator(const digits_grouping_iterator&) noexcept = default;
    constexpr STRF_HD explicit digits_grouping_iterator(digits_grouping) noexcept;

    constexpr STRF_HD bool operator==(const digits_grouping_iterator& other) const noexcept
    {
        return grps_ == other.grps_;
    }
    constexpr STRF_HD bool operator!=(const digits_grouping_iterator& other) const noexcept
    {
        return grps_ != other.grps_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD digits_grouping_iterator& operator=
        ( const digits_grouping_iterator& other ) noexcept
    {
        grps_ = other.grps_;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD unsigned current() const noexcept
    {
        STRF_ASSERT(! ended());
        return grps_ & grp_bits_mask_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void advance() noexcept
    {
        STRF_ASSERT(! ended());
        grps_ = grps_ >> grp_bits_count_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool is_last() const noexcept
    {
        STRF_ASSERT(! ended());
        return 0 == (grps_ >> (grp_bits_count_ + 2));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool shall_repeat_current() const noexcept
    {
        // Return true if this is the last group is it shall be repeated
        STRF_ASSERT(! ended());
        return (grps_ >> grp_bits_count_) == repeat_last_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool is_final() const noexcept
    {
        // Return true if this is the last group is it shall *not* be repeated
        STRF_ASSERT(! ended());
        return (grps_ >> grp_bits_count_) == dont_repeat_last_;
    }
    constexpr STRF_HD bool ended() const noexcept
    {
        return 0 == (grps_ >> 2);
    }

private:

    friend class strf::digits_grouping;

    constexpr STRF_HD digits_grouping_iterator(underlying_int_t_ grps) noexcept
        : grps_(grps)
    {
    }
    underlying_int_t_ grps_;
};

class reverse_digits_groups
{
    using common = strf::detail::digits_group_common;
    using underlying_int_t_ = common::underlying_int_t_;
    constexpr static auto grp_bits_count_ = common::grp_bits_count_;
    constexpr static auto grp_bits_mask_ = common::grp_bits_mask_;

public:

    constexpr static auto grp_max = common::grp_max;
    constexpr static auto grps_count_max = common::grps_count_max;

    constexpr reverse_digits_groups() noexcept = default;
    constexpr reverse_digits_groups(const reverse_digits_groups&) noexcept = default;
    constexpr STRF_HD bool operator==(const reverse_digits_groups& other) const noexcept
    {
        return grps_ == other.grps_;
    }
    constexpr STRF_HD bool operator!=(const reverse_digits_groups& other) const noexcept
    {
        return grps_ != other.grps_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD reverse_digits_groups& operator=
        ( const reverse_digits_groups& other ) noexcept
    {
        grps_ = other.grps_;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void push_low(unsigned grp) noexcept
    {
        STRF_ASSERT(grp != 0 && grp <=  grp_max);
        grps_ = (grps_ << grp_bits_count_) | grp;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD void pop_high() noexcept
    {
        grps_ = grps_ >> grp_bits_count_;
    }
    constexpr STRF_HD unsigned highest_group() const noexcept
    {
        return grps_ & grp_bits_mask_;
    }
    constexpr STRF_HD bool empty() const noexcept
    {
        return grps_ == 0;
    }

private:

    underlying_int_t_ grps_ = 0;
};

struct digits_distribution
{
    strf::reverse_digits_groups low_groups;
    unsigned middle_groups_count;
    unsigned highest_group;
};

class digits_grouping_creator;

class digits_grouping
{
    using common = strf::detail::digits_group_common;
    using underlying_int_t_ = common::underlying_int_t_;
    constexpr static auto grp_bits_count_ = common::grp_bits_count_;
    constexpr static auto grp_bits_mask_  = common::grp_bits_mask_;
    constexpr static auto repeat_last_      = common::repeat_last;
    constexpr static auto dont_repeat_last_ = common::dont_repeat_last;

    using grp_t_ = int;

public:

    constexpr static grp_t_ grp_max = common::grp_max;
    constexpr static unsigned grps_count_max = common::grps_count_max;

    constexpr STRF_HD digits_grouping() noexcept
        : grps_(dont_repeat_last_)
    {
    }

    template <typename... IntArgs>
    constexpr STRF_HD explicit digits_grouping
        ( grp_t_ grp0, IntArgs... grps ) noexcept
        : grps_(ctor_(grp0, grps...))
    {
        STRF_ASSERT_IN_CONSTEXPR(grps_ != 0);
    }

    STRF_HD explicit digits_grouping(const char* str) noexcept;

    constexpr digits_grouping(const digits_grouping&) noexcept = default;

    constexpr STRF_HD bool operator==(const digits_grouping& other) const noexcept
    {
        return grps_ == other.grps_;
    }
    constexpr STRF_HD bool operator!=(const digits_grouping& other) const noexcept
    {
        return grps_ != other.grps_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD digits_grouping& operator=
        ( const digits_grouping& other ) noexcept
    {
        grps_ = other.grps_;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool any_separator(int digcount) const noexcept
    {
        STRF_ASSERT_IN_CONSTEXPR(grps_ != 0);
        return grps_ != dont_repeat_last_ && digcount > (int)(grps_ & grp_bits_mask_);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD unsigned separators_count(unsigned digcount) const noexcept
    {
        STRF_ASSERT(grps_ != 0);
        if (digcount <= 1) {
            return 0;
        }
        auto it = get_iterator();
        if (it.ended()) {
            return 0;
        }
        unsigned count = 0;
        while(1) {
            auto grp = it.current();
            if (digcount <= grp) {
                return count;
            }
            if (it.is_last()) {
                return it.shall_repeat_current()
                    ? (count + (digcount - 1) / grp)
                    : count + (digcount > grp);
            }
            ++count;
            digcount -= grp;
            it.advance();
        }
    }
    constexpr STRF_HD strf::digits_grouping_iterator get_iterator() const noexcept
    {
        return strf::digits_grouping_iterator{grps_};
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::digits_distribution distribute(unsigned digcount) const noexcept
    {
        STRF_ASSERT(grps_ != 0);
        auto  git = get_iterator();
        if (git.ended()) {
            return {{}, 0, digcount};
        }
        strf::reverse_digits_groups low_groups;
        auto grp = git.current();
        while(1) {
            if (grp >= digcount) {
                return {low_groups, 0, digcount};
            }
            low_groups.push_low(grp);
            if (git.shall_repeat_current()) {
                --digcount;
                return { low_groups, digcount / grp, 1 + digcount % grp };
            }
            if (git.is_final()) {
                return {low_groups, 0, digcount - grp};
            }
            digcount -= grp;
            git.advance();
            grp = git.current();
        }
    }

    constexpr STRF_HD bool empty() const noexcept
    {
        return grps_ == dont_repeat_last_;
    }

private:
    friend class digits_grouping_creator;
    struct underlying_tag {};

    constexpr STRF_HD digits_grouping(underlying_tag, underlying_int_t_ grps)
        : grps_(grps)
    {
    }

    constexpr static STRF_HD grp_t_ last_arg_(grp_t_ x) noexcept
    {
        return x;
    }

    template <typename... Args>
    constexpr static STRF_HD grp_t_ last_arg_
        (grp_t_, grp_t_ arg1, const Args&... args) noexcept
    {
        return last_arg_(arg1, args...);
    }

    constexpr static STRF_HD underlying_int_t_ ctor2_() noexcept
    {
        return dont_repeat_last_;
    }
    constexpr static STRF_HD underlying_int_t_ ctor2_(grp_t_ last_grp) noexcept
    {
        STRF_ASSERT_IN_CONSTEXPR(last_grp == -1 || (0 < last_grp && last_grp <= grp_max));
        return last_grp == -1
            ? dont_repeat_last_
            : (( repeat_last_ << grp_bits_count_ ) | last_grp) ;
    }
    template <typename ... IntT>
    constexpr static STRF_HD underlying_int_t_ ctor2_(grp_t_ g0, grp_t_ g1, IntT... grps) noexcept
    {
        STRF_ASSERT_IN_CONSTEXPR(0 < g0 && g0 <= grp_max);
        return g0 | (ctor2_(g1, grps...) << grp_bits_count_);
    }
    template <typename... IntT>
    constexpr static STRF_HD std::size_t groups_count_(IntT... grps) noexcept
    {
        return ( last_arg_(grps...) == -1
               ? sizeof...(grps) - 1
               : sizeof...(grps) );
    }
    template <typename... IntT>
    constexpr static STRF_HD underlying_int_t_ ctor_(IntT... grps)  noexcept
    {
        STRF_ASSERT_IN_CONSTEXPR(groups_count_(grps...) <= grps_count_max);
        return ctor2_(grps...);
    }

    underlying_int_t_ grps_;
};

constexpr STRF_HD digits_grouping_iterator::digits_grouping_iterator
    ( digits_grouping g ) noexcept
    : digits_grouping_iterator(g.get_iterator())
{
    STRF_ASSERT_IN_CONSTEXPR(current() != 0);
}

class digits_grouping_creator
{
    using common = strf::detail::digits_group_common;
    using underlying_int_t_ = common::underlying_int_t_;
    constexpr static auto grp_bits_count_ = common::grp_bits_count_;
    constexpr static auto grp_bits_mask_ = common::grp_bits_mask_;
    constexpr static auto grp_max_ = common::grp_max;
    constexpr static auto grps_count_max_ = common::grps_count_max;

public:

    constexpr digits_grouping_creator() noexcept = default;
    constexpr digits_grouping_creator(const digits_grouping_creator&) noexcept = delete;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD void push_high(int grp) noexcept
    {
        if (failed_ || grp < 1 || grp > (int)grp_max_ || ! enough_space_to_push()) {
            failed_ = true;
        } else {
            reverse_grps_ = ( reverse_grps_ << grp_bits_count_ ) | grp;
        }
    }

    constexpr STRF_HD bool failed() const noexcept
    {
        return failed_;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::digits_grouping finish_no_more_sep() noexcept
    {
        if (failed_ || reverse_grps_ == 0) {
            return {};
        }
        underlying_int_t_ grps = common::dont_repeat_last;
        while (reverse_grps_) {
            grps = (grps << grp_bits_count_) | (reverse_grps_ & grp_bits_mask_);
            reverse_grps_ >>= grp_bits_count_;
        }
        return {strf::digits_grouping::underlying_tag{}, grps};
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::digits_grouping finish() noexcept
    {
        if (failed_ || reverse_grps_ == 0) {
            return {};
        }
        underlying_int_t_ grps = common::repeat_last;
        auto highest_grp = reverse_grps_ & grp_bits_mask_;
        grps = (grps << grp_bits_count_) | highest_grp;
        reverse_grps_ >>= grp_bits_count_;
        while (reverse_grps_) {
            auto grp = reverse_grps_ & grp_bits_mask_;
            if (grp != highest_grp) {
                break;
            }
            reverse_grps_ >>= grp_bits_count_;
        }
        while (reverse_grps_) {
            auto grp = reverse_grps_ & grp_bits_mask_;
            grps = (grps << grp_bits_count_) | grp;
            reverse_grps_ >>= grp_bits_count_;
        }
        return {strf::digits_grouping::underlying_tag{}, grps};
    }

private:

    constexpr STRF_HD bool enough_space_to_push() const noexcept
    {
        return reverse_grps_ < ( underlying_int_t_{1} << ((grps_count_max_ - 1) * grp_bits_count_));
    }

    underlying_int_t_ reverse_grps_ = 0;
    bool failed_ = false;
};

#if ! defined(STRF_OMIT_IMPL)

STRF_FUNC_IMPL STRF_HD digits_grouping::digits_grouping(const char* str) noexcept
{
    strf::digits_grouping_creator creator;
    while(true) {
        char ch = *str;
        if (ch == '\0') {
            *this = creator.finish();
            break;
        }
        if (0 != (ch & 0x80)) {
            *this = creator.finish_no_more_sep();
            break;
        }
        ++str;
        creator.push_high(ch);
    }
}

#endif // ! defined(STRF_OMIT_IMPL)

template <int Base> struct numpunct_c;

template <int Base>
class numpunct
{
public:
    using category = strf::numpunct_c<Base>;

    constexpr numpunct() = default;
    constexpr numpunct(const numpunct& ) = default;

    template <typename... IntArgs>
    constexpr STRF_HD explicit numpunct(int grp0, IntArgs... grps) noexcept
        : grouping_(grp0, grps...)
    {
    }

    constexpr STRF_HD explicit numpunct(const digits_grouping& grp) noexcept
        : grouping_(grp)
    {
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct& operator=(const numpunct& other) noexcept
    {
        grouping_ = other.grouping_;
        decimal_point_ = other.decimal_point_;
        thousands_sep_ = other.thousands_sep_;
    }
    constexpr STRF_HD bool operator==(const numpunct& other) const noexcept
    {
        return grouping_ == other.grouping_
            && decimal_point_ == other.decimal_point_
            && thousands_sep_ == other.thousands_sep_;
    }
    constexpr STRF_HD bool operator!=(const numpunct& other) const noexcept
    {
        return ! (*this == other);
    }
    constexpr STRF_HD strf::digits_grouping grouping() const noexcept
    {
        return grouping_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct& grouping(strf::digits_grouping grp) & noexcept
    {
        grouping_ = grp;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct&& grouping(strf::digits_grouping grp) && noexcept
    {
        grouping_ = grp;
        return static_cast<numpunct&&>(*this);
    }
    constexpr STRF_HD strf::digits_distribution distribute(unsigned digcount) const noexcept
    {
        return grouping_.distribute(digcount);
    }
    constexpr STRF_HD bool any_group_separation(int digcount) const noexcept
    {
        return grouping_.any_separator(digcount);
    }
    constexpr STRF_HD unsigned thousands_sep_count(unsigned digcount) const noexcept
    {
        return grouping_.separators_count(digcount);
    }
    constexpr STRF_HD char32_t thousands_sep() const noexcept
    {
        return thousands_sep_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct& thousands_sep(char32_t ch) & noexcept
    {
        thousands_sep_ = ch;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct&& thousands_sep(char32_t ch) && noexcept
    {
        thousands_sep_ = ch;
        return static_cast<numpunct&&>(*this);
    }
    constexpr STRF_HD char32_t decimal_point() const noexcept
    {
        return decimal_point_;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct& decimal_point(char32_t ch) & noexcept
    {
        decimal_point_ = ch;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD numpunct&& decimal_point(char32_t ch) && noexcept
    {
        decimal_point_ = ch;
        return static_cast<numpunct&&>(*this);
    }

private:
    strf::digits_grouping grouping_;
    char32_t decimal_point_ = U'.';
    char32_t thousands_sep_ = U',';
};

template <int Base>
class default_numpunct
{
public:
    using category = strf::numpunct_c<Base>;

    constexpr default_numpunct() noexcept = default;
    constexpr default_numpunct(const default_numpunct&) noexcept = default;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD default_numpunct& operator=(const default_numpunct&) noexcept
    {
    }
    constexpr STRF_HD bool operator==(const default_numpunct&) const noexcept
    {
        return true;
    }
    constexpr STRF_HD bool operator!=(const default_numpunct&) const noexcept
    {
        return false;
    }
    constexpr STRF_HD strf::digits_grouping grouping() const noexcept
    {
        return {};
    }
    constexpr STRF_HD strf::digits_distribution distribute(unsigned digcount) const noexcept
    {
        return {{}, 0, digcount};
    }
    constexpr STRF_HD bool any_group_separation(unsigned) const noexcept
    {
        return false;
    }
    constexpr STRF_HD unsigned thousands_sep_count(unsigned) const noexcept
    {
        return 0;
    }
    constexpr STRF_HD char32_t thousands_sep() const noexcept
    {
        return U',';
    }
    constexpr STRF_HD char32_t decimal_point() const noexcept
    {
        return U'.';
    }
    constexpr STRF_HD explicit operator strf::numpunct<Base> () const noexcept
    {
        return {};
    }
    void thousands_sep(char32_t) const = delete;
    void decimal_point(char32_t) const = delete;
};

template <int Base>
class no_grouping final
{
public:
    using category = strf::numpunct_c<Base>;

    constexpr no_grouping() noexcept = default;
    constexpr no_grouping(const no_grouping&) noexcept = default;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_grouping& operator=(const no_grouping& other) noexcept
    {
        decimal_point_ = other.decimal_point_;
    }
    constexpr STRF_HD bool operator==(const no_grouping& other) const noexcept
    {
        return decimal_point_ == other.decimal_point_;
    }
    constexpr STRF_HD bool operator!=(const no_grouping& other) const noexcept
    {
        return decimal_point_ != other.decimal_point_;
    }
    constexpr STRF_HD strf::digits_grouping grouping() const noexcept
    {
        return {};
    }
    constexpr STRF_HD strf::digits_distribution distribute(unsigned digcount) const noexcept
    {
        return {{}, 0, digcount};
    }
    constexpr STRF_HD bool any_group_separation(unsigned) const noexcept
    {
        return false;
    }
    constexpr STRF_HD unsigned thousands_sep_count(unsigned) const noexcept
    {
        return 0;
    }
    constexpr STRF_HD char32_t thousands_sep() const noexcept
    {
        return U',';
    }
    constexpr STRF_HD char32_t decimal_point() const noexcept
    {
        return U'.';
    }
    constexpr STRF_HD explicit operator strf::numpunct<Base> () const noexcept
    {
        return {};
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_grouping& decimal_point(char32_t ch) & noexcept
    {
        decimal_point_ = ch;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD no_grouping&& decimal_point(char32_t ch) && noexcept
    {
        decimal_point_ = ch;
        return static_cast<no_grouping&&>(*this);
    }

private:
    char32_t decimal_point_;
};

template <int Base> struct numpunct_c
{
    constexpr static bool constrainable = true;

    constexpr static int base = Base;

    constexpr static STRF_HD strf::default_numpunct<base> get_default() noexcept
    {
        return {};
    }
};

namespace detail {

template <int Base>
std::true_type has_no_grouping(const strf::no_grouping<Base>&);

template <int Base>
std::true_type has_no_grouping(const strf::default_numpunct<Base>&);

template <int Base>
std::false_type has_no_grouping(const strf::numpunct<Base>&);

template <typename CharT, typename FPack, typename InputT, unsigned Base>
class has_punct
{
public:

    static STRF_HD std::true_type  test_numpunct(...);
    static STRF_HD std::false_type test_numpunct(strf::default_numpunct<Base>);

    static STRF_HD const FPack& fp();

    using has_numpunct_type = decltype
        ( test_numpunct
            ( use_facet< strf::numpunct_c<Base>, InputT >(fp())) );

public:

    static constexpr bool value = has_numpunct_type::value;
};

} // namespace detail

#if __cpp_constexpr >= 201304

constexpr auto numpunct_aa_DJ         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_aa_ER         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_aa_ET         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_af_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_agr_PE        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ak_GH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_am_ET         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_an_ES         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_anp_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_AE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_BH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_DZ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_EG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_IQ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_JO         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_KW         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_LB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_LY         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_MA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_OM         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_QA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_SA         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_ar_SD         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_SS         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_SY         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_TN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ar_YE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_as_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ast_ES        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ayc_PE        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_az_AZ         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_az_IR         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_be_BY         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_bem_ZM        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ber_DZ        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ber_MA        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bg_BG         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_bhb_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bho_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bho_NP        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bi_VU         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bn_BD         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bn_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bo_CN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bo_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_br_FR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_brx_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_bs_BA         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_byn_ER        = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_C             = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_ca_AD         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ca_ES         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ca_FR         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ca_IT         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ce_RU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_chr_US        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ckb_IQ        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_cmn_TW        = strf::numpunct<10>{4}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_crh_UA        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_csb_PL        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_cs_CZ         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_cv_RU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_cy_GB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_da_DK         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_de_AT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_de_BE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_de_CH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U'\'');
constexpr auto numpunct_de_DE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_de_IT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_de_LI         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U'\'');
constexpr auto numpunct_de_LU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_doi_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_dv_MV         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_dz_BT         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_el_CY         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_el_GR         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_en_AG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_AU         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_BW         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_CA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_DK         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_en_GB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_HK         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_IE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_IL         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_NG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_NZ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_PH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_SC         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_SG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_US         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_ZM         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_en_ZW         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_eo            = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_eo_US         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_AR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_BO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_CL         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_CO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_CR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_es_CU         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_es_DO         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_EC         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_ES         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_GT         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_HN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_MX         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(0x202f);
constexpr auto numpunct_es_NI         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_PA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_PE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_PR         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_PY         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_SV         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_US         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_es_UY         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_es_VE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_et_EE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_eu_ES         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_eu_FR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_fa_IR         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ff_SN         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_fi_FI         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_fil_PH        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_fo_FO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_fr_BE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_fr_CA         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_fr_CH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U'\'');
constexpr auto numpunct_fr_FR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_fr_LU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_fur_IT        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_fy_DE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_fy_NL         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ga_IE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_gd_GB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_gez_ER        = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_gez_ET        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_gl_ES         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_gu_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_gv_GB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_hak_TW        = strf::numpunct<10>{4}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ha_NG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_he_IL         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_hif_FJ        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_hi_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_hne_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_hr_HR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_hsb_DE        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ht_HT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_hu_HU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_hy_AM         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ia_FR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_id_ID         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ig_NG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ik_CA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_is_IS         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_it_CH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U'\'');
constexpr auto numpunct_it_IT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_iu_CA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ja_JP         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_kab_DZ        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_ka_GE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_kk_KZ         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_kl_GL         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_km_KH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_kn_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_kok_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ko_KR         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ks_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ku_TR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_kw_GB         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ky_KG         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_lb_LU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_lg_UG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_li_BE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_lij_IT        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_li_NL         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ln_CD         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_lo_LA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_lt_LT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_lv_LV         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_lzh_TW        = strf::numpunct<10>{4}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mag_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mai_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mai_NP        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mfe_MU        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(0x202f);
constexpr auto numpunct_mg_MG         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_mhr_RU        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_mi_NZ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_miq_NI        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mjw_IN        = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mk_MK         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_ml_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mni_IN        = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mn_MN         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_mr_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ms_MY         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_mt_MT         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_my_MM         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_nan_TW        = strf::numpunct<10>{4}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_nb_NO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_nds_DE        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_nds_NL        = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ne_NP         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_nhn_MX        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(0x202f);
constexpr auto numpunct_niu_NU        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_niu_NZ        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_nl_AW         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_nl_BE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_nl_NL         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_nn_NO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_nr_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_nso_ZA        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_oc_FR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_om_ET         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_om_KE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_or_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_os_RU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_pa_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_pap_AW        = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_pap_CW        = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_pa_PK         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_pl_PL         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_POSIX         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_ps_AF         = strf::numpunct<10>{3}.decimal_point(0x66b).thousands_sep(0x66c);
constexpr auto numpunct_pt_BR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_pt_PT         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_quz_PE        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_raj_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ro_RO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ru_RU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_ru_UA         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_rw_RW         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_sa_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sat_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sc_IT         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_sd_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sd_PK         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_se_NO         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_sgs_LT        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_shn_MM        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_shs_CA        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sid_ET        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_si_LK         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sk_SK         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_sl_SI         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_sm_WS         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_so_DJ         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_so_ET         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_so_KE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_so_SO         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sq_AL         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_sq_MK         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_sr_ME         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_sr_RS         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_ss_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_st_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sv_FI         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_sv_SE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_sw_KE         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_sw_TZ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_szl_PL        = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_ta_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ta_LK         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tcy_IN        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_te_IN         = strf::numpunct<10>{3, 2}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tg_TJ         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_the_NP        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_th_TH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ti_ER         = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_ti_ET         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tig_ER        = strf::numpunct<10>{ }.decimal_point(U'.');
constexpr auto numpunct_tk_TM         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tl_PH         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tn_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_to_TO         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tpi_PG        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tr_CY         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_tr_TR         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ts_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_tt_RU         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_tt_RU_iqtelif = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_ug_CN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_uk_UA         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(0x202f);
constexpr auto numpunct_unm_US        = strf::numpunct<10>{2, 2, 2, 3}.decimal_point(U'.').thousands_sep(0x202f);
constexpr auto numpunct_ur_IN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ur_PK         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_uz_UZ         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_ve_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_vi_VN         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_wa_BE         = strf::numpunct<10>{3}.decimal_point(U',').thousands_sep(U'.');
constexpr auto numpunct_wae_CH        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U'\'');
constexpr auto numpunct_wal_ET        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_wo_SN         = strf::numpunct<10>{ }.decimal_point(U',');
constexpr auto numpunct_xh_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_yi_US         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_yo_NG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_yue_HK        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_yuw_PG        = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_zh_CN         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_zh_HK         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_zh_SG         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_zh_TW         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');
constexpr auto numpunct_zu_ZA         = strf::numpunct<10>{3}.decimal_point(U'.').thousands_sep(U',');

// Could not correctly implement locales below:
// bg_BG  : The thousands separator should only appear in numbers larger than 9999.
//          See  https://forum.opencart.com/viewtopic.php?t=144907
// kab_DZ : I could not find out what is the thousands separator.
// ln_CD  : I could not find out what is the thousands separator.

// https://lh.2xlibre.net/locale/bg_BG
// https://lh.2xlibre.net/locale/kab_DZ
// https://lh.2xlibre.net/locale/ln_CD/

#endif // __cpp_constexpr >= 201304


} // namespace strf

#endif  // STRF_DETAIL_FACETS_NUMPUNCT_HPP

