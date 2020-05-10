#ifndef STRF_DETAIL_FACETS_NUMPUNCT_HPP
#define STRF_DETAIL_FACETS_NUMPUNCT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <strf/detail/common.hpp>

namespace strf {

class digits_groups_iterator
{
    using underlying_int_t_ = std::uint32_t;
    constexpr static unsigned grp_bits_count_ = 5;
    constexpr static unsigned grp_bits_mask_ = (1 << grp_bits_count_) - 1;
    constexpr static unsigned no_more_sep_ = (1 << grp_bits_count_) - 1;

public:

    constexpr static int grp_max = 30;
    constexpr static unsigned grps_count_max = 6;

    static_assert(grp_max == no_more_sep_ - 1, "");
    static_assert(grps_count_max <= sizeof(underlying_int_t_) * 8 / grp_bits_count_, "");

    constexpr STRF_HD digits_groups_iterator() noexcept
        : grps_(no_more_sep_)
    {
    }

    constexpr STRF_HD explicit digits_groups_iterator(int grp) noexcept
        : grps_(grp & grp_bits_mask_)
    {
        STRF_ASSERT(grp == -1 || (0 < grp && grp <= grp_max));
    }

    template <typename ... IntArgs>
    constexpr STRF_HD digits_groups_iterator
        ( unsigned grp0, unsigned grp1, IntArgs... grps ) noexcept
        : grps_(ctor_(grp0, grp1, grps...))
    {
        static_assert(2 + sizeof...(grps) <= grps_count_max, "too many groups");
    }

    constexpr STRF_HD digits_groups_iterator(const char* str, std::size_t len) noexcept
    {
        if (len > grps_count_max) {
            len = grps_count_max;
        }
        if (len == 0 || str[0] == 0 || str[0] == '\xFF') {
            grps_ = no_more_sep_;
        } else {
            grps_ = 0;
            STRF_ASSERT(0 < str[0] && str[0] <= grp_max);
            char previous_grp = str[0];
            grps_ = previous_grp;
            int shift = 5;
            for(unsigned i = 1; i < len; ++i) {
                char grp = str[i];
                if (grp == 0) {
                    break;
                }
                //if (grp == previous_grp) {
                //    continue;
                //}
                STRF_ASSERT((0 < grp && grp <= grp_max) || grp == '\xFF');
                unsigned g = (grp & grp_bits_mask_) << shift;
                grps_ = grps_ | g;
                if (grp == '\xFF') {
                    break;
                }
                shift += grp_bits_count_;
                previous_grp = grp;
            }
        }
    }

    constexpr STRF_HD digits_groups_iterator(const char* str) noexcept
        : digits_groups_iterator(str, grps_count_max)
    {
    }

    constexpr digits_groups_iterator(const digits_groups_iterator&) noexcept = default;

    constexpr STRF_HD bool operator==(const digits_groups_iterator& other) const noexcept
    {
        return grps_ == other.grps_;
    }
    constexpr STRF_HD digits_groups_iterator& operator=
        ( const digits_groups_iterator& other ) noexcept
    {
        grps_ = other.grps_;
        return *this;
    }

    constexpr STRF_HD unsigned current() const noexcept
    {
        return grps_ & grp_bits_mask_;
    }
    constexpr STRF_HD void advance() noexcept
    {
        grps_ = grps_ >> grp_bits_count_;
    }
    constexpr STRF_HD unsigned next() noexcept
    {
        advance();
        return current();
    }
    constexpr STRF_HD bool is_final() const noexcept
    {
        return 0 == (grps_ >> grp_bits_count_);
    }
    constexpr STRF_HD bool no_more_sep() const noexcept
    {
        STRF_ASSERT( ! ((0x1F & grps_) == no_more_sep_ && (grps_ != no_more_sep_)));
        return grps_ == no_more_sep_;
    }

private:

    constexpr static STRF_HD underlying_int_t_ ctor_()
    {
        return 0;
    }

    constexpr static STRF_HD underlying_int_t_ ctor_(int last_grp)
    {
        STRF_ASSERT(last_grp == -1 || (0 < last_grp && last_grp < 32));
        return last_grp & 0x1F;
    }

    template <typename ... IntT>
    constexpr static STRF_HD underlying_int_t_ ctor_(int g0, int g1, IntT... grps)
    {
        STRF_ASSERT(0 < g0 && g0 < 32);
        STRF_ASSERT((0 < g1 && g1 < 32) || (g1 == -1 || sizeof...(grps) == 0));

        //if (g0 != g1) {
            return g0 | (ctor_(g1, grps...) << 5);
        //}
        //return g1 | (ctor_(grps...) << 5);
    }

    underlying_int_t_ grps_ = 0;
};

constexpr STRF_HD unsigned sep_count(strf::digits_groups_iterator it, unsigned digcount) noexcept
{
    unsigned count = 0;
    while(1) {
        auto grp = it.current();
        if (digcount <= grp || it.no_more_sep()) {
            return count;
        }
        if (it.is_final()) {
            return count + (digcount - 1) / grp;
        }
        it.advance();
        ++count;
        digcount -= grp;
    }
}

struct digits_distribution;

constexpr STRF_HD strf::digits_distribution calculate_distribution
    ( strf::digits_groups_iterator groups, unsigned digcount) noexcept;

class reverse_digits_groups_iterator
{
    using underlying_int_t_ = std::uint32_t;
    constexpr static unsigned grp_bits_count_ = 5;
    constexpr static unsigned grp_bits_mask_ = (1 << grp_bits_count_) - 1;

public:

    constexpr static unsigned grp_max = strf::digits_groups_iterator::grp_max;
    constexpr static unsigned grps_count_max = strf::digits_groups_iterator::grps_count_max;

    static_assert(grp_max == (1 << grp_bits_count_) - 2, "");
    static_assert(grps_count_max <= sizeof(underlying_int_t_) * 8 / grp_bits_count_, "");

    constexpr reverse_digits_groups_iterator() noexcept = default;
    constexpr reverse_digits_groups_iterator(const reverse_digits_groups_iterator&) noexcept = default;

    constexpr STRF_HD void push_low(unsigned grp) noexcept
    {
        STRF_ASSERT(grp != 0 && grp <=  grp_max);
        grps_ = (grps_ << grp_bits_count_) | grp;
    }
    constexpr STRF_HD void pop_high() noexcept
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
    strf::reverse_digits_groups_iterator low_groups;
    unsigned middle_groups_count;
    unsigned highest_group;
};

constexpr STRF_HD strf::digits_distribution calculate_distribution
    ( strf::digits_groups_iterator groups, unsigned digcount) noexcept
{
    strf::digits_distribution dist{{}, 1, 0};
    while(1) {
        auto grp = groups.current();
        STRF_ASSERT(grp);
        if (grp >= digcount || groups.no_more_sep()) {
            dist.highest_group = digcount;
            break;
        }
        dist.low_groups.push_low(grp);
        if (groups.is_final()) {
            --digcount;
            dist.middle_groups_count = digcount / grp;
            dist.highest_group = 1 + digcount % grp;
            break;
        }
        digcount -= grp;
        groups.advance();
    }
    return dist;
}

template <int Base> struct numpunct_c;

template <int Base>
class numpunct
{
public:
    using category = strf::numpunct_c<Base>;
    
    constexpr numpunct() = default;
    constexpr numpunct(const numpunct& ) = default;

    constexpr STRF_HD numpunct& operator=(const numpunct& other) noexcept
    {
        groups_ = other.groups_;
        decimal_point_ = other.decimal_point_;
        thousands_sep_ = other.thousands_sep_;
    }
    constexpr STRF_HD bool operator==(const numpunct& other) const noexcept
    {
        return groups_ == other.groups_
            && decimal_point_ == other.decimal_point_
            && thousands_sep_ == other.thousands_sep_;
    }    
    constexpr STRF_HD explicit numpunct(int grp0) noexcept
        : groups_(grp0)
    {
    }
    template <typename... Grps>
    constexpr STRF_HD numpunct(int grp0, int grp1, Grps... grps) noexcept
        : groups_(grp0, grp1, grps...)
    {
    }
    constexpr STRF_HD numpunct(strf::digits_groups_iterator grps) noexcept
        : groups_(grps)
    {
    }
    constexpr STRF_HD numpunct(const char* grps) noexcept
        : groups_(grps)
    {
    }
    constexpr STRF_HD numpunct(const char* grps, std::size_t grps_len) noexcept
        : groups_(grps, grps_len)
    {
    }
    constexpr STRF_HD strf::digits_groups_iterator groups() const
    {
        return groups_;
    }
    constexpr STRF_HD strf::digits_distribution groups(unsigned digcount) const noexcept
    {
        return strf::calculate_distribution(this->groups(), digcount);
    }
    constexpr STRF_HD bool no_group_separation(unsigned digcount) const noexcept
    {
        return groups_.no_more_sep() || digcount <= groups_.current();
    }
    constexpr STRF_HD unsigned thousands_sep_count(unsigned digcount) const noexcept
    {
        return strf::sep_count(groups_, digcount);
    }
    constexpr STRF_HD char32_t thousands_sep() const noexcept
    {
        return thousands_sep_;
    }
    constexpr STRF_HD numpunct& thousands_sep(char32_t ch) & noexcept
    {
        thousands_sep_ = ch;
        return *this;
    }
    constexpr STRF_HD numpunct&& thousands_sep(char32_t ch) && noexcept
    {
        thousands_sep_ = ch;
        return static_cast<numpunct&&>(*this);
    }
    constexpr STRF_HD char32_t decimal_point() const noexcept
    {
        return decimal_point_;
    }
    constexpr STRF_HD numpunct& decimal_point(char32_t ch) & noexcept
    {
        decimal_point_ = ch;
        return *this;
    }
    constexpr STRF_HD numpunct&& decimal_point(char32_t ch) && noexcept
    {
        decimal_point_ = ch;
        return static_cast<numpunct&&>(*this);
    }

private:

    strf::digits_groups_iterator groups_;
    char32_t decimal_point_ = U'.';
    char32_t thousands_sep_ = U',';
};

template <int Base>
class default_numpunct final
{
public:
    using category = strf::numpunct_c<Base>;

    default_numpunct() noexcept = default;
    default_numpunct(const default_numpunct&) noexcept = default;

    constexpr STRF_HD default_numpunct& operator=(const default_numpunct&) noexcept
    {
    }
    constexpr STRF_HD bool operator==(const default_numpunct&) const noexcept
    {
        return true;
    }
    constexpr STRF_HD strf::digits_groups_iterator groups() const
    {
        return {};
    }
    constexpr STRF_HD strf::digits_distribution groups(unsigned digcount) const noexcept
    {
        return {{}, 0, digcount};
    }
    constexpr STRF_HD bool no_group_separation(unsigned) const noexcept
    {
        return true;
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
    constexpr operator strf::numpunct<Base> () const noexcept
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

    no_grouping() noexcept = default;
    no_grouping(const no_grouping&) noexcept = default;

    constexpr STRF_HD no_grouping& operator=(const no_grouping& other) noexcept
    {
        decimal_point_ = other.decimal_point_;
    }
    constexpr STRF_HD bool operator==(const no_grouping& other) const noexcept
    {
        return decimal_point_ == other.decimal_point_;
    }
    constexpr STRF_HD strf::digits_groups_iterator groups() const
    {
        return {};
    }
    constexpr STRF_HD strf::digits_distribution groups(unsigned digcount) const noexcept
    {
        return {{}, 0, digcount};
    }
    constexpr STRF_HD bool no_group_separation(unsigned) const noexcept
    {
        return true;
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
    constexpr operator strf::numpunct<Base> () const noexcept
    {
        return {};
    }
    constexpr STRF_HD no_grouping& decimal_point(char32_t ch) & noexcept
    {
        decimal_point_ = ch;
        return *this;
    }
    constexpr STRF_HD no_grouping&& decimal_point(char32_t ch) && noexcept
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
class has_punct_impl
{
public:

    static STRF_HD std::true_type  test_numpunct(const strf::numpunct<Base>&);
    static STRF_HD std::false_type test_numpunct(const strf::default_numpunct<Base>&);

    static STRF_HD const FPack& fp();

    using has_numpunct_type = decltype
        ( test_numpunct
            ( get_facet< strf::numpunct_c<Base>, InputT >(fp())) );

public:

    static constexpr bool has_punct = has_numpunct_type::value;
};

template <typename CharT, typename FPack, typename InputT, unsigned Base>
constexpr bool has_punct = has_punct_impl<CharT, FPack, InputT, Base>::has_punct;

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_FACETS_NUMPUNCT_HPP

