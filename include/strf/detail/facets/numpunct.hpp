#ifndef STRF_DETAIL_FACETS_NUMPUNCT_HPP
#define STRF_DETAIL_FACETS_NUMPUNCT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <strf/detail/common.hpp>

namespace strf {

struct digits_distribution
{
    const std::uint8_t* low_groups;
    unsigned low_groups_count;
    unsigned middle_groups_count;
    std::uint8_t middle_groups;
    unsigned highest_group;
};

namespace detail {

class str_grouping_impl
{
public:

    str_grouping_impl(std::string grouping)
        : grouping_(std::move(grouping))
    {
    }

    str_grouping_impl(const str_grouping_impl&) = default;

    str_grouping_impl(str_grouping_impl&&) = default;

    unsigned get_thousands_sep_count(unsigned digcount) const;

    strf::digits_distribution get_groups(unsigned digcount) const;

private:

    std::string grouping_;
};

#if defined(STRF_SOURCE) || ! defined(STRF_SEPARATE_COMPILATION)

STRF_INLINE
unsigned str_grouping_impl::get_thousands_sep_count(unsigned digcount) const
{
    if (grouping_.empty() || grouping_[0] == '\0') {
        return 0;
    }
    unsigned count = 0;
    unsigned grp = 1;
    for(auto ch : grouping_) {
        if (ch == 0) {
            break;
        }
        grp = (unsigned)ch;
        if (ch & 0x80 || grp >= digcount) {
            return count;
        }
        ++ count;
        digcount -= grp;
    }

    return count + (digcount - 1) / grp;
}


STRF_INLINE strf::digits_distribution str_grouping_impl::get_groups(unsigned digcount) const
{
    STRF_ASSERT(digcount != 0); //precodition

    if (grouping_.empty() || grouping_[0] == '\0') {
        return {nullptr, 0, 0, 0, digcount};
    }
    strf::digits_distribution res{0, 0, 0, 0, 0};
    res.low_groups = reinterpret_cast<const std::uint8_t*>(grouping_.data());
    for(auto ch : grouping_) {
        if (ch == 0) {
            break;
        }
        if (ch & 0x80 || (unsigned)ch >= digcount) {
            STRF_ASSERT(digcount != 0);
            res.highest_group = digcount;
            return res;
        }
        ++ res.low_groups_count;
        digcount -= ch;
    }
    STRF_ASSERT(digcount != 0);
    --digcount;
    auto last_group = grouping_[res.low_groups_count - 1];
    res.middle_groups = last_group;
    res.middle_groups_count = digcount / last_group;
    res.highest_group = digcount % last_group + 1;

    STRF_ASSERT(res.highest_group != 0); // postcondition
    return res;    
}

#endif //defined(STRF_SOURCE) || ! defined(STRF_SEPARATE_COMPILATION)

} // namespace detail


template <int Base> struct numpunct_c;

class numpunct_base
{
public:

    STRF_HD numpunct_base
        ( unsigned first_group ) noexcept
        : first_group_(first_group)
    {
        STRF_ASSERT(first_group_ != 0);
    }

    virtual STRF_HD ~numpunct_base()
    {
    }

    STRF_HD bool no_group_separation(unsigned digcount) const
    {
        return digcount <= first_group_;
    }
    STRF_HD unsigned first_group() const noexcept
    {
        return first_group_;
    }
    virtual STRF_HD strf::digits_distribution groups(unsigned digcount) const = 0;
    /**
      return the number of thousands separators for such number of digits
     */
    virtual STRF_HD unsigned thousands_sep_count(unsigned digcount) const = 0;

    STRF_HD char32_t thousands_sep() const noexcept
    {
        return thousands_sep_;
    }
    STRF_HD char32_t decimal_point() const noexcept
    {
        return decimal_point_;
    }

protected:

    STRF_HD void thousands_sep(char32_t ch) noexcept
    {
        thousands_sep_ = ch;
    }
    STRF_HD void decimal_point(char32_t ch) noexcept
    {
        decimal_point_ = ch;
    }

    numpunct_base(const numpunct_base&) noexcept = default;

private:

    unsigned first_group_;
    char32_t decimal_point_ = U'.';
    char32_t thousands_sep_ = U',';
};

template <int Base>
class numpunct: public strf::numpunct_base
{
public:

    numpunct(unsigned first_group_size) noexcept
        : numpunct_base(first_group_size)
    {
    }

    using category = strf::numpunct_c<Base>;

protected:

    numpunct(const numpunct& other) noexcept = default;
};

template <int Base>
class no_grouping final: public strf::numpunct<Base>
{
public:

    STRF_HD no_grouping() noexcept
        : strf::numpunct<Base>((unsigned)-1)
    {
    }
    STRF_HD strf::digits_distribution groups(unsigned digcount) const override
    {
        return {nullptr, 0, 0, 0, digcount};
    }
    STRF_HD unsigned thousands_sep_count(unsigned digcount) const override
    {
        (void)digcount;
        return 0;
    }
    STRF_HD no_grouping &  decimal_point(char32_t ch) & noexcept
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD no_grouping && decimal_point(char32_t ch) && noexcept
    {
        numpunct_base::decimal_point(ch);
        return std::move(*this);
    }
    STRF_HD char32_t decimal_point() const noexcept
    {
        return numpunct_base::decimal_point();
    }
    STRF_HD char32_t thousand_sep() const noexcept
    {
        return numpunct_base::thousands_sep();
    }
};

template <int Base>
class monotonic_grouping final: public strf::numpunct<Base>
{
public:

    STRF_HD monotonic_grouping(std::uint8_t groups_size)
        : strf::numpunct<Base>(groups_size)
    {
    }
    STRF_HD strf::digits_distribution groups(unsigned digcount) const override
    {
        STRF_ASSERT(this->first_group() <= 0xFF);
        auto grp = static_cast<std::uint8_t>(this->first_group());
        --digcount;
        return { nullptr, 0, digcount / grp, grp, (digcount % grp) + 1 };
    }    
    STRF_HD unsigned thousands_sep_count(unsigned digcount) const override
    {
        STRF_ASSERT(digcount != 0);
        STRF_ASSERT(this->first_group() != 0);
        return (digcount - 1) / this->first_group();
    }
    STRF_HD monotonic_grouping &  thousands_sep(char32_t ch) & noexcept
    {
        numpunct_base::thousands_sep(ch);
        return *this;
    }
    STRF_HD monotonic_grouping && thousands_sep(char32_t ch) && noexcept
    {
        numpunct_base::thousands_sep(ch);
        return static_cast<monotonic_grouping &&>(*this);
    }
    STRF_HD monotonic_grouping &  decimal_point(char32_t ch) & noexcept
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD monotonic_grouping && decimal_point(char32_t ch) && noexcept
    {
        numpunct_base::decimal_point(ch);
        return static_cast<monotonic_grouping &&>(*this);
    }
    STRF_HD char32_t decimal_point() const noexcept
    {
        return numpunct_base::decimal_point();
    }
    STRF_HD char32_t thousand_sep() const noexcept
    {
        return numpunct_base::thousands_sep();
    }
};


template <int Base>
class str_grouping final: public strf::numpunct<Base>
{
public:

    STRF_HD str_grouping(std::string grouping)
        : strf::numpunct<Base>
            ( grouping.empty() || grouping.front() == '\0'
            ? (unsigned)-1
            : grouping.front() )
        , impl_(grouping)
    {
    }

    str_grouping(const str_grouping&) = default;

    str_grouping(str_grouping&&) = default;

    STRF_HD strf::digits_distribution groups(unsigned digcount) const override
    {
        return impl_.get_groups(digcount);
    }
    STRF_HD unsigned thousands_sep_count(unsigned digcount) const override
    {
        return impl_.get_thousands_sep_count(digcount);
    }
    STRF_HD str_grouping &  thousands_sep(char32_t ch) & noexcept
    {
        numpunct_base::thousands_sep(ch);
        return *this;
    }
    STRF_HD str_grouping && thousands_sep(char32_t ch) && noexcept
    {
        numpunct_base::thousands_sep(ch);
        return std::move(*this);
    }
    STRF_HD str_grouping &  decimal_point(char32_t ch) & noexcept
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD str_grouping && decimal_point(char32_t ch) && noexcept
    {
        numpunct_base::decimal_point(ch);
        return std::move(*this);
    }
    STRF_HD char32_t decimal_point() const noexcept
    {
        return numpunct_base::decimal_point();
    }
    STRF_HD char32_t thousand_sep() const noexcept
    {
        return numpunct_base::thousands_sep();
    }

private:

    strf::detail::str_grouping_impl impl_;
};

template <int Base>
// made final to enable the implementation of has_punct
class default_numpunct final: public strf::numpunct<Base>
{
public:

    STRF_HD default_numpunct() noexcept
        : strf::numpunct<Base>((unsigned)-1)
    {
        numpunct_base::thousands_sep(U',');
        numpunct_base::decimal_point(U'.');
    }
    STRF_HD strf::digits_distribution groups(unsigned digcount) const override
    {
        return {nullptr, 0, 0, 0, digcount};
    }
    STRF_HD unsigned thousands_sep_count(unsigned digcount) const override
    {
        (void)digcount;
        return 0;
    }
    STRF_HD char32_t thousands_sep() const noexcept
    {
        return U',';
    }
    STRF_HD char32_t decimal_point() const noexcept
    {
        return U'.';
    }

    void thousands_sep(char32_t) = delete;
    void decimal_point(char32_t) = delete;
};

template <int Base> struct numpunct_c
{
    constexpr static bool constrainable = true;

    constexpr static int base = Base;

    static STRF_HD const strf::default_numpunct<base>& get_default()
    {
#if defined (__CUDA_ARCH__)
        // TODO: find a decent alternative to this workaround:
        const strf::default_numpunct<base> * ptr = nullptr;
        return *ptr;
#else
        static const strf::default_numpunct<base> x{};
        return x;
#endif
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

    static STRF_HD std::true_type  test_numpunct(const strf::numpunct_base&);
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

