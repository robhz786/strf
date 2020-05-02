#ifndef STRF_DETAIL_FACETS_NUMPUNCT_HPP
#define STRF_DETAIL_FACETS_NUMPUNCT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <strf/detail/common.hpp>

namespace strf {

namespace detail {

STRF_HD unsigned get_monotonic_groups
    ( unsigned group_size, unsigned digcount, std::uint8_t* groups ) noexcept;

class str_grouping_impl
{
public:

    str_grouping_impl(std::string grouping)
        : grouping_(std::move(grouping))
    {
        STRF_ASSERT(!grouping_.empty());
    }

    str_grouping_impl(const str_grouping_impl&) = default;

    str_grouping_impl(str_grouping_impl&&) = default;

    unsigned get_thousands_sep_count(unsigned digcount) const;

    unsigned get_groups(unsigned digcount, std::uint8_t* groups) const;

private:

    std::string grouping_;
};

#if defined(STRF_SOURCE) || ! defined(STRF_SEPARATE_COMPILATION)

STRF_INLINE STRF_HD unsigned get_monotonic_groups
    ( unsigned groups_size, unsigned digcount, std::uint8_t* groups ) noexcept
{
    STRF_ASSERT(digcount != 0);
    STRF_ASSERT(groups_size != 0);
    std::uint8_t* const groups_begin = groups;

    while(digcount > groups_size) {
        *groups = static_cast<std::uint8_t>(groups_size);
        ++ groups;
        digcount -= groups_size;
    }
    *groups = static_cast<std::uint8_t>(digcount);
    return 1 + static_cast<unsigned>(groups - groups_begin);
}

STRF_INLINE
unsigned str_grouping_impl::get_thousands_sep_count(unsigned digcount) const
{
    STRF_ASSERT(!grouping_.empty());
    unsigned count = 0;
    for(auto ch : grouping_) {
        auto grp = static_cast<unsigned>(ch);
        if(grp == 0 || grp >= digcount) {
            return count;
        }
        if(grp < digcount) {
            ++ count;
            digcount -= grp;
        }
    }

    return count + (digcount - 1) / grouping_.back();
}


STRF_INLINE unsigned str_grouping_impl::get_groups
    ( unsigned digcount
    , std::uint8_t* groups ) const
{
    STRF_ASSERT(!grouping_.empty());

    std::uint8_t* const groups_begin = groups;
    for(auto ch : grouping_) {
        auto group_size = static_cast<unsigned>(ch);
        if (group_size == 0) {
            *groups = static_cast<std::uint8_t>(digcount);
            return 1 + (groups - groups_begin);
        }
        if (group_size < digcount) {
            *groups = static_cast<std::uint8_t>(group_size);
            digcount -= group_size;
            ++ groups;
        } else {
            *groups = static_cast<std::uint8_t>(digcount);
            return 1 + (groups - groups_begin);
        }
    }
    const unsigned last_group_size = grouping_.back();
    while(digcount > last_group_size) {
        *groups = static_cast<std::uint8_t>(last_group_size);
        ++ groups;
        digcount -= last_group_size;
    }
    *groups = static_cast<std::uint8_t>(digcount);
    return 1 + (groups - groups_begin);
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
    /**
    Caller must ensure that groups has at least digcount elements
     */
    virtual STRF_HD unsigned groups
        ( unsigned digcount
        , std::uint8_t* groups ) const = 0;

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

    STRF_HD unsigned groups
        ( unsigned digcount
        , std::uint8_t* groups ) const override
    {
        STRF_ASSERT(digcount <= 0xFF);
        *groups = static_cast<std::uint8_t>(digcount);
        return 1;
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

    STRF_HD unsigned groups(unsigned digcount, std::uint8_t* groups) const override
    {
        return strf::detail::get_monotonic_groups
            ( this->first_group(), digcount, groups );
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

    STRF_HD unsigned groups(unsigned digcount, std::uint8_t* groups) const override
    {
        return impl_.get_groups(digcount, groups);
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

    STRF_HD unsigned groups
        ( unsigned digcount
        , std::uint8_t* groups ) const override
    {
        STRF_ASSERT(digcount <= 0xFF);
        *groups = static_cast<std::uint8_t>(digcount);
        return 1;
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

