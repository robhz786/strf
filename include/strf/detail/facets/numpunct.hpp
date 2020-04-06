#ifndef STRF_DETAIL_FACETS_NUMPUNCT_HPP
#define STRF_DETAIL_FACETS_NUMPUNCT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <strf/detail/common.hpp>

namespace strf {

namespace detail {

class monotonic_grouping_impl
{
public:

    constexpr STRF_HD monotonic_grouping_impl(std::uint8_t groups_size)
        : _groups_size(groups_size)
    {
        STRF_ASSERT(_groups_size != 0);
    }

    STRF_HD unsigned get_thousands_sep_count(unsigned num_digits) const
    {
        return (_groups_size == 0 || num_digits == 0)
            ? 0
            : (num_digits - 1) / _groups_size;
    }

    STRF_HD std::uint8_t* get_groups
        ( unsigned num_digits
        , std::uint8_t* groups_array ) const;

private:

    const unsigned _groups_size;
};


class str_grouping_impl
{
public:

    str_grouping_impl(std::string grouping)
        : _grouping(std::move(grouping))
    {
        STRF_ASSERT(!_grouping.empty());
    }

    str_grouping_impl(const str_grouping_impl&) = default;

    str_grouping_impl(str_grouping_impl&&) = default;

    unsigned get_thousands_sep_count(unsigned num_digits) const;

    std::uint8_t* get_groups( unsigned num_digits
                            , std::uint8_t* groups_array ) const;

private:

    std::string _grouping;
};


#if defined(STRF_SOURCE) || ! defined(STRF_SEPARATE_COMPILATION)

STRF_INLINE STRF_HD
std::uint8_t* monotonic_grouping_impl::get_groups
    ( unsigned num_digits
    , std::uint8_t* groups_array ) const
{
    STRF_ASSERT(_groups_size != 0);
    while(num_digits > _groups_size) {
        *groups_array = static_cast<std::uint8_t>(_groups_size);
        ++ groups_array;
        num_digits -= _groups_size;
    }
    *groups_array = static_cast<std::uint8_t>(num_digits);
    return groups_array;
}

STRF_INLINE
unsigned str_grouping_impl::get_thousands_sep_count(unsigned num_digits) const
{
    STRF_ASSERT(!_grouping.empty());
    unsigned count = 0;
    for(auto ch : _grouping) {
        auto grp = static_cast<unsigned>(ch);
        if(grp == 0 || grp >= num_digits) {
            return count;
        }
        if(grp < num_digits) {
            ++ count;
            num_digits -= grp;
        }
    }

    return count + (num_digits - 1) / _grouping.back();
}


STRF_INLINE std::uint8_t* str_grouping_impl::get_groups
    ( unsigned num_digits
    , std::uint8_t* groups_array ) const
{
    STRF_ASSERT(!_grouping.empty());
    for(auto ch : _grouping) {
        auto group_size = static_cast<unsigned>(ch);
        if (group_size == 0) {
            *groups_array = static_cast<std::uint8_t>(num_digits);
            return groups_array;
        }
        if (group_size < num_digits) {
            *groups_array = static_cast<std::uint8_t>(group_size);
            num_digits -= group_size;
            ++ groups_array;
        } else {
            *groups_array = static_cast<std::uint8_t>(num_digits);
            return groups_array;
        }
    }
    const unsigned last_group_size = _grouping.back();
    while(num_digits > last_group_size) {
        *groups_array = static_cast<std::uint8_t>(last_group_size);
        ++ groups_array;
        num_digits -= last_group_size;
    }
    *groups_array = static_cast<std::uint8_t>(num_digits);
    return groups_array;
}

#endif //defined(STRF_SOURCE) || ! defined(STRF_SEPARATE_COMPILATION)

} // namespace detail


template <int Base> struct numpunct_c;

class numpunct_base
{
public:

    STRF_HD numpunct_base( unsigned first_group_size
                 , char32_t dec_point = U'.'
                 , char32_t sep = U',' ) noexcept
        : _first_group_size(first_group_size)
        , _decimal_point(dec_point)
        , _thousands_sep(sep)
    {
    }

    virtual STRF_HD ~numpunct_base()
    {
    }

    STRF_HD bool no_group_separation(unsigned num_digits) const
    {
        return num_digits <= _first_group_size;
    }

    /**
    Caller must ensure that groups_array has at least num_digits elements
     */
    virtual STRF_HD unsigned groups
        ( unsigned num_digits
        , std::uint8_t* groups_array ) const = 0;

    /**
      return the number of thousands separators for such number of digits
     */
    virtual STRF_HD unsigned thousands_sep_count(unsigned num_digits) const = 0;

    STRF_HD char32_t thousands_sep() const
    {
        return _thousands_sep;
    }
    STRF_HD numpunct_base &  thousands_sep(char32_t ch) &
    {
        _thousands_sep = ch;
        return *this;
    }
    STRF_HD numpunct_base && thousands_sep(char32_t ch) &&
    {
        _thousands_sep = ch;
        return std::move(*this);
    }
    STRF_HD char32_t decimal_point() const
    {
        return _decimal_point;
    }
    STRF_HD numpunct_base &  decimal_point(char32_t ch) &
    {
        _decimal_point = ch;
        return *this;
    }
    STRF_HD numpunct_base && decimal_point(char32_t ch) &&
    {
        _decimal_point = ch;
        return std::move(*this);
    }
    // int char_width() const
    // {
    //     return _char_width; // todo
    // }

    using no_group_sep = std::false_type;

protected:

        STRF_HD numpunct_base(const numpunct_base& other) noexcept
        : numpunct_base(other._first_group_size, other._decimal_point, other._thousands_sep) { }

private:

    unsigned _first_group_size;
    char32_t _decimal_point;
    char32_t _thousands_sep;
};

template <int Base>
class numpunct: public strf::numpunct_base
{
public:

    STRF_HD numpunct(unsigned first_group_size) noexcept
        : strf::numpunct_base(first_group_size)
    {}

    using category = strf::numpunct_c<Base>;

protected:

    STRF_HD numpunct(const numpunct& other) noexcept
        : strf::numpunct_base(other) { }
};

template <int Base>
class no_grouping final: public strf::numpunct<Base>
{
public:

    STRF_HD no_grouping()
        : strf::numpunct<Base>((unsigned)-1)
    {
    }

    STRF_HD unsigned groups( unsigned num_digits
                   , std::uint8_t* groups_array ) const override
    {
        STRF_ASSERT(num_digits <= 0xFF);
        *groups_array = static_cast<std::uint8_t>(num_digits);
        return 1;
    }
    STRF_HD unsigned thousands_sep_count(unsigned num_digits) const override
    {
        (void)num_digits;
        return 0;
    }
    STRF_HD no_grouping &  decimal_point(char32_t ch) &
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD no_grouping && decimal_point(char32_t ch) &&
    {
        numpunct_base::decimal_point(ch);
        return std::move(*this);
    }
    constexpr STRF_HD auto decimal_point() const
    {
        return numpunct_base::decimal_point();
    }
    constexpr STRF_HD auto thousand_sep() const
    {
        return numpunct_base::thousands_sep();
    }

    using no_group_sep = std::true_type;
};

template <int Base>
class monotonic_grouping: public strf::numpunct<Base>
{
public:

    constexpr STRF_HD monotonic_grouping(std::uint8_t groups_size)
        : strf::numpunct<Base>(groups_size)
        , _impl(groups_size)
    {
    }

    STRF_HD unsigned groups( unsigned num_digits
                           , std::uint8_t* groups_array ) const override
    {
        auto s = _impl.get_groups(num_digits, groups_array) - groups_array;
        return 1 + static_cast<unsigned>(s);
    }
    STRF_HD unsigned thousands_sep_count(unsigned num_digits) const override
    {
        return _impl.get_thousands_sep_count(num_digits);
    }
    STRF_HD monotonic_grouping &  thousands_sep(char32_t ch) &
    {
        numpunct_base::thousands_sep(ch);
        return *this;
    }
    STRF_HD monotonic_grouping && thousands_sep(char32_t ch) &&
    {
        numpunct_base::thousands_sep(ch);
        return std::move(*this);
    }
    STRF_HD monotonic_grouping &  decimal_point(char32_t ch) &
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD monotonic_grouping && decimal_point(char32_t ch) &&
    {
        numpunct_base::decimal_point(ch);
        return std::move(*this);
    }
    constexpr STRF_HD auto decimal_point() const
    {
        return numpunct_base::decimal_point();
    }
    constexpr STRF_HD auto thousand_sep() const
    {
        return numpunct_base::thousands_sep();
    }

private:

    strf::detail::monotonic_grouping_impl _impl;
};


template <int Base>
class str_grouping: public strf::numpunct<Base>
{
public:

    STRF_HD str_grouping(std::string grouping)
        : strf::numpunct<Base>
            ( grouping.empty() || grouping.front() == '\0'
            ? (unsigned)-1
            : grouping.front() )
        , _impl(grouping)
    {
    }

    str_grouping(const str_grouping&) = default;

    str_grouping(str_grouping&&) = default;

    STRF_HD unsigned groups( unsigned num_digits
                   , std::uint8_t* groups_array ) const override
    {
        auto s = _impl.get_groups(num_digits, groups_array) - groups_array;
        return 1 + static_cast<unsigned>(s);
    }
    STRF_HD unsigned thousands_sep_count(unsigned num_digits) const override
    {
        return _impl.get_thousands_sep_count(num_digits);
    }
    STRF_HD str_grouping &  thousands_sep(char32_t ch) &
    {
        numpunct_base::thousands_sep(ch);
        return *this;
    }
    STRF_HD str_grouping && thousands_sep(char32_t ch) &&
    {
        numpunct_base::thousands_sep(ch);
        return std::move(*this);
    }
    STRF_HD str_grouping &  decimal_point(char32_t ch) &
    {
        numpunct_base::decimal_point(ch);
        return *this;
    }
    STRF_HD str_grouping && decimal_point(char32_t ch) &&
    {
        numpunct_base::decimal_point(ch);
        return std::move(*this);
    }
    constexpr STRF_HD auto decimal_point() const
    {
        return numpunct_base::decimal_point();
    }
    constexpr STRF_HD auto thousand_sep() const
    {
        return numpunct_base::thousands_sep();
    }

private:

    strf::detail::str_grouping_impl _impl;
};

template <int Base>
// made final to enable the implementation of has_i18n
class default_numpunct final: public strf::numpunct<Base>
{
public:

    STRF_HD default_numpunct()
        : strf::numpunct<Base>((unsigned)-1)
    {}

    STRF_HD unsigned groups( unsigned num_digits
                   , std::uint8_t* groups_array ) const override
    {
        STRF_ASSERT(num_digits <= 0xFF);
        *groups_array = static_cast<std::uint8_t>(num_digits);
        return 1;
    }
    STRF_HD unsigned thousands_sep_count(unsigned num_digits) const override
    {
        (void)num_digits;
        return 0;
    }
    STRF_HD char32_t thousands_sep() const
    {
        return U',';
    }
    STRF_HD char32_t decimal_point() const
    {
        return U'.';
    }

    using no_group_sep = std::true_type;
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

