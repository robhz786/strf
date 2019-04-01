#ifndef BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/transcoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
struct encoding_category;

template <typename Facet>
class facet_trait;

template <typename CharT>
class facet_trait<stringify::v0::encoding<CharT> >
{
public:
    using category = stringify::v0::encoding_category<CharT>;
    static constexpr bool store_by_value = true;
};

template <>
struct encoding_category<char>
{
    static constexpr bool constrainable = false;

    static encoding<char> get_default()
    {
        return stringify::v0::utf8();
    }
};

template <>
struct encoding_category<char16_t>
{
    static constexpr bool constrainable = false;

    static encoding<char16_t> get_default()
    {
        return stringify::v0::utf16();
    }
};

template <>
struct encoding_category<char32_t>
{
    static constexpr bool constrainable = false;

    static encoding<char32_t> get_default()
    {
        return stringify::v0::utf32();
    }
};

template <>
struct encoding_category<wchar_t>
{
    static constexpr bool constrainable = false;

    static encoding<wchar_t> get_default()
    {
        return stringify::v0::wchar_encoding();
    }
};

struct encoding_policy_category;

class encoding_policy
{
    using _bits_type
    = typename std::underlying_type<stringify::v0::error_handling>::type;

public:

    static constexpr bool store_by_value = true;
    using category = stringify::v0::encoding_policy_category;

    constexpr encoding_policy
        ( stringify::v0::error_handling err_hdl
        , bool allow_surr = false )
        : _bits(((_bits_type)err_hdl << 1) | (_bits_type)allow_surr)
    {
    }

    constexpr encoding_policy(const encoding_policy&) = default;

    constexpr encoding_policy& operator=(const encoding_policy& other)
    {
        _bits = other._bits;
        return *this;
    }

    constexpr bool operator==(const encoding_policy& other) const
    {
        return _bits == other._bits;
    }

    constexpr bool allow_surr() const
    {
        return _bits & 1;
    }

    constexpr stringify::v0::error_handling err_hdl() const
    {
        return (stringify::v0::error_handling)(_bits >> 1);
    }

private:

    _bits_type _bits;
};

struct encoding_policy_category
{
    constexpr static bool constrainable = true;

    static stringify::v0::encoding_policy get_default()
    {
        return {stringify::v0::error_handling::replace};
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP

