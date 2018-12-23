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
};

template <>
struct encoding_category<char>
{
    static constexpr bool constrainable = false;
    static constexpr bool by_value = false;

    static const encoding<char>& get_default()
    {
        static const encoding<char> obj{ stringify::v0::utf8() };
        return obj;
    }
};

template <>
struct encoding_category<char16_t>
{
    static constexpr bool constrainable = false;
    static constexpr bool by_value = false;

    static const encoding<char16_t>& get_default()
    {
        return stringify::v0::utf16();
    }
};

template <>
struct encoding_category<char32_t>
{
    static constexpr bool constrainable = false;
    static constexpr bool by_value = false;

    static const encoding<char32_t>& get_default()
    {
        return stringify::v0::utf32();
    }
};

template <>
struct encoding_category<wchar_t>
{
    static constexpr bool constrainable = false;
    static constexpr bool by_value = false;

    static const encoding<wchar_t>& get_default()
    {
        return stringify::v0::wchar_encoding();;
    }
};

struct encoding_policy_category;

class encoding_policy
{
    using _bits_type
    = typename std::underlying_type<stringify::v0::error_handling>::type;

public:

    using category = stringify::v0::encoding_policy_category;

    constexpr encoding_policy()
        : encoding_policy(stringify::v0::error_handling::replace, true)
    {
    }

    constexpr encoding_policy
        ( stringify::v0::error_handling err_hdl
        , bool allow_surr )
        : _bits(((_bits_type)err_hdl << 1) | allow_surr)
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
    constexpr static bool constrainable = false;
    constexpr static bool by_value = true;

    static const stringify::v0::encoding_policy& get_default()
    {
        static const stringify::v0::encoding_policy obj {};
        return obj;
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_FACETS_ENCODINGS_HPP

