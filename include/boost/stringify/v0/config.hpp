#ifndef BOOST_STRINGIFY_V0_CONFIG_HPP
#define BOOST_STRINGIFY_V0_CONFIG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

#define BOOST_STRINGIFY_V0_NAMESPACE_BEGIN         \
namespace boost {                                  \
namespace stringify {                              \
inline namespace v0 {                              \

#define BOOST_STRINGIFY_V0_NAMESPACE_END  } } }

#if defined(BOOST_STRINGIFY_SOURCE) && !defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)
#define BOOST_STRINGIFY_NOT_HEADER_ONLY
#endif

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY) && !defined(BOOST_STRINGIFY_SOURCE)
#define BOOST_STRINGIFY_OMIT_IMPL
#endif

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)
#define BOOST_STRINGIFY_INLINE
#else
#define BOOST_STRINGIFY_INLINE inline
#endif

#if defined(BOOST_STRINGIFY_SOURCE)
#define BOOST_STRINGIFY_EXPLICIT_TEMPLATE template
#else
#define BOOST_STRINGIFY_EXPLICIT_TEMPLATE extern template
#endif


#if defined(_MSC_VER)
#if _MSC_VER < 1911
#define BOOST_STRINGIFY_NO_NODISCARD
#endif

#elif defined(__GNUC__) && __GNUC__ < 7
#define BOOST_STRINGIFY_NO_NODISCARD

#elif defined(__clang__)
#if __has_attribute(nodiscard) == 0
#define BOOST_STRINGIFY_NO_NODISCARD
#endif

#elif defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1800
#define BOOST_STRINGIFY_NO_NODISCARD

#elif __cplusplus < 201703L
#define BOOST_STRINGIFY_NO_NODISCARD
#endif

#if __cplusplus >= 201703L || ( defined(_MSC_VER) && defined(_HAS_CXX17))

#define BOOST_STRINGIFY_HAS_CXX17

#if ! defined(BOOST_STRINGIFY_NO_NODISCARD)
#define BOOST_STRINGIFY_HAS_NODISCARD
#endif

#if defined(__has_include)

#if __has_include(<charconv>)
#include <charconv>
#define BOOST_STRINGIFY_HAS_STD_CHARCONV
#endif

#if __has_include(<string_view>)
#define BOOST_STRINGIFY_HAS_STD_STRING_VIEW
#include <string_view>
#endif

#if __has_include(<optional>)
#define BOOST_STRINGIFY_HAS_STD_OPTIONAL
#include <optional>
BOOST_STRINGIFY_V0_NAMESPACE_BEGIN;
using in_place_t = ::std::in_place_t;
BOOST_STRINGIFY_V0_NAMESPACE_END;
#endif

#endif // defined(__has_include)

#endif // __cplusplus >= 201703L || ( defined(_MSV_VER) && defined(_HAS_CXX17))

#if defined(BOOST_STRINGIFY_HAS_NODISCARD)
#define BOOST_STRINGIFY_NODISCARD [[nodiscard]]
#else
#define BOOST_STRINGIFY_NODISCARD
#endif //defined(BOOST_STRINGIFY_HAS_NODISCARD)


#if ! defined(BOOST_STRINGIFY_HAS_STD_OPTIONAL)
BOOST_STRINGIFY_V0_NAMESPACE_BEGIN;
struct in_place_t {};
BOOST_STRINGIFY_V0_NAMESPACE_END;
#endif //! defined(BOOST_STRINGIFY_HAS_STD_OPTIONAL)


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail
{

constexpr std::integral_constant<bool, sizeof(wchar_t) == 2> wchar_is_16 {};
constexpr std::integral_constant<bool, sizeof(wchar_t) == 4> wchar_is_32 {};
using wchar_equivalent =
    typename std::conditional<sizeof(wchar_t) == 4, char32_t, char16_t>::type;

}
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_CONFIG_HPP

