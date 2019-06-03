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

#if defined(BOOST_STRINGIFY_SOURCE) && !defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#define BOOST_STRINGIFY_SEPARATE_COMPILATION
#endif

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION) && !defined(BOOST_STRINGIFY_SOURCE)
#define BOOST_STRINGIFY_OMIT_IMPL
#endif

#if defined(BOOST_STRINGIFY_SOURCE)
#define BOOST_STRINGIFY_EXPLICIT_TEMPLATE template
#elif defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#define BOOST_STRINGIFY_EXPLICIT_TEMPLATE extern template
#endif

#if defined(BOOST_STRINGIFY_SOURCE)
#define BOOST_STRINGIFY_STATIC_LINKAGE static
#define BOOST_STRINGIFY_INLINE
#elif !defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)
#define BOOST_STRINGIFY_INLINE inline
#define BOOST_STRINGIFY_STATIC_LINKAGE inline
#endif
/*
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
*/
// #if __cplusplus >= 201703L || ( defined(_MSC_VER) && defined(_HAS_CXX17) && _HAS_CXX17)
// #define BOOST_STRINGIFY_HAS_CXX17

#if defined(__cpp_lib_to_chars)
#include <charconv>
#define BOOST_STRINGIFY_HAS_STD_CHARCONV
#endif //defined(__cpp_lib_to_chars)


#if defined(__cpp_lib_string_view_)
#define BOOST_STRINGIFY_HAS_STD_STRING_VIEW
#define BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS constexpr
#include <string_view>
#else
#include <string> // char_traits
#endif // defined(__cpp_lib_string_view_)

#if defined(__cpp_lib_optional)
#define BOOST_STRINGIFY_HAS_STD_OPTIONAL
#include <optional>
BOOST_STRINGIFY_V0_NAMESPACE_BEGIN;
using in_place_t = ::std::in_place_t;
BOOST_STRINGIFY_V0_NAMESPACE_END;
#else
BOOST_STRINGIFY_V0_NAMESPACE_BEGIN;
struct in_place_t {};
BOOST_STRINGIFY_V0_NAMESPACE_END;
#endif //defined(__cpp_lib_optional)

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define BOOST_STRINGIFY_HAS_NODISCARD
#endif //__has_cpp_attribute(nodiscard)
#endif // defined(__has_cpp_attribute)

#ifndef BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
#define BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS inline
#endif

#if defined(BOOST_STRINGIFY_HAS_NODISCARD)
#define BOOST_STRINGIFY_NODISCARD [[nodiscard]]
#else
#define BOOST_STRINGIFY_NODISCARD
#endif //defined(BOOST_STRINGIFY_HAS_NODISCARD)

#if defined(__cpp_if_constexpr)
#define BOOST_STRINGIFY_IF_CONSTEXPR if constexpr
#else
#define BOOST_STRINGIFY_IF_CONSTEXPR if
#endif

#endif  // BOOST_STRINGIFY_V0_CONFIG_HPP

