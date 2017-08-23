#ifndef BOOST_STRINGIFY_V0_CONFIG_HPP
#define BOOST_STRINGIFY_V0_CONFIG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

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


#if __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<string_view>)
#define BOOST_STRINGIFY_HAS_STD_STRING_VIEW
#include <string_view>
#endif
#endif


#define BOOST_STRINGIFY_V0_NAMESPACE_BEGIN         \
namespace boost {                                  \
namespace stringify {                              \
inline namespace v0 {                              \

#define BOOST_STRINGIFY_V0_NAMESPACE_END  } } }


#endif  // BOOST_STRINGIFY_V0_CONFIG_HPP

