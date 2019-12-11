#ifndef STRF_DETAIL_COMMMON_HPP
#define STRF_DETAIL_COMMMON_HPP

// TODO: This seems to rely on some standard library headers which have host-side-only code!
// double-check and either avoid the reliance or duplicate the headers :-(

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <cassert>

#define STRF_ASSERT(x) assert(x)

#define STRF_NAMESPACE_BEGIN namespace strf {
#define STRF_NAMESPACE_END  }

#if defined(STRF_SOURCE) && !defined(STRF_SEPARATE_COMPILATION)
#define STRF_SEPARATE_COMPILATION
#endif

#if defined(STRF_SEPARATE_COMPILATION) && !defined(STRF_SOURCE)
#define STRF_OMIT_IMPL
#endif

#if defined(STRF_SOURCE)
#define STRF_EXPLICIT_TEMPLATE template
#elif defined(STRF_SEPARATE_COMPILATION)
#define STRF_EXPLICIT_TEMPLATE extern template
#endif

#if defined(STRF_SOURCE)
#define STRF_STATIC_LINKAGE static
#define STRF_INLINE
#elif !defined(STRF_SEPARATE_COMPILATION)
#define STRF_INLINE inline
#define STRF_STATIC_LINKAGE inline
#endif
/*
#if defined(_MSC_VER)
#if _MSC_VER < 1911
#define STRF_NO_NODISCARD
#endif

#elif defined(__GNUC__) && __GNUC__ < 7
#define STRF_NO_NODISCARD

#elif defined(__clang__)
#if __has_attribute(nodiscard) == 0
#define STRF_NO_NODISCARD
#endif

#elif defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1800
#define STRF_NO_NODISCARD

#elif __cplusplus < 201703L
#define STRF_NO_NODISCARD
#endif
*/
// #if __cplusplus >= 201703L || ( defined(_MSC_VER) && defined(_HAS_CXX17) && _HAS_CXX17)
// #define STRF_HAS_CXX17

#if defined(__cpp_lib_to_chars)
#include <charconv>
#define STRF_HAS_STD_CHARCONV
#endif //defined(__cpp_lib_to_chars)


#if defined(__cpp_lib_string_view_)
#define STRF_HAS_STD_STRING_VIEW
#define STRF_CONSTEXPR_CHAR_TRAITS constexpr
#include <string_view>
#else
#include <string> // char_traits
#endif // defined(__cpp_lib_string_view_)

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define STRF_HAS_NODISCARD
#endif //__has_cpp_attribute(nodiscard)
#endif // defined(__has_cpp_attribute)

#ifndef STRF_CONSTEXPR_CHAR_TRAITS
#define STRF_CONSTEXPR_CHAR_TRAITS inline
#endif

#if defined(STRF_HAS_NODISCARD)
#define STRF_NODISCARD [[nodiscard]]
#else
#define STRF_NODISCARD
#endif //defined(STRF_HAS_NODISCARD)

#if defined(__cpp_if_constexpr)
#define STRF_IF_CONSTEXPR if constexpr
#else
#define STRF_IF_CONSTEXPR if
#endif


// cxx17 guaranteed copy elision
#if defined(_MSC_VER)
#  if ((_MSC_VER < 1913) || (_MSVC_LANG < 201703))
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif defined(__clang__)
#  if (__clang_major__ < 4) || (__cplusplus < 201703L)
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif defined(__GNUC__)
#  if (__GNUC__ < 7) || (__cplusplus < 201703)
#    define STRF_NO_CXX17_COPY_ELISION
#  endif

#elif (__cplusplus < 201703)
#  define STRF_NO_CXX17_COPY_ELISION

#elif ! defined(__cpp_deduction_guides) || (__cpp_deduction_guides < 201703)
   // compilers that dont support Class template argument deductions
   // usually also dont support guaranteed copy elision
#  define STRF_NO_CXX17_COPY_ELISION
#endif

#include <strf/detail/define_specifiers.hpp>

STRF_NAMESPACE_BEGIN

namespace detail
{

#if defined(__cpp_fold_expressions)

template <bool ... C> constexpr bool fold_and = (C && ...);
template <bool ... C> constexpr bool fold_or  = (C || ...);

#else //defined(__cpp_fold_expressions)

template <bool ... > struct fold_and_impl;
template <bool ... > struct fold_or_impl;

template <> struct fold_and_impl<>
{
    constexpr static bool value = true;
};

template <> struct fold_or_impl<>
{
    constexpr static bool value = false;
};

template <bool C0, bool ... C>
struct fold_and_impl<C0, C...>
{
     constexpr static bool value = fold_and_impl<C...>::value && C0;
};

template <bool C0, bool ... C>
struct fold_or_impl<C0, C...>
{
     constexpr static bool value = fold_or_impl<C...>::value || C0;
};

template <bool ... C> constexpr bool fold_and = fold_and_impl<C...>::value;
template <bool ... C> constexpr bool fold_or = fold_or_impl<C...>::value;

#endif // defined(__cpp_fold_expressions)

} // namespace detail

struct absolute_lowest_rank
{
    explicit __hd__ absolute_lowest_rank() = default;
};

template <std::size_t N>
struct rank: rank<N - 1>
{
    explicit __hd__ rank() = default;
};

template <>
struct rank<0>: absolute_lowest_rank
{
    explicit __hd__ rank() = default;
};

template <typename ... >
struct tag
{
    explicit __hd__ tag() = default;
};

STRF_NAMESPACE_END

#include <strf/detail/undefine_specifiers.hpp>

#endif  // STRF_DETAIL_COMMMON_HPP

