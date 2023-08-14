#ifndef STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
#define STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

#if defined(__CUDA_ARCH__) && ! defined(STRF_FREESTANDING)
#define STRF_FREESTANDING
#endif

#include <type_traits>
#include <limits>
#include <new>
#include <utility>    // not freestanding, but almost
                      // std::declval, std::move, std::forward

#if ! defined(STRF_FREESTANDING)
#    define STRF_WITH_CSTRING
#    include <cstring>
#    include <algorithm>   // for std::copy_n
#    define STRF_HAS_STD_ALGORITHM
#    if defined(__cpp_lib_string_view)
#        define STRF_HAS_STD_STRING_VIEW
#        define STRF_HAS_STD_STRING_DECLARATION
#        define STRF_CONSTEXPR_CHAR_TRAITS constexpr
#        include <string_view> //for char_traits
#    else
#        define STRF_HAS_STD_STRING_DECLARATION
#        define STRF_HAS_STD_STRING_DEFINITION
#        include <string>      //for char_traits

#    endif // defined(__cpp_lib_string_view)
#endif

#if ! defined(STRF_CONSTEXPR_CHAR_TRAITS)
#    define STRF_CONSTEXPR_CHAR_TRAITS inline
#endif
#if defined(STRF_WITH_CSTRING)
#    include <cstring>
#endif

#if defined(__cpp_lib_bit_cast) || defined(__cpp_lib_bitops)
#include <bit>
#endif

namespace strf {
namespace detail {

#if defined(__cpp_lib_bitops)
#  define STRF_HAS_COUNTL_ZERO
#  define STRF_HAS_COUNTR_ZERO

// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countl_zero_l(unsigned long x) noexcept
{
    return std::countl_zero(x);
}
// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countl_zero_ll(unsigned long long x) noexcept
{
    return std::countl_zero(x);
}
// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countr_zero_l(unsigned long x) noexcept
{
    return std::countr_zero(x);
}
// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countr_zero_ll(unsigned long long x) noexcept
{
    return std::countr_zero(x);
}

#elif defined(__has_builtin)
#  if __has_builtin(__builtin_clzll)
#    define STRF_HAS_COUNTL_ZERO

// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countl_zero_l(unsigned long x) noexcept
{
    return __builtin_clzl(x);
}
// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countl_zero_ll(unsigned long long x) noexcept
{
    return __builtin_clzll(x);
}

#  endif // __has_builtin(__builtin_clzll)

#  if __has_builtin(__builtin_ctzll)
#    define STRF_HAS_COUNTR_ZERO

// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countr_zero_l(unsigned long x) noexcept
{
    return __builtin_ctzl(x);
}
// NOLINTNEXTLINE(google-runtime-int)
inline STRF_HD int countr_zero_ll(unsigned long long x) noexcept
{
    return __builtin_ctzll(x);
}

#  endif // __has_builtin(__builtin_ctzll)
#endif

#if defined(__cpp_lib_bit_cast)

template< class To, class From >
constexpr To STRF_HD bit_cast(const From& from) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "");
    return std::bit_cast<To, From>(from);
}

#else // defined(__cpp_lib_bit_cast)

template< class To, class From >
To STRF_HD bit_cast(const From& from) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "");

#if defined(STRF_WITH_CSTRING)
    To to;
    memcpy(&to, &from, sizeof(to));
    return to;
#else
    return *reinterpret_cast<const To*>(&from);
#endif
}

#endif // defined(__cpp_lib_bit_cast)

STRF_HD constexpr int slow_countl_zero_ll(std::uint64_t x)
{
    if (x <= 0xffffffffULL) {
        if (x <= 0xffffULL) {
            if (x <= 0xffULL) {
                if (x <= 0xfULL) {
                    if (x <= 3U) {
                        if (x == 0) {
                            return 64;
                        }
                        return x <= 1U ? 63 : 62;
                    }
                    return x <= 7U ? 61 : 60;
                }
                if (x <= 0x3fU) {
                    return x <= 0x1fU ? 59 : 58;
                }
                return x <= 0x7fU ? 57 : 56;
            }
            if (x <= 0xfffULL) {
                if (x <= 0x3ffULL) {
                    return x <= 0x1ffULL ? 55 :54;
                }
                return x <= 0x7ffULL ? 53 : 52;
            }
            if (x <= 0x3fffULL) {
                return x <= 0x1fffULL ? 51 :50;
            }
            return x <= 0x7fffULL ? 49 : 48;
        }
        if (x <= 0xffffffULL) {
            if (x <= 0xfffffULL) {
                if (x <= 0x3ffffULL) {
                    return x <= 0x1ffffULL ? 47 : 46;
                }
                return x <= 0x7ffffULL ? 45 : 44;
            }
            if (x <= 0x3fffffULL) {
                return x <= 0x1fffffULL ? 43 : 42 ;
            }
            return x <= 0x7fffffULL ? 41 : 40 ;
        }
        if (x <= 0xfffffffULL) {
            if (x <= 0x3ffffffULL) {
                return x <= 0x1ffffffULL ? 39 : 38 ;
            }
            return x <= 0x7ffffffULL ? 37 : 36;
        }
        if (x <= 0x3fffffffULL) {
            return x <= 0x1fffffffULL ? 35 : 34;
        }
        return x <= 0x7fffffffULL ? 33 : 32;
    }
    if (x <= 0xffffffffffffULL) {
        if (x <= 0xffffffffffULL) {
            if (x <= 0xfffffffffULL) {
                if (x <= 0x3ffffffffULL) {
                    return x <= 0x1ffffffffULL ? 31 : 30;
                }
                return x <= 0x7ffffffffULL ? 29 : 28;
            }
            if (x <= 0x3fffffffffULL) {
                return x <= 0x1fffffffffULL ? 27 : 26;
            }
            return x <= 0x7fffffffffULL ? 25 : 24;
        }
        if (x <= 0xfffffffffffULL) {
            if (x <= 0x3ffffffffffULL) {
                return x <= 0x1ffffffffffULL ? 23 : 22;
            }
            return x <= 0x7ffffffffffULL ? 21 : 20;
        }
        if (x <= 0x3fffffffffffULL) {
            return x <= 0x1fffffffffffULL ? 19 : 18;
        }
        return x <= 0x7fffffffffffULL ? 17 : 16;
    }
    if (x <= 0xffffffffffffffULL) {
        if (x <= 0xfffffffffffffULL) {
            if (x <= 0x3ffffffffffffULL) {
                return x <= 0x1ffffffffffffULL ? 15 : 14;
            }
            return x <= 0x7ffffffffffffULL ? 13 : 12;
        }
        if (x <= 0x3fffffffffffffULL) {
            return x <= 0x1fffffffffffffULL ? 11 : 10 ;
        }
        return x <= 0x7fffffffffffffULL ? 9 : 8 ;
    }
    if (x <= 0xfffffffffffffffULL) {
        if (x <= 0x3ffffffffffffffULL) {
            return x <= 0x1ffffffffffffffULL ? 7 : 6 ;
        }
        return x <= 0x7ffffffffffffffULL ? 5 : 4;
    }
    if (x <= 0x3fffffffffffffffULL) {
        return x <= 0x1fffffffffffffffULL ? 3 : 2;
    }
    return x <= 0x7fffffffffffffffULL ? 1 : 0;
}


#if ! defined(STRF_FREESTANDING)
template <typename It>
using iterator_value_type = typename std::iterator_traits<It>::value_type;

#else

template <typename It>
struct iterator_value_type_impl
{
    using type = typename It::value_type;
    //strf::remove_cvref_t<decltype(*std::declval<It>())>;
};

template <typename T>
struct iterator_value_type_impl<T*>
{
    using type = strf::detail::remove_cv_t<T>;
};

template <typename It>
using iterator_value_type = typename iterator_value_type_impl<It>::type;

#endif

template<class CharT>
STRF_CONSTEXPR_CHAR_TRAITS
STRF_HD void str_fill_n(CharT* str, std::ptrdiff_t count, CharT value)
{
#if !defined(STRF_FREESTANDING)
    std::char_traits<CharT>::assign(str, detail::safe_cast_size_t(count), value);
#else
    auto p = str;
    for (std::ptrdiff_t i = 0; i < count; ++i, ++p) {
        *p = value;
    }
#endif
}

template<>
inline
STRF_HD void str_fill_n<char>(char* str, std::ptrdiff_t count, char value)
{
#if !defined(STRF_FREESTANDING)
    std::char_traits<char>::assign(str, detail::safe_cast_size_t(count), value);
#elif defined(STRF_WITH_CSTRING)
    memset(str, value, detail::safe_cast_size_t(count));
#else
    auto p = str;
    for (std::ptrdiff_t i = 0; i < count; ++i, ++p) {
        *p = value;
    }
#endif
}

template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD std::size_t
str_length(const CharT* str)
{
#if !defined(STRF_FREESTANDING)
    return std::char_traits<CharT>::length(str);
#else
    std::size_t length { 0 };
    while (*str != CharT{0}) { ++str, ++length; }
    return length;
#endif
}

template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD std::ptrdiff_t
str_ssize(const CharT* str)
{
    return static_cast<std::ptrdiff_t>(str_length(str));
}

template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD const CharT*
str_find(const CharT* p, std::size_t count, const CharT& ch) noexcept
{
#if !defined(STRF_FREESTANDING)
    return std::char_traits<CharT>::find(p, count, ch);
#else
    for (std::size_t i = 0; i != count; ++i, ++p) {
        if (*p == ch) {
            return p;
        }
    }
    return nullptr;
#endif
}


template <class CharT>
STRF_CONSTEXPR_CHAR_TRAITS STRF_HD bool
str_equal(const CharT* str1, const CharT* str2, std::size_t count)
{
#if !defined(STRF_FREESTANDING)
    return 0 == std::char_traits<CharT>::compare(str1, str2, count);
#else
    for(;count != 0; ++str1, ++str2, --count) {
        if (*str1 != *str2) {
            return false;
        }
    }
    return true;
#endif
}

template <bool SameCharSize, typename SrcCharT, typename DstCharT>
struct interchar_copy_n_impl;

template <typename CharT>
struct interchar_copy_n_impl<true, CharT, CharT>
{
    static inline STRF_HD void copy(const CharT* src, std::size_t count, CharT* dst)
    {
#if defined(STRF_WITH_CSTRING)
        memcpy(dst, src, count * sizeof(CharT));
#else
        for(;count != 0; ++dst, ++src, --count) {
            *dst = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DstCharT>
struct interchar_copy_n_impl<true, SrcCharT, DstCharT>
{
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");

    static inline STRF_HD void copy(const SrcCharT* src, std::size_t count, DstCharT* dst)
    {
#if defined(STRF_WITH_CSTRING)
        memcpy(dst, src, count * sizeof(SrcCharT));
#else
        for(;count != 0; ++dst, ++src, --count) {
            *dst = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DstCharT>
struct interchar_copy_n_impl<false, SrcCharT, DstCharT>
{
    static_assert(sizeof(SrcCharT) != sizeof(DstCharT), "");

    static inline STRF_HD void copy(const SrcCharT* src, std::size_t count, DstCharT* dst)
    {
#if !defined(STRF_FREESTANDING)
        std::copy_n(src, count, dst);
#else
        for(;count != 0; ++dst, ++src, --count) {
            *dst = *src;
        }
#endif
    }
};

template <typename SrcCharT, typename DstCharT>
inline STRF_HD void copy_n
    ( const SrcCharT* src
    , std::size_t count
    , DstCharT* dst )
{
    using impl = strf::detail::interchar_copy_n_impl
        < sizeof(SrcCharT) == sizeof(DstCharT), SrcCharT, DstCharT >;
    impl::copy(src, count, dst);
}

// template< class T >
// constexpr T&& forward( strf::detail::remove_reference_t<T>&& t ) noexcept
// {
//     return static_cast<T&&>(t);
// }


namespace detail_tag_invoke_ns {

STRF_HD inline void tag_invoke(){}

struct tag_invoke_fn
{
    template <typename Cpo, typename ... Args>
    constexpr STRF_HD auto operator()(Cpo cpo, Args&&... args) const
        noexcept(noexcept(tag_invoke(cpo, (Args&&)(args)...)))
        -> decltype(tag_invoke(cpo, (Args&&)(args)...))
    {
        return tag_invoke(cpo, (Args&&)(args)...);
    }
};

} // namespace detail_tag_invoke_ns

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename Cpo, typename ... Args>
constexpr STRF_HD auto tag_invoke(Cpo cpo, Args&&... args)
    noexcept(noexcept(tag_invoke(cpo, (Args&&)(args)...)))
    -> decltype(tag_invoke(cpo, (Args&&)(args)...))
{
    return tag_invoke(cpo, (Args&&)(args)...);
}

#else

constexpr strf::detail::detail_tag_invoke_ns::tag_invoke_fn tag_invoke {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

// Not using generic iterators because std::distance
// is not freestanding and can't be used in CUDA devices
template <class T, class Compare>
STRF_HD const T* lower_bound
    ( const T* first
    , const T* last
    , const T& value
    , Compare comp )
{
    std::ptrdiff_t search_range_length { last - first };

    const T* iter = nullptr;
    while (search_range_length > 0) {
        auto half_range_length = search_range_length/2;
        iter = first;
        iter += half_range_length;
        if (comp(*iter, value)) {
            first = ++iter;
            search_range_length -= (half_range_length + 1);
                // the extra +1 is since we've just checked the midpoint
        }
        else {
            search_range_length = half_range_length;
        }
    }
    return first;
}


namespace int_limits_ {

// because numeric_limits::max() can't be used in CUDA devices
// without emitting a warning

template <typename IntT, bool Signed>
struct int_limits_impl;

template <typename IntT>
struct int_limits_impl<IntT, true>
{
    static_assert(std::is_integral<IntT>::value, "");

    // assuming two's complement
    using uint_t = typename std::make_unsigned<IntT>::type;
    constexpr static unsigned bits_count = sizeof(IntT) * 8;
    constexpr static uint_t min_value_bits_ = (uint_t(1) << (bits_count - 1));
    constexpr static uint_t max_value_bits_ = (uint_t(-1) >> 1);

    constexpr static IntT min_value = static_cast<IntT>(min_value_bits_);
    constexpr static IntT max_value = static_cast<IntT>(max_value_bits_);
};

template <typename UIntT>
struct int_limits_impl<UIntT, false>
{
    static_assert(std::is_integral<UIntT>::value, "");
    constexpr static UIntT min_value = 0;
    constexpr static UIntT max_value = (UIntT)-1;
};

} // namespace int_limits_

template <typename IntT>
using int_limits = int_limits_::int_limits_impl<IntT, std::is_signed<IntT>::value>;

template <typename IntT>
constexpr STRF_HD IntT int_max()
{
    return int_limits<IntT>::max_value;
}

template <typename IntT>
constexpr STRF_HD IntT int_min()
{
    return int_limits<IntT>::min_value;
}

} // namespace detail
} // namespace strf

#endif // STRF_DETAIL_STANDARD_LIB_FUNCTIONS_HPP
