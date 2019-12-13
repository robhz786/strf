// -*- C++ -*-
//
// This is a modified extract from the LLVM C++ library (libc++), version 8.0.1.
//
// This file is distributed under the University of Illinois Open Source
// License. See UIL_license.txt in the same directory as this file for details.

#ifndef STRF_DETAIL_CHAR_TRAITS_HPP
#define STRF_DETAIL_CHAR_TRAITS_HPP

#ifndef __CUDA_ARCH__
#include <cstring>     // for strlen
#endif

#include <strf/detail/define_specifiers.hpp>


#if __cplusplus >= 201703L
#define STRF_CONSTEXPR_SINCE_CXX17 __hd__ constexpr
#else
#define STRF_CONSTEXPR_SINCE_CXX17 __hd__ constexpr
#endif

#define ASSERT_WITH_MESSAGE(x, m) assert( (x) && m)
#define _LIBCPP_INLINE_VISIBILITY inline
//
//#define cpp_1997 199711L
//#define cpp_2003 cpp_1997
//#define cpp_2011 201103L
//#define cpp_2014 201402L
//#define cpp_2017 201703L


STRF_NAMESPACE_BEGIN

template <class _CharT>
struct  char_traits
{
    typedef _CharT    char_type;
    typedef int       int_type;
    typedef std::streamoff off_type;
    typedef std::streampos pos_type;
    typedef mbstate_t state_type;

    static __hd__ inline void STRF_CONSTEXPR_SINCE_CXX17
        assign(char_type& c1, const char_type& c2) noexcept {c1 = c2;}
    static __hd__ inline constexpr bool eq(char_type c1, char_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr bool lt(char_type c1, char_type c2) noexcept
        {return c1 < c2;}

    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    int compare(const char_type* s1, const char_type* s2, size_t n);
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    size_t length(const char_type* s);
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    const char_type* find(const char_type* s, size_t n, const char_type& a);
    static __hd__ char_type*       move(char_type* s1, const char_type* s2, size_t n);
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       copy(char_type* s1, const char_type* s2, size_t n);
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       assign(char_type* s, size_t n, char_type a);

    static __hd__ inline constexpr int_type  not_eof(int_type c) noexcept
        {return eq_int_type(c, eof()) ? ~eof() : c;}
    static __hd__ inline constexpr char_type to_char_type(int_type c) noexcept
        {return char_type(c);}
    static __hd__ inline constexpr int_type  to_int_type(char_type c) noexcept
        {return int_type(c);}
    static __hd__ inline constexpr bool      eq_int_type(int_type c1, int_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr int_type  eof() noexcept
        {return int_type(EOF);}
};

template <class _CharT>
STRF_CONSTEXPR_SINCE_CXX17 __hd__ int
char_traits<_CharT>::compare(const char_type* s1, const char_type* s2, size_t n)
{
    for (; n; --n, ++s1, ++s2)
    {
        if (lt(*s1, *s2))
            return -1;
        if (lt(*s2, *s1))
            return 1;
    }
    return 0;
}

template <class _CharT>
inline
STRF_CONSTEXPR_SINCE_CXX17 __hd__ size_t
char_traits<_CharT>::length(const char_type* s)
{
    size_t len = 0;
    for (; !eq(*s, char_type(0)); ++s)
        ++len;
    return len;
}

template <class _CharT>
inline
STRF_CONSTEXPR_SINCE_CXX17 __hd__ const _CharT*
char_traits<_CharT>::find(const char_type* s, size_t n, const char_type& a)
{
    for (; n; --n)
    {
        if (eq(*s, a))
            return s;
        ++s;
    }
    return 0;
}

template <class _CharT>
_CharT*
char_traits<_CharT>::move(char_type* s1, const char_type* s2, size_t n)
{
    char_type* r = s1;
    if (s1 < s2)
    {
        for (; n; --n, ++s1, ++s2)
            assign(*s1, *s2);
    }
    else if (s2 < s1)
    {
        s1 += n;
        s2 += n;
        for (; n; --n)
            assign(*--s1, *--s2);
    }
    return r;
}

template <class _CharT>
inline
_CharT*
char_traits<_CharT>::copy(char_type* s1, const char_type* s2, size_t n)
{
    ASSERT_WITH_MESSAGE(s2 < s1 || s2 >= s1+n, "char_traits::copy overlapped range");
    char_type* r = s1;
    for (; n; --n, ++s1, ++s2)
        assign(*s1, *s2);
    return r;
}

template <class _CharT>
inline
_CharT*
char_traits<_CharT>::assign(char_type* s, size_t n, char_type a)
{
    char_type* r = s;
    for (; n; --n, ++s)
        assign(*s, a);
    return r;
}

// char_traits<char>

template <>
struct  char_traits<char>
{
    typedef char      char_type;
    typedef int       int_type;
    typedef std::streamoff off_type;
    typedef std::streampos pos_type;
    typedef mbstate_t state_type;

    static __hd__ inline STRF_CONSTEXPR_SINCE_CXX17
    void assign(char_type& c1, const char_type& c2) noexcept {c1 = c2;}
    static __hd__ inline constexpr bool eq(char_type c1, char_type c2) noexcept
            {return c1 == c2;}
    static __hd__ inline constexpr bool lt(char_type c1, char_type c2) noexcept
        {return (unsigned char)c1 < (unsigned char)c2;}

    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    int compare(const char_type* s1, const char_type* s2, size_t n) noexcept;
    static __hd__ inline size_t STRF_CONSTEXPR_SINCE_CXX17
    length(const char_type* s)  noexcept {
#ifndef __CUDA_ARCH__
    	return std::strlen(s);
#else
    	const char_type * p = s;
    	while(*p != '\0') { ++p; };
    	return p - s;
#endif
    }
    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    const char_type* find(const char_type* s, size_t n, const char_type& a) noexcept;
    static __hd__ inline char_type* move(char_type* s1, const char_type* s2, size_t n) noexcept
        {return n == 0 ? s1 : (char_type*) memmove(s1, s2, n);}
    static __hd__ inline char_type* copy(char_type* s1, const char_type* s2, size_t n) noexcept
        {
            ASSERT_WITH_MESSAGE(s2 < s1 || s2 >= s1+n, "char_traits::copy overlapped range");
            return n == 0 ? s1 : (char_type*)memcpy(s1, s2, n);
        }
    static __hd__ inline char_type* assign(char_type* s, size_t n, char_type a) noexcept
        {return n == 0 ? s : (char_type*)memset(s, to_int_type(a), n);}

    static __hd__ inline constexpr int_type  not_eof(int_type c) noexcept
        {return eq_int_type(c, eof()) ? ~eof() : c;}
    static __hd__ inline constexpr char_type to_char_type(int_type c) noexcept
        {return char_type(c);}
    static __hd__ inline constexpr int_type to_int_type(char_type c) noexcept
        {return int_type((unsigned char)c);}
    static __hd__ inline constexpr bool eq_int_type(int_type c1, int_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr int_type  eof() noexcept
        {return int_type(EOF);}
};

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
int
char_traits<char>::compare(const char_type* s1, const char_type* s2, size_t n) noexcept
{
    if (n == 0)
        return 0;
    for (; n; --n, ++s1, ++s2)
    {
        if (lt(*s1, *s2))
            return -1;
        if (lt(*s2, *s1))
            return 1;
    }
    return 0;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ const char*
char_traits<char>::find(const char_type* s, size_t n, const char_type& a) noexcept
{
    if (n == 0)
        return nullptr;
    for (; n; --n)
    {
        if (eq(*s, a))
            return s;
        ++s;
    }
    return nullptr;
}


// char_traits<wchar_t>

template <>
struct  char_traits<wchar_t>
{
    typedef wchar_t   char_type;
    typedef wint_t    int_type;
    typedef std::streamoff off_type;
    typedef std::streampos pos_type;
    typedef mbstate_t state_type;

    static __hd__ inline STRF_CONSTEXPR_SINCE_CXX17
    void assign(char_type& c1, const char_type& c2) noexcept {c1 = c2;}
    static __hd__ inline constexpr bool eq(char_type c1, char_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr bool lt(char_type c1, char_type c2) noexcept
        {return c1 < c2;}

    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    int compare(const char_type* s1, const char_type* s2, size_t n) noexcept;
    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    size_t length(const char_type* s) noexcept;
    static __hd__ STRF_CONSTEXPR_SINCE_CXX17
    const char_type* find(const char_type* s, size_t n, const char_type& a) noexcept;
    static __hd__ inline char_type* move(char_type* s1, const char_type* s2, size_t n) noexcept
        {return n == 0 ? s1 : (char_type*)wmemmove(s1, s2, n);}
    static __hd__ inline char_type* copy(char_type* s1, const char_type* s2, size_t n) noexcept
        {
            ASSERT_WITH_MESSAGE(s2 < s1 || s2 >= s1+n, "char_traits::copy overlapped range");
            return n == 0 ? s1 : (char_type*)wmemcpy(s1, s2, n);
        }
    static __hd__ inline char_type* assign(char_type* s, size_t n, char_type a) noexcept
        {return n == 0 ? s : (char_type*)wmemset(s, a, n);}

    static __hd__ inline constexpr int_type  not_eof(int_type c) noexcept
        {return eq_int_type(c, eof()) ? ~eof() : c;}
    static __hd__ inline constexpr char_type to_char_type(int_type c) noexcept
        {return char_type(c);}
    static __hd__ inline constexpr int_type to_int_type(char_type c) noexcept
        {return int_type(c);}
    static __hd__ inline constexpr bool eq_int_type(int_type c1, int_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr int_type eof() noexcept
        {return int_type(WEOF);}
};

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ int
char_traits<wchar_t>::compare(const char_type* s1, const char_type* s2, size_t n) noexcept
{
    if (n == 0)
        return 0;
    for (; n; --n, ++s1, ++s2)
    {
        if (lt(*s1, *s2))
            return -1;
        if (lt(*s2, *s1))
            return 1;
    }
    return 0;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ size_t
char_traits<wchar_t>::length(const char_type* s) noexcept
{
    size_t len = 0;
    for (; !eq(*s, char_type(0)); ++s)
        ++len;
    return len;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ const wchar_t*
char_traits<wchar_t>::find(const char_type* s, size_t n, const char_type& a) noexcept
{
    if (n == 0)
        return nullptr;
    for (; n; --n)
    {
        if (eq(*s, a))
            return s;
        ++s;
    }
    return nullptr;
}


template <>
struct  char_traits<char16_t>
{
    typedef char16_t       char_type;
    typedef uint_least16_t int_type;
    typedef std::streamoff      off_type;
    typedef std::u16streampos   pos_type;
    typedef mbstate_t      state_type;

    static __hd__ inline STRF_CONSTEXPR_SINCE_CXX17
    void assign(char_type& c1, const char_type& c2) noexcept {c1 = c2;}
    static __hd__ inline constexpr bool eq(char_type c1, char_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr bool lt(char_type c1, char_type c2) noexcept
        {return c1 < c2;}

    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    int __hd__        compare(const char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    size_t __hd__    length(const char_type* s) noexcept;
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    const char_type* find(const char_type* s, size_t n, const char_type& a) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       move(char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       copy(char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       assign(char_type* s, size_t n, char_type a) noexcept;

    static __hd__ inline constexpr int_type  not_eof(int_type c) noexcept
        {return eq_int_type(c, eof()) ? ~eof() : c;}
    static __hd__ inline constexpr char_type to_char_type(int_type c) noexcept
        {return char_type(c);}
    static __hd__ inline constexpr int_type to_int_type(char_type c) noexcept
        {return int_type(c);}
    static __hd__ inline constexpr bool eq_int_type(int_type c1, int_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr int_type eof() noexcept
        {return int_type(0xFFFF);}
};

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ int
char_traits<char16_t>::compare(const char_type* s1, const char_type* s2, size_t n) noexcept
{
    for (; n; --n, ++s1, ++s2)
    {
        if (lt(*s1, *s2))
            return -1;
        if (lt(*s2, *s1))
            return 1;
    }
    return 0;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ size_t
char_traits<char16_t>::length(const char_type* s) noexcept
{
    size_t len = 0;
    for (; !eq(*s, char_type(0)); ++s)
        ++len;
    return len;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ const char16_t*
char_traits<char16_t>::find(const char_type* s, size_t n, const char_type& a) noexcept
{
    for (; n; --n)
    {
        if (eq(*s, a))
            return s;
        ++s;
    }
    return 0;
}

inline
__hd__ char16_t*
char_traits<char16_t>::move(char_type* s1, const char_type* s2, size_t n) noexcept
{
    char_type* r = s1;
    if (s1 < s2)
    {
        for (; n; --n, ++s1, ++s2)
            assign(*s1, *s2);
    }
    else if (s2 < s1)
    {
        s1 += n;
        s2 += n;
        for (; n; --n)
            assign(*--s1, *--s2);
    }
    return r;
}

inline
__hd__ char16_t*
char_traits<char16_t>::copy(char_type* s1, const char_type* s2, size_t n) noexcept
{
    ASSERT_WITH_MESSAGE(s2 < s1 || s2 >= s1+n, "char_traits::copy overlapped range");
    char_type* r = s1;
    for (; n; --n, ++s1, ++s2)
        assign(*s1, *s2);
    return r;
}

inline
__hd__ char16_t*
char_traits<char16_t>::assign(char_type* s, size_t n, char_type a) noexcept
{
    char_type* r = s;
    for (; n; --n, ++s)
        assign(*s, a);
    return r;
}

template <>
struct  char_traits<char32_t>
{
    typedef char32_t       char_type;
    typedef uint_least32_t int_type;
    typedef std::streamoff      off_type;
    typedef std::u32streampos   pos_type;
    typedef mbstate_t      state_type;

    static __hd__ inline STRF_CONSTEXPR_SINCE_CXX17
    void assign(char_type& c1, const char_type& c2) noexcept {c1 = c2;}
    static __hd__ inline constexpr bool eq(char_type c1, char_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr bool lt(char_type c1, char_type c2) noexcept
        {return c1 < c2;}

    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    int __hd__        compare(const char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    size_t __hd__    length(const char_type* s) noexcept;
    _LIBCPP_INLINE_VISIBILITY static STRF_CONSTEXPR_SINCE_CXX17 __hd__
    const char_type* find(const char_type* s, size_t n, const char_type& a) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       move(char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       copy(char_type* s1, const char_type* s2, size_t n) noexcept;
    _LIBCPP_INLINE_VISIBILITY
    static __hd__ char_type*       assign(char_type* s, size_t n, char_type a) noexcept;

    static __hd__ inline constexpr int_type  not_eof(int_type c) noexcept
        {return eq_int_type(c, eof()) ? ~eof() : c;}
    static __hd__ inline constexpr char_type to_char_type(int_type c) noexcept
        {return char_type(c);}
    static __hd__ inline constexpr int_type to_int_type(char_type c) noexcept
        {return int_type(c);}
    static __hd__ inline constexpr bool eq_int_type(int_type c1, int_type c2) noexcept
        {return c1 == c2;}
    static __hd__ inline constexpr int_type eof() noexcept
        {return int_type(0xFFFFFFFF);}
};

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ int
char_traits<char32_t>::compare(const char_type* s1, const char_type* s2, size_t n) noexcept
{
    for (; n; --n, ++s1, ++s2)
    {
        if (lt(*s1, *s2))
            return -1;
        if (lt(*s2, *s1))
            return 1;
    }
    return 0;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
__hd__ size_t
char_traits<char32_t>::length(const char_type* s) noexcept
{
    size_t len = 0;
    for (; !eq(*s, char_type(0)); ++s)
        ++len;
    return len;
}

inline STRF_CONSTEXPR_SINCE_CXX17 __hd__
const char32_t*
char_traits<char32_t>::find(const char_type* s, size_t n, const char_type& a) noexcept
{
    for (; n; --n)
    {
        if (eq(*s, a))
            return s;
        ++s;
    }
    return 0;
}

inline
__hd__ char32_t*
char_traits<char32_t>::move(char_type* s1, const char_type* s2, size_t n) noexcept
{
    char_type* r = s1;
    if (s1 < s2)
    {
        for (; n; --n, ++s1, ++s2)
            assign(*s1, *s2);
    }
    else if (s2 < s1)
    {
        s1 += n;
        s2 += n;
        for (; n; --n)
            assign(*--s1, *--s2);
    }
    return r;
}

inline
__hd__ char32_t*
char_traits<char32_t>::copy(char_type* s1, const char_type* s2, size_t n) noexcept
{
    ASSERT_WITH_MESSAGE(s2 < s1 || s2 >= s1+n, "char_traits::copy overlapped range");
    char_type* r = s1;
    for (; n; --n, ++s1, ++s2)
        assign(*s1, *s2);
    return r;
}

inline
__hd__ char32_t*
char_traits<char32_t>::assign(char_type* s, size_t n, char_type a) noexcept
{
    char_type* r = s;
    for (; n; --n, ++s)
        assign(*s, a);
    return r;
}

STRF_NAMESPACE_END

#include <strf/detail/undefine_specifiers.hpp>

#endif  // STRF_DETAIL_CHAR_TRAITS_HPP

