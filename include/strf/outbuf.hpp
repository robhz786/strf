#ifndef STRF_OUTBUF_HPP_INCLUDED
#define STRF_OUTBUF_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>
#include <strf/detail/standard_lib_functions.hpp>

#include <cwchar>
#include <cstdint>

namespace strf {

namespace detail {

class outbuf_test_tool;

template <std::size_t CharSize>
struct underlying_char_type_impl;

template <> struct underlying_char_type_impl<1>{using type = std::uint8_t;};
template <> struct underlying_char_type_impl<2>{using type = char16_t;};
template <> struct underlying_char_type_impl<4>{using type = char32_t;};

} // namespace detail

template <std::size_t CharSize>
using underlying_char_type
= typename strf::detail::underlying_char_type_impl<CharSize>::type;

template <std::size_t CharSize>
constexpr STRF_HD std::size_t min_size_after_recycle()
{
    return 64;
}

template <std::size_t CharSize>
class underlying_outbuf
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    STRF_HD underlying_outbuf(const underlying_outbuf&) = delete;
    STRF_HD underlying_outbuf(underlying_outbuf&&) = delete;
    underlying_outbuf& STRF_HD operator=(const underlying_outbuf&) = delete;
    underlying_outbuf& STRF_HD operator=(underlying_outbuf&&) = delete;

    virtual STRF_HD ~underlying_outbuf() { };

    STRF_HD char_type* pos() const noexcept
    {
        return pos_;
    }
    STRF_HD char_type* end() const noexcept
    {
        return end_;
    }
    STRF_HD std::size_t size() const noexcept
    {
        STRF_ASSERT(pos_ <= end_);
        return end_ - pos_;
    }

    STRF_HD bool good() const noexcept
    {
        return good_;
    }
    STRF_HD void advance_to(char_type* p)
    {
        STRF_ASSERT(pos_ <= p);
        STRF_ASSERT(p <= end_);
        pos_ = p;
    }
    STRF_HD void advance(std::size_t n)
    {
        STRF_ASSERT(pos() + n <= end());
        pos_ += n;
    }
    STRF_HD void advance() noexcept
    {
        STRF_ASSERT(pos() < end());
        ++pos_;
    }
    STRF_HD void require(std::size_t s)
    {
        STRF_ASSERT(s <= strf::min_size_after_recycle<CharSize>());
        if (pos() + s > end()) {
            recycle();
        }
        STRF_ASSERT(pos() + s <= end());
    }
    STRF_HD void ensure(std::size_t s)
    {
        require(s);
    }

    STRF_HD virtual void recycle() = 0;

protected:

    STRF_HD underlying_outbuf(char_type* p, char_type* e) noexcept
        : pos_(p), end_(e)
    { }

    STRF_HD underlying_outbuf(char_type* p, std::size_t s) noexcept
        : pos_(p), end_(p + s)
    { }

    STRF_HD void set_pos(char_type* p) noexcept
    { pos_ = p; };
    STRF_HD void set_end(char_type* e) noexcept
    { end_ = e; };
    STRF_HD void set_good(bool g) noexcept
    { good_ = g; };

private:

    char_type* pos_;
    char_type* end_;
    bool good_ = true;
    friend class strf::detail::outbuf_test_tool;
};

template <typename CharT>
class basic_outbuf;

template <typename CharT>
class basic_outbuf: private strf::underlying_outbuf<sizeof(CharT)>
{
    using underlying_impl_ = strf::underlying_outbuf<sizeof(CharT)>;
    using underlying_char_t_ = typename underlying_impl_::char_type;

public:

    using char_type = CharT;

    STRF_HD basic_outbuf(const basic_outbuf&) = delete;
    STRF_HD basic_outbuf(basic_outbuf&&) = delete;
    STRF_HD basic_outbuf& operator=(const basic_outbuf&) = delete;
    STRF_HD basic_outbuf& operator=(basic_outbuf&&) = delete;

    virtual STRF_HD ~basic_outbuf() { };

    STRF_HD CharT* pos() const noexcept
    {
        return reinterpret_cast<CharT*>(underlying_impl_::pos());
    }
    STRF_HD CharT* end() const noexcept
    {
        return reinterpret_cast<CharT*>(underlying_impl_::end());
    }
    STRF_HD void advance_to(CharT* p)
    {
        underlying_impl_::advance_to(reinterpret_cast<underlying_char_t_*>(p));
    }
    STRF_HD underlying_impl_& as_underlying() noexcept
    {
        return *this;
    }
    STRF_HD const underlying_impl_& as_underlying() const noexcept
    {
        return *this;
    }

    using underlying_impl_::size;
    using underlying_impl_::advance;
    using underlying_impl_::good;
    using underlying_impl_::require;
    using underlying_impl_::ensure;
    using underlying_impl_::recycle;

protected:

    STRF_HD basic_outbuf(CharT* p, CharT* e) noexcept
        : underlying_impl_( reinterpret_cast<underlying_char_t_*>(p)
                          , reinterpret_cast<underlying_char_t_*>(e) )
    { }

    STRF_HD basic_outbuf(CharT* p, std::size_t s) noexcept
        : underlying_impl_(reinterpret_cast<underlying_char_t_*>(p), s)
    { }

    STRF_HD void set_pos(CharT* p) noexcept
    {
        underlying_impl_::set_pos(reinterpret_cast<underlying_char_t_*>(p));
    }
    STRF_HD void set_end(CharT* e) noexcept
    {
        underlying_impl_::set_end(reinterpret_cast<underlying_char_t_*>(e));
    }

    using underlying_impl_::set_good;
};

template <typename CharT>
class basic_outbuf_noexcept: public basic_outbuf<CharT>
{
public:

    virtual STRF_HD void recycle() noexcept = 0;

protected:

    using basic_outbuf<CharT>::basic_outbuf;
};

// global functions

namespace  detail{

template <bool NoExcept, typename CharT>
struct basic_outbuf_noexcept_switch_impl;

template <typename CharT>
struct basic_outbuf_noexcept_switch_impl<false, CharT>
{
    using type = strf::basic_outbuf<CharT>;
};

template <typename CharT>
struct basic_outbuf_noexcept_switch_impl<true, CharT>
{
    using type = strf::basic_outbuf_noexcept<CharT>;
};

template <bool NoExcept, typename CharT>
using basic_outbuf_noexcept_switch
    = typename basic_outbuf_noexcept_switch_impl<NoExcept, CharT>
   :: type;

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#  if (__GNUC__ >= 7)
#    pragma GCC diagnostic ignored "-Wstringop-overflow"
#  endif
#endif

template <typename Outbuf, typename CharT>
STRF_HD void outbuf_write_continuation(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto space = ob.size();
    STRF_ASSERT(space < len);

    detail::str_copy_n(ob.pos(), str, space);
    str += space;
    len -= space;
    ob.advance_to(ob.end());
    while (ob.good()) {
        ob.recycle();
        space = ob.size();
        if (len <= space) {
            memcpy(ob.pos(), str, len * sizeof(CharT));
            ob.advance(len);
            break;
        }
        detail::str_copy_n(ob.pos(), str, space);
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
STRF_HD void outbuf_write(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto p = ob.pos();
    if (p + len <= ob.end()) { // the common case
        strf::detail::str_copy_n(p, str, len);
        ob.advance(len);
    } else {
        detail::outbuf_write_continuation<Outbuf, CharT>(ob, str, len);
    }
}

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
STRF_HD void outbuf_put(Outbuf& ob, CharT c)
{
    auto p = ob.pos();
    if (p != ob.end()) {
        *p = c;
        ob.advance_to(p+1);
    } else {
        ob.recycle();
        *ob.pos() = c;
        ob.advance();
    }
}

} // namespace detail

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write( strf::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline STRF_HD void write( strf::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <std::size_t CharSize>
inline STRF_HD void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_char_type<CharSize>* str
    , const strf::underlying_char_type<CharSize>* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write( strf::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline STRF_HD void write( strf::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end ) noexcept
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

inline STRF_HD void write( strf::basic_outbuf<char>& ob
                 , const char* str )
{
    strf::detail::outbuf_write(ob, str, detail::strlen(str));
}

inline STRF_HD void write( strf::basic_outbuf_noexcept<char>& ob
                 , const char* str ) noexcept
{
    strf::detail::outbuf_write(ob, str, detail::strlen(str));
}

inline STRF_HD void write( strf::basic_outbuf<wchar_t>& ob
                 , const wchar_t* str )
{
#ifndef __CUDA_ARCH__
    using std::wcslen;
#endif
    strf::detail::outbuf_write(ob, str, std::wcslen(str));
}

inline STRF_HD void write( strf::basic_outbuf_noexcept<wchar_t>& ob
                 , const wchar_t* str ) noexcept
{
#ifndef __CUDA_ARCH__
    using std::wcslen;
#endif
    strf::detail::outbuf_write(ob, str, wcslen(str));
}

template <std::size_t CharSize>
inline STRF_HD void put
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::underlying_char_type<CharSize> c )
{
    strf::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline STRF_HD void put( strf::basic_outbuf<CharT>& ob, CharT c )
{
    strf::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline STRF_HD void put( strf::basic_outbuf_noexcept<CharT>& ob, CharT c ) noexcept
{
    strf::detail::outbuf_put(ob, c);
}

namespace detail {

template<std::size_t CharSize>
void STRF_HD write_fill_continuation
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;

    std::size_t space = ob.size();
    STRF_ASSERT(space < count);
    strf::detail::char_assign<char_type>(ob.pos(), space, ch);
    count -= space;
    ob.advance_to(ob.end());
    ob.recycle();
    while (ob.good()) {
        space = ob.size();
        if (count <= space) {
            strf::detail::char_assign<char_type>(ob.pos(), count, ch);
            ob.advance(count);
            break;
        }
        strf::detail::char_assign(ob.pos(), space, ch);
        count -= space;
        ob.advance_to(ob.end());
        ob.recycle();
    }
}

template <std::size_t CharSize>
inline STRF_HD void write_fill
    ( strf::underlying_outbuf<CharSize>& ob
    , std::size_t count
    , typename strf::underlying_outbuf<CharSize>::char_type ch )
{
    using char_type = typename strf::underlying_outbuf<CharSize>::char_type;
    if (count <= ob.size()) { // the common case
        strf::detail::char_assign<char_type>(ob.pos(), count, ch);
        ob.advance(count);
    } else {
        write_fill_continuation(ob, count, ch);
    }
}

template<typename CharT>
inline STRF_HD void write_fill
    ( strf::basic_outbuf<CharT>& ob
    , std::size_t count
    , CharT ch )
{
    using u_char_type = typename strf::underlying_outbuf<sizeof(CharT)>::char_type;
    write_fill(ob.as_underlying(), count, static_cast<u_char_type>(ch));
}

} // namespace detail

// type aliases

#if defined(__cpp_lib_byte)
using bin_outbuf           = basic_outbuf<std::byte>;
using bin_outbuf_noexcept  = basic_outbuf_noexcept<std::byte>;
#endif

#if defined(__cpp_char8_t)
using u8outbuf           = basic_outbuf<char8_t>;
using u8outbuf_noexcept  = basic_outbuf_noexcept<char8_t>;
#endif

using outbuf             = basic_outbuf<char>;
using outbuf_noexcept    = basic_outbuf_noexcept<char>;
using u16outbuf          = basic_outbuf<char16_t>;
using u16outbuf_noexcept = basic_outbuf_noexcept<char16_t>;
using u32outbuf          = basic_outbuf<char32_t>;
using u32outbuf_noexcept = basic_outbuf_noexcept<char32_t>;
using woutbuf            = basic_outbuf<wchar_t>;
using woutbuf_noexcept   = basic_outbuf_noexcept<wchar_t>;

namespace detail {

class outbuf_test_tool
{
public:

    template<std::size_t CharSize>
    static STRF_HD void turn_into_bad(underlying_outbuf<CharSize>& ob)
    {
        ob.set_good(false);
    }
    template<std::size_t CharSize>
    static STRF_HD void force_set_pos
        ( underlying_outbuf<CharSize>& ob
        , strf::underlying_char_type<CharSize>* pos)
    {
        ob.set_pos(pos);
    }
};


inline STRF_HD char32_t* outbuf_garbage_buf_()
{
    constexpr std::size_t s1 = (strf::min_size_after_recycle<1>() + 1) / 4;
    constexpr std::size_t s2 = (strf::min_size_after_recycle<2>() + 1) / 2;
    constexpr std::size_t s4 = strf::min_size_after_recycle<4>();
    constexpr std::size_t max_s1_s2 = s1 > s2 ? s1 : s2;
    constexpr std::size_t max_s1_s2_s4 = max_s1_s2 > s4 ? max_s1_s2 : s4;

    static char32_t arr[max_s1_s2_s4];
    return arr;
}

} // namespace detail

template <typename CharT>
inline STRF_HD CharT* outbuf_garbage_buf()
{
    return reinterpret_cast<CharT*>(strf::detail::outbuf_garbage_buf_());
}

template <typename CharT>
inline STRF_HD CharT* outbuf_garbage_buf_end()
{
    return strf::outbuf_garbage_buf<CharT>()
        + strf::min_size_after_recycle<sizeof(CharT)>();
}

template <typename CharT>
class basic_cstr_writer final: public strf::basic_outbuf_noexcept<CharT>
{
public:

    STRF_HD basic_cstr_writer(CharT* dest, CharT* dest_end)
        : basic_outbuf_noexcept<CharT>(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD basic_cstr_writer(CharT* dest, std::size_t len)
        : basic_outbuf_noexcept<CharT>(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    STRF_HD basic_cstr_writer(CharT (&dest)[N])
        : basic_outbuf_noexcept<CharT>(dest, dest + N - 1)
    {
    }

    STRF_HD basic_cstr_writer(basic_cstr_writer&& r)
        : basic_cstr_writer(r.pos(), r.end())
    {}

    STRF_HD void recycle() noexcept override
    {
        if (this->good()) {
            it_ = this->pos();
            this->set_good(false);
            this->set_end(outbuf_garbage_buf_end<CharT>());
        }
        this->set_pos(outbuf_garbage_buf<CharT>());
    }

    struct result
    {
        CharT* ptr;
        bool truncated;
    };

    STRF_HD result finish()
    {
        bool g = this->good();
        if (g) {
            it_ = this->pos();
            this->set_good(false);
        }
        this->set_pos(outbuf_garbage_buf<CharT>());
        this->set_end(outbuf_garbage_buf_end<CharT>());

        *it_ = CharT();

        return { it_, ! g };
    }

private:

    CharT* it_ = nullptr;
};


#if defined(__cpp_char8_t)
using u8cstr_writer = basic_cstr_writer<char8_t>;
#endif
using cstr_writer = basic_cstr_writer<char>;
using u16cstr_writer = basic_cstr_writer<char16_t>;
using u32cstr_writer = basic_cstr_writer<char32_t>;
using wcstr_writer = basic_cstr_writer<wchar_t>;

template <typename CharT>
class discarded_outbuf final
    : public strf::basic_outbuf_noexcept<CharT>
{
public:

    STRF_HD discarded_outbuf()
        : basic_outbuf_noexcept<CharT>
            { strf::outbuf_garbage_buf<CharT>()
            , strf::outbuf_garbage_buf_end<CharT>() }
    {
        this->set_good(false);
    }

    STRF_HD ~discarded_outbuf()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        this->set_pos(strf::outbuf_garbage_buf<CharT>());
    }
};

} // namespace strf

#endif  // STRF_OUTBUF_HPP_INCLUDED

