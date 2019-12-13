#ifndef BOOST_OUTBUF_HPP
#define BOOST_OUTBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/common.hpp>
#ifndef __CUDA_ARCH__
#include <cstring>
	// TODO: Is this used for anything other than memcpy?
#else
	// TODO: If necessary, I have my
	// own - implemented but not properly tested - <string.h>
	// GPU-side per-thread implementation which might be used
	// for this purpose. But - how much of it do we really need?
#endif
#include <cwchar>
#include <cstdint>

#include <strf/detail/define_specifiers.hpp>

STRF_NAMESPACE_BEGIN

namespace detail {

class outbuf_test_tool;

template <std::size_t CharSize>
struct underlying_outbuf_char_type_impl;

template <> struct underlying_outbuf_char_type_impl<1>{using type = std::uint8_t;};
template <> struct underlying_outbuf_char_type_impl<2>{using type = char16_t;};
template <> struct underlying_outbuf_char_type_impl<4>{using type = char32_t;};

} // namespace detail

template <std::size_t CharSize>
using underlying_outbuf_char_type
= typename strf::detail::underlying_outbuf_char_type_impl<CharSize>::type;

template <typename CharT>
constexpr __hd__ std::size_t min_size_after_recycle()
{
    return 64;
}

template <std::size_t CharSize>
class underlying_outbuf
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    __hd__ underlying_outbuf(const underlying_outbuf&) = delete;
    __hd__ underlying_outbuf(underlying_outbuf&&) = delete;
    underlying_outbuf& __hd__ operator=(const underlying_outbuf&) = delete;
    underlying_outbuf& __hd__ operator=(underlying_outbuf&&) = delete;

    virtual __hd__ ~underlying_outbuf() = default;

    __hd__ char_type* pos() const noexcept
    {
        return _pos;
    }
    __hd__ char_type* end() const noexcept
    {
        return _end;
    }
    __hd__ std::size_t size() const noexcept
    {
        STRF_ASSERT(_pos <= _end);
        return _end - _pos;
    }

    __hd__ bool good() const noexcept
    {
        return _good;
    }
    __hd__ void advance_to(char_type* p)
    {
        STRF_ASSERT(_pos <= p);
        STRF_ASSERT(p <= _end);
        _pos = p;
    }
    __hd__ void advance(std::size_t n)
    {
        STRF_ASSERT(pos() + n <= end());
        _pos += n;
    }
    __hd__ void advance() noexcept
    {
        STRF_ASSERT(pos() < end());
        ++_pos;
    }
    __hd__ void require(std::size_t s)
    {
        STRF_ASSERT(s <= strf::min_size_after_recycle<char_type>());
        if (pos() + s > end())
        {
            recycle();
        }
        STRF_ASSERT(pos() + s <= end());
    }
    __hd__ void ensure(std::size_t s)
    {
        require(s);
    }

    __hd__ virtual void recycle() = 0;

protected:

    __hd__ underlying_outbuf(char_type* pos_, char_type* end_) noexcept
        : _pos(pos_), _end(end_)
    { }

    __hd__ underlying_outbuf(char_type* pos_, std::size_t s) noexcept
        : _pos(pos_), _end(pos_ + s)
    { }

    __hd__ void set_pos(char_type* p) noexcept
    { _pos = p; };
    __hd__ void set_end(char_type* e) noexcept
    { _end = e; };
    __hd__ void set_good(bool g) noexcept
    { _good = g; };

private:

    char_type* _pos;
    char_type* _end;
    bool _good = true;
    friend class strf::detail::outbuf_test_tool;
};

template <typename CharT>
class basic_outbuf;

template <typename CharT>
class basic_outbuf: private strf::underlying_outbuf<sizeof(CharT)>
{
    using _underlying_impl = strf::underlying_outbuf<sizeof(CharT)>;
    using _underlying_char_t = typename _underlying_impl::char_type;

public:

    using char_type = CharT;

    __hd__ basic_outbuf(const basic_outbuf&) = delete;
    __hd__ basic_outbuf(basic_outbuf&&) = delete;
    __hd__ basic_outbuf& operator=(const basic_outbuf&) = delete;
    __hd__ basic_outbuf& operator=(basic_outbuf&&) = delete;

    virtual __hd__ ~basic_outbuf() = default;

    __hd__ CharT* pos() const noexcept
    {
        return reinterpret_cast<CharT*>(_underlying_impl::pos());
    }
    __hd__ CharT* end() const noexcept
    {
        return reinterpret_cast<CharT*>(_underlying_impl::end());
    }
    __hd__ void advance_to(CharT* p)
    {
        _underlying_impl::advance_to(reinterpret_cast<_underlying_char_t*>(p));
    }
    __hd__ _underlying_impl& as_underlying() noexcept
    {
        return *this;
    }
    __hd__ const _underlying_impl& as_underlying() const noexcept
    {
        return *this;
    }

    using _underlying_impl::size;
    using _underlying_impl::advance;
    using _underlying_impl::good;
    using _underlying_impl::require;
    using _underlying_impl::ensure;
    using _underlying_impl::recycle;

protected:

    __hd__ basic_outbuf(CharT* pos_, CharT* end_) noexcept
        : _underlying_impl( reinterpret_cast<_underlying_char_t*>(pos_)
                          , reinterpret_cast<_underlying_char_t*>(end_) )
    { }

    __hd__ basic_outbuf(CharT* pos_, std::size_t s) noexcept
        : _underlying_impl(reinterpret_cast<_underlying_char_t*>(pos_), s)
    { }

    __hd__ void set_pos(CharT* p) noexcept
    {
        _underlying_impl::set_pos(reinterpret_cast<_underlying_char_t*>(p));
    }
    __hd__ void set_end(CharT* e) noexcept
    {
        _underlying_impl::set_end(reinterpret_cast<_underlying_char_t*>(e));
    }

    using _underlying_impl::set_good;
};

template <typename CharT>
class basic_outbuf_noexcept: public basic_outbuf<CharT>
{
public:

    virtual __hd__ void recycle() noexcept = 0;

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
__hd__ void outbuf_write_continuation(Outbuf& ob, const CharT* str, std::size_t len)
{
#ifndef __CUDA_ARCH__
	using std::memcpy;
#else
	// CUDA has a device-side memcpy(); see:
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations
#endif
    auto space = ob.size();
    STRF_ASSERT(space < len);

    memcpy(ob.pos(), str, space * sizeof(CharT));
    str += space;
    len -= space;
    ob.advance_to(ob.end());
    while (ob.good())
    {
        ob.recycle();
        space = ob.size();
        if (len <= space)
        {
            memcpy(ob.pos(), str, len * sizeof(CharT));
            ob.advance(len);
            break;
        }
        memcpy(ob.pos(), str, space * sizeof(CharT));
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
__hd__ void outbuf_write(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto p = ob.pos();
    if (p + len <= ob.end()) // the common case
    {
#ifndef __CUDA_ARCH__
    	using std::memcpy;
#endif
        memcpy(p, str, len * sizeof(CharT));
        ob.advance(len);
    }
    else
    {
        detail::outbuf_write_continuation<Outbuf, CharT>(ob, str, len);
    }
}

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
__hd__ void outbuf_put(Outbuf& ob, CharT c)
{
    auto p = ob.pos();
    if (p != ob.end())
    {
        *p = c;
        ob.advance_to(p+1);
    }
    else
    {
        ob.recycle();
        *ob.pos() = c;
        ob.advance();
    }
}

} // namespace detail

template <std::size_t CharSize>
inline __hd__ void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_outbuf_char_type<CharSize>* str
    , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline __hd__ void write( strf::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline __hd__ void write( strf::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    strf::detail::outbuf_write(ob, str, len);
}

template <std::size_t CharSize>
inline __hd__ void write
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::underlying_outbuf_char_type<CharSize>* str
    , const strf::underlying_outbuf_char_type<CharSize>* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline __hd__ void write( strf::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end )
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline __hd__ void write( strf::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end ) noexcept
{
    STRF_ASSERT(str_end >= str);
    strf::detail::outbuf_write(ob, str, str_end - str);
}

inline __hd__ void write( strf::basic_outbuf<char>& ob
                 , const char* str )
{
#ifndef __CUDA_ARCH__
	using std::strlen;
#endif
    strf::detail::outbuf_write(ob, str, strlen(str));
}

inline __hd__ void write( strf::basic_outbuf_noexcept<char>& ob
                 , const char* str ) noexcept
{
#ifndef __CUDA_ARCH__
	using std::strlen;
#endif
    strf::detail::outbuf_write(ob, str, strlen(str));
}

inline __hd__ void write( strf::basic_outbuf<wchar_t>& ob
                 , const wchar_t* str )
{
#ifndef __CUDA_ARCH__
	using std::wcslen;
#endif
strf::detail::outbuf_write(ob, str, std::wcslen(str));
}

inline __hd__ void write( strf::basic_outbuf_noexcept<wchar_t>& ob
                 , const wchar_t* str ) noexcept
{
#ifndef __CUDA_ARCH__
	using std::wcslen;
#endif
    strf::detail::outbuf_write(ob, str, wcslen(str));
}

template <std::size_t CharSize>
inline __hd__ void put
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::underlying_outbuf_char_type<CharSize> c )
{
    strf::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline __hd__ void put( strf::basic_outbuf<CharT>& ob, CharT c )
{
    strf::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline __hd__ void put( strf::basic_outbuf_noexcept<CharT>& ob, CharT c ) noexcept
{
    strf::detail::outbuf_put(ob, c);
}
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
    static __hd__ void turn_into_bad(underlying_outbuf<CharSize>& ob)
    {
        ob.set_good(false);
    }
    template<std::size_t CharSize>
    static __hd__ void force_set_pos
        ( underlying_outbuf<CharSize>& ob
        , strf::underlying_outbuf_char_type<CharSize>* pos)
    {
        ob.set_pos(pos);
    }
};


inline __hd__ char32_t* _outbuf_garbage_buf()
{
    constexpr std::size_t s1
        = (strf::min_size_after_recycle<char>() + 1) / 4;
    constexpr std::size_t s2
        = (strf::min_size_after_recycle<char16_t>() + 1) / 2;
    constexpr std::size_t s4
        = strf::min_size_after_recycle<char32_t>();
    constexpr std::size_t max_s1_s2 = s1 > s2 ? s1 : s2;
    constexpr std::size_t max_s1_s2_s4 = max_s1_s2 > s4 ? max_s1_s2 : s4;

    static char32_t arr[max_s1_s2_s4];
    return arr;
}

} // namespace detail

template <typename CharT>
inline __hd__ CharT* outbuf_garbage_buf()
{
    return reinterpret_cast<CharT*>(strf::detail::_outbuf_garbage_buf());
}

template <typename CharT>
inline __hd__ CharT* outbuf_garbage_buf_end()
{
    return strf::outbuf_garbage_buf<CharT>()
        + strf::min_size_after_recycle<CharT>();
}

template <typename CharT>
class basic_cstr_writer final: public strf::basic_outbuf_noexcept<CharT>
{
public:

	__hd__ basic_cstr_writer(CharT* dest, CharT* dest_end)
        : basic_outbuf_noexcept<CharT>(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

	__hd__ basic_cstr_writer(CharT* dest, std::size_t len)
        : basic_outbuf_noexcept<CharT>(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    __hd__ basic_cstr_writer(CharT (&dest)[N])
        : basic_outbuf_noexcept<CharT>(dest, dest + N - 1)
    {
    }

    __hd__ basic_cstr_writer(basic_cstr_writer&& r)
        : basic_cstr_writer(r.pos(), r.end())
    {}

    __hd__ void recycle() noexcept override
    {
        if (this->good())
        {
            _it = this->pos();
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

    __hd__ result finish()
    {
        bool g = this->good();
        if (g)
        {
            _it = this->pos();
            this->set_good(false);
        }
        this->set_pos(outbuf_garbage_buf<CharT>());
        this->set_end(outbuf_garbage_buf_end<CharT>());

        *_it = CharT();

        return { _it, ! g };
    }

private:

    CharT* _it = nullptr;
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

	__hd__ discarded_outbuf()
        : basic_outbuf_noexcept<CharT>
            { strf::outbuf_garbage_buf<CharT>()
            , strf::outbuf_garbage_buf_end<CharT>() }
    {
        this->set_good(false);
    }

    __hd__ ~discarded_outbuf() = default;

    __hd__ void recycle() noexcept override
    {
        this->set_pos(strf::outbuf_garbage_buf<CharT>());
    }
};

STRF_NAMESPACE_END

#include <strf/detail/undefine_specifiers.hpp>

#endif  // BOOST_OUTBUF_HPP

