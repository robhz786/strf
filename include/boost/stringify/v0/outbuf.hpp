#ifndef BOOST_OUTBUF_HPP
#define BOOST_OUTBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <cstring>
#include <cwchar>
#include <cstdint>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

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
= typename stringify::v0::detail::underlying_outbuf_char_type_impl<CharSize>::type;

template <typename CharT>
constexpr std::size_t min_size_after_recycle()
{
    return 64;
}

template <std::size_t CharSize>
class underlying_outbuf
{
public:

    using char_type = stringify::v0::underlying_outbuf_char_type<CharSize>;

    underlying_outbuf(const underlying_outbuf&) = delete;
    underlying_outbuf(underlying_outbuf&&) = delete;
    underlying_outbuf& operator=(const underlying_outbuf&) = delete;
    underlying_outbuf& operator=(underlying_outbuf&&) = delete;

    virtual ~underlying_outbuf() = default;

    char_type* pos() const noexcept
    {
        return _pos;
    }
    char_type* end() const noexcept
    {
        return _end;
    }
    std::size_t size() const noexcept
    {
        STRF_ASSERT(_pos <= _end);
        return _end - _pos;
    }

    bool good() const noexcept
    {
        return _good;
    }
    void advance_to(char_type* p)
    {
        STRF_ASSERT(_pos <= p);
        STRF_ASSERT(p <= _end);
        _pos = p;
    }
    void advance(std::size_t n)
    {
        STRF_ASSERT(pos() + n <= end());
        _pos += n;
    }
    void advance() noexcept
    {
        STRF_ASSERT(pos() < end());
        ++_pos;
    }
    void require(std::size_t s)
    {
        STRF_ASSERT(s <= stringify::v0::min_size_after_recycle<char_type>());
        if (pos() + s > end())
        {
            recycle();
        }
        STRF_ASSERT(pos() + s <= end());
    }
    void ensure(std::size_t s)
    {
        require(s);
    }

    virtual void recycle() = 0;

protected:

    underlying_outbuf(char_type* pos_, char_type* end_) noexcept
        : _pos(pos_), _end(end_)
    { }

    underlying_outbuf(char_type* pos_, std::size_t s) noexcept
        : _pos(pos_), _end(pos_ + s)
    { }

    void set_pos(char_type* p) noexcept
    { _pos = p; };
    void set_end(char_type* e) noexcept
    { _end = e; };
    void set_good(bool g) noexcept
    { _good = g; };

private:

    char_type* _pos;
    char_type* _end;
    bool _good = true;
    friend class stringify::v0::detail::outbuf_test_tool;
};

template <typename CharT>
class basic_outbuf;

template <typename CharT>
class basic_outbuf: private stringify::v0::underlying_outbuf<sizeof(CharT)>
{
    using _underlying_impl = stringify::v0::underlying_outbuf<sizeof(CharT)>;
    using _underlying_char_t = typename _underlying_impl::char_type;

public:

    using char_type = CharT;

    basic_outbuf(const basic_outbuf&) = delete;
    basic_outbuf(basic_outbuf&&) = delete;
    basic_outbuf& operator=(const basic_outbuf&) = delete;
    basic_outbuf& operator=(basic_outbuf&&) = delete;

    virtual ~basic_outbuf() = default;

    CharT* pos() const noexcept
    {
        return reinterpret_cast<CharT*>(_underlying_impl::pos());
    }
    CharT* end() const noexcept
    {
        return reinterpret_cast<CharT*>(_underlying_impl::end());
    }
    void advance_to(CharT* p)
    {
        _underlying_impl::advance_to(reinterpret_cast<_underlying_char_t*>(p));
    }
    _underlying_impl& as_underlying() noexcept
    {
        return *this;
    }
    const _underlying_impl& as_underlying() const noexcept
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

    basic_outbuf(CharT* pos_, CharT* end_) noexcept
        : _underlying_impl( reinterpret_cast<_underlying_char_t*>(pos_)
                          , reinterpret_cast<_underlying_char_t*>(end_) )
    { }

    basic_outbuf(CharT* pos_, std::size_t s) noexcept
        : _underlying_impl(reinterpret_cast<_underlying_char_t*>(pos_), s)
    { }

    void set_pos(CharT* p) noexcept
    {
        _underlying_impl::set_pos(reinterpret_cast<_underlying_char_t*>(p));
    }
    void set_end(CharT* e) noexcept
    {
        _underlying_impl::set_end(reinterpret_cast<_underlying_char_t*>(e));
    }

    using _underlying_impl::set_good;
};

template <typename CharT>
class basic_outbuf_noexcept: public basic_outbuf<CharT>
{
public:

    virtual void recycle() noexcept = 0;

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
    using type = stringify::v0::basic_outbuf<CharT>;
};

template <typename CharT>
struct basic_outbuf_noexcept_switch_impl<true, CharT>
{
    using type = stringify::v0::basic_outbuf_noexcept<CharT>;
};

template <bool NoExcept, typename CharT>
using basic_outbuf_noexcept_switch
    = typename basic_outbuf_noexcept_switch_impl<NoExcept, CharT>
   :: type;

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

template <typename Outbuf, typename CharT>
void outbuf_write_continuation(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto space = ob.size();
    STRF_ASSERT(space < len);
    std::memcpy(ob.pos(), str, space * sizeof(CharT));
    str += space;
    len -= space;
    ob.advance_to(ob.end());
    while (ob.good())
    {
        ob.recycle();
        space = ob.size();
        if (len <= space)
        {
            std::memcpy(ob.pos(), str, len * sizeof(CharT));
            ob.advance(len);
            break;
        }
        std::memcpy(ob.pos(), str, space * sizeof(CharT));
        len -= space;
        str += space;
        ob.advance_to(ob.end());
    }
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
void outbuf_write(Outbuf& ob, const CharT* str, std::size_t len)
{
    auto p = ob.pos();
    if (p + len <= ob.end()) // the common case
    {
        std::memcpy(p, str, len * sizeof(CharT));
        ob.advance(len);
    }
    else
    {
        detail::outbuf_write_continuation<Outbuf, CharT>(ob, str, len);
    }
}

template <typename Outbuf, typename CharT = typename Outbuf::char_type>
void outbuf_put(Outbuf& ob, CharT c)
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
inline void write
    ( stringify::v0::underlying_outbuf<CharSize>& ob
    , const stringify::v0::underlying_outbuf_char_type<CharSize>* str
    , std::size_t len )
{
    stringify::v0::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline void write( stringify::v0::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    stringify::v0::detail::outbuf_write(ob, str, len);
}

template <typename CharT>
inline void write( stringify::v0::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , std::size_t len )
{
    stringify::v0::detail::outbuf_write(ob, str, len);
}

template <std::size_t CharSize>
inline void write
    ( stringify::v0::underlying_outbuf<CharSize>& ob
    , const stringify::v0::underlying_outbuf_char_type<CharSize>* str
    , const stringify::v0::underlying_outbuf_char_type<CharSize>* str_end )
{
    STRF_ASSERT(str_end >= str);
    stringify::v0::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline void write( stringify::v0::basic_outbuf<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end )
{
    STRF_ASSERT(str_end >= str);
    stringify::v0::detail::outbuf_write(ob, str, str_end - str);
}

template <typename CharT>
inline void write( stringify::v0::basic_outbuf_noexcept<CharT>& ob
                 , const CharT* str
                 , const CharT* str_end ) noexcept
{
    STRF_ASSERT(str_end >= str);
    stringify::v0::detail::outbuf_write(ob, str, str_end - str);
}

inline void write( stringify::v0::basic_outbuf<char>& ob
                 , const char* str )
{
    stringify::v0::detail::outbuf_write(ob, str, std::strlen(str));
}

inline void write( stringify::v0::basic_outbuf_noexcept<char>& ob
                 , const char* str ) noexcept
{
    stringify::v0::detail::outbuf_write(ob, str, std::strlen(str));
}

inline void write( stringify::v0::basic_outbuf<wchar_t>& ob
                 , const wchar_t* str )
{
    stringify::v0::detail::outbuf_write(ob, str, std::wcslen(str));
}

inline void write( stringify::v0::basic_outbuf_noexcept<wchar_t>& ob
                 , const wchar_t* str ) noexcept
{
    stringify::v0::detail::outbuf_write(ob, str, std::wcslen(str));
}

template <std::size_t CharSize>
inline void put
    ( stringify::v0::underlying_outbuf<CharSize>& ob
    , stringify::v0::underlying_outbuf_char_type<CharSize> c )
{
    stringify::v0::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline void put( stringify::v0::basic_outbuf<CharT>& ob, CharT c )
{
    stringify::v0::detail::outbuf_put(ob, c);
}

template <typename CharT>
inline void put( stringify::v0::basic_outbuf_noexcept<CharT>& ob, CharT c ) noexcept
{
    stringify::v0::detail::outbuf_put(ob, c);
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
    static void turn_into_bad(underlying_outbuf<CharSize>& ob)
    {
        ob.set_good(false);
    }
    template<std::size_t CharSize>
    static void force_set_pos
        ( underlying_outbuf<CharSize>& ob
        , stringify::v0::underlying_outbuf_char_type<CharSize>* pos)
    {
        ob.set_pos(pos);
    }
};


inline char32_t* _outbuf_garbage_buf()
{
    constexpr std::size_t s1
        = (stringify::v0::min_size_after_recycle<char>() + 1) / 4;
    constexpr std::size_t s2
        = (stringify::v0::min_size_after_recycle<char16_t>() + 1) / 2;
    constexpr std::size_t s4
        = stringify::v0::min_size_after_recycle<char32_t>();
    constexpr std::size_t max_s1_s2 = s1 > s2 ? s1 : s2;
    constexpr std::size_t max_s1_s2_s4 = max_s1_s2 > s4 ? max_s1_s2 : s4;

    static char32_t arr[max_s1_s2_s4];
    return arr;
}

} // namespace detail

template <typename CharT>
inline CharT* outbuf_garbage_buf()
{
    return reinterpret_cast<CharT*>(stringify::v0::detail::_outbuf_garbage_buf());
}

template <typename CharT>
inline CharT* outbuf_garbage_buf_end()
{
    return stringify::v0::outbuf_garbage_buf<CharT>()
        + stringify::v0::min_size_after_recycle<CharT>();
}

template <typename CharT>
class basic_cstr_writer final: public stringify::v0::basic_outbuf_noexcept<CharT>
{
public:

    basic_cstr_writer(CharT* dest, CharT* dest_end)
        : basic_outbuf_noexcept<CharT>(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

    basic_cstr_writer(CharT* dest, std::size_t len)
        : basic_outbuf_noexcept<CharT>(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    basic_cstr_writer(CharT (&dest)[N])
        : basic_outbuf_noexcept<CharT>(dest, dest + N - 1)
    {
    }

    basic_cstr_writer(basic_cstr_writer&& r)
        : basic_cstr_writer(r.pos(), r.end())
    {}

    void recycle() noexcept override
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

    result finish()
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
    : public stringify::v0::basic_outbuf_noexcept<CharT>
{
public:

    discarded_outbuf()
        : basic_outbuf_noexcept<CharT>
            { stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() }
    {
        this->set_good(false);
    }

    ~discarded_outbuf() = default;

    void recycle() noexcept override
    {
        this->set_pos(stringify::v0::outbuf_garbage_buf<CharT>());
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_OUTBUF_HPP

