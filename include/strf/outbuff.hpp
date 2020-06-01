#ifndef STRF_OUTBUFF_HPP_INCLUDED
#define STRF_OUTBUFF_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>
#include <cstdint>

namespace strf {
namespace detail {

class outbuff_test_tool;

template <std::size_t CharSize>
struct underlying_char_type_impl
{
    static_assert( CharSize == 1 || CharSize == 2 || CharSize == 4, "Invalid CharSize");
};

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
class underlying_outbuff
{
public:

    using char_type = strf::underlying_char_type<CharSize>;

    STRF_HD underlying_outbuff(const underlying_outbuff&) = delete;
    STRF_HD underlying_outbuff(underlying_outbuff&&) = delete;
    underlying_outbuff& STRF_HD operator=(const underlying_outbuff&) = delete;
    underlying_outbuff& STRF_HD operator=(underlying_outbuff&&) = delete;

    virtual STRF_HD ~underlying_outbuff() { };

    STRF_HD char_type* pointer() const noexcept
    {
        return pointer_;
    }
    STRF_HD char_type* end() const noexcept
    {
        return end_;
    }
    STRF_HD std::size_t size() const noexcept
    {
        STRF_ASSERT(pointer_ <= end_);
        return end_ - pointer_;
    }
    STRF_HD bool good() const noexcept
    {
        return good_;
    }
    STRF_HD void advance_to(char_type* p)
    {
        STRF_ASSERT(pointer_ <= p);
        STRF_ASSERT(p <= end_);
        pointer_ = p;
    }
    STRF_HD void advance(std::size_t n)
    {
        STRF_ASSERT(pointer() + n <= end());
        pointer_ += n;
    }
    STRF_HD void advance() noexcept
    {
        STRF_ASSERT(pointer() < end());
        ++pointer_;
    }
    STRF_HD void require(std::size_t s)
    {
        STRF_ASSERT(s <= strf::min_size_after_recycle<CharSize>());
        if (pointer() + s > end()) {
            recycle();
        }
        STRF_ASSERT(pointer() + s <= end());
    }
    STRF_HD void ensure(std::size_t s)
    {
        require(s);
    }

    STRF_HD virtual void recycle() = 0;

protected:

    STRF_HD underlying_outbuff(char_type* p, char_type* e) noexcept
        : pointer_(p), end_(e)
    { }

    STRF_HD underlying_outbuff(char_type* p, std::size_t s) noexcept
        : pointer_(p), end_(p + s)
    { }

    STRF_HD void set_pointer(char_type* p) noexcept
    { pointer_ = p; };
    STRF_HD void set_end(char_type* e) noexcept
    { end_ = e; };
    STRF_HD void set_good(bool g) noexcept
    { good_ = g; };

private:

    char_type* pointer_;
    char_type* end_;
    bool good_ = true;
    friend class strf::detail::outbuff_test_tool;
};

template <typename CharT>
class basic_outbuff: private strf::underlying_outbuff<sizeof(CharT)>
{
    using underlying_impl_ = strf::underlying_outbuff<sizeof(CharT)>;
    using underlying_char_t_ = typename underlying_impl_::char_type;

public:

    using char_type = CharT;

    STRF_HD basic_outbuff(const basic_outbuff&) = delete;
    STRF_HD basic_outbuff(basic_outbuff&&) = delete;
    STRF_HD basic_outbuff& operator=(const basic_outbuff&) = delete;
    STRF_HD basic_outbuff& operator=(basic_outbuff&&) = delete;

    virtual STRF_HD ~basic_outbuff() { };

    STRF_HD CharT* pointer() const noexcept
    {
        return reinterpret_cast<CharT*>(underlying_impl_::pointer());
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

    STRF_HD basic_outbuff(CharT* p, CharT* e) noexcept
        : underlying_impl_( reinterpret_cast<underlying_char_t_*>(p)
                          , reinterpret_cast<underlying_char_t_*>(e) )
    { }

    STRF_HD basic_outbuff(CharT* p, std::size_t s) noexcept
        : underlying_impl_(reinterpret_cast<underlying_char_t_*>(p), s)
    { }

    STRF_HD void set_pointer(CharT* p) noexcept
    {
        underlying_impl_::set_pointer(reinterpret_cast<underlying_char_t_*>(p));
    }
    STRF_HD void set_end(CharT* e) noexcept
    {
        underlying_impl_::set_end(reinterpret_cast<underlying_char_t_*>(e));
    }

    using underlying_impl_::set_good;
};

template <typename CharT>
class basic_outbuff_noexcept: public basic_outbuff<CharT>
{
public:

    virtual STRF_HD void recycle() noexcept = 0;

protected:

    using basic_outbuff<CharT>::basic_outbuff;
};

namespace detail {

template <bool NoExcept, typename CharT>
struct basic_outbuff_noexcept_switch_impl;

template <typename CharT>
struct basic_outbuff_noexcept_switch_impl<false, CharT>
{
    using type = strf::basic_outbuff<CharT>;
};

template <typename CharT>
struct basic_outbuff_noexcept_switch_impl<true, CharT>
{
    using type = strf::basic_outbuff_noexcept<CharT>;
};

template <bool NoExcept, typename CharT>
using basic_outbuff_noexcept_switch
    = typename basic_outbuff_noexcept_switch_impl<NoExcept, CharT>
   :: type;

template <typename Outbuff, typename CharT = typename Outbuff::char_type>
STRF_HD void outbuff_put(Outbuff& ob, CharT c)
{
    auto p = ob.pointer();
    if (p != ob.end()) {
        *p = c;
        ob.advance_to(p+1);
    } else {
        ob.recycle();
        *ob.pointer() = c;
        ob.advance();
    }
}

} // namespace detail

template <std::size_t CharSize>
inline STRF_HD void put
    ( strf::underlying_outbuff<CharSize>& ob
    , strf::underlying_char_type<CharSize> c )
{
    strf::detail::outbuff_put(ob, c);
}

template <typename CharT>
inline STRF_HD void put( strf::basic_outbuff<CharT>& ob, CharT c )
{
    strf::detail::outbuff_put(ob, c);
}

template <typename CharT>
inline STRF_HD void put( strf::basic_outbuff_noexcept<CharT>& ob, CharT c ) noexcept
{
    strf::detail::outbuff_put(ob, c);
}

// type aliases

#if defined(__cpp_lib_byte)
using bin_outbuff           = basic_outbuff<std::byte>;
using bin_outbuff_noexcept  = basic_outbuff_noexcept<std::byte>;
#endif

#if defined(__cpp_char8_t)
using u8outbuff           = basic_outbuff<char8_t>;
using u8outbuff_noexcept  = basic_outbuff_noexcept<char8_t>;
#endif

using outbuff             = basic_outbuff<char>;
using outbuff_noexcept    = basic_outbuff_noexcept<char>;
using u16outbuff          = basic_outbuff<char16_t>;
using u16outbuff_noexcept = basic_outbuff_noexcept<char16_t>;
using u32outbuff          = basic_outbuff<char32_t>;
using u32outbuff_noexcept = basic_outbuff_noexcept<char32_t>;
using woutbuff            = basic_outbuff<wchar_t>;
using woutbuff_noexcept   = basic_outbuff_noexcept<wchar_t>;

namespace detail {

class outbuff_test_tool
{
public:

    template<std::size_t CharSize>
    static STRF_HD void turn_into_bad(underlying_outbuff<CharSize>& ob)
    {
        ob.set_good(false);
    }
    template<std::size_t CharSize>
    static STRF_HD void force_set_pointer
        ( underlying_outbuff<CharSize>& ob
        , strf::underlying_char_type<CharSize>* pointer)
    {
        ob.set_pointer(pointer);
    }
};


inline STRF_HD char32_t* outbuff_garbage_buf_()
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
inline STRF_HD CharT* outbuff_garbage_buf() noexcept
{
    return reinterpret_cast<CharT*>(strf::detail::outbuff_garbage_buf_());
}

template <typename CharT>
inline STRF_HD CharT* outbuff_garbage_buf_end() noexcept
{
    return strf::outbuff_garbage_buf<CharT>()
        + strf::min_size_after_recycle<sizeof(CharT)>();
}

template <typename CharT>
class basic_cstr_writer final: public strf::basic_outbuff_noexcept<CharT>
{
public:

    STRF_HD basic_cstr_writer(CharT* dest, CharT* dest_end) noexcept
        : basic_outbuff_noexcept<CharT>(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD basic_cstr_writer(CharT* dest, std::size_t len) noexcept
        : basic_outbuff_noexcept<CharT>(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    STRF_HD basic_cstr_writer(CharT (&dest)[N]) noexcept
        : basic_outbuff_noexcept<CharT>(dest, dest + N - 1)
    {
    }

    STRF_HD basic_cstr_writer(basic_cstr_writer&& r) noexcept
        : basic_cstr_writer(r.pointer(), r.end())
    {}

    STRF_HD void recycle() noexcept override
    {
        if (this->good()) {
            it_ = this->pointer();
            this->set_good(false);
            this->set_end(outbuff_garbage_buf_end<CharT>());
        }
        this->set_pointer(outbuff_garbage_buf<CharT>());
    }

    struct result
    {
        CharT* ptr;
        bool truncated;
    };

    STRF_HD result finish() noexcept
    {
        bool g = this->good();
        if (g) {
            it_ = this->pointer();
            this->set_good(false);
        }
        this->set_pointer(outbuff_garbage_buf<CharT>());
        this->set_end(outbuff_garbage_buf_end<CharT>());

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
class discarded_outbuff final
    : public strf::basic_outbuff_noexcept<CharT>
{
public:

    STRF_HD discarded_outbuff() noexcept
        : basic_outbuff_noexcept<CharT>
            { strf::outbuff_garbage_buf<CharT>()
            , strf::outbuff_garbage_buf_end<CharT>() }
    {
        this->set_good(false);
    }

    STRF_HD ~discarded_outbuff()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        this->set_pointer(strf::outbuff_garbage_buf<CharT>());
    }
};

} // namespace strf

#endif  // STRF_OUTBUFF_HPP_INCLUDED

