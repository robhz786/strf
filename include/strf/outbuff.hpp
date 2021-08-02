#ifndef STRF_OUTBUFF_HPP_INCLUDED
#define STRF_OUTBUFF_HPP_INCLUDED

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>
#include <cstdint>
#if ! defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
#    include <cstring>
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace strf {
namespace detail {
class outbuff_test_tool;
} // namespace detail

#define STRF_MIN_SPACE_AFTER_RECYCLE 64

template <typename CharT>
STRF_DEPRECATED
constexpr STRF_HD std::size_t min_size_after_recycle()
{
    return STRF_MIN_SPACE_AFTER_RECYCLE;
}

template <typename CharT>
constexpr STRF_HD std::size_t min_space_after_recycle()
{
    return STRF_MIN_SPACE_AFTER_RECYCLE;
}


template <typename CharT>
class basic_outbuff
{
public:

    using char_type = CharT;

    STRF_HD basic_outbuff(const basic_outbuff&) = delete;
    STRF_HD basic_outbuff(basic_outbuff&&) = delete;
    basic_outbuff& operator=(const basic_outbuff&) = delete;
    basic_outbuff& operator=(basic_outbuff&&) = delete;

    virtual STRF_HD ~basic_outbuff() { };

    STRF_HD char_type* pointer() const noexcept
    {
        return pointer_;
    }
    STRF_HD char_type* end() const noexcept
    {
        return end_;
    }
    STRF_DEPRECATED
    STRF_HD std::size_t size() const noexcept
    {
        STRF_ASSERT(pointer_ <= end_);
        return end_ - pointer_;
    }
    STRF_HD std::size_t space() const noexcept
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
        STRF_ASSERT(s <= strf::min_space_after_recycle<CharT>());
        STRF_IF_UNLIKELY (pointer() + s > end()) {
            recycle();
        }
        STRF_ASSERT(pointer() + s <= end());
    }
    STRF_HD void ensure(std::size_t s)
    {
        require(s);
    }

    STRF_HD virtual void recycle() = 0;

    STRF_HD void write(const char_type* str, std::size_t str_len)
    {
        STRF_IF_LIKELY (str_len <= space()) {
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
            memcpy(pointer_, str, str_len * sizeof(char_type));
            pointer_ += str_len;
#else
            for(; str_len != 0; ++pointer_, ++str, --str_len) {
                *pointer_ = *str;
            }
#endif
        } else {
            do_write(str, str_len);
        }
    }

protected:

    STRF_HD basic_outbuff(char_type* p, char_type* e) noexcept
        : pointer_(p), end_(e)
    { }

    STRF_HD basic_outbuff(char_type* p, std::size_t s) noexcept
        : pointer_(p), end_(p + s)
    { }

    STRF_HD void set_pointer(char_type* p) noexcept
    { pointer_ = p; };
    STRF_HD void set_end(char_type* e) noexcept
    { end_ = e; };
    STRF_HD void set_good(bool g) noexcept
    { good_ = g; };

    STRF_HD virtual void do_write(const char_type* src, std::size_t src_size);

private:

    char_type* pointer_;
    char_type* end_;
    bool good_ = true;
    friend class strf::detail::outbuff_test_tool;
};

template <typename CharT>
void basic_outbuff<CharT>::do_write(const CharT* str, std::size_t str_len)
{
    for(;;) {
        std::size_t s = space();
        std::size_t sub_count = (str_len <= s ? str_len : s);
        str_len -= sub_count;

#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
        memcpy(pointer_, str, sub_count * sizeof(char_type));
        str += sub_count;
        pointer_ += sub_count;
#else
        for(; sub_count != 0; ++pointer_, ++str, --sub_count) {
            *pointer_ = *str;
        }
#endif
        if (str_len == 0) {
            break;
        }
        recycle();
        if (!good_) {
            break;
        }
    }
}

template <typename CharT>
inline STRF_HD void put(strf::basic_outbuff<CharT>& ob, CharT c)
{
    auto p = ob.pointer();
    STRF_IF_LIKELY (p != ob.end()) {
        *p = c;
        ob.advance_to(p + 1);
    } else {
        ob.recycle();
        *ob.pointer() = c;
        ob.advance();
    }
}

// type aliases

#if defined(__cpp_lib_byte)
using bin_outbuff           = basic_outbuff<std::byte>;
#endif

#if defined(__cpp_char8_t)
using u8outbuff           = basic_outbuff<char8_t>;
#endif

using outbuff             = basic_outbuff<char>;
using u16outbuff          = basic_outbuff<char16_t>;
using u32outbuff          = basic_outbuff<char32_t>;
using woutbuff            = basic_outbuff<wchar_t>;

namespace detail {

class outbuff_test_tool
{
public:
    template<typename CharT>
    static STRF_HD void turn_into_bad(basic_outbuff<CharT>& ob)
    {
        ob.set_good(false);
    }
};

} // namespace detail

template <typename CharT>
inline STRF_HD CharT* garbage_buff() noexcept
{
    static CharT arr[ STRF_MIN_SPACE_AFTER_RECYCLE ];
    return arr;
}

template <typename CharT>
inline STRF_HD CharT* garbage_buff_end() noexcept
{
    return strf::garbage_buff<CharT>() + strf::min_space_after_recycle<CharT>();
}

template <typename CharT>
class basic_cstr_writer final: public strf::basic_outbuff<CharT>
{
public:

    struct range{ CharT* dest; CharT* dest_end; };

    STRF_HD basic_cstr_writer(range r) noexcept
        : basic_outbuff<CharT>(r.dest, r.dest_end - 1)
    {
        STRF_ASSERT(r.dest < r.dest_end);
    }

    STRF_HD basic_cstr_writer(CharT* dest, CharT* dest_end) noexcept
        : basic_outbuff<CharT>(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD basic_cstr_writer(CharT* dest, std::size_t len) noexcept
        : basic_outbuff<CharT>(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    STRF_HD basic_cstr_writer(CharT (&dest)[N]) noexcept
        : basic_outbuff<CharT>(dest, dest + N - 1)
    {
    }

    basic_cstr_writer(const basic_cstr_writer& r) = delete;

    STRF_HD void recycle() noexcept override
    {
        if (this->good()) {
            it_ = this->pointer();
            this->set_good(false);
            this->set_end(garbage_buff_end<CharT>());
        }
        this->set_pointer(garbage_buff<CharT>());
    }

    struct result
    {
        CharT* ptr;
        bool truncated;
    };

    STRF_HD result finish() noexcept
    {
        bool g = this->good();
        STRF_IF_LIKELY (g) {
            it_ = this->pointer();
            this->set_good(false);
        }
        this->set_pointer(garbage_buff<CharT>());
        this->set_end(garbage_buff_end<CharT>());

        *it_ = CharT();

        return { it_, ! g };
    }

private:


    STRF_HD void do_write(const CharT* str, std::size_t) noexcept override
    {
        auto sub_count = this->space();
        auto ptr = this->pointer();
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
        memcpy(ptr, str, sub_count * sizeof(CharT));
#else
        for(; sub_count != 0; ++ptr, ++str, --sub_count) {
            *ptr = *str;
        }
#endif
        it_ = this->end();
        this->set_pointer(garbage_buff<CharT>());
        this->set_end(garbage_buff_end<CharT>());
        this->set_good(false);
    }

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
class basic_char_array_writer final: public strf::basic_outbuff<CharT>
{
public:

    struct range{ CharT* dest; CharT* dest_end; };

    STRF_HD basic_char_array_writer(range r) noexcept
        : basic_outbuff<CharT>(r.dest, r.dest_end)
    {
        STRF_ASSERT(r.dest <= r.dest_end);
    }

    STRF_HD basic_char_array_writer(CharT* dest, CharT* dest_end) noexcept
        : basic_outbuff<CharT>(dest, dest_end)
    {
        STRF_ASSERT(dest <= dest_end);
    }

    STRF_HD basic_char_array_writer(CharT* dest, std::size_t len) noexcept
        : basic_outbuff<CharT>(dest, dest + len)
    {
    }

    template <std::size_t N>
    STRF_HD basic_char_array_writer(CharT (&dest)[N]) noexcept
        : basic_outbuff<CharT>(dest, dest + N)
    {
    }

    STRF_HD basic_char_array_writer(const basic_char_array_writer& r) noexcept
        : basic_outbuff<CharT>(r.pointer(), r.end())
        , it_(r.it_)
    {
        this->set_good(r.good());
    }

    STRF_HD basic_char_array_writer& operator=(const basic_char_array_writer& r) noexcept
    {
        this->set_good(r.good());
        this->set_pointer(r.pointer());
        this->set_end(r.end());
        it_ = r.it_;
        return *this;
    }
    STRF_HD bool operator==(const basic_char_array_writer& r) const noexcept
    {
        if (this->good()) {
            return ( r.good()
                  && this->pointer() == r.pointer()
                  && this->end() == r.end() );
        }
        return ! r.good() && it_ == r.it_;
    }
    STRF_HD void recycle() noexcept override
    {
        if (this->good()) {
            it_ = this->pointer();
            this->set_good(false);
            this->set_end(garbage_buff_end<CharT>());
        }
        this->set_pointer(garbage_buff<CharT>());
    }

    struct result
    {
        CharT* ptr;
        bool truncated;
    };

    STRF_HD result finish() noexcept
    {
        bool truncated = ! this->good();
        CharT* ptr = truncated ? it_ : this->pointer();
        return { ptr, truncated };
    }

private:

    STRF_HD void do_write(const CharT* str, std::size_t) noexcept override
    {
        auto sub_count = this->space();
        auto ptr = this->pointer();
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CHAR_ARRAYING)
        memcpy(ptr, str, sub_count * sizeof(CharT));
#else
        for(; sub_count != 0; ++ptr, ++str, --sub_count) {
            *ptr = *str;
        }
#endif
        it_ = this->end();
        this->set_pointer(garbage_buff<CharT>());
        this->set_end(garbage_buff_end<CharT>());
        this->set_good(false);
    }

    CharT* it_ = nullptr;
};


#if defined(__cpp_char8_t)
using u8char_array_writer = basic_char_array_writer<char8_t>;
#endif
using char_array_writer = basic_char_array_writer<char>;
using u16char_array_writer = basic_char_array_writer<char16_t>;
using u32char_array_writer = basic_char_array_writer<char32_t>;
using wchar_array_writer = basic_char_array_writer<wchar_t>;

template <typename CharT>
class discarded_outbuff final
    : public strf::basic_outbuff<CharT>
{
public:

    STRF_HD discarded_outbuff() noexcept
        : basic_outbuff<CharT>
            { strf::garbage_buff<CharT>()
            , strf::garbage_buff_end<CharT>() }
    {
        this->set_good(false);
    }

    STRF_HD ~discarded_outbuff()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        this->set_pointer(strf::garbage_buff<CharT>());
    }

private:

    STRF_HD void do_write(const CharT*, std::size_t) noexcept override
    {
        this->set_pointer(strf::garbage_buff<CharT>());
    }
};

} // namespace strf

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#endif  // STRF_OUTBUFF_HPP_INCLUDED

