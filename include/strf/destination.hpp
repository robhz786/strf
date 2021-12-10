#ifndef STRF_DESTINATION_HPP_INCLUDED
#define STRF_DESTINATION_HPP_INCLUDED

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
class output_buffer_test_tool;
} // namespace detail

template <typename T, unsigned Log2BufferSpace>
class output_buffer: public output_buffer<T, Log2BufferSpace - 1>
{
public:
    static constexpr std::size_t buffer_space_after_recycle
        = (std::size_t)1 << Log2BufferSpace;

    STRF_HD void ensure(std::size_t s) {
        STRF_ASSERT(s <= buffer_space_after_recycle);
        STRF_IF_UNLIKELY (this->buffer_ptr() + s > this->buffer_end()) {
            this->recycle_buffer();
        }
        STRF_ASSERT(this->buffer_ptr() + s <= this->buffer_end());
    }

    STRF_HD void flush() {
        this->recycle_buffer();
        STRF_ASSERT(this->buffer_space() >= buffer_space_after_recycle);
    }

protected:
    using output_buffer<T, Log2BufferSpace - 1>::output_buffer;
};

template <typename T>
class output_buffer<T, 0>
{
public:
    static constexpr std::size_t buffer_space_after_recycle = 1;
    using value_type = T;

    STRF_HD output_buffer(const output_buffer&) = delete;
    STRF_HD output_buffer(output_buffer&&) = delete;
    output_buffer& operator=(const output_buffer&) = delete;
    output_buffer& operator=(output_buffer&&) = delete;

    virtual STRF_HD ~output_buffer() { };

    STRF_HD value_type* buffer_ptr() const noexcept
    {
        return pointer_;
    }
    STRF_HD value_type* buffer_end() const noexcept
    {
        return end_;
    }
    STRF_DEPRECATED
    STRF_HD std::size_t size() const noexcept
    {
        STRF_ASSERT(pointer_ <= end_);
        return end_ - pointer_;
    }
    STRF_HD std::size_t buffer_space() const noexcept
    {
        STRF_ASSERT(pointer_ <= end_);
        return end_ - pointer_;
    }
    STRF_HD bool good() const noexcept
    {
        return good_;
    }
    STRF_HD void advance_to(value_type* p)
    {
        STRF_ASSERT(pointer_ <= p);
        STRF_ASSERT(p <= end_);
        pointer_ = p;
    }
    STRF_HD void advance(std::size_t n)
    {
        STRF_ASSERT(buffer_ptr() + n <= buffer_end());
        pointer_ += n;
    }
    STRF_HD void advance() noexcept
    {
        STRF_ASSERT(buffer_ptr() < buffer_end());
        ++pointer_;
    }
    STRF_HD void ensure(std::size_t s)
    {
        STRF_ASSERT(s <= 1);
        STRF_IF_UNLIKELY (buffer_ptr() + s > buffer_end()) {
            recycle_buffer();
        }
        STRF_ASSERT(buffer_ptr() + s <= buffer_end());
    }
    STRF_HD void flush()
    {
        recycle_buffer();
        STRF_ASSERT(buffer_space() != 0);
    }

    STRF_HD virtual void recycle_buffer() = 0;

    STRF_HD void write(const value_type* data, std::size_t count)
    {
        STRF_IF_LIKELY (count <= buffer_space()) {
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
            memcpy(pointer_, data, count * sizeof(value_type));
            pointer_ += count;
#else
            for(; count != 0; ++pointer_, ++data, --count) {
                *pointer_ = *data;
            }
#endif
        } else {
            do_write(data, count);
        }
    }

    // old names keeped to preserve backwards compatibiliy
    STRF_HD value_type* pointer() const noexcept
    {
        return buffer_ptr();
    }
    STRF_HD value_type* end() const noexcept
    {
        return buffer_end();
    }
    STRF_HD std::size_t space() const noexcept
    {
        return buffer_space();
    }
    STRF_HD void require(std::size_t s)
    {
        ensure(s);
    }

protected:

    STRF_HD output_buffer(value_type* p, value_type* e) noexcept
        : pointer_(p), end_(e)
    { }

    STRF_HD output_buffer(value_type* p, std::size_t s) noexcept
        : pointer_(p), end_(p + s)
    { }

    STRF_HD void set_buffer_ptr(value_type* p) noexcept
    { pointer_ = p; };
    STRF_HD void set_buffer_end(value_type* e) noexcept
    { end_ = e; };
    STRF_HD void set_good(bool g) noexcept
    { good_ = g; };

    STRF_HD virtual void do_write(const value_type* src, std::size_t src_size);

    // old names for backwards compatibility
    STRF_HD void set_pointer(value_type* p) noexcept
    { pointer_ = p; };
    STRF_HD void set_end(value_type* e) noexcept
    { end_ = e; };

private:

    value_type* pointer_;
    value_type* end_;
    bool good_ = true;
    friend class strf::detail::output_buffer_test_tool;
};

template <typename T>
void output_buffer<T, 0>::do_write(const T* data, std::size_t count)
{
    for(;;) {
        std::size_t s = buffer_space();
        std::size_t sub_count = (count <= s ? count : s);
        count -= sub_count;

#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
        memcpy(pointer_, data, sub_count * sizeof(value_type));
        data += sub_count;
        pointer_ += sub_count;
#else
        for(; sub_count != 0; ++pointer_, ++data, --sub_count) {
            *pointer_ = *data;
        }
#endif
        if (count == 0) {
            break;
        }
        recycle_buffer();
        if (!good_) {
            break;
        }
    }
}

template <typename T>
inline STRF_HD void put(strf::output_buffer<T, 0>& dest, T c)
{
    auto p = dest.buffer_ptr();
    STRF_IF_LIKELY (p != dest.buffer_end()) {
        *p = c;
        dest.advance_to(p + 1);
    } else {
        dest.recycle_buffer();
        *dest.buffer_ptr() = c;
        dest.advance();
    }
}

namespace detail {

class output_buffer_test_tool
{
public:
    template<typename T>
    static STRF_HD void turn_into_bad(output_buffer<T, 0>& dest)
    {
        dest.set_good(false);
    }
};

} // namespace detail

constexpr unsigned log2_garbage_buff_size = 6;
constexpr std::size_t garbage_buff_size = 1 << log2_garbage_buff_size;

template <typename T>
inline STRF_HD T* garbage_buff() noexcept
{
    static T arr[ garbage_buff_size ];
    return arr;
}

template <typename T>
inline STRF_HD T* garbage_buff_end() noexcept
{
    return strf::garbage_buff<T>() + garbage_buff_size;
}

template <typename CharT>
class basic_cstr_destination final
    : public strf::output_buffer<CharT, strf::log2_garbage_buff_size>
{
    using dest_t_ = strf::output_buffer<CharT, strf::log2_garbage_buff_size>;

public:

    struct range{ CharT* dest; CharT* dest_end; };

    STRF_HD basic_cstr_destination(range r) noexcept
        : dest_t_(r.dest, r.dest_end - 1)
    {
        STRF_ASSERT(r.dest < r.dest_end);
    }

    STRF_HD basic_cstr_destination(CharT* dest, CharT* dest_end) noexcept
        : dest_t_(dest, dest_end - 1)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD basic_cstr_destination(CharT* dest, std::size_t len) noexcept
        : dest_t_(dest, dest + len - 1)
    {
        STRF_ASSERT(len != 0);
    }

    template <std::size_t N>
    STRF_HD basic_cstr_destination(CharT (&dest)[N]) noexcept
        : dest_t_(dest, dest + N - 1)
    {
    }

    basic_cstr_destination(const basic_cstr_destination& r) = delete;

    STRF_HD void recycle_buffer() noexcept override
    {
        if (this->good()) {
            it_ = this->buffer_ptr();
            this->set_good(false);
            this->set_buffer_end(garbage_buff_end<CharT>());
        }
        this->set_buffer_ptr(garbage_buff<CharT>());
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
            it_ = this->buffer_ptr();
            this->set_good(false);
        }
        this->set_buffer_ptr(garbage_buff<CharT>());
        this->set_buffer_end(garbage_buff_end<CharT>());

        *it_ = CharT();

        return { it_, ! g };
    }

private:


    STRF_HD void do_write(const CharT* data, std::size_t) noexcept override
    {
        auto sub_count = this->buffer_space();
        auto ptr = this->buffer_ptr();
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CSTRING)
        memcpy(ptr, data, sub_count * sizeof(CharT));
#else
        for(; sub_count != 0; ++ptr, ++data, --sub_count) {
            *ptr = *data;
        }
#endif
        it_ = this->buffer_end();
        this->set_buffer_ptr(garbage_buff<CharT>());
        this->set_buffer_end(garbage_buff_end<CharT>());
        this->set_good(false);
    }

    CharT* it_ = nullptr;
};

constexpr unsigned log2_destination_space_after_flush = 6;

constexpr std::size_t destination_space_after_flush =
    (std::size_t)1 << strf::log2_destination_space_after_flush;

static_assert(destination_space_after_flush == 64, "");

template <typename CharT>
using destination = strf::output_buffer<CharT, log2_destination_space_after_flush>;

#if defined(__cpp_char8_t)
using u8cstr_destination  = basic_cstr_destination<char8_t>;
#endif
using cstr_destination    = basic_cstr_destination<char>;
using u16cstr_destination = basic_cstr_destination<char16_t>;
using u32cstr_destination = basic_cstr_destination<char32_t>;
using wcstr_destination   = basic_cstr_destination<wchar_t>;

template <typename CharT>
using basic_cstr_writer
STRF_DEPRECATED_MSG("basic_cstr_writer renamed to basic_cstr_destination")
= basic_cstr_destination<CharT>;

#if defined(__cpp_char8_t)
using u8cstr_writer
STRF_DEPRECATED_MSG("u8cstr_writer renamed to u8cstr_destination")
= basic_cstr_destination<char8_t>;
#endif

using cstr_writer
STRF_DEPRECATED_MSG("cstr_writer renamed to cstr_destination")
= basic_cstr_destination<char>;

using u16cstr_writer
STRF_DEPRECATED_MSG("u16cstr_writer renamed to u16cstr_destination")
= basic_cstr_destination<char16_t>;

using u32cstr_writer
STRF_DEPRECATED_MSG("u32cstr_writer renamed to u32cstr_destination")
= basic_cstr_destination<char32_t>;

using wcstr_writer
STRF_DEPRECATED_MSG("wcstr_writer renamed to wcstr_destination")
= basic_cstr_destination<wchar_t>;

template <typename CharT>
class array_destination final
    : public strf::output_buffer<CharT, strf::log2_garbage_buff_size>
{
    using dest_t_ = strf::output_buffer<CharT, strf::log2_garbage_buff_size>;

public:

    struct range{ CharT* dest; CharT* dest_end; };

    STRF_HD array_destination(range r) noexcept
        : dest_t_(r.dest, r.dest_end)
    {
        STRF_ASSERT(r.dest <= r.dest_end);
    }

    STRF_HD array_destination(CharT* dest, CharT* dest_end) noexcept
        : dest_t_(dest, dest_end)
    {
        STRF_ASSERT(dest <= dest_end);
    }

    STRF_HD array_destination(CharT* dest, std::size_t len) noexcept
        : dest_t_(dest, dest + len)
    {
    }

    template <std::size_t N>
    STRF_HD array_destination(CharT (&dest)[N]) noexcept
        : dest_t_(dest, dest + N)
    {
    }

    STRF_HD array_destination(const array_destination& r) noexcept
        : dest_t_(r.buffer_ptr(), r.buffer_end())
        , it_(r.it_)
    {
        this->set_good(r.good());
    }

    STRF_HD array_destination& operator=(const array_destination& r) noexcept
    {
        this->set_good(r.good());
        this->set_buffer_ptr(r.buffer_ptr());
        this->set_buffer_end(r.buffer_end());
        it_ = r.it_;
        return *this;
    }
    STRF_HD bool operator==(const array_destination& r) const noexcept
    {
        if (this->good()) {
            return ( r.good()
                  && this->buffer_ptr() == r.buffer_ptr()
                  && this->buffer_end() == r.buffer_end() );
        }
        return ! r.good() && it_ == r.it_;
    }
    STRF_HD void recycle_buffer() noexcept override
    {
        if (this->good()) {
            it_ = this->buffer_ptr();
            this->set_good(false);
            this->set_buffer_end(garbage_buff_end<CharT>());
        }
        this->set_buffer_ptr(garbage_buff<CharT>());
    }

    struct result
    {
        CharT* ptr;
        bool truncated;
    };

    STRF_HD result finish() noexcept
    {
        bool truncated = ! this->good();
        CharT* ptr = truncated ? it_ : this->buffer_ptr();
        return { ptr, truncated };
    }

private:

    STRF_HD void do_write(const CharT* data, std::size_t) noexcept override
    {
        auto sub_count = this->buffer_space();
        auto ptr = this->buffer_ptr();
#if !defined(STRF_FREESTANDING) || defined(STRF_WITH_CHAR_ARRAYING)
        memcpy(ptr, data, sub_count * sizeof(CharT));
#else
        for(; sub_count != 0; ++ptr, ++data, --sub_count) {
            *ptr = *data;
        }
#endif
        it_ = this->buffer_end();
        this->set_buffer_ptr(garbage_buff<CharT>());
        this->set_buffer_end(garbage_buff_end<CharT>());
        this->set_good(false);
    }

    CharT* it_ = nullptr;
};

template <typename CharT>
using basic_char_array_writer
STRF_DEPRECATED_MSG("basic_char_array_writer renamed to array_destination")
= array_destination<CharT>;

#if defined(__cpp_char8_t)
using u8char_array_writer  STRF_DEPRECATED = array_destination<char8_t>;
#endif
using char_array_writer    STRF_DEPRECATED = array_destination<char>;
using u16char_array_writer STRF_DEPRECATED = array_destination<char16_t>;
using u32char_array_writer STRF_DEPRECATED = array_destination<char32_t>;
using wchar_array_writer   STRF_DEPRECATED = array_destination<wchar_t>;

template <typename T>
class discarder final
    : public strf::output_buffer<T, strf::log2_garbage_buff_size>
{
    using dest_t_ = strf::output_buffer<T, strf::log2_garbage_buff_size>;

public:

    STRF_HD discarder() noexcept
        : dest_t_{strf::garbage_buff<T>(), strf::garbage_buff_end<T>()}
    {
        this->set_good(false);
    }

    STRF_HD ~discarder()
    {
    }

    STRF_HD void recycle_buffer() noexcept override
    {
        this->set_buffer_ptr(strf::garbage_buff<T>());
    }

private:

    STRF_HD void do_write(const T*, std::size_t) noexcept override
    {
        this->set_buffer_ptr(strf::garbage_buff<T>());
    }
};

} // namespace strf

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#endif  // STRF_DESTINATION_HPP_INCLUDED

