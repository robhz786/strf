#ifndef STRF_TO_CFILE_HPP
#define STRF_TO_CFILE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <cstdio>
#include <cstring>
#include <cwchar>

#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

namespace strf {

namespace detail {

template <typename CharT>
class narrow_cfile_writer_traits {
public:
    narrow_cfile_writer_traits() = delete;

    STRF_HD explicit narrow_cfile_writer_traits(FILE* file)
        : file_(file)
    {
    }

    STRF_HD std::size_t write(const CharT* ptr, std::size_t count) const noexcept {
        return std::fwrite(ptr, sizeof(CharT), count, file_);
    }

private:
    FILE* file_;
};

class wide_cfile_writer_traits {
public:
    wide_cfile_writer_traits() = delete;

    STRF_HD explicit wide_cfile_writer_traits(FILE* file)
        : file_(file)
    {
    }

    STRF_HD std::size_t write(const wchar_t* ptr, std::size_t count) const noexcept {
        std::size_t successful_count = 0;
        for (; successful_count < count; ++successful_count) {
            if (std::fputwc(*ptr, file_) == WEOF) {
                break;
            }
            ++ptr;
        }
        return successful_count;
    }

private:
    FILE* file_;
};

struct cfile_writer_result {
    std::size_t count;
    bool success;
};

// The purpose of the Traits template parameter is to enable the
// unit tests to simulate unsuccessful writings.
// Also, to reduce the number of warning messages emitted by
// nvcc ( CUDA compiler ) because of a __host__ __device__ functions
// ( recycle and do_write ) calling a __host__ function ( fwrite )
template <typename CharT, typename Traits>
class cfile_writer_base
    : public strf::destination<CharT>
{
    static_assert(noexcept(std::declval<Traits>().write(nullptr, 0)), "");

public:
    cfile_writer_base(const cfile_writer_base&) = delete;
    cfile_writer_base(cfile_writer_base&&) = delete;
    cfile_writer_base& operator=(const cfile_writer_base&) = delete;
    cfile_writer_base& operator=(cfile_writer_base&&) = delete;

    template <typename... TraitsInitArgs>
    STRF_HD cfile_writer_base
        ( CharT* buff
        , std::size_t buff_size
        , TraitsInitArgs&&... args)
        : strf::destination<CharT>(buff, buff_size)
        , buff_(buff)
        , traits_(std::forward<TraitsInitArgs>(args)...)
    {
    }

    ~cfile_writer_base() override = default;

    STRF_HD void recycle() noexcept override {
        auto *p = this->buffer_ptr();
        this->set_buffer_ptr(buff_);
        STRF_IF_LIKELY (this->good()) {
            const std::size_t count = detail::safe_cast_size_t(p - buff_);
            auto count_inc = traits_.write(buff_, count);
            count_ += count_inc;
            const bool success = count_inc == count;
            this->set_good(success);
        }
    }

    using result = strf::detail::cfile_writer_result;

    STRF_HD result finish() {
        bool g = this->good();
        this->set_good(false);
        STRF_IF_LIKELY (g) {
            const auto count = detail::safe_cast_size_t(this->buffer_ptr() - buff_);
            auto count_inc = traits_.write(buff_, count);
            count_ += count_inc;
            g = (count == count_inc);
        }
        return {count_, g};
    }

protected:

    STRF_HD void destroy() // must be called in the destructor of derivate class
    {
        if (this->good()) {
            traits_.write(buff_, detail::safe_cast_size_t(this->buffer_ptr() - buff_));
        }
    }

private:

    STRF_HD void do_write(const CharT* str, std::size_t str_len) noexcept override {
        auto *p = this->buffer_ptr();
        this->set_buffer_ptr(buff_);
        STRF_IF_LIKELY (this->good()) {
            const std::size_t count = detail::safe_cast_size_t(p - buff_);
            auto count_inc = traits_.write(buff_, count);
            if (count_inc == count) {
                count_inc += traits_.write(str, str_len);
            }
            count_ += count_inc;
            this->set_good(count_inc == count + str_len);
        }
    }


    std::size_t count_ = 0;
    CharT* const buff_;
    Traits traits_;
};

} // namespace detail

template <typename CharT, std::size_t BuffSize>
class narrow_cfile_writer final
    : public strf::detail::cfile_writer_base
        < CharT, strf::detail::narrow_cfile_writer_traits<CharT> >
{
    static_assert(BuffSize >= strf::min_destination_buffer_size, "BuffSize too small");

    using impl_ = strf::detail::cfile_writer_base
        < CharT, strf::detail::narrow_cfile_writer_traits<CharT> >;
public:

    explicit STRF_HD narrow_cfile_writer(std::FILE* file)
        : impl_(buf_, BuffSize, file)
    {
        STRF_ASSERT(file != nullptr);
    }

    narrow_cfile_writer() = delete;

    STRF_HD ~narrow_cfile_writer() override
    {
        impl_::destroy();
    }

    narrow_cfile_writer(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer(narrow_cfile_writer&&) = delete;
    narrow_cfile_writer& operator=(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer& operator=(narrow_cfile_writer&&) = delete;

    using result = typename impl_::result;
    using impl_::recycle;
    using impl_::finish;

private:

    CharT buf_[BuffSize];
};

class wide_cfile_writer final
    : public strf::detail::cfile_writer_base
        < wchar_t, strf::detail::wide_cfile_writer_traits >
{
    using impl_ = strf::detail::cfile_writer_base
        < wchar_t, strf::detail::wide_cfile_writer_traits >;
public:

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    explicit STRF_HD wide_cfile_writer(std::FILE* file)
        : impl_(buf_, buf_size_, file)
    {
        STRF_ASSERT(file != nullptr);
    }

    wide_cfile_writer() = delete;

    STRF_HD ~wide_cfile_writer() override
    {
        impl_::destroy();
    }

    wide_cfile_writer(const wide_cfile_writer&) = delete;
    wide_cfile_writer(wide_cfile_writer&&) = delete;
    wide_cfile_writer& operator=(const wide_cfile_writer&) = delete;
    wide_cfile_writer& operator=(wide_cfile_writer&&) = delete;

    using result = typename impl_::result;
    using impl_::recycle;
    using impl_::finish;

private:
    static constexpr std::size_t buf_size_ = strf::min_destination_buffer_size;
    wchar_t buf_[buf_size_];
};

namespace detail {

template <typename CharT>
class narrow_cfile_writer_creator
{
public:

    using char_type = CharT;
    using destination_type = strf::narrow_cfile_writer<CharT, strf::min_destination_buffer_size>;
    using finish_type = typename destination_type::result;

    constexpr STRF_HD explicit narrow_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    STRF_HD FILE* create() const
    {
        return file_;
    }

private:
    FILE* file_;
};

class wide_cfile_writer_creator
{
public:

    using char_type = wchar_t;
    using destination_type = strf::wide_cfile_writer;
    using finish_type = typename destination_type::result;

    constexpr STRF_HD explicit wide_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    STRF_HD FILE* create() const noexcept
    {
        return file_;
    }

private:

    FILE* file_;
};

} // namespace detail


template <typename CharT = char>
STRF_HD inline auto to(std::FILE* destfile)
    -> strf::printing_syntax<strf::detail::narrow_cfile_writer_creator<CharT>>
{
    return strf::make_printing_syntax
        ( strf::detail::narrow_cfile_writer_creator<CharT>(destfile) );
}

STRF_HD inline auto wto(std::FILE* destfile)
    -> strf::printing_syntax<strf::detail::wide_cfile_writer_creator>
{
    return strf::make_printing_syntax
        ( strf::detail::wide_cfile_writer_creator(destfile) );
}

} // namespace strf

#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic pop
#endif

#endif // STRF_TO_CFILE_HPP

