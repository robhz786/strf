#ifndef STRF_DETAIL_OUTPUT_TYPES_FILE_HPP
#define STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace strf {

template <typename CharT, std::size_t BuffSize>
class narrow_cfile_writer final: public strf::basic_outbuff<CharT>
{
    static_assert(BuffSize >= strf::min_space_after_recycle<CharT>(), "BuffSize too small");

public:

    explicit STRF_HD narrow_cfile_writer(std::FILE* d)
        : strf::basic_outbuff<CharT>(buf_, BuffSize)
        , dest_(d)
    {
        STRF_ASSERT(d != nullptr);
    }

    STRF_HD narrow_cfile_writer() = delete;

    narrow_cfile_writer(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer(narrow_cfile_writer&&) = delete;

    STRF_HD ~narrow_cfile_writer()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        auto p = this->pointer();
        this->set_pointer(buf_);
        STRF_IF_LIKELY (this->good()) {
            std::size_t count = p - buf_;
            auto count_inc = std::fwrite(buf_, sizeof(CharT), count, dest_);
            count_ += count_inc;
            this->set_good(count == count_inc);
        }
    }

    struct result
    {
        std::size_t count;
        bool success;
    };

    STRF_HD result finish()
    {
        bool g = this->good();
        this->set_good(false);
        STRF_IF_LIKELY (g) {
            std::size_t count = this->pointer() - buf_;
            auto count_inc = std::fwrite(buf_, sizeof(CharT), count, dest_);
            count_ += count_inc;
            g = (count == count_inc);
        }
        return {count_, g};
    }

private:

    STRF_HD void do_write(const CharT* str, std::size_t str_len) noexcept override
    {
#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
        auto p = this->pointer();
        this->set_pointer(buf_);
        STRF_IF_LIKELY (this->good()) {
            std::size_t count = p - buf_;
            auto count_inc = std::fwrite(buf_, sizeof(CharT), count, dest_);
            count_inc += std::fwrite(str, sizeof(CharT), str_len, dest_);
            count_ += count_inc;
            this->set_good(count_inc == count + str_len);
        }

#if defined(__GNUC__) && (__GNUC__ >= 11)
#  pragma GCC diagnostic pop
#endif
    }

    std::FILE* dest_;
    std::size_t count_ = 0;
    CharT buf_[BuffSize];
};

class wide_cfile_writer final: public strf::basic_outbuff<wchar_t>
{
public:

    STRF_HD explicit wide_cfile_writer(std::FILE* d)
        : strf::basic_outbuff<wchar_t>(buf_, buf_size_)
        , dest_(d)
    {
        STRF_ASSERT(d != nullptr);
    }

    wide_cfile_writer() = delete;
    wide_cfile_writer(const wide_cfile_writer&) = delete;
    wide_cfile_writer(wide_cfile_writer&&) = delete;

    STRF_HD ~wide_cfile_writer()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        auto p = this->pointer();
        this->set_pointer(buf_);
        STRF_IF_LIKELY (this->good()) {
            for (auto it = buf_; it != p; ++it, ++count_) {
                STRF_IF_UNLIKELY (std::fputwc(*it, dest_) == WEOF) {
                    this->set_good(false);
                    break;
                }
            }
        }
    }

    struct result
    {
        std::size_t count;
        bool success;
    };

    STRF_HD result finish()
    {
        recycle();
        auto g = this->good();
        this->set_good(false);
        return {count_, g};
    }

private:

    STRF_HD void do_write(const wchar_t* str, std::size_t str_len) noexcept override
    {
        auto p = this->pointer();
        this->set_pointer(buf_);
        STRF_IF_LIKELY (this->good()) {
            for (auto it = buf_; it != p; ++it, ++count_) {
                STRF_IF_UNLIKELY (std::fputwc(*it, dest_) == WEOF) {
                    this->set_good(false);
                    return;
                }
            }
            for (; str_len != 0; ++str, --str_len, ++count_) {
                STRF_IF_UNLIKELY (std::fputwc(*str, dest_) == WEOF) {
                    this->set_good(false);
                    return;
                }
            }
        }
    }

    std::FILE* dest_;
    std::size_t count_ = 0;
    static constexpr std::size_t buf_size_ = strf::min_space_after_recycle<wchar_t>();
    wchar_t buf_[buf_size_];
};

namespace detail {

template <typename CharT>
class narrow_cfile_writer_creator
{
public:

    using char_type = CharT;
    using outbuff_type = strf::narrow_cfile_writer<CharT, strf::min_space_after_recycle<CharT>()>;
    using finish_type = typename outbuff_type::result;

    constexpr narrow_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    constexpr narrow_cfile_writer_creator
        (const narrow_cfile_writer_creator&) = default;

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
    using outbuff_type = strf::wide_cfile_writer;
    using finish_type = typename outbuff_type::result;

    constexpr wide_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    constexpr wide_cfile_writer_creator(const wide_cfile_writer_creator&) = default;

    STRF_HD FILE* create() const noexcept
    {
        return file_;
    }

private:

    FILE* file_;
};

} // namespace detail


template <typename CharT = char>
STRF_HD inline auto to(std::FILE* destination)
    -> strf::destination_no_reserve<strf::detail::narrow_cfile_writer_creator<CharT>>
{
    return strf::destination_no_reserve
        < strf::detail::narrow_cfile_writer_creator<CharT> >
        (destination);
}

STRF_HD inline auto wto(std::FILE* destination)
    -> strf::destination_no_reserve<strf::detail::wide_cfile_writer_creator>
{
    return strf::destination_no_reserve
        < strf::detail::wide_cfile_writer_creator >
        (destination);
}


} // namespace strf

#endif  // STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

