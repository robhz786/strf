#ifndef STRF_DETAIL_OUTPUT_TYPES_FILE_HPP
#define STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace strf {

template <typename CharT>
class narrow_cfile_writer final: public strf::basic_outbuf_noexcept<CharT>
{
public:

    explicit STRF_HD narrow_cfile_writer(std::FILE* d)
        : strf::basic_outbuf_noexcept<CharT>(buf_, buf_size_)
        , dest_(d)
    {
        STRF_ASSERT(d != nullptr);
    }

    STRF_HD narrow_cfile_writer() = delete;

#ifdef STRF_NO_CXX17_COPY_ELISION

    STRF_HD narrow_cfile_writer(narrow_cfile_writer&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    narrow_cfile_writer(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer(narrow_cfile_writer&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    STRF_HD ~narrow_cfile_writer()
    {
    }

    STRF_HD void recycle() noexcept
    {
        auto p = this->pointer();
        this->set_pointer(buf_);
        if (this->good()) {
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
        if (g) {
            std::size_t count = this->pointer() - buf_;
            auto count_inc = std::fwrite(buf_, sizeof(CharT), count, dest_);
            count_ += count_inc;
            g = (count == count_inc);
        }
        return {count_, g};
    }

private:

    std::FILE* dest_;
    std::size_t count_ = 0;
    static constexpr std::size_t buf_size_
        = strf::min_size_after_recycle<sizeof(CharT)>();
    CharT buf_[buf_size_];
};

class wide_cfile_writer final: public strf::basic_outbuf_noexcept<wchar_t>
{
public:

    explicit wide_cfile_writer(std::FILE* d)
        : strf::basic_outbuf_noexcept<wchar_t>(buf_, buf_size_)
        , dest_(d)
    {
        STRF_ASSERT(d != nullptr);
    }

    wide_cfile_writer() = delete;

#ifdef STRF_NO_CXX17_COPY_ELISION

    wide_cfile_writer(wide_cfile_writer&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    wide_cfile_writer(const wide_cfile_writer&) = delete;
    wide_cfile_writer(wide_cfile_writer&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    STRF_HD ~wide_cfile_writer()
    {
    }

    STRF_HD void recycle() noexcept override
    {
        auto p = this->pointer();
        this->set_pointer(buf_);
        if (this->good()) {
            for (auto it = buf_; it != p; ++it, ++count_) {
                if(std::fputwc(*it, dest_) == WEOF) {
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

    std::FILE* dest_;
    std::size_t count_ = 0;
    static constexpr std::size_t buf_size_
        = strf::min_size_after_recycle<sizeof(wchar_t)>();
    wchar_t buf_[buf_size_];
};

namespace detail {

template <typename CharT>
class narrow_cfile_writer_creator
{
public:

    using char_type = CharT;
    using outbuf_type = strf::narrow_cfile_writer<CharT>;
    using finish_type = typename outbuf_type::result;

    constexpr narrow_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    constexpr narrow_cfile_writer_creator
        (const narrow_cfile_writer_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{file_};
    }

private:
    FILE* file_;
};

class wide_cfile_writer_creator
{
public:

    using char_type = wchar_t;
    using outbuf_type = strf::wide_cfile_writer;
    using finish_type = typename outbuf_type::result;

    constexpr wide_cfile_writer_creator(FILE* file) noexcept
        : file_(file)
    {}

    constexpr wide_cfile_writer_creator(const wide_cfile_writer_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{file_};
    }

private:

    FILE* file_;
};

} // namespace detail


template <typename CharT = char>
inline auto to(std::FILE* destination)
{
    return strf::destination_no_reserve
        < strf::detail::narrow_cfile_writer_creator<CharT> >
        (destination);
}

inline auto wto(std::FILE* destination)
{
    return strf::destination_no_reserve
        < strf::detail::wide_cfile_writer_creator >
        (destination);
}


} // namespace strf

#endif  // STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

