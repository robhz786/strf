#ifndef STRF_DETAIL_OUTPUT_TYPES_FILE_HPP
#define STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <strf/destination.hpp>

STRF_NAMESPACE_BEGIN

template <typename CharT>
class narrow_cfile_writer final: public strf::basic_outbuf_noexcept<CharT>
{
public:

    explicit narrow_cfile_writer(std::FILE* dest_)
        : strf::basic_outbuf_noexcept<CharT>(_buf, _buf_size)
        , _dest(dest_)
    {
        STRF_ASSERT(dest_ != nullptr);
    }

    narrow_cfile_writer() = delete;

#ifdef STRF_NO_CXX17_COPY_ELISION

    narrow_cfile_writer(narrow_cfile_writer&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    narrow_cfile_writer(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer(narrow_cfile_writer&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    ~narrow_cfile_writer()
    {
    }

    void recycle() noexcept
    {
        auto p = this->pos();
        this->set_pos(_buf);
        if (this->good())
        {
            std::size_t count = p - _buf;
            auto count_inc = std::fwrite(_buf, sizeof(CharT), count, _dest);
            _count += count_inc;
            this->set_good(count == count_inc);
        }
    }

    struct result
    {
        std::size_t count;
        bool success;
    };

    result finish()
    {
        bool g = this->good();
        this->set_good(false);
        if (g)
        {
            std::size_t count = this->pos() - _buf;
            auto count_inc = std::fwrite(_buf, sizeof(CharT), count, _dest);
            _count += count_inc;
            g = (count == count_inc);
        }
        return {_count, g};
    }

private:

    std::FILE* _dest;
    std::size_t _count = 0;
    static constexpr std::size_t _buf_size
        = strf::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

class wide_cfile_writer final: public strf::basic_outbuf_noexcept<wchar_t>
{
public:

    explicit wide_cfile_writer(std::FILE* dest_)
        : strf::basic_outbuf_noexcept<wchar_t>(_buf, _buf_size)
        , _dest(dest_)
    {
        STRF_ASSERT(dest_ != nullptr);
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
#ifdef __CUDA_ARCH__
        asm("trap;");
#endif
        // This will only be compiled as device-side code;
        // the host-side version simply doesn't have object
        // code, so using it should fail linking
        auto p = this->pos();
        this->set_pos(_buf);
        if (this->good())
        {
            for (auto it = _buf; it != p; ++it, ++_count)
            {
                if(std::fputwc(*it, _dest) == WEOF)
                {
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
        return {_count, g};
    }

  private:

    std::FILE* _dest;
    std::size_t _count = 0;
    static constexpr std::size_t _buf_size
        = strf::min_size_after_recycle<wchar_t>();
    wchar_t _buf[_buf_size];
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
        : _file(file)
    {}

    constexpr narrow_cfile_writer_creator
        (const narrow_cfile_writer_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{_file};
    }

private:
    FILE* _file;
};

class wide_cfile_writer_creator
{
public:

    using char_type = wchar_t;
    using outbuf_type = strf::wide_cfile_writer;
    using finish_type = typename outbuf_type::result;

    constexpr wide_cfile_writer_creator(FILE* file) noexcept
        : _file(file)
    {}

    constexpr wide_cfile_writer_creator(const wide_cfile_writer_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{_file};
    }

private:

    FILE* _file;
};

} // namespace detail


template <typename CharT = char>
inline auto to(std::FILE* destination)
{
#ifndef __CUDA_ARCH__
    return strf::destination_no_reserve
        < strf::detail::narrow_cfile_writer_creator<CharT> >
        (destination);
#else
    return 0;
#endif
}

inline auto wto(std::FILE* destination)
{
#ifndef __CUDA_ARCH__
    return strf::destination_no_reserve
        < strf::detail::wide_cfile_writer_creator >
        (destination);
#else
    return 0;
#endif
}


STRF_NAMESPACE_END

#endif  // STRF_DETAIL_OUTPUT_TYPES_FILE_HPP

