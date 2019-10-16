#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
class narrow_cfile_writer final: public stringify::v0::basic_outbuf_noexcept<CharT>
{
public:

    explicit narrow_cfile_writer(std::FILE* dest_)
        : stringify::v0::basic_outbuf_noexcept<CharT>(_buf, _buf_size)
        , _dest(dest_)
    {
        BOOST_ASSERT(dest_ != nullptr);
    }

    narrow_cfile_writer() = delete;
    narrow_cfile_writer(const narrow_cfile_writer&) = delete;
    narrow_cfile_writer(narrow_cfile_writer&&) = delete;

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
        = stringify::v0::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

class wide_cfile_writer final: public stringify::v0::basic_outbuf_noexcept<wchar_t>
{
public:

    explicit wide_cfile_writer(std::FILE* dest_)
        : stringify::v0::basic_outbuf_noexcept<wchar_t>(_buf, _buf_size)
        , _dest(dest_)
    {
        BOOST_ASSERT(dest_ != nullptr);
    }

    wide_cfile_writer() = delete;
    wide_cfile_writer(const wide_cfile_writer&) = delete;
    wide_cfile_writer(wide_cfile_writer&&) = delete;

    ~wide_cfile_writer()
    {
    }

    void recycle() noexcept
    {
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

    result finish()
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
        = stringify::v0::min_size_after_recycle<wchar_t>();
    wchar_t _buf[_buf_size];
};

template <typename CharT = char>
inline auto write(std::FILE* destination)
{
    using writer = stringify::v0::narrow_cfile_writer<CharT>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE* >
        (destination);
}

inline auto wwrite(std::FILE* destination)
{
    using writer = stringify::v0::wide_cfile_writer;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE* >
        (destination);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

