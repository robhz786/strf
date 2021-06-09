//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


// NOTE: This example depends on the following external libraries:
//     - https://github.com/ned14/outcome
//     - https://github.com/ned14/quickcpplib
//     - https://github.com/ned14/llfio

#include <strf/outbuff.hpp>
#include <llfio.hpp>

namespace llfio = LLFIO_V2_NAMESPACE;

template <typename CharT, std::size_t BufferSize>
class llfio_file_writer final: public strf::basic_outbuff<CharT>
{
public:

    llfio_file_writer(llfio::file_handle&& file, std::size_t offset = 0)
        : strf::basic_outbuff<CharT>{buffer_, BufferSize}
        , file_{std::move(file)}
        , offset_{offset}
    {
    }

    llfio_file_writer(const llfio_file_writer&) = delete;

    llfio_file_writer(llfio_file_writer&& other)
        : strf::basic_outbuff<CharT>{buffer_, BufferSize}
        , file_{std::move(other.file_)}
        , offset_{other.offset_}
        , error_{other.error_}
    {
        if (other.good()) {
            std::size_t chars_count = other.pointer() - other.buffer_;
            memcpy(buffer_, other.buffer_, chars_count * sizeof(CharT));
            this->advance(chars_count);
        } else {
            set_bad_();
        }
    }

    ~llfio_file_writer()
    {
        (void) close();
    }

    void recycle() noexcept override;

    llfio::result<void> close() noexcept;

    llfio::file_handle release_handle() noexcept;

    llfio::error_info get_error() const noexcept
    {
        return error_;
    }

private:

    void set_bad_() noexcept
    {
        this->set_good(false);
        this->set_pointer(strf::garbage_buff<CharT>());
        this->set_end(strf::garbage_buff_end<CharT>());
    }

    llfio::result<llfio::file_handle::const_buffers_type>
    write_(std::size_t chars_count) noexcept
    {
        auto bytes_count = chars_count * sizeof(CharT);
        auto bytes_ptr = reinterpret_cast<const llfio::byte *>(buffer_);
        llfio::file_handle::const_buffer_type buff{bytes_ptr, bytes_count};
        auto result = file_.write({{&buff, 1}, offset_ });
        offset_ += bytes_count;
        return result;
    }

    llfio::file_handle file_;
    std::size_t offset_ = 0;
    llfio::error_info error_;
    CharT buffer_[BufferSize];
    static_assert(BufferSize >= strf::min_space_after_recycle<CharT>());
};

template <typename CharT, std::size_t BufferSize>
void llfio_file_writer<CharT, BufferSize>::recycle() noexcept
{
    if (this->good()) {
        std::size_t chars_count = this->pointer() - buffer_;
        if (chars_count) {
            auto result = write_(chars_count);
            if (! result.has_error()) {
                this->set_pointer(buffer_);
            } else {
                error_ = result.assume_error();
                set_bad_();
            }
        }
    } else {
        this->set_pointer(strf::garbage_buff<CharT>());
    }
}

template <typename CharT, std::size_t BufferSize>
llfio::result<void> llfio_file_writer<CharT, BufferSize>::close() noexcept
{
    if (this->good()) {
        std::size_t chars_count = this->pointer() - buffer_;
        if (chars_count) {
            auto result = write_(chars_count);
            if (result.has_error()) {
                error_ = result.assume_error();
            }
        }
        set_bad_();
    }
    auto close_result = file_.close();
    if (error_ != llfio::error_info{}) {
        return llfio::result<void>{error_};
    }
    return close_result;
}

template <typename CharT, std::size_t BufferSize>
llfio::file_handle llfio_file_writer<CharT, BufferSize>::release_handle() noexcept
{
    if (this->good()) {
        std::size_t chars_count = this->pointer() - buffer_;
        if (chars_count) {
            auto result = write_(chars_count);
            if (result.has_error()) {
                error_ = result.assume_error();
            }
        }
        set_bad_();
    }
    return std::move(file_);
}

#include <strf.hpp>

int main()
{
    llfio::file_handle fh = llfio::file
        ( {}
        , "llfio_example_output.txt"
        , llfio::file_handle::mode::write
        , llfio::file_handle::creation::if_needed )
        .value();

    llfio_file_writer<char, 4096> file(std::move(fh));

    strf::to(file) ("Begin of content\n");
    for (int line = 1; line <= 10000000; ++line) {
        strf::to(file) ("Line number ", strf::right(line, 50, '.'), '\n');
    }
    strf::to(file) ("End of content\n");
    file.close().value();

    return 0;
}
