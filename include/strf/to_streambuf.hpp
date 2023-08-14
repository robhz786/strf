#ifndef STRF_TO_STREAMBUF_HPP
#define STRF_TO_STREAMBUF_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <strf.hpp>

namespace strf {

template <typename CharT, typename Traits = std::char_traits<CharT> >
class basic_streambuf_writer final: public strf::destination<CharT>
{
public:

    explicit basic_streambuf_writer(std::basic_streambuf<CharT, Traits>& d)
        : strf::destination<CharT>(buf_, buf_size_)
        , dst_(d)
    {
    }
    explicit basic_streambuf_writer(std::basic_streambuf<CharT, Traits>* d)
        : strf::destination<CharT>(buf_, buf_size_)
        , dst_(*d)
    {
    }

    basic_streambuf_writer() = delete;

    basic_streambuf_writer(const basic_streambuf_writer&) = delete;
    basic_streambuf_writer(basic_streambuf_writer&&) = delete;
    basic_streambuf_writer& operator=(const basic_streambuf_writer&) = delete;
    basic_streambuf_writer& operator=(basic_streambuf_writer&&) = delete;

    ~basic_streambuf_writer() override {
        if (this->good()) {
            const std::streamsize count = this->buffer_ptr() - buf_;

#if defined __cpp_exceptions
            try { dst_.sputn(buf_, count); } catch(...) {};
#else
            dst_.sputn(buf_, count);
#endif
        }
    }

    void recycle() override {
        const std::streamsize count = this->buffer_ptr() - buf_;
        this->set_buffer_ptr(buf_);
        STRF_IF_LIKELY (this->good()) {
            auto count_inc = dst_.sputn(buf_, count);
            count_ += count_inc;
            this->set_good(count_inc == count);
        }
    }

    struct result {
        std::streamsize count;
        bool success;
    };

    result finish() {
        const std::streamsize count = this->buffer_ptr() - buf_;
        auto g = this->good();
        this->set_buffer_ptr(buf_);
        this->set_good(false);
        STRF_IF_LIKELY (g) {
            this->set_good(false);
            auto count_inc = dst_.sputn(buf_, count);
            count_ += count_inc;
            g = (count_inc == count);
        }
        return {count_, g};
    }

private:

    void do_write(const CharT* str, std::size_t str_len) override {
        auto str_slen = static_cast<std::streamsize>(str_len);
        auto count = this->buffer_ptr() - buf_;
        count = count >= 0 ? count : 0;
        this->set_buffer_ptr(buf_);
        STRF_IF_LIKELY (this->good()) {
            this->set_good(false);
            auto count_inc = dst_.sputn(buf_, count);
            count_inc += dst_.sputn(str,  str_slen);
            count_ += count_inc;
            this->set_good(count_inc == count + str_slen);
        }
    }

    std::basic_streambuf<CharT, Traits>& dst_;
    std::streamsize count_ = 0;
    static constexpr std::size_t buf_size_
        = strf::min_destination_buffer_size;
    CharT buf_[buf_size_];
};

using streambuf_writer  = strf::basic_streambuf_writer<char>;
using wstreambuf_writer = strf::basic_streambuf_writer<wchar_t>;

namespace detail {

template <typename CharT, typename Traits>
class basic_streambuf_writer_creator
{

public:

    using char_type = CharT;
    using destination_type = strf::basic_streambuf_writer<CharT, Traits>;
    using finish_type = typename destination_type::result;

    explicit basic_streambuf_writer_creator
        ( std::basic_streambuf<CharT, Traits>& dst )
        : dst_(dst)
    {
    }

    std::basic_streambuf<CharT, Traits>& create() const noexcept
    {
        return dst_;
    }

private:

    std::basic_streambuf<CharT, Traits>& dst_;
};


} // namespace detail


template <typename CharT, typename Traits>
inline auto to( std::basic_streambuf<CharT, Traits>& dst )
    -> strf::printing_syntax
        < strf::detail::basic_streambuf_writer_creator<CharT, Traits> >
{
    return strf::make_printing_syntax
        ( strf::detail::basic_streambuf_writer_creator<CharT, Traits>(dst) );
}


template<typename CharT, typename Traits>
inline auto to( std::basic_streambuf<CharT, Traits>* dst )
    -> strf::printing_syntax
        < strf::detail::basic_streambuf_writer_creator<CharT, Traits> >
{
    return strf::to(*dst);
}

} // namespace strf

#endif // STRF_TO_STREAMBUF_HPP

