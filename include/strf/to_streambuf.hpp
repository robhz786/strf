#ifndef STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <strf.hpp>

namespace strf {

template <typename CharT, typename Traits = std::char_traits<CharT> >
class basic_streambuf_writer final: public strf::basic_outbuff<CharT>
{
public:

    explicit basic_streambuf_writer(std::basic_streambuf<CharT, Traits>& d)
        : strf::basic_outbuff<CharT>(buf_, buf_size_)
        , dest_(d)
    {
    }
    explicit basic_streambuf_writer(std::basic_streambuf<CharT, Traits>* d)
        : strf::basic_outbuff<CharT>(buf_, buf_size_)
        , dest_(*d)
    {
    }

    basic_streambuf_writer() = delete;

    basic_streambuf_writer(const basic_streambuf_writer&) = delete;
    basic_streambuf_writer(basic_streambuf_writer&&) = delete;

    ~basic_streambuf_writer()
    {
    }

    void recycle() override
    {
        std::streamsize count = this->pointer() - buf_;
        this->set_pointer(buf_);
        if (this->good()) {
            auto count_inc = dest_.sputn(buf_, count);
            count_ += count_inc;
            this->set_good(count_inc == count);
        }
    }

    struct result
    {
        std::streamsize count;
        bool success;
    };

    result finish()
    {
        std::streamsize count = this->pointer() - buf_;
        auto g = this->good();
        this->set_pointer(buf_);
        this->set_good(false);
        if (g) {
            this->set_good(false);
            auto count_inc = dest_.sputn(buf_, count);
            count_ += count_inc;
            g = (count_inc == count);
        }
        return {count_, g};
    }

private:

    void do_write(const CharT* str, std::size_t str_len) override
    {
        std::streamsize count = this->pointer() - buf_;
        this->set_pointer(buf_);
        if (this->good()) {
            this->set_good(false);
            auto count_inc = dest_.sputn(buf_, count);
            count_inc += dest_.sputn(str, str_len);
            count_ += count_inc;
            this->set_good(count_inc == static_cast<std::streamsize>(count + str_len));
        }
    }

    std::basic_streambuf<CharT, Traits>& dest_;
    std::streamsize count_ = 0;
    static constexpr std::size_t buf_size_
        = strf::min_space_after_recycle<CharT>();
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
    using outbuff_type = strf::basic_streambuf_writer<CharT, Traits>;
    using finish_type = typename outbuff_type::result;

    explicit basic_streambuf_writer_creator
        ( std::basic_streambuf<CharT, Traits>& dest )
        : dest_(dest)
    {
    }

    basic_streambuf_writer_creator(const basic_streambuf_writer_creator&) = default;

    std::basic_streambuf<CharT, Traits>& create() const noexcept
    {
        return dest_;
    }

private:

    std::basic_streambuf<CharT, Traits>& dest_;
};


} // namespace detail


template <typename CharT, typename Traits>
inline auto to( std::basic_streambuf<CharT, Traits>& dest )
{
    return strf::destination_no_reserve
        < strf::detail::basic_streambuf_writer_creator<CharT, Traits> >
        (dest);
}


template<typename CharT, typename Traits>
inline auto to( std::basic_streambuf<CharT, Traits>* dest )
{
    return strf::to(*dest);
}

} // namespace strf

#endif  // STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

