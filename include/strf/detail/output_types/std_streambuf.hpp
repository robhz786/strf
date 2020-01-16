#ifndef STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <strf/destination.hpp>
#include <strf/outbuf.hpp>

STRF_NAMESPACE_BEGIN

template <typename CharT, typename Traits = std::char_traits<CharT> >
class basic_streambuf_writer final: public strf::basic_outbuf<CharT>
{
public:

    explicit basic_streambuf_writer(std::basic_streambuf<CharT, Traits>& dest_)
        : strf::basic_outbuf<CharT>(_buf, _buf_size)
        , _dest(dest_)
    {
    }

    basic_streambuf_writer() = delete;

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_streambuf_writer(basic_streambuf_writer&& other);

#else  // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_streambuf_writer(const basic_streambuf_writer&) = delete;
    basic_streambuf_writer(basic_streambuf_writer&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    ~basic_streambuf_writer()
    {
    }

    void recycle() override
    {
        std::streamsize count = this->pos() - _buf;
        this->set_pos(_buf);
        if (this->good()) {
            auto count_inc = _dest.sputn(_buf, count);
            _count += count_inc;
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
        std::streamsize count = this->pos() - _buf;
        auto g = this->good();
        this->set_pos(_buf);
        this->set_good(false);
        if (g) {
            auto count_inc = _dest.sputn(_buf, count);
            _count += count_inc;
            g = (count_inc == count);
        }
        return {_count, g};
    }

private:

    std::basic_streambuf<CharT, Traits>& _dest;
    std::streamsize _count = 0;
    static constexpr std::size_t _buf_size
        = strf::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

using streambuf_writer
    = strf::basic_streambuf_writer< char, std::char_traits<char> >;

using wstreambuf_writer
    = strf::basic_streambuf_writer< wchar_t, std::char_traits<wchar_t> >;

namespace detail {

template <typename CharT, typename Traits>
class basic_streambuf_writer_creator
{
    using _outbuf_type = strf::basic_streambuf_writer<CharT, Traits>;
    using _finish_type = typename _outbuf_type::result;

public:

    using char_type = CharT;

    basic_streambuf_writer_creator
        ( std::basic_streambuf<CharT, Traits>& dest )
        : _dest(dest)
    {
    }

    basic_streambuf_writer_creator(const basic_streambuf_writer_creator&) = default;

    _outbuf_type create() const
    {
        return _outbuf_type{_dest};
    }

private:

    std::basic_streambuf<CharT, Traits>& _dest;
};


} // namespace detail


template <typename CharT, typename Traits = std::char_traits<CharT> >
inline auto to( std::basic_streambuf<CharT, Traits>& dest )
{
    return strf::destination_no_reserve
        < strf::detail::basic_streambuf_writer_creator<CharT, Traits> >
        (dest);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
inline auto to( std::basic_streambuf<CharT, Traits>* dest )
{
    return strf::to(*dest);
}

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

