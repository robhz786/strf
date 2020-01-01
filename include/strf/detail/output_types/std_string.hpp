#ifndef STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>
#include <strf/destination.hpp>
#include <string>

STRF_NAMESPACE_BEGIN

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_appender final: public strf::basic_outbuf<CharT>
{
    using _string_type = std::basic_string<CharT, Traits, Allocator>;

public:

    basic_string_appender(_string_type& str)
        : strf::basic_outbuf<CharT>(_buf, _buf_size)
        , _str(str)
    {
    }
    basic_string_appender( _string_type& str
                         , std::size_t size )
        : strf::basic_outbuf<CharT>(_buf, _buf_size)
        , _str(str)
    {
        _str.reserve(size);
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_appender(basic_string_appender&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_appender(const basic_string_appender&) = delete;
    basic_string_appender(basic_string_appender&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    void recycle() override
    {
        auto * p = this->pos();
        this->set_pos(_buf);
        if (this->good())
        {
            this->set_good(false);
            _str.append(_buf, p);
            this->set_good(true);
        }
    }

    void finish()
    {
        auto * p = this->pos();
        if (this->good())
        {
            this->set_good(false);
            _str.append(_buf, p);
        }
    }

private:

    _string_type& _str;
    static constexpr std::size_t _buf_size
        = strf::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker final: public strf::basic_outbuf<CharT>
{
    using _string_type = std::basic_string<CharT, Traits, Allocator>;

public:

    basic_string_maker() : strf::basic_outbuf<CharT>(_buf, _buf_size)
    {
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_maker(basic_string_maker&&);

#else  // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_maker(const basic_string_maker&) = delete;
    basic_string_maker(basic_string_maker&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    ~basic_string_maker() = default;

    void recycle() override
    {
        auto * p = this->pos();
        this->set_pos(_buf);
        if (this->good())
        {
            this->set_good(false);
            _str.append(_buf, p);
            this->set_good(true);
        }
    }

    _string_type finish()
    {
        auto * p = this->pos();
        if (this->good())
        {
            this->set_good(false);
            _str.append(_buf, p);
        }
        return std::move(_str);
    }

private:

    _string_type _str;
    static constexpr std::size_t _buf_size
        = strf::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_pre_sized_string_maker final
    : public strf::basic_outbuf<CharT>
{
public:

    basic_pre_sized_string_maker(std::size_t count)
        : strf::basic_outbuf<CharT>(nullptr, nullptr)
        , _str(count, (CharT)0)
    {
        this->set_pos(&*_str.begin());
        this->set_end(&*_str.begin() + count);
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_pre_sized_string_maker(basic_pre_sized_string_maker&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_pre_sized_string_maker(const basic_pre_sized_string_maker&) = delete;
    basic_pre_sized_string_maker(basic_pre_sized_string_maker&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    void recycle() override
    {
        std::size_t original_size = this->pos() - _str.data();
        auto append_size = std::max
            ( original_size
            , strf::min_size_after_recycle<CharT>() );
        _str.append(append_size, (CharT)0);
        this->set_pos(&*_str.begin() + original_size);
        this->set_end(&*_str.begin() + original_size + append_size);
    }

    std::basic_string<CharT, Traits, Allocator> finish()
    {
        _str.resize(this->pos() - _str.data());
        return std::move(_str);
    }

private:

    std::basic_string<CharT, Traits, Allocator> _str;
};

using string_appender = basic_string_appender<char>;
using u16string_appender = basic_string_appender<char16_t>;
using u32string_appender = basic_string_appender<char32_t>;
using wstring_appender = basic_string_appender<wchar_t>;

using string_maker = basic_string_maker<char>;
using u16string_maker = basic_string_maker<char16_t>;
using u32string_maker = basic_string_maker<char32_t>;
using wstring_maker = basic_string_maker<wchar_t>;

using pre_sized_string_maker = basic_pre_sized_string_maker<char>;
using pre_sized_u16string_maker = basic_pre_sized_string_maker<char16_t>;
using pre_sized_u32string_maker = basic_pre_sized_string_maker<char32_t>;
using pre_sized_wstring_maker = basic_pre_sized_string_maker<wchar_t>;

#if defined(__cpp_char8_t)

using u8string_appender = basic_string_appender<char8_t>;
using u8string_maker = basic_string_maker<char8_t>;
using pre_sized_u8string_maker = basic_pre_sized_string_maker<char8_t>;

#endif

namespace detail {

template <typename CharT, typename Traits, typename Allocator>
class basic_string_appender_creator
{
public:

    using char_type = CharT;
    using outbuf_type = strf::basic_string_appender<CharT, Traits, Allocator>;
    using finish_type = void;

    basic_string_appender_creator
        ( std::basic_string<CharT, Traits, Allocator>& str )
        : _str(str)
    {
    }

    basic_string_appender_creator(const basic_string_appender_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{_str};
    }

    outbuf_type create(std::size_t size) const
    {
        _str.reserve(_str.size() + size);
        return outbuf_type{_str};
    }

private:

    std::basic_string<CharT, Traits, Allocator>& _str;
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker_creator
{
public:

    using char_type = CharT;
    using finish_type = std::basic_string<CharT, Traits, Allocator>;

    strf::basic_string_maker<CharT, Traits, Allocator> create() const
    {
        return strf::basic_string_maker<CharT, Traits, Allocator>{};
    }
    strf::basic_pre_sized_string_maker<CharT, Traits, Allocator>
    create(std::size_t size) const
    {
        return strf::basic_pre_sized_string_maker<CharT, Traits, Allocator>{size};
    }
};

}

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    return strf::destination_no_reserve
        < strf::detail::basic_string_appender_creator<CharT, Traits, Allocator> >
        { str };
}

template <typename CharT, typename Traits, typename Allocator>
auto assign(std::basic_string<CharT, Traits, Allocator>& str)
{
    str.clear();
    return append(str);
}

template< typename CharT
        , typename Traits = std::char_traits<CharT>
        , typename Allocator = std::allocator<CharT> >
constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<CharT, Traits, Allocator> >
    to_basic_string{};

#if defined(__cpp_char8_t)

constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<char8_t> >
    to_u8string{};

#endif

constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<char> >
    to_string{};

constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<char16_t> >
    to_u16string{};

constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<char32_t> >
    to_u32string{};

constexpr strf::destination_no_reserve
    < strf::detail::basic_string_maker_creator<wchar_t> >
    to_wstring{};


STRF_NAMESPACE_END

#endif  // STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

