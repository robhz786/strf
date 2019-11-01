#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/outbuf.hpp>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename T, bool NoExcept, typename CharT>
class string_writer_mixin;

template <typename T, typename CharT>
class string_writer_mixin<T, false, CharT>
{
public:

    string_writer_mixin() = default;

    void do_recycle()
    {
        auto * p = static_cast<T*>(this)->pos();
        static_cast<T*>(this)->set_pos(_buf);
        if (static_cast<T*>(this)->good())
        {
            static_cast<T*>(this)->set_good(false);
            static_cast<T*>(this)->_append(buf_begin(), p);
            static_cast<T*>(this)->set_good(true);
        }
    }

    void do_finish()
    {
        auto * p = static_cast<T*>(this)->pos();
        if (static_cast<T*>(this)->good())
        {
            static_cast<T*>(this)->set_good(false);
            static_cast<T*>(this)->_append(buf_begin(), p);
        }
    }

    CharT* buf_begin()
    {
        return _buf;
    }
    CharT* buf_end()
    {
        return _buf + _buf_size;
    }

private:

    static constexpr std::size_t _buf_size
        = stringify::v0::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

#if defined(__cpp_exceptions)

template <typename T, typename CharT>
class string_writer_mixin<T, true, CharT>
{
public:

    string_writer_mixin() = default;

    void do_recycle() noexcept
    {
        auto * p = static_cast<T*>(this)->pos();
        static_cast<T*>(this)->set_pos(buf_begin());
        if (static_cast<T*>(this)->good())
        {
            try
            {
                static_cast<T*>(this)->_append(buf_begin(), p);
            }
            catch(...)
            {
                _eptr = std::current_exception();
                static_cast<T*>(this)->set_good(false);
            }
        }
    }

    void do_finish()
    {
        if (_eptr != nullptr)
        {
            std::rethrow_exception(_eptr);
        }
        BOOST_ASSERT(static_cast<T*>(this)->good());
        auto * p = static_cast<T*>(this)->pos();
        static_cast<T*>(this)->set_good(false);
        static_cast<T*>(this)->_append(buf_begin(), p);
    }

    CharT* buf_begin()
    {
        return _buf;
    }
    CharT* buf_end()
    {
        return _buf + _buf_size;
    }

private:

    std::exception_ptr _eptr = nullptr;
    static constexpr std::size_t _buf_size
        = stringify::v0::min_size_after_recycle<CharT>();
    CharT _buf[_buf_size];
};

#else // defined(__cpp_exceptions)

template <typename T, typename CharT>
class string_writer_mixin<T, true, CharT>
    : public string_writer_mixin<T, false, CharT>
{
};

#endif // defined(__cpp_exceptions)

template < bool NoExcept
         , typename CharT
         , typename Traits
         , typename Allocator >
class basic_string_appender_impl
    : public stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
    , protected stringify::v0::detail::string_writer_mixin
        < basic_string_appender_impl<NoExcept, CharT, Traits, Allocator>
        , NoExcept
        , CharT >
{
public:

    using string_type = std::basic_string<CharT, Traits>;

    basic_string_appender_impl(string_type& str_)
        : stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
            ( stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() )
        , _str(str_)
    {
        this->set_pos(this->buf_begin());
        this->set_end(this->buf_end());
    }

    basic_string_appender_impl(string_type& str_, std::size_t size)
        : stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
            ( stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() )
        , _str(str_)
    {
        this->set_pos(this->buf_begin());
        this->set_end(this->buf_end());
        str_.reserve(str_.size() + size);
    }

    basic_string_appender_impl() = delete;
    basic_string_appender_impl(const basic_string_appender_impl&) = delete;
    basic_string_appender_impl(basic_string_appender_impl&&) = delete;

    ~basic_string_appender_impl() = default;

    void finish()
    {
        this->do_finish();
    }

private:

    template <typename, bool, typename>
    friend class detail::string_writer_mixin;

    void _append(const CharT* begin, const CharT* end)
    {
        _str.append(begin, end);
    }

    string_type& _str;
};

template < bool NoExcept
         , typename CharT
         , typename Traits
         , typename Allocator >
class basic_string_maker_impl
    : public stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
    , protected stringify::v0::detail::string_writer_mixin
        < basic_string_maker_impl<NoExcept, CharT, Traits, Allocator>
        , NoExcept
        , CharT >
{
public:

    using string_type = std::basic_string<CharT, Traits>;

    basic_string_maker_impl()
        : stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
            ( stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() )
    {
        this->set_pos(this->buf_begin());
        this->set_end(this->buf_end());
    }

    basic_string_maker_impl(std::size_t size)
        : stringify::v0::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
            ( stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() )
    {
        this->set_pos(this->buf_begin());
        this->set_end(this->buf_end());
        _str.reserve(size);
    }

    basic_string_maker_impl(const basic_string_maker_impl&) = delete;
    basic_string_maker_impl(basic_string_maker_impl&&) = delete;

    ~basic_string_maker_impl() = default;

    string_type finish()
    {
        this->do_finish();
        return std::move(_str);
    }

private:

    template <typename, bool, typename>
    friend class detail::string_writer_mixin;

    void _append(const CharT* begin, const CharT* end)
    {
        _str.append(begin, end);
    }

    string_type _str;
};

} // namespace detail

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT>  >
class basic_string_appender_noexcept final
    : public stringify::v0::detail::basic_string_appender_impl
        < true, CharT, Traits, Allocator >
{
public:

    using stringify::v0::detail::basic_string_appender_impl
        < true, CharT, Traits, Allocator >
        ::basic_string_appender_impl;

    void recycle() noexcept(true) override
    {
        this->do_recycle();
    }
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_appender final
    : public stringify::v0::detail::basic_string_appender_impl
        < false, CharT, Traits, Allocator >
{
public:

    using stringify::v0::detail::basic_string_appender_impl
        < false, CharT, Traits, Allocator >
        ::basic_string_appender_impl;

    void recycle() override
    {
        this->do_recycle();
    }
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker_noexcept final
    : public stringify::v0::detail::basic_string_maker_impl
        < true, CharT, Traits, Allocator >
{
public:

    using stringify::v0::detail::basic_string_maker_impl
        < true, CharT, Traits, Allocator >
        ::basic_string_maker_impl;

    void recycle() noexcept(true) override
    {
        this->do_recycle();
    }
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker final
    : public stringify::v0::detail::basic_string_maker_impl
        < false, CharT, Traits, Allocator >
{
public:

    using stringify::v0::detail::basic_string_maker_impl
        < false, CharT, Traits, Allocator >
        ::basic_string_maker_impl;

    void recycle() override
    {
        this->do_recycle();
    }
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class pre_sized_basic_string_maker final
    : public stringify::v0::basic_outbuf<CharT>
{
public:

    // pre_sized_basic_string_maker()
    //     : stringify::v0::basic_outbuf<CharT>
    //         ( stringify::v0::outbuf_garbage_buf<CharT>()
    //         , stringify::v0::outbuf_garbage_buf_end<CharT>() )
    // {
    //     _str.append(_str.capacity(), (CharT)0);
    //     this->set_pos(_str.data());
    //     this->set_end(_str.data() + _str.size());
    // }

    pre_sized_basic_string_maker(std::size_t count)
        : stringify::v0::basic_outbuf<CharT>
            ( stringify::v0::outbuf_garbage_buf<CharT>()
            , stringify::v0::outbuf_garbage_buf_end<CharT>() )
        , _str(count, (CharT)0)
    {
        this->set_pos(&*_str.begin());
        this->set_end(&*_str.begin() + count);
    }

    void recycle() override
    {
        std::size_t original_size = this->pos() - _str.data();
        auto append_size = std::max
            ( original_size
            , stringify::v0::min_size_after_recycle<CharT>() );
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

using string_appender_noexcept = basic_string_appender_noexcept<char>;
using u16string_appender_noexcept = basic_string_appender_noexcept<char16_t>;
using u32string_appender_noexcept = basic_string_appender_noexcept<char32_t>;
using wstring_appender_noexcept = basic_string_appender_noexcept<wchar_t>;

using string_maker_noexcept = basic_string_maker_noexcept<char>;
using u16string_maker_noexcept = basic_string_maker_noexcept<char16_t>;
using u32string_maker_noexcept = basic_string_maker_noexcept<char32_t>;
using wstring_maker_noexcept = basic_string_maker_noexcept<wchar_t>;

#if defined(__cpp_char8_t)

using u8string_appender = basic_string_appender<char8_t>;
using u8string_maker = basic_string_maker<char8_t>;
using u8string_appender_noexcept = basic_string_appender_noexcept<char8_t>;
using u8string_maker_noexcept = basic_string_maker_noexcept<char8_t>;

#endif


namespace detail {

template <typename CharT, typename Traits, typename Allocator>
class basic_string_appender_creator
{
public:

    using char_type = CharT;
    using finish_type = void;

    basic_string_appender_creator
        ( std::basic_string<CharT, Traits, Allocator>& str )
        : _str(str)
    {
    }

    basic_string_appender_creator(const basic_string_appender_creator&) = default;

    template <typename ... Printers>
    void write(const Printers& ... printers) const
    {
        stringify::v0::basic_string_appender<CharT, Traits, Allocator> ob(_str);
        stringify::v0::detail::write_args(ob, printers...);;
        ob.finish();
    }

    template <typename ... Printers>
    void sized_write(std::size_t size, const Printers& ... printers) const
    {
        _str.reserve(_str.size() + size);
        stringify::v0::basic_string_appender<CharT, Traits, Allocator> ob(_str);
        stringify::v0::detail::write_args(ob, printers...);;
        ob.finish();
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

    template <typename ... Printers>
    finish_type write(const Printers& ... printers) const
    {
        stringify::v0::basic_string_maker<CharT, Traits, Allocator> ob;
        stringify::v0::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    template <typename ... Printers>
    finish_type sized_write(std::size_t size, const Printers& ... printers) const
    {
        stringify::v0::pre_sized_basic_string_maker<CharT, Traits, Allocator>
            ob(size);
        stringify::v0::detail::write_args(ob, printers...);;
        return ob.finish();
    }
};

}

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    return stringify::v0::dispatcher_no_reserve
        < stringify::v0::detail::basic_string_appender_creator<CharT, Traits, Allocator> >
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
constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<CharT, Traits, Allocator> >
    to_basic_string{};

#if defined(__cpp_char8_t)

constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<char8_t> >
    to_u8string{};

#endif

constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<char> >
    to_string{};

constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<char16_t> >
    to_u16string{};

constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<char32_t> >
    to_u32string{};

constexpr stringify::v0::dispatcher_no_reserve
    < stringify::v0::detail::basic_string_maker_creator<wchar_t> >
    to_wstring{};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

