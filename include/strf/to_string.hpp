#ifndef STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuf.hpp>
#include <strf.hpp>
#include <string>

namespace strf {

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_appender final: public strf::basic_outbuf<CharT>
{
    using string_type_ = std::basic_string<CharT, Traits, Allocator>;

public:

    basic_string_appender(string_type_& str)
        : strf::basic_outbuf<CharT>(buf_, buf_size_)
        , str_(str)
    {
    }
    basic_string_appender( string_type_& str
                         , std::size_t size )
        : strf::basic_outbuf<CharT>(buf_, buf_size_)
        , str_(str)
    {
        str_.reserve(size);
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_appender(basic_string_appender&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_string_appender(const basic_string_appender&) = delete;
    basic_string_appender(basic_string_appender&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    void recycle() override
    {
        auto * p = this->pointer();
        this->set_pointer(buf_);
        if (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
            this->set_good(true);
        }
    }

    void finish()
    {
        auto * p = this->pointer();
        if (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
        }
    }

private:

    string_type_& str_;
    static constexpr std::size_t buf_size_
        = strf::min_size_after_recycle<sizeof(CharT)>();
    CharT buf_[buf_size_];
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker final: public strf::basic_outbuf<CharT>
{
    using string_type_ = std::basic_string<CharT, Traits, Allocator>;

public:

    basic_string_maker() : strf::basic_outbuf<CharT>(buf_, buf_size_)
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
        auto * p = this->pointer();
        this->set_pointer(buf_);
        if (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
            this->set_good(true);
        }
    }

    string_type_ finish()
    {
        auto * p = this->pointer();
        if (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
        }
        return std::move(str_);
    }

private:

    string_type_ str_;
    static constexpr std::size_t buf_size_
        = strf::min_size_after_recycle<sizeof(CharT)>();
    CharT buf_[buf_size_];
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
        , str_(count, (CharT)0)
    {
        this->set_pointer(&*str_.begin());
        this->set_end(&*str_.begin() + count);
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    basic_pre_sized_string_maker(basic_pre_sized_string_maker&&);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    basic_pre_sized_string_maker(const basic_pre_sized_string_maker&) = delete;
    basic_pre_sized_string_maker(basic_pre_sized_string_maker&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    void recycle() override
    {
        std::size_t original_size = this->pointer() - str_.data();
        auto append_size = std::max
            ( original_size
            , strf::min_size_after_recycle<sizeof(CharT)>() );
        str_.append(append_size, (CharT)0);
        this->set_pointer(&*str_.begin() + original_size);
        this->set_end(&*str_.begin() + original_size + append_size);
    }

    std::basic_string<CharT, Traits, Allocator> finish()
    {
        str_.resize(this->pointer() - str_.data());
        return std::move(str_);
    }

private:

    std::basic_string<CharT, Traits, Allocator> str_;
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
        : str_(str)
    {
    }

    basic_string_appender_creator(const basic_string_appender_creator&) = default;

    outbuf_type create() const
    {
        return outbuf_type{str_};
    }

    outbuf_type create(std::size_t size) const
    {
        str_.reserve(str_.size() + size);
        return outbuf_type{str_};
    }

private:

    std::basic_string<CharT, Traits, Allocator>& str_;
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


} // namespace strf

#endif  // STRF_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

