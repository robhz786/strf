#ifndef STRF_TO_STRING_HPP
#define STRF_TO_STRING_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/destination.hpp>
#include <strf.hpp>
#include <string>

namespace strf {

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_appender final: public strf::destination<CharT>
{
    using string_type_ = std::basic_string<CharT, Traits, Allocator>;

public:

    explicit basic_string_appender(string_type_& str)
        : strf::destination<CharT>(buf_, buf_size_)
        , str_(str)
    {
    }
    basic_string_appender(string_type_& str, std::size_t size )
        : strf::destination<CharT>(buf_, buf_size_)
        , str_(str)
    {
        str_.reserve(size);
    }

    basic_string_appender() = delete;
    ~basic_string_appender() override = default;
    basic_string_appender(const basic_string_appender&) = delete;
    basic_string_appender(basic_string_appender&&) = delete;
    basic_string_appender& operator=(const basic_string_appender&) = delete;
    basic_string_appender& operator=(basic_string_appender&&) = delete;

    void recycle() override
    {
        auto * p = this->buffer_ptr();
        this->set_buffer_ptr(buf_);
        STRF_IF_LIKELY (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
            this->set_good(true);
        }
    }

    void finish()
    {
        auto * p = this->buffer_ptr();
        STRF_IF_LIKELY (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
        }
    }

private:

    void do_write(const CharT* str, std::size_t str_len) override
    {
        auto * p = this->buffer_ptr();
        this->set_buffer_ptr(buf_);
        STRF_IF_LIKELY (this->good()) {
            this->set_good(false);
            str_.append(buf_, p);
            str_.append(str, str_len);
            this->set_good(true);
        }
    }

    string_type_& str_;
    static constexpr std::size_t buf_size_
        = strf::min_destination_buffer_size;
    CharT buf_[buf_size_];
};

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_string_maker final: public strf::destination<CharT>
{
    using string_type_ = std::basic_string<CharT, Traits, Allocator>;

public:

    basic_string_maker()
        : strf::destination<CharT>(buf_, buf_size_)
    {
    }

    explicit basic_string_maker(strf::tag<void>)
        : basic_string_maker()
    {
    }

    basic_string_maker(const basic_string_maker&) = delete;
    basic_string_maker(basic_string_maker&&) = delete;
    basic_string_maker& operator=(const basic_string_maker&) = delete;
    basic_string_maker& operator=(basic_string_maker&&) = delete;

    ~basic_string_maker() override
    {
        if (string_initialized_) {
            string_ptr()->~string_type_();
        }
    }

    void recycle() override;

    string_type_ finish()
    {
        STRF_IF_LIKELY (this->good()) {
            this->set_good(false);
            const std::size_t count = detail::safe_cast_size_t(this->buffer_ptr() - buf_);
            STRF_IF_LIKELY ( ! string_initialized_) {
                return {buf_, count};
            }
            string_ptr() -> append(buf_, count);
            return std::move(*string_ptr());
        }
        return {};
    }

private:

    void do_write(const CharT* str, std::size_t str_len) override;

    bool string_initialized_ = false;

    string_type_* string_ptr()
    {
        void* ptr = &string_obj_space_;
        return reinterpret_cast<string_type_*>(ptr);
    }

    static constexpr std::size_t buf_size_ = strf::min_destination_buffer_size;
    CharT buf_[buf_size_];

    using string_storage_type_ = typename std::aligned_storage
        < sizeof(string_type_), alignof(string_type_) >
        :: type;

    string_storage_type_ string_obj_space_;
};

template < typename CharT, typename Traits, typename Allocator >
void basic_string_maker<CharT, Traits, Allocator>::recycle()
{
    const std::size_t count = detail::safe_cast_size_t(this->buffer_ptr() - buf_);
    this->set_buffer_ptr(buf_);
    STRF_IF_LIKELY (this->good()) {
        this->set_good(false); // in case the following code throws
        if ( ! string_initialized_) {
            new (string_ptr()) string_type_{buf_, count};
            string_initialized_ = true;
        } else {
            string_ptr() -> append(buf_, count);
        }
        this->set_good(true);
    }
}

template < typename CharT, typename Traits, typename Allocator >
void basic_string_maker<CharT, Traits, Allocator>::do_write(const CharT* str, std::size_t str_len)
{
    STRF_IF_LIKELY (this->good()) {
        const std::size_t buf_count = detail::safe_cast_size_t(this->buffer_ptr() - buf_);
        this->set_buffer_ptr(buf_);
        this->set_good(false); // in case the following code throws
        if ( ! string_initialized_) {
            new (string_ptr()) string_type_();
            string_ptr()->reserve((buf_count + str_len) << 1);
            string_ptr()->append(buf_, buf_count);
            string_ptr()->append(str, str_len);
            string_initialized_ = true;
        } else {
            string_ptr()->append(buf_, buf_count);
            string_ptr()->append(str, str_len);
        }
        this->set_good(true);
    }
}

template < typename CharT
         , typename Traits = std::char_traits<CharT>
         , typename Allocator = std::allocator<CharT> >
class basic_sized_string_maker final
    : public strf::destination<CharT>
{
public:

    explicit basic_sized_string_maker(std::size_t count)
        : strf::destination<CharT>(nullptr, nullptr)
        , str_(count ? count : 1, (CharT)0)
    {
        this->set_buffer_ptr(&*str_.begin());
        this->set_buffer_end(&*str_.begin() + (count ? count : 1));
    }
    basic_sized_string_maker() = delete;
    ~basic_sized_string_maker() override = default;
    basic_sized_string_maker(const basic_sized_string_maker&) = delete;
    basic_sized_string_maker(basic_sized_string_maker&&) = delete;
    basic_sized_string_maker& operator=(const basic_sized_string_maker&) = delete;
    basic_sized_string_maker& operator=(basic_sized_string_maker&&) = delete;

    void recycle() override
    {
        const auto original_size = detail::safe_cast_size_t(this->buffer_ptr() - str_.data());
        constexpr std::size_t min_buff_size = strf::min_destination_buffer_size;
        auto append_size = strf::detail::max<std::size_t>(original_size, min_buff_size);
        str_.append(append_size, (CharT)0);
        this->set_buffer_ptr(&*str_.begin() + original_size);
        this->set_buffer_end(&*str_.begin() + original_size + append_size);
    }

    std::basic_string<CharT, Traits, Allocator> finish()
    {
        str_.resize(detail::safe_cast_size_t(this->buffer_ptr() - str_.data()));
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

using sized_string_maker = basic_sized_string_maker<char>;
using sized_u16string_maker = basic_sized_string_maker<char16_t>;
using sized_u32string_maker = basic_sized_string_maker<char32_t>;
using sized_wstring_maker = basic_sized_string_maker<wchar_t>;

#if defined(__cpp_char8_t)

using u8string_appender = basic_string_appender<char8_t>;
using u8string_maker = basic_string_maker<char8_t>;
using pre_sized_u8string_maker = basic_sized_string_maker<char8_t>;

#endif

namespace detail {

template <typename CharT, typename Traits, typename Allocator>
class basic_string_appender_creator
{
public:

    using char_type = CharT;
    using destination_type = strf::basic_string_appender<CharT, Traits, Allocator>;
    using sized_destination_type = destination_type;
    using finish_type = void;

    explicit basic_string_appender_creator
        ( std::basic_string<CharT, Traits, Allocator>& str )
        : str_(str)
    {
    }

    std::basic_string<CharT, Traits, Allocator>& create() const noexcept
    {
        return str_;
    }
    std::basic_string<CharT, Traits, Allocator>& create(std::size_t size) const noexcept
    {
        if (size > 0) {
            str_.reserve(str_.size() + size);
        }
        return str_;
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
    using destination_type = strf::basic_string_maker<CharT, Traits, Allocator>;
    using sized_destination_type = strf::basic_sized_string_maker<CharT, Traits, Allocator>;

    strf::tag<void> create() const noexcept
    {
        return strf::tag<void>{};
    }
    std::size_t create(std::size_t size) const noexcept
    {
        return size;
    }
};

} // namespace detail

template <typename CharT, typename Traits, typename Allocator>
inline auto append(std::basic_string<CharT, Traits, Allocator>& str)
    -> strf::printing_syntax
        < strf::detail::basic_string_appender_creator<CharT, Traits, Allocator> >
{
    return strf::make_printing_syntax
        ( strf::detail::basic_string_appender_creator<CharT, Traits, Allocator>{str} );
}

template <typename CharT, typename Traits, typename Allocator>
inline auto assign(std::basic_string<CharT, Traits, Allocator>& str)
    -> strf::printing_syntax
        < strf::detail::basic_string_appender_creator<CharT, Traits, Allocator> >
{
    str.clear();
    return append(str);
}

#if defined(STRF_HAS_VARIABLE_TEMPLATES)

template< typename CharT
        , typename Traits = std::char_traits<CharT>
        , typename Allocator = std::allocator<CharT> >
constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<CharT, Traits, Allocator> >
    to_basic_string{};

#endif // defined(STRF_HAS_VARIABLE_TEMPLATES)

#if defined(__cpp_char8_t)

constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<char8_t> >
    to_u8string{};

#endif

constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<char> >
    to_string{};

constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<char16_t> >
    to_u16string{};

constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<char32_t> >
    to_u32string{};

constexpr strf::printing_syntax
    < strf::detail::basic_string_maker_creator<wchar_t> >
    to_wstring{};


} // namespace strf

#endif // STRF_TO_STRING_HPP

