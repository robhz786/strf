#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/make_destination.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template<typename CharOut>
class ec_char_ptr_writer: public output_buffer<CharOut>
{
    using Traits = std::char_traits<CharOut>;

public:

    using char_type = CharOut;

    ec_char_ptr_writer
        ( CharOut* dest
        , CharOut* dest_end
        , std::size_t* count )
        : output_buffer<CharOut>{dest, dest_end - 1}
        , _begin(dest)
        , _count_ptr{count}
    {
        BOOST_ASSERT(dest < dest_end);
    }

    ~ec_char_ptr_writer()
    {
        * this->pos() = 0;
        if (_count_ptr != nullptr)
        {
            *_count_ptr = this->pos() - _begin;
        }
    }

    bool recycle() override;

    stringify::v0::nodiscard_error_code finish() noexcept;

private:

    CharOut* _begin;
    std::size_t* _count_ptr = nullptr;
};

template<typename CharOut>
bool ec_char_ptr_writer<CharOut>::recycle()
{
    this->set_error(std::errc::result_out_of_range);
    return false;
}

template<typename CharOut>
inline stringify::v0::nodiscard_error_code
ec_char_ptr_writer<CharOut>::finish() noexcept
{

    * this->pos() = 0;
    return this->get_error();
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_char_ptr_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_char_ptr_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_char_ptr_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_char_ptr_writer<wchar_t>;

#endif

} // namespace detail

template<std::size_t N>
inline auto ec_write(char (&dest)[N], std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*, std::size_t*>
        (dest, dest + N, count_ptr);
}

template<std::size_t N>
inline auto ec_write(char16_t (&dest)[N], std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*, std::size_t*>
        (dest, dest + N, count_ptr);
}

template<std::size_t N>
inline auto ec_write(char32_t (&dest)[N], std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*, std::size_t*>
        (dest, dest + N, count_ptr);
}

template<std::size_t N>
inline auto ec_write(wchar_t (&dest)[N], std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*, std::size_t*>
        (dest, dest + N, count_ptr);
}

inline auto ec_write(char* dest, char* end, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*, std::size_t*>
        (dest, end, count_ptr);
}

inline auto ec_write(char16_t* dest, char16_t* end, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*, std::size_t*>
        (dest, end, count_ptr);
}

inline auto ec_write(char32_t* dest, char32_t* end, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*, std::size_t*>
        (dest, end, count_ptr);
}

inline auto ec_write(wchar_t* dest, wchar_t* end, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*, std::size_t*>
        (dest, end, count_ptr);
}

inline auto ec_write(char* dest, std::size_t count, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*, std::size_t*>
        (dest, dest + count, count_ptr);
}

inline auto ec_write(char16_t* dest, std::size_t count, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*, std::size_t*>
        (dest, dest + count, count_ptr);
}

inline auto ec_write(char32_t* dest, std::size_t count, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*, std::size_t*>
        (dest, dest + count, count_ptr);
}

inline auto ec_write(wchar_t* dest, std::size_t count, std::size_t* count_ptr = nullptr)
{
    using writer = stringify::v0::detail::ec_char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*, std::size_t*>
        (dest, dest + count, count_ptr);
}

#if ! defined(BOOST_NO_EXCEPTION)

namespace detail{

template<typename CharOut>
class char_ptr_writer: public output_buffer<CharOut>
{
    using Traits = std::char_traits<CharOut>;

public:

    using char_type = CharOut;

    char_ptr_writer(CharOut* dest, CharOut* end )
        : output_buffer<CharOut>{dest, end - 1}
        , _begin(dest)
    {
        BOOST_ASSERT(dest < end);
    }

    ~char_ptr_writer()
    {
        if ( ! _finish_called)
        {
            * _begin = 0;
        }
    }

    bool recycle() override;

    std::size_t finish();

private:

    CharOut* _begin;
    bool _finish_called = false;
};

template<typename CharOut>
bool char_ptr_writer<CharOut>::recycle()
{
    this->set_error(std::errc::result_out_of_range);
    return false;
}

template<typename CharOut>
inline std::size_t char_ptr_writer<CharOut>::finish()
{
    _finish_called = true;
    if ( this->has_error() )
    {
        *_begin = 0;
        throw stringify::v0::stringify_error{this->get_error()};
    }
    * this->pos() = 0;
    return this->pos() - _begin;
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<wchar_t>;

#endif


} // namespace detail

template<std::size_t N>
inline auto write(char (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*>
        (dest, dest + N);
}

template<std::size_t N>
inline auto write(char16_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*>
        (dest, dest + N);
}

template<std::size_t N>
inline auto write(char32_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*>
        (dest, dest + N);
}

template<std::size_t N>
inline auto write(wchar_t (&dest)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*>
        (dest, dest + N);
}

inline auto write(char* dest, char* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (dest, end);
}

inline auto write(char16_t* dest, char16_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (dest, end);
}

inline auto write(char32_t* dest, char32_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (dest, end);
}

inline auto write(wchar_t* dest, wchar_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (dest, end);
}

inline auto write(char* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (dest, dest + count);
}

inline auto write(char16_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (dest, dest + count);
}

inline auto write(char32_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (dest, dest + count);
}

inline auto write(wchar_t* dest, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (dest, dest + count);
}

#endif // ! defined(BOOST_NO_EXCEPTION)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP */

