#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/syntax.hpp>
#include <boost/stringify/v0/expected.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template<typename CharOut>
class char_ptr_writer: public buffer_recycler<CharOut>
{
    using Traits = std::char_traits<CharOut>;

public:

    using char_type = CharOut;

    char_ptr_writer(CharOut* destination, CharOut* end )
        : m_begin{destination}
        , m_end{end}
    {
        BOOST_ASSERT(m_begin <= m_end);
    }

    ~char_ptr_writer()
    {
    }

    stringify::v0::expected_buff_it<CharOut> start() noexcept
    {
        return { boost::stringify::v0::in_place_t{}
               , stringify::v0::buff_it<CharOut>{m_begin, m_end - 1} };
    }
   
    stringify::v0::expected_buff_it<CharOut> recycle(CharOut* it) override
    {
        BOOST_ASSERT(it < m_end);
        (void) it;
        return { stringify::v0::unexpect_t{}
               , std::make_error_code(std::errc::result_out_of_range) };
    }

    stringify::v0::expected<std::size_t, std::error_code> finish(CharOut *it) noexcept
    {
        BOOST_ASSERT(it < m_end);
        *it = 0;
        return { boost::stringify::v0::in_place_t{}, it - m_begin };
    }
 
private:

    CharOut* m_begin;
    CharOut* m_end;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<wchar_t>;

#endif


} // namespace detail


template<std::size_t N>
auto write(char (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(char16_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(char32_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(wchar_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*>
        (destination, destination + N);
}

inline auto write(char* destination, char* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (destination, end);
}

inline auto write(char16_t* destination, char16_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (destination, end);
}

inline auto write(char32_t* destination, char32_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (destination, end);
}

inline auto write(wchar_t* destination, wchar_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (destination, end);
}

inline auto write(char* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (destination, destination + count);
}

inline auto write(char16_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (destination, destination + count);
}

inline auto write(char32_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (destination, destination + count);
}

inline auto write(wchar_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (destination, destination + count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP */

